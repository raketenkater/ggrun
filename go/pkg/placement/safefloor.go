package placement

import (
	"fmt"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// The safe floor is the answer to a structural gap: ggrun is fail-closed at
// every stage and had nowhere to stand when no stage could prove a placement
// safe. Three separate terminal failures on one rig in one day -- an unmeasured
// hot-expert baseline, a CUDA OOM inside the verification canary, and a
// wrong-role compute reading that over-packed a card -- were each a PREDICTION
// error escalated into "does not launch at all".
//
// This is deliberately a serviceability mechanism, not an evidence mechanism.
// The probe cache is keyed by model|ctx|ubatch|kvQuality|kvPlacement|backend|
// gpuSig|parallel, so a floor that moves any of those coordinates writes a
// DIFFERENT key and teaches the requested plan nothing; and llama-fit-params
// already measures per-GPU model/context/compute for any argv in about a second
// without loading. The one quantity a real floor launch uniquely produces is
// runtime graph growth (RecordRuntimeGraphGrowth*), which no oracle predicts --
// and RelatedModelRuntimeGraphGrowth only transfers it when the slot count
// matches. That is why Parallel is never surrendered by any rung below.

// SafeFloorConstraints names the coordinates the user pinned explicitly.
//
// Contract invariant 3 -- "explicit user choices are constraints; automatic
// search may move only coordinates the user left automatic" -- is what makes an
// automatic floor legal at all. Moving an automatic coordinate needs no opt-in;
// surrendering a pinned one requires AllowExplicit, and without it the floor
// refuses that rung and the launch fails as it did before.
type SafeFloorConstraints struct {
	ContextExplicit     bool
	KVQualityExplicit   bool
	KVPlacementExplicit bool
	UBatchExplicit      bool
	BatchExplicit       bool
	HotExpertsExplicit  bool
	SpecExplicit        bool
	SWAFullExplicit     bool
	// AllowExplicit permits a rung to surrender a coordinate the user pinned.
	//
	// Callers set it at a TERMINAL stage, where the alternative is not launching
	// at all. Invariant 3 protects a user's explicit choice from being quietly
	// traded away by an optimiser looking for speed; it is not a licence to
	// refuse to serve. Somebody who asked for hot experts wants a server without
	// them far more than they want no server -- so the floor surrenders the pin,
	// names it in the banner, and serves. The --safe-floor flag pre-authorises
	// the same thing for non-terminal callers. Parallel is never surrendered.
	AllowExplicit bool
}

// SafeFloorRung describes the tier a safe-floor placement came from, so the
// launcher can name every coordinate it gave up rather than degrade silently.
type SafeFloorRung struct {
	Name       string
	Surrenders []string
}

// String renders the rung for the launch banner and the activation record.
func (r SafeFloorRung) String() string {
	if len(r.Surrenders) == 0 {
		return r.Name
	}
	return r.Name + " (" + strings.Join(r.Surrenders, ", ") + ")"
}

// safeFloorTier is one rung: a pinned-Options mutation plus the coordinates it
// surrenders and the explicit-pin checks that would block it.
type safeFloorTier struct {
	name string
	// blockedBy reports whether this rung needs a coordinate the user pinned.
	blockedBy func(SafeFloorConstraints) []string
	// apply pins the rung's coordinates onto a copy of the request options.
	apply func(*Options, *detect.Capabilities, *ModelProfile)
	// surrenders lists what this rung gives up, for the banner.
	surrenders []string
}

// ComputeSafeFloor descends a bounded ladder of progressively more conservative
// placements and returns the first one Compute can resolve. Every rung is a
// complete placement produced by the ordinary Compute path -- never a partial
// argv overlay (invariant 2) and never a static fudge reserve (invariant 8) --
// so the caller's exact-argv admission still gates it exactly as it gates any
// other placement.
//
// The last rung is CPU-only. It is the only tier that is near-certain on an
// arbitrary rig: every GPU rung still depends on what companions occupy and on
// the driver-side CUDA-graph executable that no no-alloc oracle predicts.
func ComputeSafeFloor(caps *detect.Capabilities, model *ModelProfile, opts Options, cons SafeFloorConstraints) (*Strategy, SafeFloorRung, error) {
	if model == nil {
		return nil, SafeFloorRung{}, fmt.Errorf("safe floor needs a model profile")
	}
	var refused []string
	for _, tier := range safeFloorTiers() {
		if blocked := tier.blockedBy(cons); len(blocked) > 0 && !cons.AllowExplicit {
			refused = append(refused, fmt.Sprintf("%s needs %s", tier.name, strings.Join(blocked, "+")))
			continue
		}
		next := opts
		// Never reuse a cached placement for a floor: the whole point is to
		// derive a fresh conservative layout, and the cached one is the layout
		// that just failed.
		next.SkipPlacementCache = true
		tier.apply(&next, caps, model)
		strategy, err := Compute(caps, model, next)
		if err != nil || strategy == nil {
			continue
		}
		// A floor placement is a serviceability decision, never a measured
		// winner: it must not suppress the optimizer or be replayed as proof
		// (invariant 7).
		strategy.PerformanceTuned = false
		strategy.VerifiedConfigReused = false
		strategy.BatchTuned = false
		strategy.ContextFitTier = safeFloorContextTier
		return strategy, SafeFloorRung{Name: tier.name, Surrenders: tier.surrenders}, nil
	}
	if len(refused) > 0 {
		return nil, SafeFloorRung{}, fmt.Errorf(
			"no safe-floor placement is available without overriding an explicit setting (%s); re-run with --safe-floor to allow it",
			strings.Join(refused, "; "))
	}
	return nil, SafeFloorRung{}, fmt.Errorf("no safe-floor placement could be resolved for this model and hardware")
}

// safeFloorContextTier tags a floor placement's evidence so a record produced
// under it is never mistaken for an ordinary automatic-context decision.
const safeFloorContextTier = "safe-floor"

// IsSafeFloorStrategy reports whether a strategy came from the safe-floor
// ladder. Callers use it to refuse promotion and to keep the banner truthful.
func IsSafeFloorStrategy(s *Strategy) bool {
	return s != nil && s.ContextFitTier == safeFloorContextTier
}

func safeFloorTiers() []safeFloorTier {
	// Every rung inherits the previous rung's concessions, so the ladder is
	// monotonically more conservative. Parallel is absent from all of them by
	// design (see the package comment).
	base := func(o *Options, caps *detect.Capabilities, cons SafeFloorConstraints) {
		// One split owner: firstLaunchComputeBufMBForGPUParallel charges EVERY
		// layer-owning device a full ubatch graph, so collapsing to a single
		// owner is the largest single reduction available.
		if owner := largestVRAMGPUIndex(caps); owner >= 0 {
			ownerCopy := owner
			o.MoESplitOwnerGPU = &ownerCopy
		}
		if !cons.UBatchExplicit {
			o.UBatchSize = safeFloorUBatch
		}
		if !cons.BatchExplicit {
			o.BatchSize = safeFloorUBatch
		}
		if !cons.HotExpertsExplicit {
			o.HotExperts = "off"
			o.HotExpertCacheSlots = 0
		}
		if !cons.SpecExplicit {
			o.SpecMode = "off"
		}
		if !cons.SWAFullExplicit {
			o.SWAFull = false
		}
	}
	return []safeFloorTier{
		{
			name:       "single-owner",
			surrenders: []string{"multi-GPU layer split", "hot experts", "speculative decode", "microbatch 64"},
			blockedBy: func(c SafeFloorConstraints) []string {
				return pinnedAmong(c, "ubatch", "batch", "hot-experts", "spec")
			},
			apply: func(o *Options, caps *detect.Capabilities, _ *ModelProfile) {
				base(o, caps, SafeFloorConstraints{})
			},
		},
		{
			name:       "no-gpu-companion",
			surrenders: []string{"the above", "GPU-seated reviewer/companion"},
			blockedBy: func(c SafeFloorConstraints) []string {
				return pinnedAmong(c, "ubatch", "batch", "hot-experts", "spec")
			},
			apply: func(o *Options, caps *detect.Capabilities, _ *ModelProfile) {
				base(o, caps, SafeFloorConstraints{})
				o.Companions = nil
			},
		},
		{
			name:       "kv-host",
			surrenders: []string{"the above", "GPU-resident KV"},
			blockedBy: func(c SafeFloorConstraints) []string {
				return pinnedAmong(c, "ubatch", "batch", "hot-experts", "spec", "kv-placement")
			},
			apply: func(o *Options, caps *detect.Capabilities, _ *ModelProfile) {
				base(o, caps, SafeFloorConstraints{})
				o.Companions = nil
				o.KVPlacement = "cpu"
			},
		},
		{
			name:       "min-context",
			surrenders: []string{"the above", "requested context"},
			blockedBy: func(c SafeFloorConstraints) []string {
				return pinnedAmong(c, "ubatch", "batch", "hot-experts", "spec", "kv-placement", "context")
			},
			apply: func(o *Options, caps *detect.Capabilities, _ *ModelProfile) {
				base(o, caps, SafeFloorConstraints{})
				o.Companions = nil
				o.KVPlacement = "cpu"
				o.ContextSize = contextMinimum
				o.AutoContextMax = contextMinimum
			},
		},
		{
			name:       "cpu-only",
			surrenders: []string{"the above", "all GPU offload"},
			blockedBy: func(c SafeFloorConstraints) []string {
				return pinnedAmong(c, "ubatch", "batch", "hot-experts", "spec", "kv-placement", "context")
			},
			apply: func(o *Options, caps *detect.Capabilities, _ *ModelProfile) {
				base(o, caps, SafeFloorConstraints{})
				o.Companions = nil
				o.KVPlacement = "cpu"
				o.ContextSize = contextMinimum
				o.AutoContextMax = contextMinimum
				o.CPUMode = true
			},
		},
	}
}

// safeFloorUBatch is llama.cpp's smallest practical microbatch. It is not a
// tuning choice: the floor's only job is to load.
const safeFloorUBatch = 64

func pinnedAmong(c SafeFloorConstraints, names ...string) []string {
	var out []string
	for _, n := range names {
		switch n {
		case "ubatch":
			if c.UBatchExplicit {
				out = append(out, "--ubatch-size")
			}
		case "batch":
			if c.BatchExplicit {
				out = append(out, "--batch-size")
			}
		case "hot-experts":
			if c.HotExpertsExplicit {
				out = append(out, "--hot-experts")
			}
		case "spec":
			if c.SpecExplicit {
				out = append(out, "--spec")
			}
		case "kv-placement":
			if c.KVPlacementExplicit {
				out = append(out, "--kv-placement")
			}
		case "context":
			if c.ContextExplicit {
				out = append(out, "--ctx-size")
			}
		}
	}
	return out
}

// largestVRAMGPUIndex returns the CUDA index of the card with the most free
// VRAM, or -1 when there is no usable GPU. Free rather than total: a companion
// already seated on a card is exactly what makes it the wrong split owner.
func largestVRAMGPUIndex(caps *detect.Capabilities) int {
	if caps == nil || len(caps.GPUs) == 0 {
		return -1
	}
	best, bestFree := -1, -1
	for _, g := range caps.GPUs {
		if free := g.VRAMFreeMB(); free > bestFree {
			best, bestFree = g.Index, free
		}
	}
	return best
}
