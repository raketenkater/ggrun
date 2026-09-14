package main

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// launchMemoryRecovery is the single state machine shared by contained
// preflight and real-startup allocation recovery. The two paths produce
// different evidence, but neither may resurrect an argv that this launch has
// already disproved.
type launchMemoryRecovery struct {
	rejected map[string]struct{}
	// rejectedContext is the smallest automatic context this launch has proven
	// does not fit. The argv identity ledger cannot carry this: a later
	// recompute from the original automatic request proposes a *different* argv
	// at a context already disproved, so no identity ever matches and the loop
	// climbs back into the rejected range until the re-plan budget expires.
	rejectedContext int
	// rejectedReclaimTokens is how far below rejectedContext the next proposal
	// must land, derived from the measured deficit via KV size per token.
	rejectedReclaimTokens int
	// acceptedContext is the smallest automatic context an exact preflight has
	// proven fits this launch. The backend-measured re-plan exists to refine
	// placement from measured buffers, never to spend that proof on a larger
	// context: observed twice on GLM-5.3-Flash, it discarded an accepted
	// 555,008-token plan for 562,176 and failed against the same limit that had
	// just rejected 563,200.
	acceptedContext int
	// acceptedUBatch is the smallest ubatch an exact preflight has proven fits.
	// recoverPreflightOOM already refuses to recompute a derated ubatch back up;
	// the measured re-plan did not, so a plan accepted at ubatch 128 came back
	// at 256 and overshot every device at the same context.
	acceptedUBatch int
}

func newLaunchMemoryRecovery() *launchMemoryRecovery {
	return &launchMemoryRecovery{rejected: map[string]struct{}{}}
}

func (r *launchMemoryRecovery) reject(args []string) {
	if r == nil {
		return
	}
	if r.rejected == nil {
		r.rejected = map[string]struct{}{}
	}
	r.rejected[launchArgsIdentity(args)] = struct{}{}
}

func (r *launchMemoryRecovery) isRejected(args []string) bool {
	if r == nil {
		return false
	}
	_, rejected := r.rejected[launchArgsIdentity(args)]
	return rejected
}

// rejectContext records an automatic context proven not to fit. An explicit
// context is a user constraint, not a coordinate this launch may move, so only
// automatic contexts are recorded.
//
// reclaimTokens is how far below the rejected context the next proposal has to
// land to have any chance of covering the measured deficit. Without it the
// ceiling sits one granule down, and the search creeps 1,024 tokens at a time:
// GLM-5.3-Flash at --parallel 2 went 870,400 -> 817,152 -> 816,128 against
// deficits of 2,763, 912 and 4,636 MiB and spent its whole budget getting
// nowhere. Pass 0 when the deficit cannot be converted to tokens; absent is
// not zero, so the ceiling then falls back to the one-granule step.
func (r *launchMemoryRecovery) rejectContext(strategy *placement.Strategy, reclaimTokens int) {
	if r == nil || strategy == nil || !strategy.ContextAuto || strategy.ContextSize <= 0 {
		return
	}
	if r.rejectedContext == 0 || strategy.ContextSize < r.rejectedContext {
		r.rejectedContext = strategy.ContextSize
		r.rejectedReclaimTokens = 0
	}
	// Only a reclaim for the context that currently sets the ceiling counts, and
	// only the largest such step: a later, bigger deficit at the same context
	// means the earlier step was not enough.
	if strategy.ContextSize == r.rejectedContext && reclaimTokens > r.rejectedReclaimTokens {
		r.rejectedReclaimTokens = reclaimTokens
	}
}

// contextReclaimTokens converts a measured VRAM deficit into the number of
// context tokens whose KV cache covers it. KV is the part of the memory a
// context change actually moves, so its size per token is the honest exchange
// rate. Returns 0 when the geometry is unknown rather than guessing.
func contextReclaimTokens(model *placement.ModelProfile, strategy *placement.Strategy, args []string, deficitMB int) int {
	if model == nil || strategy == nil || deficitMB <= 0 || strategy.ContextSize <= 0 {
		return 0
	}
	kvType := strategy.KVType
	if kvType == "" {
		kvType = effectiveMemoryArgValues(args)["cache-k"]
	}
	if kvType == "" {
		return 0
	}
	kvTotalMB := placement.EstimateKVCacheMB(model, strategy.ContextSize, kvType, hasArg(args, "--swa-full"))
	if kvTotalMB <= 0 {
		return 0
	}
	// tokens = deficit / (kvTotal / ctx), with the same safety margin the other
	// recovery levers require of themselves.
	required := recoveryRequiredMB(deficitMB)
	tokens := int(int64(required) * int64(strategy.ContextSize) / int64(kvTotalMB))
	if tokens <= 0 {
		return 0
	}
	if tokens >= strategy.ContextSize {
		tokens = strategy.ContextSize - 1
	}
	return tokens
}

// acceptContext records the shape an exact preflight proved fits. Only
// measured evidence may be recorded; a planner estimate is not proof.
func (r *launchMemoryRecovery) acceptContext(strategy *placement.Strategy) {
	if r == nil || strategy == nil {
		return
	}
	// ubatch is recorded whatever the context policy, because it is a compute
	// lever recovery derates on its own and the re-plan recomputes it from the
	// original automatic request otherwise.
	if strategy.UBatchSize > 0 && (r.acceptedUBatch == 0 || strategy.UBatchSize < r.acceptedUBatch) {
		r.acceptedUBatch = strategy.UBatchSize
	}
	if !strategy.ContextAuto || strategy.ContextSize <= 0 {
		return
	}
	if r.acceptedContext == 0 || strategy.ContextSize < r.acceptedContext {
		r.acceptedContext = strategy.ContextSize
	}
}

// automaticContextCeiling is the largest automatic context this launch may
// still propose, and it only ever ratchets down.
//
// A rejected context contributes a step below itself: at least one token, so
// that placement.Compute's granule floor drops the rejected value, and as much
// as the measured deficit requires when that is known. An accepted context
// contributes itself, because re-proposing exactly the plan that passed is the
// fixed point this loop is trying to reach.
func (r *launchMemoryRecovery) automaticContextCeiling() int {
	if r == nil {
		return 0
	}
	ceiling := 0
	if r.rejectedContext > 1 {
		step := maxPreflightInt(r.rejectedReclaimTokens, 1)
		if step >= r.rejectedContext {
			step = r.rejectedContext - 1
		}
		ceiling = r.rejectedContext - step
	}
	if r.acceptedContext > 0 && (ceiling == 0 || r.acceptedContext < ceiling) {
		ceiling = r.acceptedContext
	}
	return ceiling
}

// boundByProvenLimits constrains a recompute to the shapes this launch has not
// already disproved. It is the single place that applies them, so production
// and tests exercise the same rule rather than two descriptions of it.
//
// An explicit context is untouched: AutoContextMax caps only an automatic/fit
// context. ubatch is pinned the same way recoverPreflightOOM pins it, so a
// recompute cannot undo a derating this launch had to make.
func boundByProvenLimits(opts placement.Options, r *launchMemoryRecovery) placement.Options {
	if ceiling := r.automaticContextCeiling(); ceiling > 0 &&
		(opts.AutoContextMax <= 0 || ceiling < opts.AutoContextMax) {
		opts.AutoContextMax = ceiling
	}
	if r != nil && r.acceptedUBatch > 0 && (opts.UBatchSize <= 0 || r.acceptedUBatch < opts.UBatchSize) {
		opts.UBatchSize = r.acceptedUBatch
	}
	return opts
}

func (r *launchMemoryRecovery) hasRejections() bool {
	return r != nil && len(r.rejected) > 0
}

func (r *launchMemoryRecovery) rejectionCount() int {
	if r == nil {
		return 0
	}
	return len(r.rejected)
}

// recomputeDecision reports whether next changes the exact process argv and,
// if so, whether that argv has already failed this launch's memory checks.
func (r *launchMemoryRecovery) recomputeDecision(current, next []string) (changed, rejected bool) {
	if launchArgsIdentity(current) == launchArgsIdentity(next) {
		return false, false
	}
	if r == nil {
		return true, false
	}
	_, rejected = r.rejected[launchArgsIdentity(next)]
	return true, rejected
}

// launchArgsIdentity is deliberately the complete argv, not only the common
// CUDA flags: backend-specific switches such as full SWA cache and checkpoint
// modes also change memory. NUL is not valid inside an argv element, so this is
// collision-free for process arguments.
func launchArgsIdentity(args []string) string {
	return strings.Join(args, "\x00")
}

// recoverPreflightOOM converts one measured allocation failure into a strictly
// different launch shape. Failed graph allocations are useful component
// measurements, but are never promoted to complete allocation evidence.
func recoverPreflightOOM(
	req *launchRequest,
	cfg *config.Config,
	model *placement.ModelProfile,
	be *backendInfo,
	caps, runtimeCaps *detect.Capabilities,
	visibleToPhysical map[int]int,
	strategy *placement.Strategy,
	serverArgs []string,
	oomPenalty map[int]int,
	outcome preflightOutcome,
	recovery *launchMemoryRecovery,
) (*placement.Strategy, []string, string, error) {
	if !outcome.DoesNotFit || outcome.Device < 0 || model == nil || strategy == nil {
		return nil, nil, "", fmt.Errorf("invalid preflight allocation failure")
	}
	if outcome.AllocMB <= 0 {
		outcome.AllocMB = maxPreflightInt(outcome.DeficitMB, 1)
		outcome.AllocMBMeasured = false
	}
	if outcome.DeficitMB <= 0 {
		outcome.DeficitMB = 1
	}

	var candidate *placement.Strategy
	var replanErr error
	if outcome.IsComputeBuffer && outcome.AllocMBMeasured && cfg != nil && be != nil && runtimeCaps != nil {
		cacheBackendTag := scopedProbeBackendTagForStrategy(req, model, be, strategy)
		recordErr := placement.RecordMeasuredComputeBuffers(
			cfg.CacheDir, model, strategy.ContextSize, strategy.UBatchSize,
			strategy.KVQuality, strategy.KVPlacement, cacheBackendTag,
			runtimeCaps.GPUs, strategy.Parallel, map[int]int{outcome.Device: outcome.AllocMB},
		)
		if recordErr == nil {
			opts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
			// Preserve a prior derating: a retry at ubatch 256 must never be
			// recomputed from the original automatic request back to 512. The
			// same holds for context, which this ledger carries: at --parallel 2
			// this candidate climbed back to 524,288 after recovery had derated
			// to 301,056, and the launch spent its whole re-plan budget there.
			opts.UBatchSize = strategy.UBatchSize
			opts.SkipPlacementCache = true
			opts.CacheFile = ""
			opts = boundByProvenLimits(opts, recovery)
			candidate, replanErr = placement.Compute(caps, model, opts)
		}
	}

	if candidate == nil && !outcome.IsComputeBuffer {
		physicalDev := physicalGPUIndex(outcome.Device, visibleToPhysical)
		oomPenalty[physicalDev] += outcome.DeficitMB
		candidate, replanErr = placement.ReplanAfterOOM(
			caps, model,
			boundByProvenLimits(placementOptionsFromRequest(req, model, be, cfg.CacheDir), recovery),
			oomPenalty,
		)
	}

	var candidateArgs []string
	if candidate != nil {
		candidateArgs = buildLaunchServerArgs(req, cfg, be, caps, model, candidate)
	}
	// The no-allocation oracle reports a total device deficit, not a failed
	// compute allocation. Its fully recomputed automatic-context plan must not
	// be discarded by the partial-overlay selector below. Return it for the
	// caller's next exact preflight; this is a candidate, never fit proof.
	oracleReplanEligible := outcome.Evidence.Level == memoryEvidenceOraclePlanned &&
		req != nil && automaticContextRequest(req) && strategy.ContextAuto &&
		candidate != nil && candidate.ContextAuto &&
		candidate.ContextSize > 0 && candidate.ContextSize < strategy.ContextSize &&
		candidate.Parallel == strategy.Parallel && candidate.KVType == strategy.KVType &&
		candidate.KVTypeV == strategy.KVTypeV && candidate.KVPlacement == strategy.KVPlacement &&
		candidate.MMapRequired == strategy.MMapRequired && candidate.UBatchSize <= strategy.UBatchSize
	// A smaller context wins outright only when it reclaims enough KV to cover
	// the deficit. Taking it unconditionally let a single 1,024-token granule
	// worth about 7 MiB satisfy a ~100 MiB shortfall and return before any other
	// lever was considered, so Qwen3.8-Flash-Next never launched: 101 -> 94 ->
	// 87 MiB with ubatch still at 256. The codebase names this shape in
	// TestGLMContextNudgeCannotBeatFailedDeviceExpertRelief.
	//
	// This is a preference, not a veto. An under-sized drop is still better than
	// failing closed, so it is retained below as a last resort when no other
	// lever moves.
	if oracleReplanEligible && oracleContextDropCoversDeficit(model, strategy, serverArgs, candidate, outcome) {
		candidate.PerformanceTuned = false
		return candidate, candidateArgs, "context-replanned", nil
	}
	nextStrategy, nextArgs, method, changed := applyMemoryRecoverySelection(
		req, strategy, serverArgs, candidate, model, runtimeCaps, outcome, candidateArgs,
	)
	if !changed {
		// Context is the last automatic compute-memory lever. It cannot be an
		// argv patch: context changes KV, graph, CRAM, checkpoint, placement-cache,
		// and host-ledger state together. Recompute the complete configuration at
		// one deficit-sized target, then let the normal exact preflight prove it.
		contextCandidate, contextArgs, contextErr := recomputeAutomaticContextRecovery(
			req, cfg, model, be, caps, strategy, serverArgs, outcome,
		)
		if contextErr != nil {
			return nil, nil, "", contextErr
		}
		if contextCandidate != nil {
			return contextCandidate, contextArgs, "context-derate", nil
		}
	}
	if !changed && oracleReplanEligible {
		// No lever sized to the deficit exists. An under-sized context drop is a
		// worse recovery than one that covers the shortfall, and a better one
		// than refusing to launch.
		candidate.PerformanceTuned = false
		return candidate, candidateArgs, "context-replanned", nil
	}
	if !changed {
		detail := ""
		if replanErr != nil {
			detail = ": " + replanErr.Error()
		}
		return nil, nil, "", fmt.Errorf(
			"CUDA%d allocation of %d MiB exceeded the guard by %d MiB, but neither exact re-planning nor deterministic derating changed the effective memory configuration%s",
			outcome.Device, outcome.AllocMB, outcome.DeficitMB, detail,
		)
	}
	return nextStrategy, nextArgs, method, nil
}

// applyMemoryRecoverySelection turns a selected non-context memory recovery
// action into the next Strategy+argv pair. Production callers pass the complete
// candidate argv rebuilt from its Strategy; the optional form keeps the pure
// selector testable. Context recovery is deliberately outside this function
// because it must re-enter placement.Compute before any argv is emitted.
func applyMemoryRecoverySelection(
	req *launchRequest,
	current *placement.Strategy,
	currentArgs []string,
	candidate *placement.Strategy,
	model *placement.ModelProfile,
	caps *detect.Capabilities,
	outcome preflightOutcome,
	completeCandidateArgs ...[]string,
) (*placement.Strategy, []string, string, bool) {
	// Withdraw a ggrun-generated --swa-full before shedding any weights.
	//
	// It is the largest single reclaimable block on a memory failure and the
	// cheapest to give up: measured 2026-08-03 on DeepSeek-V4-Flash, it cost
	// 6196 MiB of KV on CUDA0 against 871 MiB without — 5.3 GiB, more than an
	// expert layer and a half — while the very runs that carried it still logged
	// "forcing full prompt re-processing due to lack of cache data", so it was
	// buying no prefix reuse at all. Dropping it only ever frees VRAM, so the
	// placement computed against the larger cache stays valid; the next loop
	// re-plans around the headroom. A --swa-full typed on the command line is an
	// instruction and is never withdrawn here.
	if req != nil && hasArg(currentArgs, "--swa-full") && !userExplicitBackendFlag(req, "--swa-full") {
		// Compare the argv directly, not the memory fingerprint: --swa-full is
		// not one of the canonical memory args, so the fingerprint is identical
		// with and without it even though the KV allocation differs sevenfold.
		withdrawn := setPassthroughBoolFlag(currentArgs, "--swa-full", false)
		if len(withdrawn) != len(currentArgs) {
			req.ExtraArgs = setPassthroughBoolFlag(req.ExtraArgs, "--swa-full", false)
			disableBackendFlag(req, "--swa-full", "withdrawn after a memory failure; it reserves a full-context KV cache this model does not reuse")
			return current, withdrawn, "swa-full-withdrawn", true
		}
	}
	// Context changes need a complete placement recompute. Reject any
	// contradictory partial candidate here; an explicit context is immutable,
	// while the automatic path below derives one deficit-sized target and sends
	// it back through Compute.
	if candidate != nil && current != nil && candidate.ContextSize > 0 &&
		candidate.ContextSize != current.ContextSize {
		// Even an automatically-selected context cannot be a partial candidate
		// argv. The caller will derive one deficit-sized target and fully recompute
		// it after the non-context levers are exhausted.
		candidate = nil
	}
	var candidateArgs []string
	if candidate != nil {
		if len(completeCandidateArgs) > 0 {
			candidateArgs = completeCandidateArgs[0]
		} else {
			candidateArgs = patchPlacementArgs(currentArgs, candidate)
		}
	}
	nextArgs, entry, method, changed := selectChangedPreflightRecoveryWithArgs(
		currentArgs, candidate, candidateArgs, model, caps, outcome,
	)
	if !changed {
		return nil, nil, "", false
	}
	if method == "replanned" {
		if candidate == nil {
			return nil, nil, "", false
		}
		return candidate, nextArgs, method, true
	}
	if current == nil || entry == nil {
		return nil, nil, "", false
	}
	applyDeratedPlacementEntry(current, entry)
	return current, nextArgs, method, true
}

// selectChangedPreflightRecovery accepts a packer result only when it changes
// the backend's effective memory flags. Otherwise it applies a monotonic
// black-box fallback. A known failed CUDA device first loses one expert layer;
// only when that device has no movable expert layer does a graph OOM lower
// ubatch. Returning changed=false forbids an identical reload.
func selectChangedPreflightRecovery(
	currentArgs []string,
	candidate *placement.Strategy,
	model *placement.ModelProfile,
	caps *detect.Capabilities,
	outcome preflightOutcome,
) ([]string, *placement.CacheEntry, string, bool) {
	var candidateArgs []string
	if candidate != nil {
		candidateArgs = patchPlacementArgs(currentArgs, candidate)
	}
	return selectChangedPreflightRecoveryWithArgs(currentArgs, candidate, candidateArgs, model, caps, outcome)
}

func selectChangedPreflightRecoveryWithArgs(
	currentArgs []string,
	candidate *placement.Strategy,
	candidateArgs []string,
	model *placement.ModelProfile,
	caps *detect.Capabilities,
	outcome preflightOutcome,
) ([]string, *placement.CacheEntry, string, bool) {
	currentFingerprint := effectiveMemoryArgsFingerprint(currentArgs)
	candidateLowersUBatch := false
	if candidate != nil {
		currentUB := placement.CurrentUBatch(currentArgs)
		if outcome.IsComputeBuffer && currentUB > 0 {
			switch {
			case candidate.UBatchSize > currentUB:
				candidate = nil
				candidateArgs = nil
			case candidate.UBatchSize > 0 && candidate.UBatchSize < currentUB:
				candidateLowersUBatch = true
			}
		}
	}
	if candidate != nil {
		candidateArgs = append([]string(nil), candidateArgs...)
		if effectiveMemoryArgsFingerprint(candidateArgs) == currentFingerprint {
			candidateArgs = nil
		}
	}
	// A same-ubatch re-pack wins early only when it actually relieves the failed
	// device. Context is part of the memory fingerprint, so the old generic
	// changed-fingerprint test accepted GLM's 1,024-token nudges (4-5 MiB each)
	// against a 2.5 GiB deficit until the retry budget expired.
	candidateRelievesTopology := candidateArgs != nil && candidateRelievesFailedDevice(currentArgs, candidateArgs, outcome.Device)
	candidateUBatchCovers := candidateArgs != nil && candidateLowersUBatch && ubatchCandidateCoversDeficit(currentArgs, candidateArgs, outcome)
	if candidateArgs != nil && !candidateLowersUBatch && candidateRelievesTopology {
		return candidateArgs, nil, "replanned", true
	}

	// The exact deficit is the minimum memory that must be reclaimed from the
	// loaded state. Passing the full allocation here could drop many expert
	// layers even when the guard was exceeded by only a few MiB.
	if outcome.IsComputeBuffer {
		// The failed device is stronger evidence than a generic compute-buffer
		// classification. Move one routed expert layer off that exact GPU before
		// reducing global compute throughput via ubatch.
		nextArgs, entry, ok := placement.DerateCUDAOOMArgsForDeficit(
			currentArgs, model, caps, outcome.Device,
			maxPreflightInt(outcome.AllocMB, 1), maxPreflightInt(outcome.DeficitMB, 1), false,
		)
		if ok && effectiveMemoryArgsFingerprint(nextArgs) != currentFingerprint {
			return nextArgs, entry, "expert-derate", true
		}
	}
	if candidateArgs != nil && (candidateRelievesTopology || candidateUBatchCovers) {
		return candidateArgs, nil, "replanned", true
	}
	nextArgs, entry, ok := placement.DerateCUDAOOMArgsForDeficit(
		currentArgs, model, caps, outcome.Device,
		maxPreflightInt(outcome.AllocMB, 1), maxPreflightInt(outcome.DeficitMB, 1), outcome.IsComputeBuffer,
	)
	if ok && effectiveMemoryArgsFingerprint(nextArgs) != currentFingerprint {
		method := "expert-derate"
		if outcome.IsComputeBuffer && entry != nil && entry.UBatchSize > 0 {
			method = "ubatch-derate"
		}
		return nextArgs, entry, method, true
	}
	return nil, nil, "", false
}

func recoveryRequiredMB(deficitMB int) int {
	deficitMB = maxPreflightInt(deficitMB, 1)
	margin := (deficitMB + 9) / 10
	if margin < 64 {
		margin = 64
	}
	return deficitMB + margin
}

func effectiveMemoryArgValues(args []string) map[string]string {
	values := map[string]string{}
	for i := 0; i < len(args); i++ {
		canonical, ok := memoryArgCanonical[args[i]]
		if !ok || i+1 >= len(args) || strings.HasPrefix(args[i+1], "-") {
			continue
		}
		values[canonical] = args[i+1]
		i++
	}
	for _, flag := range []string{"--no-kv-offload", "--no-mmap", "--mmap"} {
		if hasArg(args, flag) {
			values[flag] = "true"
		}
	}
	return values
}

func memoryValueInt(values map[string]string, key string) int {
	v, _ := strconv.Atoi(values[key])
	return v
}

func tensorSplitShare(values map[string]string, device int) float64 {
	parts := strings.Split(values["tensor-split"], ",")
	if device < 0 || device >= len(parts) {
		return 0
	}
	shares := make([]float64, len(parts))
	total := 0.0
	for i, part := range parts {
		share, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil || share < 0 {
			return 0
		}
		shares[i] = share
		total += share
	}
	if total <= 0 {
		return 0
	}
	return shares[device] / total
}

// candidateRelievesFailedDevice proves a topology candidate reduces pressure on
// the device that failed. A generic changed argv is not enough: lateral split
// churn and context-only changes do not address a device-local allocation.
func candidateRelievesFailedDevice(currentArgs, candidateArgs []string, device int) bool {
	currentLayers := placement.CurrentGPUExpertLayers(currentArgs, device)
	candidateLayers := placement.CurrentGPUExpertLayers(candidateArgs, device)
	if currentLayers > 0 && candidateLayers < currentLayers {
		return true
	}
	current := effectiveMemoryArgValues(currentArgs)
	candidate := effectiveMemoryArgValues(candidateArgs)
	if currentLayers > 0 && memoryValueInt(candidate, "n-cpu-moe") > memoryValueInt(current, "n-cpu-moe") {
		return true
	}
	currentShare := tensorSplitShare(current, device)
	candidateShare := tensorSplitShare(candidate, device)
	if currentShare > 0 && candidateShare >= 0 && candidateShare < currentShare {
		return true
	}
	currentLayersGeneric := memoryValueInt(current, "gpu-layers")
	candidateLayersGeneric := memoryValueInt(candidate, "gpu-layers")
	return currentLayersGeneric > 0 && candidateLayersGeneric >= 0 && candidateLayersGeneric < currentLayersGeneric
}

func ubatchCandidateCoversDeficit(currentArgs, candidateArgs []string, outcome preflightOutcome) bool {
	current := memoryValueInt(effectiveMemoryArgValues(currentArgs), "ubatch")
	next := memoryValueInt(effectiveMemoryArgValues(candidateArgs), "ubatch")
	if current <= 0 || next <= 0 || next >= current {
		return false
	}
	if !outcome.AllocMBMeasured || outcome.AllocMB <= 0 {
		return true
	}
	return outcome.AllocMB*(current-next)/current >= recoveryRequiredMB(outcome.DeficitMB)
}

func contextCandidateCoversDeficit(currentArgs, candidateArgs []string, model *placement.ModelProfile, outcome preflightOutcome) bool {
	if !outcome.IsComputeBuffer || !outcome.AllocMBMeasured || outcome.AllocMB <= 0 {
		return false
	}
	currentValues := effectiveMemoryArgValues(currentArgs)
	candidateValues := effectiveMemoryArgValues(candidateArgs)
	currentCtx := memoryValueInt(currentValues, "ctx")
	nextCtx := memoryValueInt(candidateValues, "ctx")
	if currentCtx <= 0 || nextCtx <= 0 || nextCtx >= currentCtx {
		return false
	}
	// Allocation scaling plus total KV reduction is an optimistic upper bound
	// for one device. If even that bound misses the deficit, the candidate is
	// disproved. If it passes, exact preflight remains authoritative.
	reclaim := outcome.AllocMB * (currentCtx - nextCtx) / currentCtx
	kvType := currentValues["cache-k"]
	if kvType == "" {
		kvType = "q8_0"
	}
	currentKV := placement.EstimateKVCacheMB(model, currentCtx, kvType, hasArg(currentArgs, "--swa-full"))
	nextKV := placement.EstimateKVCacheMB(model, nextCtx, kvType, hasArg(candidateArgs, "--swa-full"))
	if currentKV > nextKV {
		reclaim += currentKV - nextKV
	}
	return reclaim >= recoveryRequiredMB(outcome.DeficitMB)
}

func automaticContextRecoveryTarget(req *launchRequest, current *placement.Strategy, currentArgs []string, outcome preflightOutcome) (int, bool) {
	if req == nil || current == nil || !current.ContextAuto || !outcome.IsComputeBuffer ||
		!outcome.AllocMBMeasured || outcome.AllocMB <= 0 {
		return 0, false
	}
	if !automaticContextRequest(req) {
		return 0, false
	}
	currentCtx := memoryValueInt(effectiveMemoryArgValues(currentArgs), "ctx")
	if currentCtx <= 0 {
		currentCtx = current.ContextSize
	}
	minimum := 32768
	if req.ClaudeCode {
		slots := maxPreflightInt(current.Parallel, 1)
		minimum = maxPreflightInt(minimum, slots*claudeSlotMin)
	}
	if currentCtx <= minimum {
		return 0, false
	}
	required := recoveryRequiredMB(outcome.DeficitMB)
	target := minimum
	if required < outcome.AllocMB {
		target = currentCtx * (outcome.AllocMB - required) / outcome.AllocMB
	}
	target = target / 1024 * 1024
	if target < minimum {
		target = minimum
	}
	if target >= currentCtx {
		target = (currentCtx - 1024) / 1024 * 1024
	}
	if target < minimum || target >= currentCtx {
		return 0, false
	}
	return target, true
}

// recomputeAutomaticContextRecovery turns the measured target into one complete
// placement. A context change is never applied to the current Strategy in
// place: all context-derived memory and cache state must come from Compute.
func recomputeAutomaticContextRecovery(req *launchRequest, cfg *config.Config, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, current *placement.Strategy, currentArgs []string, outcome preflightOutcome) (*placement.Strategy, []string, error) {
	target, ok := automaticContextRecoveryTarget(req, current, currentArgs, outcome)
	if !ok || cfg == nil || model == nil || be == nil || caps == nil {
		return nil, nil, nil
	}
	opts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	opts.ContextSize = target
	opts.AutoContextMax = 0
	opts.Parallel = maxPreflightInt(current.Parallel, 1)
	opts.AutoParallel = false
	opts.BatchSize = current.BatchSize
	opts.UBatchSize = current.UBatchSize
	opts.SkipPlacementCache = true
	opts.CacheFile = ""
	opts.VerifiedConfigScopeKey = ""
	next, err := placement.Compute(caps, model, opts)
	if err != nil {
		return nil, nil, fmt.Errorf("full context-recovery re-plan at %d tokens: %w", target, err)
	}
	if next == nil || next.ContextSize != target {
		return nil, nil, fmt.Errorf("full context-recovery re-plan did not preserve target %d", target)
	}
	next.ContextAuto = true
	next.ContextFitRejected = current.ContextSize
	next.ContextFitTier = current.ContextFitTier
	if next.ContextFitTier == "" {
		next.ContextFitTier = "recovered"
	}
	next.ContextFitEvidence = fmt.Sprintf("exact CUDA%d compute allocation missed guard by %d MiB", outcome.Device, outcome.DeficitMB)
	next.PerformanceTuned = false
	nextArgs := buildLaunchServerArgs(req, cfg, be, caps, model, next)
	if effectiveMemoryArgsFingerprint(nextArgs) == effectiveMemoryArgsFingerprint(currentArgs) {
		return nil, nil, fmt.Errorf("full context-recovery re-plan at %d tokens did not change effective memory arguments", target)
	}
	if !contextCandidateCoversDeficit(currentArgs, nextArgs, model, outcome) {
		return nil, nil, fmt.Errorf("full context-recovery re-plan at %d tokens cannot cover the measured %d MiB deficit", target, outcome.DeficitMB)
	}
	return next, nextArgs, nil
}

func automaticContextRequest(req *launchRequest) bool {
	if req == nil {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(req.CtxFlag)) {
	case "", "fit", "auto":
		return true
	default:
		return false
	}
}

var memoryArgCanonical = map[string]string{
	"-m": "model", "--model": "model",
	"-c": "ctx", "--ctx-size": "ctx", "--ctx": "ctx",
	"-b": "batch", "--batch-size": "batch",
	"-ub": "ubatch", "--ubatch-size": "ubatch",
	"-ctk": "cache-k", "--cache-type-k": "cache-k",
	"-ctv": "cache-v", "--cache-type-v": "cache-v",
	"-np": "parallel", "--parallel": "parallel",
	"-ngl": "gpu-layers", "--n-gpu-layers": "gpu-layers", "--gpu-layers": "gpu-layers",
	"-ts": "tensor-split", "--tensor-split": "tensor-split",
	"-sm": "split-mode", "--split-mode": "split-mode",
	"-ot": "override-tensor", "--override-tensor": "override-tensor",
	"-ncmoe": "n-cpu-moe", "--n-cpu-moe": "n-cpu-moe",
	"-fa": "flash-attn", "--flash-attn": "flash-attn",
	"-mg": "main-gpu", "--main-gpu": "main-gpu",
	"-dev": "device", "--device": "device",
}

// effectiveMemoryArgsFingerprint follows the backend's last-value-wins argv
// behavior. This catches retries where ggrun changed an earlier generated flag
// but a later user override kept the effective placement identical.
func effectiveMemoryArgsFingerprint(args []string) string {
	values := effectiveMemoryArgValues(args)
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var parts []string
	for _, key := range keys {
		parts = append(parts, key+"="+values[key])
	}
	return strings.Join(parts, "\n")
}

// deficitProgress reports whether a re-plan materially reduced the measured
// shortfall. "Materially" is deliberate: a deficit that creeps down by a few
// MiB per round is the nudge pathology, not convergence, and must still be
// charged against the re-plan budget. The first round has nothing to compare
// against and is never treated as progress.
func deficitProgress(previousMB, currentMB int) bool {
	if previousMB <= 0 || currentMB <= 0 || currentMB >= previousMB {
		return false
	}
	// At least a fifth of the previous shortfall must be gone.
	return previousMB-currentMB >= previousMB/5
}

// oracleContextDropCoversDeficit reports whether shrinking to the candidate's
// context reclaims enough KV to cover the oracle's measured device deficit.
//
// The no-allocation oracle reports a total device shortfall rather than a
// failed allocation, so contextCandidateCoversDeficit -- which needs a measured
// compute-buffer allocation -- cannot judge it. KV is what a context change
// moves, so its size per token is the exchange rate, and contextReclaimTokens
// already converts a deficit into that many tokens.
//
// Unknown geometry returns true: without an exchange rate there is no basis to
// reject a candidate the rest of the guard accepted, and the exact preflight
// that follows remains the authority either way.
func oracleContextDropCoversDeficit(
	model *placement.ModelProfile, current *placement.Strategy, currentArgs []string,
	candidate *placement.Strategy, outcome preflightOutcome,
) bool {
	if current == nil || candidate == nil {
		return false
	}
	needed := contextReclaimTokens(model, current, currentArgs, outcome.DeficitMB)
	if needed <= 0 {
		return true
	}
	return current.ContextSize-candidate.ContextSize >= needed
}
