package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/config"
	modelstore "github.com/raketenkater/ggrun/pkg/models"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

// A file-backed sparse lookup tier (DeepSeek-V4.1 Engram) is a third host
// memory class beside resident execution weights and non-reclaimable runtime.
// Its resident share is page cache: the kernel may drop it, and dropping it
// costs storage reads, never correctness. The memory scope gives it an
// explicit, bounded allowance on top of the non-reclaimable ledger rather than
// counting it as resident (which would refuse the model) or ignoring it (which
// would let lookup traffic hit memory.max).

// argsFileBackedSparseLookup reports whether the emitted argv keeps lookup
// tables lazily mapped on a backend whose loader does so. Last value wins, as
// in the backend's own parser.
func argsFileBackedSparseLookup(be *backendInfo, serverArgs []string) bool {
	if backendSparseLookupCapability(be) != placement.SparseLookupFileBacked {
		return false
	}
	mode := ""
	for i, a := range serverArgs {
		flag, value, inline := strings.Cut(a, "=")
		if flag != placement.SparseLookupLazyFlag && flag != "-lzm" {
			continue
		}
		if inline {
			mode = value
		} else if i+1 < len(serverArgs) {
			mode = serverArgs[i+1]
		}
	}
	return mode == "on"
}

// sparseLookupScopeLimits sizes the running scope for a plan with a
// file-backed lookup tier. nonReclaimableCeilMB is the Fix-B ceiling for
// anonymous state (measured+headroom or the plan floor, already clamped).
// The lookup allowance is what remains below the user's ceiling, capped at the
// table size. memory.high = non-reclaimable ceiling + allowance, so reclaim
// keeps the lookup cache inside its allowance; memory.max adds one headroom
// band so a burst is throttled before it is killed. Anonymous growth first
// squeezes the lookup cache and then meets memory.max inside its own scope.
// keepMax preserves an existing larger hard ceiling (mapped CPU experts).
func sparseLookupScopeLimits(strategy *placement.Strategy, nonReclaimableCeilMB, headroomMB, ceilingMB int, keepMax bool) (highMB, maxMB, lookupMB int) {
	if !placement.StrategyUsesFileBackedSparseLookup(strategy) || nonReclaimableCeilMB <= 0 || ceilingMB <= 0 {
		return 0, 0, 0
	}
	if nonReclaimableCeilMB > ceilingMB {
		nonReclaimableCeilMB = ceilingMB
	}
	lookupMB = ceilingMB - nonReclaimableCeilMB
	if lookupMB > strategy.SparseLookupMB {
		lookupMB = strategy.SparseLookupMB
	}
	if lookupMB < 0 {
		lookupMB = 0
	}
	highMB = nonReclaimableCeilMB + lookupMB
	if keepMax {
		return highMB, 0, lookupMB
	}
	maxMB = highMB + headroomMB
	if maxMB > ceilingMB {
		maxMB = ceilingMB
	}
	if highMB > maxMB {
		highMB = maxMB
	}
	return highMB, maxMB, lookupMB
}

func applySparseLookupScope(p *server.Process, strategy *placement.Strategy, measured, plannedFloor, nonReclaimableCeilMB, headroomMB, ceilingMB int, keepMax bool) {
	highMB, maxMB, lookupMB := sparseLookupScopeLimits(strategy, nonReclaimableCeilMB, headroomMB, ceilingMB, keepMax)
	if highMB <= 0 {
		return
	}
	if err := p.SetMemoryHighMB(highMB); err != nil {
		fmt.Fprintf(os.Stderr, "[launch] lookup-cache reclaim threshold re-size to %d MiB failed: %v\n", highMB, err)
		return
	}
	if maxMB > 0 {
		if err := p.SetMemoryMaxMB(maxMB); err != nil {
			fmt.Fprintf(os.Stderr, "[launch] lookup-cache hard ceiling re-size to %d MiB failed: %v\n", maxMB, err)
			return
		}
	}
	fmt.Fprintf(os.Stderr, "[launch] memory scope: non-reclaimable ceiling %d MiB (measured %d MiB, planned floor %d MiB) + lookup cache %d of %d MiB = high %d MiB, max %s\n",
		nonReclaimableCeilMB, measured, plannedFloor, lookupMB, strategy.SparseLookupMB, highMB, scopeMaxLabel(maxMB))
	if lookupMB < strategy.SparseLookupMB/16 {
		fmt.Fprintf(os.Stderr, "[launch] warning: only %d MiB of RAM can cache %d MiB of lookup tables; most lookups will read storage\n",
			lookupMB, strategy.SparseLookupMB)
	}
}

func scopeMaxLabel(maxMB int) string {
	if maxMB <= 0 {
		return "preserved"
	}
	return strconv.Itoa(maxMB) + " MiB"
}

// mappedFileUsage sums /proc/<pid>/smaps entries whose backing path is one of
// the model files. Anonymous or locked pages inside those mappings, or no
// mapping at all, contradict a file-backed lookup plan.
type mappedFileUsage struct {
	Mappings    int
	SizeKB      int64
	RssKB       int64
	AnonymousKB int64
	LockedKB    int64
}

func parseSmapsModelUsage(r io.Reader, files map[string]bool) mappedFileUsage {
	var usage mappedFileUsage
	inModel := false
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		if !strings.HasSuffix(fields[0], ":") {
			// Mapping header: "start-end perms offset dev inode [path]".
			inModel = false
			if len(fields) >= 6 && files[strings.Join(fields[5:], " ")] {
				inModel = true
				usage.Mappings++
			}
			continue
		}
		if !inModel || len(fields) < 2 {
			continue
		}
		value, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			continue
		}
		switch fields[0] {
		case "Size:":
			usage.SizeKB += value
		case "Rss:":
			usage.RssKB += value
		case "Anonymous:":
			usage.AnonymousKB += value
		case "Locked:":
			usage.LockedKB += value
		}
	}
	return usage
}

// sparseLookupContradiction applies the observation rules. A nil usage means
// the host could not expose mappings, which is not evidence either way.
func sparseLookupContradiction(strategy *placement.Strategy, usage *mappedFileUsage, nonReclaimableMB int) string {
	if strategy == nil || !placement.StrategyUsesFileBackedSparseLookup(strategy) {
		return ""
	}
	if usage != nil {
		switch {
		case usage.Mappings == 0:
			return "no model-file mapping is present, so the lookup tables cannot be file-backed"
		case usage.SizeKB < int64(strategy.SparseLookupMB-1)*1024: // tier MiB is rounded up
			return fmt.Sprintf("model-file mappings cover %d MiB, less than the %d MiB lookup tier", usage.SizeKB/1024, strategy.SparseLookupMB)
		case usage.AnonymousKB > 0:
			return fmt.Sprintf("%d KiB of model-file mappings are anonymous copies", usage.AnonymousKB)
		case usage.LockedKB > 0:
			return fmt.Sprintf("%d KiB of model-file mappings are locked", usage.LockedKB)
		}
	}
	if nonReclaimableMB > 0 {
		// Same envelope rule as the CPU-expert check: allocator noise up to the
		// larger of 1 GiB or an eighth of the tier; a copying loader exceeds
		// both that and half the tier.
		slack := maxInt(1024, strategy.SparseLookupMB/8)
		limit := strategy.PlannedHostFootprintMB + strategy.CRAM + slack
		if nonReclaimableMB > limit && nonReclaimableMB > strategy.SparseLookupMB/2 {
			return fmt.Sprintf("%d MiB non-reclaimable exceeds the %d MiB envelope for a %d MiB file-backed lookup tier",
				nonReclaimableMB, limit, strategy.SparseLookupMB)
		}
	}
	return ""
}

func observeModelMappings(model *placement.ModelProfile, root int) (*mappedFileUsage, error) {
	if model == nil || root <= 0 {
		return nil, fmt.Errorf("no backend process")
	}
	shards, _, err := modelstore.ResolveGGUFShardFiles(model.Path)
	if err != nil {
		return nil, err
	}
	files := map[string]bool{}
	for _, shard := range shards {
		files[shard] = true
		if abs, err := filepath.Abs(shard); err == nil {
			files[abs] = true
		}
		if real, err := filepath.EvalSymlinks(shard); err == nil {
			files[real] = true
		}
	}
	pids := processTreePIDs(root)
	if len(pids) == 0 {
		pids = map[int]bool{root: true}
	}
	var total mappedFileUsage
	read := false
	for pid := range pids {
		f, err := os.Open(filepath.Join("/proc", strconv.Itoa(pid), "smaps"))
		if err != nil {
			continue
		}
		u := parseSmapsModelUsage(f, files)
		f.Close()
		read = true
		total.Mappings += u.Mappings
		total.SizeKB += u.SizeKB
		total.RssKB += u.RssKB
		total.AnonymousKB += u.AnonymousKB
		total.LockedKB += u.LockedKB
	}
	if !read {
		return nil, fmt.Errorf("process mappings unavailable")
	}
	return &total, nil
}

// validateObservedSparseLookup checks, after load, that the lookup tier really
// is file-backed. Missing observations leave the capability as probed; they
// are logged and never recorded as proof. A contradiction is persisted for
// this exact model/backend so the next plan refuses instead of reloading.
func validateObservedSparseLookup(cfg *config.Config, model *placement.ModelProfile, be *backendInfo, strategy *placement.Strategy, process *server.Process) error {
	if cfg == nil || model == nil || be == nil || process == nil || !placement.StrategyUsesFileBackedSparseLookup(strategy) {
		return nil
	}
	usage, mapErr := observeModelMappings(model, serverProcessPID(process))
	measuredMB, statErr := process.ScopeNonReclaimableMB()
	if statErr != nil {
		measuredMB = 0
	}
	if mapErr != nil && measuredMB <= 0 {
		fmt.Fprintf(os.Stderr, "[sparse] lookup-tier mapping not observable (%v); capability remains help-probed\n", mapErr)
		return nil
	}
	if reason := sparseLookupContradiction(strategy, usage, measuredMB); reason != "" {
		evidence := "live contradiction: " + reason
		be.SparseLookupCapability = placement.SparseLookupAnonymous
		be.SparseLookupEvidence = evidence
		if err := persistBackendSparseLookupCapability(cfg.CacheDir, model, be, placement.SparseLookupAnonymous, evidence); err != nil {
			fmt.Fprintf(os.Stderr, "[sparse] warning: could not persist lookup-tier contradiction: %v\n", err)
		}
		return fmt.Errorf("backend did not keep the %d MiB lookup tier file-backed (%s)", strategy.SparseLookupMB, evidence)
	}
	if usage != nil {
		fmt.Fprintf(os.Stderr, "[sparse] live lookup-tier check: %d model mappings, %d MiB mapped, %d MiB resident, 0 anonymous, 0 locked; %d MiB non-reclaimable\n",
			usage.Mappings, usage.SizeKB/1024, usage.RssKB/1024, measuredMB)
	}
	return nil
}

// validateObservedHostTiers runs the CPU-expert and lookup-tier observations.
// They cover separate tiers and neither result stands in for the other.
func validateObservedHostTiers(cfg *config.Config, model *placement.ModelProfile, be *backendInfo, strategy *placement.Strategy, process *server.Process) error {
	if err := validateObservedMMapPageability(cfg, model, be, strategy, process); err != nil {
		return err
	}
	return validateObservedSparseLookup(cfg, model, be, strategy, process)
}

// oracleDevicesExcludingLookupTier removes lazily mapped lookup tables from
// the fit pass's Host model row. llama-fit-params counts every CPU-buffer
// tensor as host model memory, including TENSOR_READ_LAZY tables the server
// maps from the file and never makes resident. On the pinned fork and the
// IQ2_XS headers the planned argv reported Host model 333,327 MiB with
// --lazy-mode on: 134,102 MiB of resident CPU experts plus the 199,225 MiB
// Engram tier. Only that exact tier is removed, and only when the argv and the
// backend keep it lazy; a row too small to contain it is left untouched.
func oracleDevicesExcludingLookupTier(devs []preflightDevice, be *backendInfo, strategy *placement.Strategy, serverArgs []string) []preflightDevice {
	if !placement.StrategyUsesFileBackedSparseLookup(strategy) || !argsFileBackedSparseLookup(be, serverArgs) {
		return devs
	}
	out := append([]preflightDevice(nil), devs...)
	for i := range out {
		if out[i].Name != "Host" || out[i].ModelMB < strategy.SparseLookupMB {
			continue
		}
		out[i].ModelMB -= strategy.SparseLookupMB
		fmt.Fprintf(os.Stderr, "[sparse] fit pass host model %d MiB includes the %d MiB file-backed lookup tier; resident host model is %d MiB\n",
			devs[i].ModelMB, strategy.SparseLookupMB, out[i].ModelMB)
	}
	return out
}
