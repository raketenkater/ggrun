package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/config"
	modelstore "github.com/raketenkater/ggrun/pkg/models"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

// A sparse lookup tier (DeepSeek-V4.1 Engram is the first) is a third host
// memory class beside resident execution weights and non-reclaimable runtime.
// Placement decides how a plan holds it (see pkg/placement/sparselookup.go);
// this file supplies the backend facts that decision needs and enforces them
// at launch:
//
//   - the loader contract: whether the exact backend build keeps lazily read
//     tables file-backed, unpinned and off the GPUs (reviewed source, never a
//     help line);
//   - the fit-oracle accounting contract: whether that build's fit pass counts
//     lazily mapped tables in its Host model row;
//   - the running memory scope and the post-load mapping check.

// sparseLookupOracleAccounting says what a fit pass's Host model row holds for
// lazily mapped lookup tables. It is versioned separately from the loader
// contract: a build can read tables lazily and still report them differently.
type sparseLookupOracleAccounting string

const (
	sparseLookupOracleUnknown sparseLookupOracleAccounting = "unknown"
	// The Host model row includes every lazily mapped table byte.
	sparseLookupOracleInclusiveV1 sparseLookupOracleAccounting = "host-row-includes-lazy-tables/v1"
	// The Host model row already excludes lazily mapped tables.
	sparseLookupOracleResidentOnlyV1 sparseLookupOracleAccounting = "host-row-resident-only/v1"
)

// sparseLookupLoaderContract is a reviewed lazy-loader implementation. It is
// matched against the source commit a backend build reports, the model's table
// kind and the platform the review covered.
type sparseLookupLoaderContract struct {
	ID        string
	Commits   []string // source commits (prefixes) whose loader was reviewed
	Kinds     []string // sparse lookup table kinds from the parser's adapters
	Platforms []string // runtime.GOOS values the review covered
	Flag      string
	Oracle    sparseLookupOracleAccounting
	Review    string
	Synthetic bool // test fixture only; never a support claim
}

var reviewedSparseLookupLoaders = []sparseLookupLoaderContract{{
	ID:        "llama.cpp-tensor-read-lazy/v1",
	Commits:   []string{"425badfe"},
	Kinds:     []string{"engram"},
	Platforms: []string{"linux"},
	Flag:      placement.SparseLookupLazyFlag,
	Oracle:    sparseLookupOracleInclusiveV1,
	Review: "smalinin/llama.cpp 425badfe source review 2026-10-19: with --lazy-mode on, TENSOR_READ_LAZY " +
		"tables take the CPU buffer before any -ot, are mapped from the file under every --load-mode, get " +
		"MADV_RANDOM and stay out of MAP_POPULATE and mlock; its llama-fit-params counts them in the Host " +
		"model row (333,327 = 134,102 + 199,225 MiB on the pinned IQ2_XS headers)",
}}

var backendCommitPattern = regexp.MustCompile(`\bcommit ([0-9a-f]{7,40})\b`)

// backendSourceCommit returns the source commit a build reports in --version.
func backendSourceCommit(version string) string {
	if m := backendCommitPattern.FindStringSubmatch(version); m != nil {
		return m[1]
	}
	return ""
}

// backendAdvertisedLazyFlag is the lazy-read option the backend's help lists.
// It is flag availability only: it lets ggrun state "off" for tables charged
// to their layer, and never authorizes reclaimable memory by itself.
func backendAdvertisedLazyFlag(help string) string {
	if helpHasExactFlag(help, placement.SparseLookupLazyFlag) {
		return placement.SparseLookupLazyFlag
	}
	return ""
}

func sparseLookupLoaderFor(be *backendInfo, kind, goos string) (*sparseLookupLoaderContract, string) {
	if be == nil {
		return nil, "backend probe unavailable"
	}
	name := strings.TrimSpace(be.Identity)
	if name == "" {
		name = filepath.Base(be.Path)
	}
	if strings.TrimSpace(be.Help) == "" || backendLoaderFailed(be.Help) {
		return nil, "lookup-table loader not proven: " + name
	}
	flag := backendAdvertisedLazyFlag(be.Help)
	if flag == "" {
		return nil, "backend has no lazy-read option: " + name
	}
	commit := backendSourceCommit(be.Version)
	for i := range reviewedSparseLookupLoaders {
		c := &reviewedSparseLookupLoaders[i]
		if c.Flag != flag || !containsFold(c.Kinds, kind) || !containsFold(c.Platforms, goos) {
			continue
		}
		for _, reviewed := range c.Commits {
			if commit != "" && (strings.HasPrefix(commit, reviewed) || strings.HasPrefix(reviewed, commit)) {
				return c, c.ID + " (" + name + ")"
			}
		}
	}
	return nil, fmt.Sprintf("%s is advertised but no reviewed lazy-loader contract covers %s tables in %s (commit %q, %s)",
		flag, kind, name, commit, goos)
}

func containsFold(list []string, v string) bool {
	for _, item := range list {
		if strings.EqualFold(item, v) {
			return true
		}
	}
	return false
}

// sparseLookupCapabilityFor resolves the placement inputs for one model on one
// backend: the verified capability, its evidence and the advertised flag. A
// live contradiction persisted for this model/backend overrides any contract.
func sparseLookupCapabilityFor(be *backendInfo, model *placement.ModelProfile) (placement.SparseLookupCapability, string, string) {
	if model == nil || model.SparseLookupBytes <= 0 || be == nil {
		return placement.SparseLookupUnknown, "", ""
	}
	flag := backendAdvertisedLazyFlag(be.Help)
	if placement.NormalizeSparseLookupCapability(string(be.SparseLookupCapability)) == placement.SparseLookupAnonymous {
		return placement.SparseLookupAnonymous, be.SparseLookupEvidence, flag
	}
	contract, evidence := sparseLookupLoaderFor(be, model.SparseLookupKind, runtime.GOOS)
	if contract == nil {
		return placement.SparseLookupUnknown, evidence, flag
	}
	return placement.SparseLookupFileBacked, evidence, flag
}

// sparseLookupOracleFor returns the fit-oracle accounting contract for the
// model's tables on this backend; unknown without a reviewed contract.
func sparseLookupOracleFor(be *backendInfo, model *placement.ModelProfile) sparseLookupOracleAccounting {
	if model == nil {
		return sparseLookupOracleUnknown
	}
	contract, _ := sparseLookupLoaderFor(be, model.SparseLookupKind, runtime.GOOS)
	if contract == nil || contract.Oracle == "" {
		return sparseLookupOracleUnknown
	}
	return contract.Oracle
}

// backendReviewedLazyLoader reports whether any reviewed contract covers this
// build on this platform. Placement emits "--lazy-mode on" only under such a
// contract, so the start-time scope can trust the argv.
func backendReviewedLazyLoader(be *backendInfo) bool {
	if be == nil {
		return false
	}
	for _, c := range reviewedSparseLookupLoaders {
		for _, kind := range c.Kinds {
			if contract, _ := sparseLookupLoaderFor(be, kind, runtime.GOOS); contract != nil {
				return true
			}
		}
	}
	return false
}

// argsLazySparseLookup reports whether the emitted argv keeps lookup tables
// lazily mapped on a backend with a reviewed loader. Last value wins, as in
// the backend's own parser.
func argsLazySparseLookup(be *backendInfo, serverArgs []string) bool {
	if !backendReviewedLazyLoader(be) {
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

// applyNativeLazyMode folds a passthrough lazy-read choice into the request.
// ggrun re-emits the flag from the plan, so the token is consumed: "on" and
// "off" are explicit constraints, "auto" leaves the choice to placement.
func applyNativeLazyMode(req *launchRequest, args []string, i int) int {
	flag, value, inline := strings.Cut(args[i], "=")
	if flag != placement.SparseLookupLazyFlag && flag != "-lzm" {
		return 0
	}
	consumed := 1
	if !inline {
		if i+1 >= len(args) {
			return 0
		}
		value, consumed = args[i+1], 2
	}
	switch value {
	case "on", "off":
		req.SparseLookupLazy = value
	case "auto":
		req.SparseLookupLazy = ""
	default:
		return 0
	}
	return consumed
}

// sparseLookupScopeLimits sizes the running scope of a paged-tier plan after
// the canary. nonReclaimableCeilMB is the Fix-B ceiling for anonymous state
// (measured + headroom, or the plan floor), ceilingMB the user's total limit
// for this backend and headroomMB the configured reclaim band.
//
// The reclaim band is reserved before any lookup cache is granted:
//
//	memory.max  = min(ceiling, non-reclaimable + tier + band)
//	memory.high = memory.max - band
//	lookup      = memory.high - non-reclaimable   (0 .. tier)
//
// This is one cgroup, not separate kernel budgets: above memory.high the
// kernel reclaims the scope's clean file pages (the lookup rows) and throttles
// further growth; anonymous growth past memory.max is killed inside the scope.
// When the ceiling cannot hold the non-reclaimable state plus a full band,
// memory.high stays at the non-reclaimable ceiling (never throttling the
// required working set), the band shrinks to what is left, lookup gets 0 and
// short reports it. keepMax preserves an existing larger hard ceiling
// (explicitly mapped CPU experts) and only sets memory.high.
func sparseLookupScopeLimits(strategy *placement.Strategy, nonReclaimableCeilMB, headroomMB, ceilingMB int, keepMax bool) (highMB, maxMB, lookupMB int, short bool) {
	if !placement.StrategyUsesFileBackedSparseLookup(strategy) || nonReclaimableCeilMB <= 0 || ceilingMB <= 0 {
		return 0, 0, 0, false
	}
	if headroomMB < 0 {
		headroomMB = 0
	}
	if nonReclaimableCeilMB > ceilingMB {
		nonReclaimableCeilMB = ceilingMB
	}
	maxMB = nonReclaimableCeilMB + strategy.SparseLookupMB + headroomMB
	if maxMB > ceilingMB {
		maxMB = ceilingMB
	}
	highMB = maxMB - headroomMB
	if highMB < nonReclaimableCeilMB {
		highMB, short = nonReclaimableCeilMB, true
	}
	lookupMB = highMB - nonReclaimableCeilMB
	if keepMax {
		maxMB = 0
	}
	return highMB, maxMB, lookupMB, short
}

// sparseLookupStartLimits sizes the scope at process start for a lazy-tier
// argv, before any measurement. budgetMB is the plan ceiling the scope would
// otherwise get, hostCeilingMB the larger reclaim ceiling a derived percent
// target allows (0 when the user named an explicit budget). The band between
// memory.high and memory.max is at least headroomMB whenever the limits allow
// it, so page-cache growth is reclaimed before the hard limit.
func sparseLookupStartLimits(budgetMB, hostCeilingMB, headroomMB int) (highMB, maxMB int) {
	if budgetMB <= 0 {
		return 0, 0
	}
	highMB, maxMB = budgetMB, budgetMB
	if hostCeilingMB > budgetMB {
		maxMB = hostCeilingMB
	}
	if headroomMB > 0 && maxMB-highMB < headroomMB && maxMB > headroomMB {
		highMB = maxMB - headroomMB
	}
	return highMB, maxMB
}

func applySparseLookupScope(p *server.Process, strategy *placement.Strategy, measured, plannedFloor, nonReclaimableCeilMB, headroomMB, ceilingMB int, keepMax bool) {
	highMB, maxMB, lookupMB, short := sparseLookupScopeLimits(strategy, nonReclaimableCeilMB, headroomMB, ceilingMB, keepMax)
	if highMB <= 0 {
		return
	}
	// memory.high first: reclaim starts at the new threshold before the hard
	// limit moves.
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
	if short {
		fmt.Fprintf(os.Stderr, "[launch] warning: the %d MiB ceiling leaves less than the %d MiB reclaim band above the non-reclaimable state; lookups will read storage and growth meets the hard limit sooner\n",
			ceilingMB, headroomMB)
	} else if lookupMB < strategy.SparseLookupMB/16 {
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
// mapping at all, contradict a lazily mapped lookup plan.
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
// the host could not expose mappings, which is not evidence either way. The
// non-reclaimable envelope applies only to a paged tier: a resident tier is
// already charged in the plan's host footprint.
func sparseLookupContradiction(strategy *placement.Strategy, usage *mappedFileUsage, nonReclaimableMB int) string {
	if strategy == nil || !placement.StrategyUsesLazySparseLookup(strategy) {
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
	if nonReclaimableMB > 0 && placement.StrategyUsesFileBackedSparseLookup(strategy) {
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

// validateObservedSparseLookup checks, after load, that a lazily mapped tier
// really is file-backed. Missing observations leave the reviewed contract as
// the only evidence; they are logged and never recorded as proof. A
// contradiction is persisted for this exact model/backend so the next plan
// charges the tables to their layers instead of reloading the same failure.
func validateObservedSparseLookup(cfg *config.Config, model *placement.ModelProfile, be *backendInfo, strategy *placement.Strategy, process *server.Process) error {
	if cfg == nil || model == nil || be == nil || process == nil || !placement.StrategyUsesLazySparseLookup(strategy) {
		return nil
	}
	usage, mapErr := observeModelMappings(model, serverProcessPID(process))
	measuredMB, statErr := process.ScopeNonReclaimableMB()
	if statErr != nil {
		measuredMB = 0
	}
	if mapErr != nil && measuredMB <= 0 {
		fmt.Fprintf(os.Stderr, "[sparse] lookup-tier mapping not observable (%v); relying on the reviewed loader contract only\n", mapErr)
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
		fmt.Fprintf(os.Stderr, "[sparse] live lookup-tier check (%s): %d model mappings, %d MiB mapped, %d MiB resident, 0 anonymous, 0 locked; %d MiB non-reclaimable\n",
			strategy.SparseLookupPolicy, usage.Mappings, usage.SizeKB/1024, usage.RssKB/1024, measuredMB)
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

// oracleDevicesExcludingLookupTier converts a fit pass's Host model row to
// resident bytes for a paged lookup tier. It acts only when the backend's
// versioned oracle accounting contract says the row includes lazily mapped
// tables; a resident-only or unknown oracle is left as reported, which keeps
// admission conservative. The subtraction uses the tier's whole MiB (rounded
// down) so it can never remove more than the tables, applies at most once per
// row, and is skipped when the row is too small to contain the tier (that
// contradicts the contract).
func oracleDevicesExcludingLookupTier(devs []preflightDevice, accounting sparseLookupOracleAccounting, strategy *placement.Strategy) []preflightDevice {
	if !placement.StrategyUsesFileBackedSparseLookup(strategy) {
		return devs
	}
	if accounting != sparseLookupOracleInclusiveV1 {
		fmt.Fprintf(os.Stderr, "[sparse] fit pass lookup accounting %q: Host row used as reported\n", accounting)
		return devs
	}
	tierMB := strategy.SparseLookupMB - 1
	if strategy.SparseLookupBytes > 0 {
		tierMB = int(strategy.SparseLookupBytes >> 20)
	}
	if tierMB <= 0 {
		return devs
	}
	out := append([]preflightDevice(nil), devs...)
	for i := range out {
		if out[i].Name != "Host" || out[i].SparseLookupExcludedMB > 0 {
			continue
		}
		if out[i].ModelMB < tierMB {
			fmt.Fprintf(os.Stderr, "[sparse] fit pass Host model %d MiB cannot include the %d MiB lookup tier its contract says it counts; used as reported\n",
				out[i].ModelMB, tierMB)
			continue
		}
		out[i].ModelMB -= tierMB
		out[i].SparseLookupExcludedMB = tierMB
		fmt.Fprintf(os.Stderr, "[sparse] fit pass host model %d MiB includes the %d MiB paged lookup tier (%s); resident host model is %d MiB\n",
			devs[i].ModelMB, tierMB, accounting, out[i].ModelMB)
	}
	return out
}
