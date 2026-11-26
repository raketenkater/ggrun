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
	"github.com/raketenkater/ggrun/pkg/gguf"
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

// mappedFileUsage summarizes the /proc/<pid>/smaps mappings whose backing
// file is one of the model's shards. Totals cover every model mapping; Maps
// keeps each mapping's file range so lookup-table pages can be told apart from
// execution weights in the same shard.
type mappedFileUsage struct {
	Mappings    int
	SizeKB      int64
	RssKB       int64
	AnonymousKB int64
	LockedKB    int64
	Maps        []smapsModelMapping
}

// smapsModelMapping is one mapping of a model shard: the file byte range it
// maps and its page counters.
type smapsModelMapping struct {
	Shard       string // shard basename as the model profile names it
	Offset      int64  // file offset of the mapping start
	Length      int64
	RssKB       int64
	AnonymousKB int64
	LockedKB    int64
}

// parseSmapsModelUsage reads smaps text. files maps every path spelling of a
// model shard (as given, absolute, symlink target) to the shard's basename.
func parseSmapsModelUsage(r io.Reader, files map[string]string) mappedFileUsage {
	var usage mappedFileUsage
	var cur *smapsModelMapping
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) == 0 {
			continue
		}
		if !strings.HasSuffix(fields[0], ":") {
			// Mapping header: "start-end perms offset dev inode [path]".
			cur = nil
			if len(fields) < 6 {
				continue
			}
			shard, ok := files[strings.Join(fields[5:], " ")]
			if !ok {
				continue
			}
			lo, hi, okRange := strings.Cut(fields[0], "-")
			start, err1 := strconv.ParseUint(lo, 16, 64)
			end, err2 := strconv.ParseUint(hi, 16, 64)
			off, err3 := strconv.ParseUint(fields[2], 16, 64)
			m := smapsModelMapping{Shard: shard, Offset: -1}
			if okRange && err1 == nil && err2 == nil && err3 == nil && end >= start {
				m.Offset, m.Length = int64(off), int64(end-start)
			}
			usage.Maps = append(usage.Maps, m)
			cur = &usage.Maps[len(usage.Maps)-1]
			usage.Mappings++
			continue
		}
		if cur == nil || len(fields) < 2 {
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
			cur.RssKB += value
		case "Anonymous:":
			usage.AnonymousKB += value
			cur.AnonymousKB += value
		case "Locked:":
			usage.LockedKB += value
			cur.LockedKB += value
		}
	}
	return usage
}

// lookupMappingAttribution splits the model mappings by the lookup tables'
// file ranges. A mapping that maps only table bytes (up to one partial page at
// each end, since the loader's MADV_RANDOM split is page-aligned) is a table
// mapping; one that maps no table bytes is an execution mapping and is not
// judged here; anything else mixes both and cannot be attributed.
type lookupMappingAttribution struct {
	Known        bool  // ranges known for every table and every mapping's range parsed
	CoveredBytes int64 // table bytes mapped by any model mapping
	TableAnonKB  int64
	TableLockKB  int64
	MixedAnonKB  int64
	MixedLockKB  int64
}

const smapsPageBytes = 4096

func attributeLookupMappings(usage *mappedFileUsage, ranges []placement.SparseLookupRange) lookupMappingAttribution {
	a := lookupMappingAttribution{Known: len(ranges) > 0}
	if usage == nil || !a.Known {
		return a
	}
	byShard := map[string][]placement.SparseLookupRange{}
	for _, r := range ranges {
		byShard[r.Shard] = append(byShard[r.Shard], r)
	}
	for _, m := range usage.Maps {
		if m.Offset < 0 {
			a.Known = false
			continue
		}
		var overlap int64
		for _, r := range byShard[m.Shard] {
			lo := maxInt64(m.Offset, r.Offset)
			hi := minInt64(m.Offset+m.Length, r.Offset+r.Bytes)
			if hi > lo {
				overlap += hi - lo
			}
		}
		a.CoveredBytes += overlap
		switch {
		case overlap == 0:
		case m.Length-overlap < 2*smapsPageBytes:
			a.TableAnonKB += m.AnonymousKB
			a.TableLockKB += m.LockedKB
		default:
			a.MixedAnonKB += m.AnonymousKB
			a.MixedLockKB += m.LockedKB
		}
	}
	return a
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func minInt64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// sparseLookupContradiction applies the observation rules and returns the
// contradiction ("" if none) plus a note for observations it cannot attribute.
// A nil usage means the host could not expose mappings, which is not evidence
// either way. Only pages attributable to the lookup tables' own file ranges
// can contradict the loader: locked or anonymous execution weights in the same
// or another shard are not lookup evidence. Without table ranges, locks and
// anonymous pages stay unknown. The non-reclaimable envelope applies only to a
// paged tier: a resident tier is already charged in the plan's host footprint.
func sparseLookupContradiction(strategy *placement.Strategy, usage *mappedFileUsage, nonReclaimableMB int) (string, string) {
	if strategy == nil || !placement.StrategyUsesLazySparseLookup(strategy) {
		return "", ""
	}
	note := ""
	if usage != nil {
		tierBytes := strategy.SparseLookupBytes
		if tierBytes <= 0 {
			tierBytes = int64(strategy.SparseLookupMB-1) << 20 // MiB is rounded up
		}
		a := attributeLookupMappings(usage, strategy.SparseLookupRanges)
		switch {
		case usage.Mappings == 0:
			return "no model-file mapping is present, so the lookup tables cannot be file-backed", ""
		case a.Known && a.CoveredBytes < tierBytes-2*smapsPageBytes:
			return fmt.Sprintf("model-file mappings cover %d MiB of the %d MiB lookup tables", a.CoveredBytes>>20, tierBytes>>20), ""
		case !a.Known && usage.SizeKB*1024 < tierBytes:
			return fmt.Sprintf("model-file mappings cover %d MiB, less than the %d MiB lookup tier", usage.SizeKB/1024, strategy.SparseLookupMB), ""
		case a.Known && a.TableAnonKB > 0:
			return fmt.Sprintf("%d KiB of lookup-table mappings are anonymous copies", a.TableAnonKB), ""
		case a.Known && a.TableLockKB > 0:
			return fmt.Sprintf("%d KiB of lookup-table mappings are locked", a.TableLockKB), ""
		}
		switch {
		case a.Known && (a.MixedAnonKB > 0 || a.MixedLockKB > 0):
			note = fmt.Sprintf("%d KiB locked / %d KiB anonymous in mappings that mix lookup tables and other tensors; not attributable", a.MixedLockKB, a.MixedAnonKB)
		case !a.Known && (usage.AnonymousKB > 0 || usage.LockedKB > 0):
			note = fmt.Sprintf("%d KiB locked / %d KiB anonymous in model mappings; lookup table ranges unknown, not attributable", usage.LockedKB, usage.AnonymousKB)
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
				nonReclaimableMB, limit, strategy.SparseLookupMB), note
		}
	}
	return "", note
}

func observeModelMappings(model *placement.ModelProfile, root int) (*mappedFileUsage, error) {
	if model == nil || root <= 0 {
		return nil, fmt.Errorf("no backend process")
	}
	shards, _, err := modelstore.ResolveGGUFShardFiles(model.Path)
	if err != nil {
		return nil, err
	}
	files := map[string]string{}
	for _, shard := range shards {
		base := filepath.Base(shard)
		files[shard] = base
		if abs, err := filepath.Abs(shard); err == nil {
			files[abs] = base
		}
		if real, err := filepath.EvalSymlinks(shard); err == nil {
			files[real] = base
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
		total.Maps = append(total.Maps, u.Maps...)
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
	reason, note := sparseLookupContradiction(strategy, usage, measuredMB)
	if note != "" {
		fmt.Fprintf(os.Stderr, "[sparse] lookup-tier observation unknown: %s\n", note)
	}
	if reason != "" {
		evidence := "live contradiction: " + reason
		be.SparseLookupCapability = placement.SparseLookupAnonymous
		be.SparseLookupEvidence = evidence
		if err := persistBackendSparseLookupCapability(cfg.CacheDir, model, be, placement.SparseLookupAnonymous, evidence); err != nil {
			fmt.Fprintf(os.Stderr, "[sparse] warning: could not persist lookup-tier contradiction: %v\n", err)
		}
		return fmt.Errorf("backend did not keep the %d MiB lookup tier file-backed (%s)", strategy.SparseLookupMB, evidence)
	}
	if usage != nil {
		fmt.Fprintf(os.Stderr, "[sparse] %s\n", sparseLookupObservationSummary(strategy, usage, measuredMB, note))
	}
	return nil
}

// sparseLookupObservationSummary states what a passing live check saw. It
// claims clean table pages only when every table byte was attributed; mixed
// or unattributed mappings stay explicitly unknown (never a rejection).
func sparseLookupObservationSummary(strategy *placement.Strategy, usage *mappedFileUsage, measuredMB int, note string) string {
	verdict := "no table page anonymous or locked"
	switch {
	case note != "":
		verdict = "table pages unknown: " + note
	case !attributeLookupMappings(usage, strategy.SparseLookupRanges).Known:
		verdict = "table pages unknown: lookup-table ranges not attributable; no model-mapping page anonymous or locked"
	}
	return fmt.Sprintf("live lookup-tier check (%s): %d model mappings, %d MiB mapped, %d MiB resident; %s; %d MiB non-reclaimable",
		strategy.SparseLookupPolicy, usage.Mappings, usage.SizeKB/1024, usage.RssKB/1024, verdict, measuredMB)
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

// sparseLookupRangesFromInfo converts parser ranges into placement's type.
func sparseLookupRangesFromInfo(in []gguf.SparseLookupRange) []placement.SparseLookupRange {
	if len(in) == 0 {
		return nil
	}
	out := make([]placement.SparseLookupRange, len(in))
	for i, r := range in {
		out[i] = placement.SparseLookupRange{Shard: r.Shard, Offset: r.Offset, Bytes: r.Bytes}
	}
	return out
}
