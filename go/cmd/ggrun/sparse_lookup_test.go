package main

import (
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

const smapsFixture = `7f0000000000-7f1000000000 r--s 00000000 08:02 212 /models/ds-00008-of-00010.gguf
Size:           102001640 kB
Rss:              131072 kB
Anonymous:             0 kB
Locked:                0 kB
7f2000000000-7f2000100000 rw-p 00000000 00:00 0
Size:               1024 kB
Rss:                1024 kB
Anonymous:          1024 kB
Locked:                0 kB
7f3000000000-7f4000000000 r--s 00000000 08:02 213 /models/ds-00009-of-00010.gguf
Size:           102004432 kB
Rss:               65536 kB
Anonymous:             0 kB
Locked:                0 kB
7f5000000000-7f5000100000 r--s 00000000 08:02 999 /models/unrelated.gguf
Size:               1024 kB
Rss:                1024 kB
Anonymous:             0 kB
Locked:             1024 kB
`

// The pinned fork's real --help/--version lines.
const (
	reviewedLazyHelp    = "-lm,   --load-mode MODE   model loading mode (default: auto)\n-lzm,  --lazy-mode MODE   on-demand reading of certain tensors, for example per-layer embeddings"
	reviewedLazyVersion = "version: 0.4.0-dev (build 11074, commit 425badfe)\nbuilt with GNU 13.3.0 for Linux x86_64"
	engramTierBytes     = int64(208902215200) // pinned IQ2_XS headers: 199,225 MiB rounded up
)

func reviewedLazyBackend() *backendInfo {
	return &backendInfo{Path: "/b/llama-server", Identity: "llama-server-test", Help: reviewedLazyHelp, Version: reviewedLazyVersion}
}

func engramModel() *placement.ModelProfile {
	return &placement.ModelProfile{ModelArch: "deepseek41", SparseLookupBytes: engramTierBytes, SparseLookupKind: "engram"}
}

func lookupStrategy() *placement.Strategy {
	return &placement.Strategy{
		SparseLookupMB: 199225, SparseLookupBytes: engramTierBytes,
		SparseLookupPolicy: placement.SparseLookupPaged, SparseLookupCapability: placement.SparseLookupFileBacked,
		SparseLookupFlag: placement.SparseLookupLazyFlag, PlannedHostFootprintMB: 140000, CRAM: 8192,
	}
}

func TestSmapsCountsOnlyModelFileMappings(t *testing.T) {
	files := map[string]string{"/models/ds-00008-of-00010.gguf": "ds-00008-of-00010.gguf", "/models/ds-00009-of-00010.gguf": "ds-00009-of-00010.gguf"}
	u := parseSmapsModelUsage(strings.NewReader(smapsFixture), files)
	if u.Mappings != 2 || u.SizeKB != 204006072 || u.RssKB != 196608 || u.AnonymousKB != 0 || u.LockedKB != 0 || len(u.Maps) != 2 {
		t.Fatalf("model mappings mis-summed (anonymous heap and unrelated files must not count): %+v", u)
	}
	if m := u.Maps[0]; m.Shard != "ds-00008-of-00010.gguf" || m.Offset != 0 || m.Length != 0x1000000000 {
		t.Fatalf("mapping range not parsed: %+v", m)
	}
}

// Split-VMA fixture: the loader's MADV_RANDOM on each table range splits the
// shard mapping, so smaps lists the table separately from the execution
// weights around it. The table is 64 MiB at file offset 1 MiB + 512 B (not
// page aligned): its mapping starts one partial page early.
const (
	tableShard  = "ds-00001-of-00010.gguf"
	tableOffset = int64(1<<20 + 512)
	tableBytes  = int64(64 << 20)
)

func splitSmaps(execLockedKB, tableLockedKB, tableAnonKB int) string {
	return fmt.Sprintf(`7f0000000000-7f0000100000 r--s 00000000 08:02 300 /models/ds-00001-of-00010.gguf
Size:               1024 kB
Rss:                1024 kB
Anonymous:             0 kB
Locked:             %d kB
7f0000100000-7f0004101000 r--s 00100000 08:02 300 /models/ds-00001-of-00010.gguf
Size:              65540 kB
Rss:                4096 kB
Anonymous:            %d kB
Locked:               %d kB
7f0004101000-7f0004201000 r--s 04101000 08:02 300 /models/ds-00001-of-00010.gguf
Size:               1024 kB
Rss:                1024 kB
Anonymous:             0 kB
Locked:             %d kB
7f1000000000-7f1000100000 r--s 00000000 08:02 301 /models/ds-00002-of-00010.gguf
Size:               1024 kB
Rss:                1024 kB
Anonymous:             0 kB
Locked:             %d kB
`, execLockedKB, tableAnonKB, tableLockedKB, execLockedKB, execLockedKB)
}

func splitFiles() map[string]string {
	return map[string]string{"/models/ds-00001-of-00010.gguf": tableShard, "/models/ds-00002-of-00010.gguf": "ds-00002-of-00010.gguf"}
}

func rangedStrategy(policy placement.SparseLookupPolicy) *placement.Strategy {
	return &placement.Strategy{
		SparseLookupMB: 64, SparseLookupBytes: tableBytes, SparseLookupPolicy: policy,
		SparseLookupCapability: placement.SparseLookupFileBacked, PlannedHostFootprintMB: 140000, CRAM: 8192,
		SparseLookupRanges: []placement.SparseLookupRange{{Shard: tableShard, Offset: tableOffset, Bytes: tableBytes}},
	}
}

// PR #93 re-review finding 3: only pages attributable to the lookup tables'
// own file ranges can contradict the loader.
func TestSparseLookupObservationAttributesTableRanges(t *testing.T) {
	for _, tc := range []struct {
		name       string
		smaps      string
		strategy   *placement.Strategy
		nonMB      int
		want, note string
	}{
		{"clean table", splitSmaps(0, 0, 0), rangedStrategy(placement.SparseLookupPaged), 140000, "", ""},
		// Re-review reproducer: --mlock execution weights in the table's shard
		// and in another shard are not lookup evidence.
		{"locked execution weights beside clean tables", splitSmaps(1024, 0, 0), rangedStrategy(placement.SparseLookupPaged), 140000, "", ""},
		{"locked table pages", splitSmaps(0, 4, 0), rangedStrategy(placement.SparseLookupPaged), 140000, "lookup-table mappings are locked", ""},
		{"anonymous table pages", splitSmaps(0, 0, 4), rangedStrategy(placement.SparseLookupResident), 0, "anonymous copies", ""},
		{"table range unmapped", splitSmaps(0, 0, 0), func() *placement.Strategy {
			s := rangedStrategy(placement.SparseLookupPaged)
			s.SparseLookupRanges[0].Shard = "ds-00003-of-00010.gguf"
			return s
		}(), 140000, "cover 0 MiB", ""},
		{"copied into anonymous memory", splitSmaps(0, 0, 0), rangedStrategy(placement.SparseLookupPaged), 340000, "non-reclaimable exceeds", ""},
		// A whole-shard mapping that was never split mixes table and other
		// bytes: its locks are unknown, not a contradiction.
		{"unsplit mapping with locks", smapsFixture + `7f6000000000-7f6000100000 r--s 00000000 08:02 214 /models/ds-00001-of-00010.gguf
Size:               1024 kB
Rss:                1024 kB
Anonymous:             0 kB
Locked:             1024 kB
`, func() *placement.Strategy {
			s := rangedStrategy(placement.SparseLookupPaged)
			s.SparseLookupRanges[0].Offset = 0
			s.SparseLookupRanges[0].Bytes = 8 << 10
			s.SparseLookupBytes, s.SparseLookupMB = 8<<10, 1
			return s
		}(), 140000, "", "not attributable"},
		{"ranges unknown", splitSmaps(1024, 4, 0), func() *placement.Strategy {
			s := rangedStrategy(placement.SparseLookupPaged)
			s.SparseLookupRanges = nil
			return s
		}(), 140000, "", "ranges unknown"},
	} {
		files := splitFiles()
		files["/models/ds-00008-of-00010.gguf"] = "ds-00008-of-00010.gguf"
		files["/models/ds-00009-of-00010.gguf"] = "ds-00009-of-00010.gguf"
		u := parseSmapsModelUsage(strings.NewReader(tc.smaps), files)
		got, note := sparseLookupContradiction(tc.strategy, &u, tc.nonMB)
		if tc.want == "" && got != "" || tc.want != "" && !strings.Contains(got, tc.want) {
			t.Errorf("%s: contradiction %q, want %q", tc.name, got, tc.want)
		}
		if tc.note == "" && note != "" || tc.note != "" && !strings.Contains(note, tc.note) {
			t.Errorf("%s: note %q, want %q", tc.name, note, tc.note)
		}
	}
}

func TestSparseLookupObservationRules(t *testing.T) {
	files := map[string]string{"/models/ds-00008-of-00010.gguf": "ds-00008-of-00010.gguf", "/models/ds-00009-of-00010.gguf": "ds-00009-of-00010.gguf"}
	healthy := parseSmapsModelUsage(strings.NewReader(smapsFixture), files)
	cases := []struct {
		name  string
		usage *mappedFileUsage
		nonMB int
		want  string
	}{
		{"healthy", &healthy, 150000, ""},
		{"no observation is not proof either way", nil, 0, ""},
		{"no mapping", &mappedFileUsage{}, 150000, "no model-file mapping"},
		{"short mapping without ranges", &mappedFileUsage{Mappings: 1, SizeKB: 1024 * 1024}, 150000, "less than"},
		// Without table ranges, locks and anonymous pages are not attributable.
		{"locks without ranges", &mappedFileUsage{Mappings: 2, SizeKB: 204006072, LockedKB: 4}, 150000, ""},
		{"copied into anonymous memory", &healthy, 340000, "non-reclaimable exceeds"},
		{"copy seen only through memory.stat", nil, 340000, "non-reclaimable exceeds"},
	}
	for _, tc := range cases {
		got, _ := sparseLookupContradiction(lookupStrategy(), tc.usage, tc.nonMB)
		if tc.want == "" && got != "" || tc.want != "" && !strings.Contains(got, tc.want) {
			t.Errorf("%s: got %q, want %q", tc.name, got, tc.want)
		}
	}
	// A resident tier is charged in the host footprint: its non-reclaimable
	// total is not a contradiction, but a missing mapping still is.
	resident := lookupStrategy()
	resident.SparseLookupPolicy = placement.SparseLookupResident
	resident.PlannedHostFootprintMB = 340000
	if got, _ := sparseLookupContradiction(resident, &healthy, 345000); got != "" {
		t.Fatalf("resident tier judged by the paged envelope: %q", got)
	}
	if got, _ := sparseLookupContradiction(resident, &mappedFileUsage{}, 0); !strings.Contains(got, "no model-file mapping") {
		t.Fatalf("resident lazy tier without a mapping: %q", got)
	}
	layer := lookupStrategy()
	layer.SparseLookupPolicy = placement.SparseLookupLayer
	if got, _ := sparseLookupContradiction(layer, nil, 999999); got != "" {
		t.Fatalf("tables loaded with their layers must not be judged as lazy: %q", got)
	}
	if got, _ := sparseLookupContradiction(&placement.Strategy{}, nil, 999999); got != "" {
		t.Fatalf("a plan without a lookup tier must not be judged by it: %q", got)
	}
}

// PR #93 review finding 5: the reclaim band is reserved before any lookup
// cache is granted, the user's ceiling is never exceeded, and an insufficient
// ceiling never throttles the non-reclaimable working set.
func TestSparseLookupScopeReservesReclaimBand(t *testing.T) {
	s := lookupStrategy()
	for _, tc := range []struct {
		name                    string
		nonRecl, headroom, ceil int
		keepMax                 bool
		high, max, lookup       int
		short                   bool
	}{
		// Review reproducer: 150,000 non-reclaimable, 4,096 band, 200,000 ceiling.
		{"constrained host keeps the band", 150000, 4096, 200000, false, 195904, 200000, 45904, false},
		{"roomy host caps the cache at the tier", 150000, 4096, 400000, false, 349225, 353321, 199225, false},
		{"explicit small budget", 60000, 4096, 70000, false, 65904, 70000, 5904, false},
		{"ceiling below state + band", 198000, 4096, 200000, false, 198000, 200000, 0, true},
		{"state above ceiling is clamped", 210000, 4096, 200000, false, 200000, 200000, 0, true},
		{"no configured band", 150000, 0, 200000, false, 200000, 200000, 50000, false},
		{"mapped CPU experts keep their hard ceiling", 150000, 4096, 200000, true, 195904, 0, 45904, false},
	} {
		high, max, lookup, short := sparseLookupScopeLimits(s, tc.nonRecl, tc.headroom, tc.ceil, tc.keepMax)
		if high != tc.high || max != tc.max || lookup != tc.lookup || short != tc.short {
			t.Errorf("%s: high=%d max=%d lookup=%d short=%v, want %d %d %d %v",
				tc.name, high, max, lookup, short, tc.high, tc.max, tc.lookup, tc.short)
		}
		if !tc.keepMax && max > tc.ceil {
			t.Errorf("%s: memory.max %d exceeds the user ceiling %d", tc.name, max, tc.ceil)
		}
		if !tc.short && !tc.keepMax && max-high < tc.headroom {
			t.Errorf("%s: band %d below configured %d", tc.name, max-high, tc.headroom)
		}
	}
	for _, other := range []*placement.Strategy{{}, {SparseLookupMB: 199225, SparseLookupPolicy: placement.SparseLookupResident}} {
		if high, _, _, _ := sparseLookupScopeLimits(other, 150000, 4096, 200000, false); high != 0 {
			t.Fatalf("policy %q keeps the ordinary Fix-B rule, got high=%d", other.SparseLookupPolicy, high)
		}
	}
	// Missing telemetry: no measured state or no platform ceiling leave the
	// scope alone.
	if high, max, _, _ := sparseLookupScopeLimits(s, 0, 4096, 200000, false); high != 0 || max != 0 {
		t.Fatalf("unmeasured state resized the scope: high=%d max=%d", high, max)
	}
	if high, max, _, _ := sparseLookupScopeLimits(s, 150000, 4096, 0, false); high != 0 || max != 0 {
		t.Fatalf("no platform ceiling resized the scope: high=%d max=%d", high, max)
	}
}

func TestSparseLookupStartLimitsKeepABand(t *testing.T) {
	for _, tc := range []struct {
		name                       string
		budget, hostCeil, headroom int
		high, max                  int
	}{
		{"percent target below free RAM", 190000, 205000, 4096, 190000, 205000},
		{"thin derived band widened to the headroom", 200000, 202000, 4096, 197904, 202000},
		{"explicit budget (no host ceiling)", 150000, 0, 4096, 145904, 150000},
		{"no headroom configured", 150000, 0, 0, 150000, 150000},
		{"no scope on this platform", 0, 0, 4096, 0, 0},
	} {
		high, max := sparseLookupStartLimits(tc.budget, tc.hostCeil, tc.headroom)
		if high != tc.high || max != tc.max {
			t.Errorf("%s: high=%d max=%d, want %d %d", tc.name, high, max, tc.high, tc.max)
		}
	}
}

func TestLazyLookupArgvGetsReclaimBandAtStart(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("backend memory scopes and the reviewed loader are Linux-only")
	}
	req := &launchRequest{RAMLimitPercent: 90, CgroupHeadroomMB: 4096}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 200000, FreeMB: 200000}}
	be := reviewedLazyBackend()
	plain := backendStartOptions(req, caps, be, nil, []string{"--no-mmap"})
	lazy := backendStartOptions(req, caps, be, nil, []string{"--no-mmap", "--lazy-mode", "on"})
	if plain.MemoryHighMB != plain.MemoryMaxMB {
		t.Fatalf("resident plan keeps high == max: %+v", plain)
	}
	if lazy.MemoryHighMB != plain.MemoryHighMB || lazy.MemoryMaxMB-lazy.MemoryHighMB < req.CgroupHeadroomMB {
		t.Fatalf("lazy lookup plan needs a reclaim band above the same high: plain=%+v lazy=%+v", plain, lazy)
	}
	// An explicit budget is never exceeded; the band comes out of it.
	req.RamBudgetMB = 150000
	budgeted := backendStartOptions(req, caps, be, nil, []string{"--lazy-mode", "on"})
	if budgeted.MemoryMaxMB != 150000 || budgeted.MemoryHighMB != 150000-4096 {
		t.Fatalf("explicit budget: %+v", budgeted)
	}
	// Help alone (no reviewed contract) never earns the band.
	unreviewed := &backendInfo{Path: "/x/llama-server", Help: reviewedLazyHelp, Version: "version: 1 (build 1, commit deadbeef)"}
	req.RamBudgetMB = 0
	if got := backendStartOptions(req, caps, unreviewed, nil, []string{"--lazy-mode", "on"}); got.MemoryHighMB != got.MemoryMaxMB {
		t.Fatalf("unreviewed lazy loader got a reclaim band: %+v", got)
	}
}

func TestSparseLookupArgvNeedsReviewedLoaderAndLastValue(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the reviewed loader contract covers Linux only")
	}
	be := reviewedLazyBackend()
	for _, tc := range []struct {
		args []string
		want bool
	}{
		{[]string{"--no-mmap", "--lazy-mode", "on"}, true},
		{[]string{"--lazy-mode=on"}, true},
		{[]string{"--lazy-mode", "on", "-lzm", "off"}, false},
		{[]string{"--lazy-mode", "auto"}, false},
		{nil, false},
	} {
		if got := argsLazySparseLookup(be, tc.args); got != tc.want {
			t.Errorf("%v: got %v want %v", tc.args, got, tc.want)
		}
	}
	if argsLazySparseLookup(&backendInfo{Help: reviewedLazyHelp}, []string{"--lazy-mode", "on"}) {
		t.Fatal("a backend without a reviewed loader must not get the lookup reclaim band")
	}
}

// PR #93 review finding 2: an advertised flag is availability, not loader
// semantics. Only a reviewed contract for the exact build, table kind and
// platform authorizes a file-backed tier.
func TestSparseLookupCapabilityNeedsReviewedContract(t *testing.T) {
	model := engramModel()
	for _, tc := range []struct {
		name string
		be   *backendInfo
		want placement.SparseLookupCapability
		flag string
	}{
		{"review reproducer: unreviewed build advertising the flag",
			&backendInfo{Path: "/unknown/llama-server", Identity: "unreviewed-build", Help: "--lazy-mode MODE defer tensor initialization"},
			placement.SparseLookupUnknown, placement.SparseLookupLazyFlag},
		{"other commit", &backendInfo{Help: reviewedLazyHelp, Version: "version: 9 (build 9, commit 0123abcd)"}, placement.SparseLookupUnknown, placement.SparseLookupLazyFlag},
		{"no flag", &backendInfo{Help: "--no-mmap --mlock", Version: reviewedLazyVersion}, placement.SparseLookupUnknown, ""},
		{"broken help", &backendInfo{Help: "error while loading shared libraries", Version: reviewedLazyVersion}, placement.SparseLookupUnknown, ""},
		{"empty probe", &backendInfo{}, placement.SparseLookupUnknown, ""},
	} {
		got, evidence, flag := sparseLookupCapabilityFor(tc.be, model)
		if got != tc.want || flag != tc.flag || evidence == "" {
			t.Errorf("%s: got %q flag %q (%s), want %q flag %q", tc.name, got, flag, evidence, tc.want, tc.flag)
		}
	}
	got, evidence, _ := sparseLookupCapabilityFor(reviewedLazyBackend(), model)
	want := placement.SparseLookupUnknown
	if runtime.GOOS == "linux" {
		want = placement.SparseLookupFileBacked
	}
	if got != want {
		t.Fatalf("reviewed build on %s: got %q (%s), want %q", runtime.GOOS, got, evidence, want)
	}
	// The review covered Linux only.
	if c, _ := sparseLookupLoaderFor(reviewedLazyBackend(), "engram", "windows"); c != nil {
		t.Fatal("Windows matched a Linux-only loader review")
	}
	// The contract covers the reviewed table kind only.
	other := engramModel()
	other.SparseLookupKind = "synthetic-rowtable"
	if got, _, _ := sparseLookupCapabilityFor(reviewedLazyBackend(), other); got != placement.SparseLookupUnknown {
		t.Fatalf("another table kind inherited the engram contract: %q", got)
	}
	// A persisted live contradiction overrides the contract.
	contradicted := reviewedLazyBackend()
	contradicted.SparseLookupCapability, contradicted.SparseLookupEvidence = placement.SparseLookupAnonymous, "live contradiction: test"
	if got, _, _ := sparseLookupCapabilityFor(contradicted, model); got != placement.SparseLookupAnonymous {
		t.Fatalf("persisted contradiction ignored: %q", got)
	}
	if got, evidence, flag := sparseLookupCapabilityFor(reviewedLazyBackend(), &placement.ModelProfile{}); got != placement.SparseLookupUnknown || evidence != "" || flag != "" {
		t.Fatalf("a model without a tier gets no lookup inputs: %q %q %q", got, evidence, flag)
	}
}

// SYNTHETIC: a second table kind with its own (test-only) reviewed loader.
// This exercises contract matching for a non-Engram adapter; it is not a
// claim that any real second architecture is supported.
func TestSyntheticSecondKindNeedsItsOwnContract(t *testing.T) {
	saved := reviewedSparseLookupLoaders
	t.Cleanup(func() { reviewedSparseLookupLoaders = saved })
	reviewedSparseLookupLoaders = append(append([]sparseLookupLoaderContract(nil), saved...), sparseLookupLoaderContract{
		ID: "synthetic-rowtable/v1", Commits: []string{"5e7e7e7e"}, Kinds: []string{"synthetic-rowtable"},
		Platforms: []string{runtime.GOOS}, Flag: placement.SparseLookupLazyFlag, Oracle: sparseLookupOracleResidentOnlyV1, Synthetic: true,
	})
	model := &placement.ModelProfile{ModelArch: "synthrow", SparseLookupBytes: 1 << 30, SparseLookupKind: "synthetic-rowtable"}
	be := &backendInfo{Help: reviewedLazyHelp, Version: "version: 1 (build 1, commit 5e7e7e7e)"}
	if got, _, _ := sparseLookupCapabilityFor(be, model); got != placement.SparseLookupFileBacked {
		t.Fatalf("synthetic kind with its contract: %q", got)
	}
	if got := sparseLookupOracleFor(be, model); got != sparseLookupOracleResidentOnlyV1 {
		t.Fatalf("synthetic oracle accounting: %q", got)
	}
	if got, _, _ := sparseLookupCapabilityFor(be, engramModel()); got != placement.SparseLookupUnknown {
		t.Fatalf("the synthetic build inherited the engram contract: %q", got)
	}
}

// PR #93 review finding 3: the fit oracle's Host row is converted only under
// an explicit, versioned accounting contract, and only once.
func TestFitPassHostRowFollowsOracleContract(t *testing.T) {
	s := lookupStrategy()
	devs := []preflightDevice{{Name: "CUDA0", ModelMB: 10316, ComputeMB: 2913}, {Name: "Host", ModelMB: 333327, ContextMB: 185, ComputeMB: 237}}
	got := oracleDevicesExcludingLookupTier(devs, sparseLookupOracleInclusiveV1, s)
	// 208,902,215,200 B is 199,224.6 MiB: the whole MiB (199,224) is removed,
	// never the rounded-up charge, so the resident row cannot be undercounted.
	if got[1].ModelMB != 333327-199224 || got[1].SparseLookupExcludedMB != 199224 || got[0] != devs[0] || devs[1].ModelMB != 333327 {
		t.Fatalf("inclusive oracle must drop exactly the tier without mutating input: %+v (input %+v)", got, devs)
	}
	if again := oracleDevicesExcludingLookupTier(got, sparseLookupOracleInclusiveV1, s); !reflect.DeepEqual(again, got) {
		t.Fatalf("second application changed the row: %+v -> %+v", got, again)
	}
	residentRow := []preflightDevice{{Name: "Host", ModelMB: 250000}}
	for name, tc := range map[string]struct {
		accounting sparseLookupOracleAccounting
		s          *placement.Strategy
		devs       []preflightDevice
	}{
		// Review reproducer: a resident-only or unknown oracle reporting
		// 250,000 MiB must not become 50,775.
		"resident-only oracle": {sparseLookupOracleResidentOnlyV1, s, residentRow},
		"unknown oracle":       {sparseLookupOracleUnknown, s, residentRow},
		"empty accounting":     {"", s, residentRow},
		"row lacks the tier":   {sparseLookupOracleInclusiveV1, s, []preflightDevice{{Name: "Host", ModelMB: 134102}}},
		"no tier":              {sparseLookupOracleInclusiveV1, &placement.Strategy{}, devs},
		"resident policy":      {sparseLookupOracleInclusiveV1, &placement.Strategy{SparseLookupMB: 199225, SparseLookupPolicy: placement.SparseLookupResident}, devs},
		"layer policy":         {sparseLookupOracleInclusiveV1, &placement.Strategy{SparseLookupMB: 199225, SparseLookupPolicy: placement.SparseLookupLayer}, devs},
	} {
		if got := oracleDevicesExcludingLookupTier(tc.devs, tc.accounting, tc.s); !reflect.DeepEqual(got, tc.devs) {
			t.Errorf("%s: rows changed to %+v", name, got)
		}
	}
	// The contract, not the loader capability, selects the accounting.
	if runtime.GOOS == "linux" {
		if got := sparseLookupOracleFor(reviewedLazyBackend(), engramModel()); got != sparseLookupOracleInclusiveV1 {
			t.Fatalf("reviewed fork oracle accounting: %q", got)
		}
	}
	if got := sparseLookupOracleFor(&backendInfo{Help: reviewedLazyHelp}, engramModel()); got != sparseLookupOracleUnknown {
		t.Fatalf("unreviewed oracle accounting: %q", got)
	}
}

func TestDryRunSummaryLabelsOnlyMappedWeightsReclaimable(t *testing.T) {
	s := lookupStrategy()
	s.Residency = "tight-resident"
	s.MMap = false
	s.ReclaimableHostWeightsMB = 132864
	s.CPUExpertMMapCapability = placement.CPUExpertMMapFileBacked
	s.ResourceLedger = &placement.ResourceLedger{Host: placement.HostResourceLedger{FreeMB: 209258, RequiredMB: 134927, Reclaimable: 132864}}
	text := strings.Join(optimizationSummaryLines("dry-run", s, true), "\n")
	if !strings.Contains(text, "reclaimable weights 0 MiB") {
		t.Fatalf("resident (--no-mmap) experts were labelled reclaimable:\n%s", text)
	}
	if !strings.Contains(text, "lookup tables 199225 MiB paged") {
		t.Fatalf("lookup tier missing from the summary:\n%s", text)
	}
	s.SparseLookupPolicy = placement.SparseLookupResident
	if text = strings.Join(optimizationSummaryLines("dry-run", s, true), "\n"); !strings.Contains(text, "199225 MiB resident") {
		t.Fatalf("resident tier label:\n%s", text)
	}
	s.SparseLookupPolicy = placement.SparseLookupLayer
	if text = strings.Join(optimizationSummaryLines("dry-run", s, true), "\n"); !strings.Contains(text, "loaded with their layers") {
		t.Fatalf("layer tier label:\n%s", text)
	}
	s.MMap = true
	if text = strings.Join(optimizationSummaryLines("dry-run", s, true), "\n"); !strings.Contains(text, "reclaimable weights 132864 MiB") {
		t.Fatalf("mapped experts on a file-backed loader are reclaimable:\n%s", text)
	}
}

// The pinned fork keeps --mmap/--no-mmap as deprecated aliases next to
// --load-mode (default auto); production mainline has only --load-mode; ik has
// neither. Both --load-mode builds must receive an explicit mapping choice.
func TestLoadModeBackendsIncludeDeprecatedAliasBuilds(t *testing.T) {
	fork := "--mmap, --no-mmap   DEPRECATED in favor of `--load-mode`\n-lm,   --load-mode MODE   model loading mode (default: auto)"
	mainline := "-lm,   --load-mode MODE   model loading mode (default: auto)"
	ik := "--no-mmap   do not memory-map model\n--mlock"
	if !backendParsesLoadMode(fork) || !backendParsesLoadMode(mainline) || backendParsesLoadMode(ik) {
		t.Fatalf("load-mode dialect: fork=%v mainline=%v ik=%v", backendParsesLoadMode(fork), backendParsesLoadMode(mainline), backendParsesLoadMode(ik))
	}
}

// PR #93 review finding 1, request side: a passthrough native load mode is a
// user constraint that the planner sees too, so planning, admission, cache
// identity and argv agree.
func TestPassthroughLoadModeBecomesPlannerConstraint(t *testing.T) {
	for _, tc := range []struct {
		extra             []string
		noMMap, forceMMap bool
		keep              []string
	}{
		{[]string{"--load-mode", "none"}, true, false, nil},
		{[]string{"--load-mode=mmap"}, false, true, nil},
		{[]string{"-lm", "mlock"}, true, false, []string{"--mlock"}},
		{[]string{"-lm=mmap+mlock"}, false, true, []string{"--mlock"}},
		{[]string{"--load-mode", "dio"}, true, false, []string{"--load-mode", "dio"}},
		{[]string{"--no-mmap", "--load-mode", "auto"}, false, false, nil},
		// Last spelling wins, as in the backend.
		{[]string{"--load-mode", "none", "--mmap"}, false, true, nil},
		{[]string{"--load-mode", "fast"}, false, false, []string{"--load-mode", "fast"}},
		{[]string{"--lazy-mode", "on", "-c", "4096"}, false, false, []string{"-c", "4096"}},
	} {
		req := &launchRequest{}
		got := normalizePlacementAwareExtraArgs(req, append([]string(nil), tc.extra...))
		if len(got) == 0 {
			got = nil
		}
		if req.NoMMap != tc.noMMap || req.ForceMMap != tc.forceMMap || !reflect.DeepEqual(got, tc.keep) {
			t.Errorf("%v: no-mmap=%v force=%v passthrough=%v, want %v %v %v", tc.extra, req.NoMMap, req.ForceMMap, got, tc.noMMap, tc.forceMMap, tc.keep)
		}
	}
}

func TestPassthroughLazyModeIsExplicitChoice(t *testing.T) {
	for _, tc := range []struct {
		extra []string
		want  string
	}{
		{[]string{"--lazy-mode", "on"}, "on"},
		{[]string{"-lzm=off"}, "off"},
		{[]string{"--lazy-mode", "on", "--lazy-mode", "auto"}, ""},
	} {
		req := &launchRequest{}
		if got := normalizePlacementAwareExtraArgs(req, append([]string(nil), tc.extra...)); len(got) != 0 || req.SparseLookupLazy != tc.want {
			t.Errorf("%v: lazy=%q passthrough=%v, want %q and no passthrough", tc.extra, req.SparseLookupLazy, got, tc.want)
		}
	}
	plain := requestedLaunchPolicyIdentity(&launchRequest{}, nil)
	if lazyOn := requestedLaunchPolicyIdentity(&launchRequest{SparseLookupLazy: "on"}, nil); lazyOn == plain {
		t.Fatal("an explicit lazy choice must scope the launch policy")
	}
}

// Cache and recovery identities treat both spellings of one loading state as
// one configuration, and accumulate every -ot like the backend does.
func TestMemoryFingerprintFoldsLoadModeSpellings(t *testing.T) {
	base := []string{"s", "-c", "4096"}
	same := [][]string{
		append(append([]string(nil), base...), "--no-mmap"),
		append(append([]string(nil), base...), "--load-mode", "none"),
		append(append([]string(nil), base...), "--mmap", "-lm=none"),
	}
	want := effectiveMemoryArgsFingerprint(same[0])
	for _, args := range same[1:] {
		if got := effectiveMemoryArgsFingerprint(args); got != want {
			t.Errorf("%v fingerprint %q != %q", args, got, want)
		}
	}
	if effectiveMemoryArgsFingerprint(append(append([]string(nil), base...), "--mlock", "--no-mmap")) == want {
		t.Fatal("mlock must change the allocation identity")
	}
	one := effectiveMemoryArgsFingerprint([]string{"s", "-ot", "a=CPU"})
	two := effectiveMemoryArgsFingerprint([]string{"s", "-ot", "b=CUDA0", "-ot", "a=CPU"})
	if one == two {
		t.Fatal("a second -ot was dropped from the identity")
	}
}

// PR #93 re-review finding 2: the whole request parser folds classic and
// native loading flags once, in argv order; a later native mode replaces
// earlier locks and retained native tokens.
func TestParseLaunchArgsFoldsLoadingModesInOrder(t *testing.T) {
	for _, tc := range []struct {
		args              []string
		noMMap, forceMMap bool
		want              string // folded mode of the emitted classic flags + passthrough
	}{
		{[]string{"model.gguf", "--load-mode", "none", "--mmap"}, false, true, "mmap"},
		{[]string{"model.gguf", "--load-mode", "mmap", "--no-mmap"}, true, false, "none"},
		{[]string{"model.gguf", "--mmap", "--load-mode", "none"}, true, false, "none"},
		{[]string{"model.gguf", "--no-mmap", "-lm=mmap"}, false, true, "mmap"},
		{[]string{"model.gguf", "--load-mode", "mlock", "--load-mode", "none"}, true, false, "none"},
		{[]string{"model.gguf", "--load-mode", "dio", "--load-mode", "mmap"}, false, true, "mmap"},
		{[]string{"model.gguf", "--load-mode", "mmap", "--load-mode", "dio"}, true, false, "dio"},
		{[]string{"model.gguf", "--mlock", "--load-mode", "mmap"}, false, true, "mmap"},
		{[]string{"model.gguf", "--load-mode", "none", "--mlock"}, true, false, "mlock"},
		{[]string{"model.gguf", "--no-mmap", "--load-mode", "auto"}, false, false, "mmap"},
		{[]string{"model.gguf", "--no-mmap=false"}, false, false, "mmap"},
		{[]string{"model.gguf", "--mmap", "--no-mmap=false"}, false, true, "mmap"},
	} {
		req, err := parseLaunchArgs(tc.args)
		if err != nil {
			t.Fatalf("%v: %v", tc.args, err)
		}
		emitted := []string{"s"}
		if req.NoMMap {
			emitted = append(emitted, "--no-mmap")
		} else if req.ForceMMap {
			emitted = append(emitted, "--mmap")
		}
		emitted = append(emitted, req.ExtraArgs...)
		mode, err := server.ParseLoadMode(emitted)
		if err != nil {
			t.Fatalf("%v: %v", tc.args, err)
		}
		if req.NoMMap != tc.noMMap || req.ForceMMap != tc.forceMMap || mode.Spelling() != tc.want {
			t.Errorf("%v -> no-mmap=%v force=%v emitted=%v (%s); want %v %v %s",
				tc.args, req.NoMMap, req.ForceMMap, emitted, mode.Spelling(), tc.noMMap, tc.forceMMap, tc.want)
		}
	}
}

// PR #93 re-review finding 1, command side: every evidence writer scopes by
// the plan's lookup policy exactly as placement's readers do.
func TestEvidenceTagScopesSparsePolicy(t *testing.T) {
	be := reviewedLazyBackend()
	base := scopedProbeBackendTagForStrategy(nil, nil, be, &placement.Strategy{})
	if strings.Contains(base, "lookup=") {
		t.Fatalf("plans without a tier keep their namespace: %q", base)
	}
	paged := scopedProbeBackendTagForStrategy(nil, nil, be, &placement.Strategy{SparseLookupPolicy: placement.SparseLookupPaged})
	resident := scopedProbeBackendTagForStrategy(nil, nil, be, &placement.Strategy{SparseLookupPolicy: placement.SparseLookupResident})
	if paged != placement.SparseLookupScopedTag(base, placement.SparseLookupPaged) || paged == resident {
		t.Fatalf("policy scoping: base=%q paged=%q resident=%q", base, paged, resident)
	}
	if placement.SparseLookupScopedTag(paged, placement.SparseLookupPaged) != paged {
		t.Fatal("policy scoping must be idempotent")
	}
}

// PR #93 final review P2, command side: a previous Claude run's served log is
// recorded under the plan's scoped tag, and the relaunch refresh (request
// options, full profile) must read that measurement back. Another policy and
// an ordinary model stay isolated.
func TestPreviousClaudePromptCacheReachesRuntimeRefresh(t *testing.T) {
	const served = "srv alloc: - prompt state size 9650.558 MiB exceeds cache size limit 3584.000 MiB, skipping\n"
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564}},
		RAM:  detect.RAMInfo{TotalMB: 400000, FreeMB: 380000},
	}
	model := func(tier int64) *placement.ModelProfile {
		m := engramModel()
		m.Path, m.SizeBytes, m.TotalSizeMB, m.NumLayers, m.CTXTrain = "/models/ds-00001-of-00010.gguf", 393292445504, 375072, 40, 1048576
		if tier == 0 {
			m.ModelArch, m.SparseLookupBytes, m.SparseLookupKind = "deepseek4", 0, ""
		}
		return m
	}
	plan := func(policy placement.SparseLookupPolicy) *placement.Strategy {
		s := lookupStrategy()
		s.ContextSize, s.UBatchSize, s.Parallel, s.KVQuality, s.KVPlacement = 65536, 512, 1, "high", "gpu"
		s.SparseLookupPolicy = policy
		if policy == "" {
			s.SparseLookupMB, s.SparseLookupBytes, s.SparseLookupCapability, s.SparseLookupFlag = 0, 0, "", ""
		}
		return s
	}
	for _, tc := range []struct {
		name           string
		tier           int64
		writer, reader placement.SparseLookupPolicy
		want           float64
	}{
		{"paged plan", engramTierBytes, placement.SparseLookupPaged, placement.SparseLookupPaged, 9650.558},
		{"other policy", engramTierBytes, placement.SparseLookupPaged, placement.SparseLookupResident, 0},
		{"ordinary model", 0, "", "", 9650.558},
	} {
		cfg := &config.Config{CacheDir: t.TempDir()}
		req := &launchRequest{ModelPath: "/models/ds-00001-of-00010.gguf", Port: 18080, ClaudeCode: true, KVPlacement: "gpu", KVQuality: "high"}
		be := reviewedLazyBackend()
		m := model(tc.tier)
		if !recordPreviousClaudePromptCache(req, cfg, m, plan(tc.writer), be, caps, served) {
			t.Fatalf("%s: the served measurement was not recorded as growth", tc.name)
		}
		before := plan(tc.reader)
		next := placement.RefreshRuntimeCachePolicy(caps, m, before, placementOptionsFromRequest(req, m, be, cfg.CacheDir))
		if got := next.MeasuredPromptCacheEntryMB; got < tc.want-1 || got > tc.want+1 { // stored at 0.1 MiB
			t.Fatalf("%s: refresh read %.3f MiB, want %.3f (writer %q)", tc.name, next.MeasuredPromptCacheEntryMB, tc.want,
				scopedProbeBackendTagForStrategy(req, m, be, plan(tc.writer)))
		}
		if tc.want > 0 && next.CRAM <= before.CRAM {
			t.Fatalf("%s: measured entry did not resize -cram: %d -> %d", tc.name, before.CRAM, next.CRAM)
		}
	}
}

// PR #93 final review: a passing live check claims clean table pages only
// when every table byte was attributed.
func TestSparseLookupObservationSummaryStaysUnknown(t *testing.T) {
	files := splitFiles()
	clean := parseSmapsModelUsage(strings.NewReader(splitSmaps(0, 0, 0)), files)
	if got := sparseLookupObservationSummary(rangedStrategy(placement.SparseLookupPaged), &clean, 140000, ""); !strings.Contains(got, "no table page anonymous or locked") {
		t.Fatalf("attributed clean tables: %q", got)
	}
	unranged := rangedStrategy(placement.SparseLookupPaged)
	unranged.SparseLookupRanges = nil
	for name, tc := range map[string]struct {
		s    *placement.Strategy
		note string
	}{
		"ranges unknown":          {unranged, ""},
		"mixed mapping with note": {rangedStrategy(placement.SparseLookupPaged), "4 KiB locked / 0 KiB anonymous in mappings that mix lookup tables and other tensors; not attributable"},
	} {
		got := sparseLookupObservationSummary(tc.s, &clean, 140000, tc.note)
		if strings.Contains(got, "no table page anonymous or locked") || !strings.Contains(got, "table pages unknown") {
			t.Errorf("%s: summary overstates the observation: %q", name, got)
		}
	}
}
