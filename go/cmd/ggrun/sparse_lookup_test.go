package main

import (
	"runtime"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
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

func lookupStrategy() *placement.Strategy {
	return &placement.Strategy{
		SparseLookupMB: 199225, SparseLookupCapability: placement.SparseLookupFileBacked,
		PlannedHostFootprintMB: 140000, CRAM: 8192,
	}
}

func TestSmapsCountsOnlyModelFileMappings(t *testing.T) {
	files := map[string]bool{"/models/ds-00008-of-00010.gguf": true, "/models/ds-00009-of-00010.gguf": true}
	u := parseSmapsModelUsage(strings.NewReader(smapsFixture), files)
	if u.Mappings != 2 || u.SizeKB != 204006072 || u.RssKB != 196608 || u.AnonymousKB != 0 || u.LockedKB != 0 {
		t.Fatalf("model mappings mis-summed (anonymous heap and unrelated files must not count): %+v", u)
	}
}

func TestSparseLookupObservationRules(t *testing.T) {
	healthy := &mappedFileUsage{Mappings: 2, SizeKB: 204006072, RssKB: 196608}
	cases := []struct {
		name  string
		usage *mappedFileUsage
		nonMB int
		want  string
	}{
		{"healthy", healthy, 150000, ""},
		{"no observation is not proof either way", nil, 0, ""},
		{"no mapping", &mappedFileUsage{}, 150000, "no model-file mapping"},
		{"short mapping", &mappedFileUsage{Mappings: 1, SizeKB: 1024 * 1024}, 150000, "less than"},
		{"anonymous copy", &mappedFileUsage{Mappings: 2, SizeKB: 204006072, AnonymousKB: 4}, 150000, "anonymous copies"},
		{"locked", &mappedFileUsage{Mappings: 2, SizeKB: 204006072, LockedKB: 4}, 150000, "locked"},
		{"copied into anonymous memory", healthy, 340000, "non-reclaimable exceeds"},
		{"copy seen only through memory.stat", nil, 340000, "non-reclaimable exceeds"},
	}
	for _, tc := range cases {
		got := sparseLookupContradiction(lookupStrategy(), tc.usage, tc.nonMB)
		if tc.want == "" && got != "" || tc.want != "" && !strings.Contains(got, tc.want) {
			t.Errorf("%s: got %q, want %q", tc.name, got, tc.want)
		}
	}
	if got := sparseLookupContradiction(&placement.Strategy{}, nil, 999999); got != "" {
		t.Fatalf("a plan without a lookup tier must not be judged by it: %q", got)
	}
}

func TestSparseLookupScopeSharesOneLedger(t *testing.T) {
	s := lookupStrategy()
	// Plenty of room: the allowance is capped by the table size.
	high, max, lookup := sparseLookupScopeLimits(s, 150000, 4096, 400000, false)
	if lookup != s.SparseLookupMB || high != 150000+s.SparseLookupMB || max != high+4096 {
		t.Fatalf("roomy host: high=%d max=%d lookup=%d", high, max, lookup)
	}
	// Typical host: the allowance is what the user's ceiling leaves, and the
	// hard ceiling never exceeds it.
	high, max, lookup = sparseLookupScopeLimits(s, 150000, 4096, 200000, false)
	if lookup != 50000 || high != 200000 || max != 200000 {
		t.Fatalf("bounded host: high=%d max=%d lookup=%d", high, max, lookup)
	}
	// No room left: the non-reclaimable ceiling is never raised past the
	// user's ceiling and the lookup cache gets nothing.
	high, max, lookup = sparseLookupScopeLimits(s, 210000, 4096, 200000, false)
	if lookup != 0 || high != 200000 || max != 200000 {
		t.Fatalf("over-committed host: high=%d max=%d lookup=%d", high, max, lookup)
	}
	// Mapped CPU experts keep their existing larger hard ceiling.
	if _, max, _ = sparseLookupScopeLimits(s, 150000, 4096, 200000, true); max != 0 {
		t.Fatalf("keepMax must preserve memory.max, got %d", max)
	}
	if high, _, _ = sparseLookupScopeLimits(&placement.Strategy{}, 150000, 4096, 200000, false); high != 0 {
		t.Fatal("plans without a lookup tier keep the ordinary Fix-B rule")
	}
}

func TestSparseLookupArgvNeedsCapabilityAndLastValue(t *testing.T) {
	be := &backendInfo{SparseLookupCapability: placement.SparseLookupFileBacked}
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
		if got := argsFileBackedSparseLookup(be, tc.args); got != tc.want {
			t.Errorf("%v: got %v want %v", tc.args, got, tc.want)
		}
	}
	if argsFileBackedSparseLookup(&backendInfo{}, []string{"--lazy-mode", "on"}) {
		t.Fatal("an unclassified backend must not get the lookup reclaim band")
	}
}

func TestSparseLookupCapabilityProbe(t *testing.T) {
	for _, tc := range []struct {
		help string
		want placement.SparseLookupCapability
	}{
		{"-lzm,  --lazy-mode MODE   on-demand reading of certain tensors\n--no-mmap", placement.SparseLookupFileBacked},
		{"--no-mmap  --mlock", placement.SparseLookupUnknown},
		{"--lazy-moded X", placement.SparseLookupUnknown},
		{"", placement.SparseLookupUnknown},
	} {
		got, evidence := probedSparseLookupCapability(&backendInfo{Path: "/b/llama-server", Help: tc.help})
		if got != tc.want || evidence == "" {
			t.Errorf("help %q: got %q (%s), want %q", tc.help, got, evidence, tc.want)
		}
	}
}

func TestLazyLookupArgvGetsReclaimBandAtStart(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("backend memory scopes are Linux-only")
	}
	req := &launchRequest{RAMLimitPercent: 90}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 200000, FreeMB: 200000}}
	be := &backendInfo{SparseLookupCapability: placement.SparseLookupFileBacked}
	plain := backendStartOptions(req, caps, be, nil, []string{"--no-mmap"})
	lazy := backendStartOptions(req, caps, be, nil, []string{"--no-mmap", "--lazy-mode", "on"})
	if plain.MemoryHighMB != plain.MemoryMaxMB {
		t.Fatalf("resident plan keeps high == max: %+v", plain)
	}
	if lazy.MemoryHighMB != plain.MemoryHighMB || lazy.MemoryMaxMB <= lazy.MemoryHighMB {
		t.Fatalf("lazy lookup plan needs a reclaim band above the same high: plain=%+v lazy=%+v", plain, lazy)
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
	if !strings.Contains(text, "lookup tables 199225 MiB file-backed") {
		t.Fatalf("lookup tier missing from the summary:\n%s", text)
	}
	s.MMap = true
	if text = strings.Join(optimizationSummaryLines("dry-run", s, true), "\n"); !strings.Contains(text, "reclaimable weights 132864 MiB") {
		t.Fatalf("mapped experts on a file-backed loader are reclaimable:\n%s", text)
	}
}

// Measured on the pinned fork (llama-fit-params, 425badfe) against the real
// IQ2_XS headers and the planned argv.
func TestFitPassHostRowExcludesOnlyTheLazyTier(t *testing.T) {
	be := &backendInfo{SparseLookupCapability: placement.SparseLookupFileBacked}
	s := lookupStrategy()
	devs := []preflightDevice{{Name: "CUDA0", ModelMB: 10316, ComputeMB: 2913}, {Name: "Host", ModelMB: 333327, ContextMB: 185, ComputeMB: 237}}
	lazy := []string{"--no-mmap", "--lazy-mode", "on"}
	got := oracleDevicesExcludingLookupTier(devs, be, s, lazy)
	if got[1].ModelMB != 134102 || got[0] != devs[0] || devs[1].ModelMB != 333327 {
		t.Fatalf("host row must drop exactly the lookup tier without mutating input: %+v (input %+v)", got, devs)
	}
	for name, tc := range map[string]struct {
		be   *backendInfo
		s    *placement.Strategy
		args []string
		devs []preflightDevice
	}{
		"lazy off":           {be, s, []string{"--lazy-mode", "off"}, devs},
		"unclassified":       {&backendInfo{}, s, lazy, devs},
		"no tier":            {be, &placement.Strategy{}, lazy, devs},
		"row lacks the tier": {be, s, lazy, []preflightDevice{{Name: "Host", ModelMB: 134102}}},
	} {
		if got := oracleDevicesExcludingLookupTier(tc.devs, tc.be, tc.s, tc.args); got[len(got)-1].ModelMB != tc.devs[len(tc.devs)-1].ModelMB {
			t.Errorf("%s: host row changed to %d", name, got[len(got)-1].ModelMB)
		}
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
