package placement

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// An explicit flag the verified scope key leaves out must never be silently
// replaced by a saved config; automatic values and matching flags keep reuse.
func TestVerifiedConfigExplicitConflict(t *testing.T) {
	saved := &VerifiedConfig{StrategyType: MoEOffload, KVPlacement: "cpu", CRAM: 4096, CRAMSet: true, MMProjPath: "/m/proj.gguf"}
	cases := []struct {
		name string
		vc   *VerifiedConfig
		opts Options
		want string // "" = reuse allowed, else the flag the conflict names
	}{
		{"no record", nil, Options{KVPlacement: "gpu"}, ""},
		{"no explicit flag", saved, Options{}, ""},
		{"automatic kv placement", saved, Options{KVPlacement: "auto"}, ""},
		{"matching kv placement", saved, Options{KVPlacement: "cpu"}, ""},
		{"matching kv placement any case", saved, Options{KVPlacement: " CPU "}, ""},
		{"conflicting kv placement", saved, Options{KVPlacement: "gpu"}, "--kv-placement gpu"},
		{"kv placement missing from record", &VerifiedConfig{StrategyType: MoEOffload}, Options{KVPlacement: "cpu"}, "--kv-placement cpu"},
		{"matching cache-ram", saved, Options{CacheRAMMB: 4096}, ""},
		{"conflicting cache-ram", saved, Options{CacheRAMMB: 8192}, "--cache-ram 8192"},
		{"served zero cache-ram replaces the flag", &VerifiedConfig{StrategyType: MoEOffload, CRAMSet: true}, Options{CacheRAMMB: 8192}, "--cache-ram 8192"},
		{"legacy record re-derives cache-ram", &VerifiedConfig{StrategyType: MoEOffload}, Options{CacheRAMMB: 8192}, ""},
		{"matching mmproj", saved, Options{MMProjPath: "/m/proj.gguf"}, ""},
		{"conflicting mmproj", saved, Options{MMProjPath: "/m/other.gguf"}, "--mmproj /m/other.gguf"},
		{"mmproj missing from record", &VerifiedConfig{StrategyType: MoEOffload}, Options{MMProjPath: "/m/proj.gguf"}, "--mmproj /m/proj.gguf"},
		{"cpu mode against a gpu record", saved, Options{CPUMode: true}, "--cpu"},
		{"cpu mode against a cpu record", &VerifiedConfig{StrategyType: CPUOnly, KVPlacement: "cpu"}, Options{CPUMode: true, KVPlacement: "gpu"}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := verifiedConfigExplicitConflict(tc.vc, tc.opts)
			if tc.want == "" && got != "" {
				t.Fatalf("reuse must be kept, got conflict %q", got)
			}
			if tc.want != "" && !strings.HasPrefix(got, tc.want) {
				t.Fatalf("conflict = %q, want it to name %q", got, tc.want)
			}
		})
	}
}

// The relaunch defect: a verified config planned with KV on the CPU was reused
// for a launch that asked for --kv-placement gpu, and only --no-cached-config
// applied the flag. The conflicting flag must re-plan; the same record still
// serves a launch that leaves KV placement automatic or asks for what it holds.
func TestComputeExplicitKVPlacementOverridesVerifiedConfig(t *testing.T) {
	dir := t.TempDir()
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "3090", VRAMTotalMB: 24576, VRAMUsedMB: 512, BandwidthMBps: 30000}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	model := &ModelProfile{
		Path: filepath.Join(dir, "target.gguf"), SizeBytes: 4 << 30, TotalSizeMB: 4096,
		ModelArch: "llama", NumLayers: 32, EmbeddingLength: 4096, ContextSize: 32768,
	}
	opts := Options{
		ContextSize: 8192, KVQuality: "mid", KVPlacement: "gpu", Parallel: 1,
		CacheDir: dir, BackendIdentity: "b", BackendTag: "llama",
	}
	key := NewCalibrationScopeKey(model, caps, opts, nil).String()
	first, err := Compute(caps, model, opts)
	if err != nil {
		t.Fatal(err)
	}
	if first.KVPlacement != "gpu" {
		t.Fatalf("fixture must plan KV on the GPU, got %q", first.KVPlacement)
	}
	// The saved record holds KV on the CPU, as the stale relaunch record did.
	vc := VerifiedConfigToRecord(key, model.Basename, first, "b", "/p", "", "")
	vc.KVPlacement = "cpu"
	if _, err := SaveVerifiedConfig(dir, vc); err != nil {
		t.Fatal(err)
	}
	opts.VerifiedConfigScopeKey = key

	explicit := opts
	explicit.KVPlacement = "gpu"
	got, err := Compute(caps, model, explicit)
	if err != nil {
		t.Fatalf("re-plan after the conflict failed: %v", err)
	}
	if got.VerifiedConfigReused || got.PlacementCacheHit {
		t.Fatal("an explicit --kv-placement gpu reused a record planned with KV on the CPU")
	}
	if got.KVPlacement != "gpu" {
		t.Fatalf("the re-plan dropped the explicit KV placement: %q", got.KVPlacement)
	}

	for _, kv := range []string{"", "auto", "cpu"} {
		keep := opts
		keep.KVPlacement = kv
		got, err := Compute(caps, model, keep)
		if err != nil {
			t.Fatalf("kv-placement %q: %v", kv, err)
		}
		if !got.VerifiedConfigReused || got.KVPlacement != "cpu" {
			t.Fatalf("kv-placement %q must keep reusing the saved config: reused=%v kv=%q",
				kv, got.VerifiedConfigReused, got.KVPlacement)
		}
	}
}

// The conflict check exists only for coordinates the scope key leaves out. The
// plan-shaping flags that are in the key (--ctx, --parallel, --kv-quality,
// --gpus) must keep moving it, so a different value is a clean miss; the ones
// that are not (--kv-placement, --cpu, --cache-ram, --mmproj) must stay out of
// it, or the conflict check would be dead code hiding a key change.
func TestVerifiedScopeKeyCoversOnlyTheInKeyFlags(t *testing.T) {
	model := &ModelProfile{Path: "m.gguf", SizeBytes: 1 << 30, TotalSizeMB: 1024, ModelArch: "llama", NumLayers: 32, EmbeddingLength: 4096}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "3090", VRAMTotalMB: 24576}, {Index: 1, Name: "3060", VRAMTotalMB: 12288}},
		RAM:  detect.RAMInfo{TotalMB: 65536, FreeMB: 60000},
		CPU:  detect.CPUInfo{Cores: 8},
	}
	base := Options{ContextSize: 8192, KVQuality: "mid", KVPlacement: "auto", Parallel: 1, BackendIdentity: "b"}
	key := func(o Options) string { return NewCalibrationScopeKey(model, caps, o, nil).String() }
	baseKey := key(base)

	inKey := map[string]func(*Options){
		"--ctx":        func(o *Options) { o.ContextSize = 16384 },
		"--parallel":   func(o *Options) { o.Parallel = 2; o.ParallelExplicit = true },
		"--kv-quality": func(o *Options) { o.KVQuality = "q8_0" },
		"--gpus":       func(o *Options) { o.GPUs = []int{1} },
	}
	for flag, set := range inKey {
		o := base
		set(&o)
		if key(o) == baseKey {
			t.Errorf("%s no longer moves the verified scope key: a saved config for another value would be reused", flag)
		}
	}
	outOfKey := map[string]func(*Options){
		"--kv-placement": func(o *Options) { o.KVPlacement = "gpu" },
		"--cpu":          func(o *Options) { o.CPUMode = true },
		"--cache-ram":    func(o *Options) { o.CacheRAMMB = 8192 },
		"--mmproj":       func(o *Options) { o.MMProjPath = "/m/proj.gguf" },
	}
	for flag, set := range outOfKey {
		o := base
		set(&o)
		if key(o) != baseKey {
			t.Errorf("%s now moves the scope key; drop it from verifiedConfigExplicitConflict", flag)
		}
	}
}
