package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// Golden launch plans over a synthetic hardware matrix (review W11, step 1).
//
// Each case is planned through the same steps, in the same order, that
// cmdDryRun --emit-server-argv-json runs, and records the same
// ggrun-server-plan-v1 fields: backend, server argv, environment additions and
// memory limit. The W11 planLaunch refactor must leave this file unchanged.
//
// Regenerate (Linux only):
//
//	go test ./cmd/ggrun -run TestGoldenPlanSyntheticMatrix -update-launch-plans
//
// When the file is missing or differs, the test also logs the computed golden
// as gzip+base64 lines, so it can be recovered from a CI log without a local
// build:
//
//	grep -o 'launch-plan-golden-b64 [0-9]* [A-Za-z0-9+/=]*' log | sort -k2,2n |
//	  awk '{printf "%s", $3}' | base64 -d | gunzip > testdata/launch_plans_golden.json
//
// and the logged sha256 checks the result.
//
// cmdDryRun cannot run in a test as is. Three inputs are injected, and these
// are the seams planLaunch needs:
//   - hardware: an --inventory file, read by loadInventory as dry-run does; the
//     detect.Detect call before it is skipped (it would probe this host).
//   - model: a synthetic ModelProfile stands in for parseModel, which needs a
//     GGUF on disk and the Python parser.
//   - backend: a fixed backendInfo stands in for selectBackendForModel, which
//     discovers and runs backend binaries. What resolveLaunchBackend does after
//     selection (applyBackendFeatureCompatibility, preflightBackendArch) runs.
//
// Excluded because they need a live probe, network or nvidia-smi: backend
// discovery, GGUF parsing, live VRAM readings (the fit guard's driver free
// VRAM and live usage are stubbed from the inventory), the Vulkan device
// listing (the fixtures are CUDA backends; a call fails the test), draft-model discovery (speculation
// stays off) and the "Model does not fit" process hint of placementErrorMessage
// (the raw error is recorded). Until planLaunch exists this guards the pieces
// cmdDryRun calls, not cmdDryRun's own ordering.

var updateGoldenPlans = flag.Bool("update-launch-plans", false, "rewrite testdata/launch_plans_golden.json")

const goldenPlanPath = "testdata/launch_plans_golden.json"

// goldenPlanCase is one dry run: a hardware inventory, a model and the ggrun
// flags that follow the model path.
type goldenPlanCase struct {
	name    string
	backend string // see goldenPlanBackend
	hw      string
	model   string
	args    []string
}

// goldenPlanRecord is one case's outcome. Plan mirrors the ggrun-server-plan-v1
// object cmdDryRun encodes, field for field and tag for tag.
type goldenPlanRecord struct {
	Case    string          `json:"case"`
	Args    []string        `json:"args"`
	Outcome string          `json:"outcome"`
	Error   string          `json:"error,omitempty"`
	Plan    *goldenPlanJSON `json:"plan,omitempty"`
}

type goldenPlanJSON struct {
	Schema               string                          `json:"schema"`
	ModelPath            string                          `json:"model_path"`
	BackendTag           string                          `json:"backend_tag"`
	BackendID            string                          `json:"backend_identity"`
	ClaudeProfile        string                          `json:"claude_profile,omitempty"`
	LaunchScope          string                          `json:"launch_scope,omitempty"`
	MemoryMaxMB          int                             `json:"memory_max_mb,omitempty"`
	Environment          map[string]string               `json:"environment"`
	ServerArgv           []string                        `json:"server_argv"`
	Residency            placement.ResidencyClass        `json:"residency,omitempty"`
	Bottleneck           string                          `json:"optimization_bottleneck,omitempty"`
	ResourceLedger       *placement.ResourceLedger       `json:"resource_ledger,omitempty"`
	OptimizationBoundary *placement.OptimizationBoundary `json:"optimization_boundary,omitempty"`
	CachedOptimization   *placement.CalibrationDecision  `json:"cached_optimization,omitempty"`
}

// Hardware inventories in the --inventory format. 128 GB host, 16 cores.
var goldenPlanInventories = map[string]string{
	"cpu":    goldenPlanInventory(),
	"gpu12":  goldenPlanInventory(goldenPlanGPU(0, 12288, 16, 504000)),
	"gpu24":  goldenPlanInventory(goldenPlanGPU(0, 24576, 16, 1008000)),
	"mixed3": goldenPlanInventory(goldenPlanGPU(0, 12288, 16, 504000), goldenPlanGPU(1, 24576, 16, 1008000), goldenPlanGPU(2, 12288, 8, 360000)),
}

func goldenPlanGPU(index, vramMB, lanes, memBW int) string {
	return fmt.Sprintf(`{"index":%d,"name":"NVIDIA synthetic %d GB","vram_total_mb":%d,"vram_used_mb":300,"driver":"570.00","pci_gen":3,"pci_lanes":%d,"bandwidth_mbps":%d,"mem_bandwidth_mbps":%d,"pci_bus_id":"00000000:%02X:00.0","compute_cap":"8.6"}`,
		index, vramMB/1024, vramMB, lanes, lanes*985, memBW, index+1)
}

func goldenPlanInventory(gpus ...string) string {
	return `{"os":"linux","arch":"amd64","gpus":[` + strings.Join(gpus, ",") +
		`],"ram":{"total_mb":131072,"free_mb":122880},"cpu":{"model":"synthetic 16-core","cores":16,"threads":32},"backends":[],"host_memory_bandwidth_mbps":80000}`
}

var goldenPlanModels = map[string]func() *placement.ModelProfile{
	// About 5 GB: fits one 12 GB card with room for context.
	"dense-small": func() *placement.ModelProfile {
		return goldenPlanDense("synthetic/dense-small.gguf", 32, 4096, 32, 14336, 130<<20, 131072)
	},
	// About 30 GB: larger than any one card, fits the three together.
	"dense-large": func() *placement.ModelProfile {
		return goldenPlanDense("synthetic/dense-large.gguf", 64, 8192, 64, 28672, 440<<20, 131072)
	},
	// About 80 GB MoE: experts must leave the GPUs.
	"moe": goldenPlanMoE,
}

func goldenPlanDense(path string, layers, hidden, heads, ff int, layerBytes int64, ctx int) *placement.ModelProfile {
	const vocab = 128256
	embd := int64(vocab) * int64(hidden) * 9 / 16 // about Q4_K
	m := &placement.ModelProfile{
		Path: path, Basename: strings.TrimSuffix(filepath.Base(path), ".gguf"), ModelArch: "llama", QuantType: "Q4_K_M",
		NumLayers: layers, CTXTrain: ctx, ContextSize: ctx,
		HiddenSize: hidden, EmbeddingLength: hidden, HeadCount: heads, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
		FeedForwardLength: ff, VocabSize: vocab, TokenEmbdBytes: embd, OutputBytes: embd,
	}
	for il := 0; il < layers; il++ {
		m.NonExpertLayerBytes = append(m.NonExpertLayerBytes, layerBytes)
		m.NonExpertBytes += layerBytes
	}
	m.NonExpertBytes += m.TokenEmbdBytes + m.OutputBytes
	m.SizeBytes = m.NonExpertBytes
	m.TotalSizeMB = int(m.SizeBytes >> 20)
	return m
}

// goldenPlanMoE follows syntheticMoEProfile in pkg/placement: 40 layers, three
// leading dense, about 2 GB of routed experts per MoE layer.
func goldenPlanMoE() *placement.ModelProfile {
	const n, dense = 40, 3
	m := &placement.ModelProfile{
		Path: "synthetic/moe.gguf", Basename: "moe", ModelArch: "qwen3moe", QuantType: "Q4_K_M",
		IsMoE: true, NumExperts: 128, ExpertUsedCount: 8,
		CTXTrain: 262144, ContextSize: 262144, NumLayers: n, LeadingDense: dense,
		HeadCount: 32, HeadCountKV: 4, KeyLength: 128, ValueLength: 128,
		EmbeddingLength: 4096, HiddenSize: 4096, FeedForwardLength: 12288, ExpertFF: 1536, ExpertSharedFF: 1536, HasShexp: 1,
		VocabSize: 151936, OutputBytes: 622329856, TokenEmbdBytes: 622329856,
	}
	for il := 0; il < n; il++ {
		nonExpert := int64(110 << 20)
		var routed, shexp, aux int64
		if il < dense {
			nonExpert = 400 << 20
		} else {
			routed, shexp, aux = 2<<30, 30<<20, 2<<20
		}
		m.NonExpertLayerBytes = append(m.NonExpertLayerBytes, nonExpert)
		m.RoutedExpertLayerBytes = append(m.RoutedExpertLayerBytes, routed)
		m.ShexpLayerBytes = append(m.ShexpLayerBytes, shexp)
		m.ExpertAuxLayerBytes = append(m.ExpertAuxLayerBytes, aux)
		m.ExpertLayerBytes = append(m.ExpertLayerBytes, routed+shexp)
		m.NonExpertBytes += nonExpert + aux
		m.ExpertBytes += routed + shexp
		m.ShexpBytes += shexp
		m.ExpertAuxBytes += aux
	}
	m.NonExpertBytes += m.OutputBytes + m.TokenEmbdBytes
	m.SizeBytes = m.NonExpertBytes + m.ExpertBytes
	m.TotalSizeMB = int(m.SizeBytes >> 20)
	return m
}

// goldenPlanBaseHelp is the help surface every fixture backend shares.
const goldenPlanBaseHelp = `-c, --ctx-size N
-ngl, --gpu-layers N
-ot, --override-tensor <tensor name pattern>=<buffer type>,...
--n-cpu-moe N
-ts, --tensor-split N0,N1,N2,...
-sm, --split-mode {none,layer,row}
-mg, --main-gpu INDEX
-fa, --flash-attn [on|off|auto]
-ctk, --cache-type-k TYPE
-ctv, --cache-type-v TYPE
--cache-reuse N
-cram, --cache-ram N
--ctx-checkpoints N
--swa-full
-np, --parallel N
-kvu, --kv-unified
--no-mmap
--mlock
--jinja
--reasoning [on|off|auto]
--reasoning-budget N
--samplers SAMPLERS
--metrics
-lv, --verbosity, --log-verbosity N
--op-offload, --no-op-offload
`

// goldenPlanMainlineFitHelp is mainline llama.cpp's fit surface
// (common/arg.cpp): a valued --fit and its target options, no --fit-margin.
const goldenPlanMainlineFitHelp = `-fit,   --fit [on|off]                  whether to adjust unset arguments to fit in device memory ('on' or 'off', default: 'on')
-fitt,  --fit-target MiB0,MiB1,MiB2,... target margin per device for --fit, comma-separated list of values, single value is broadcast across all devices, default: 1024
-fitc,  --fit-ctx N                     minimum ctx size that can be set by --fit option, default: 4096
`

// goldenPlanIKFitHelp is ik_llama.cpp's fit surface (common/common.cpp): a
// boolean --fit and --fit-margin, which binds the fit to the allocation guard.
const goldenPlanIKFitHelp = `       --fit-margin N         safety margin in MiB when auto-fitting model offloading
       --fit                  automatically determine which tensors to offload to the GPU(s)
`

// goldenPlanBackend is a fresh CUDA backend per case. Paths are under
// /usr/bin, so no library directory is derived from them, and do not exist,
// so no architecture probe reads a file.
//   - "" (mainline): llama.cpp with its valued --fit.
//   - "ik": ik_llama.cpp with boolean --fit and --fit-margin.
//   - "nofit": a backend without --fit. Real backends all advertise it; the
//     case records a known latent defect (see goldenPlanCases).
func goldenPlanBackend(kind string) *backendInfo {
	switch kind {
	case "ik":
		return &backendInfo{
			Path: "/usr/bin/ggrun-golden-ik-llama-server", Tag: "ik_llama", Dialect: "ik_llama", IsIK: true,
			Identity: "golden-ik-build", SupportsReasoning: true, Help: goldenPlanBaseHelp + goldenPlanIKFitHelp,
		}
	case "nofit":
		return &backendInfo{
			Path: "/usr/bin/ggrun-golden-nofit-llama-server", Tag: "llama", Dialect: "llama",
			Identity: "golden-nofit-build", SupportsReasoning: true, Help: goldenPlanBaseHelp,
		}
	}
	return &backendInfo{
		Path: "/usr/bin/ggrun-golden-llama-server", Tag: "llama", Dialect: "llama",
		Identity: "golden-mainline-build", SupportsReasoning: true, Help: goldenPlanBaseHelp + goldenPlanMainlineFitHelp,
	}
}

func goldenPlanCases() []goldenPlanCase {
	var cases []goldenPlanCase
	addOn := func(backend, hw, model string, args ...string) {
		flags := []string{}
		for _, a := range args {
			flags = append(flags, strings.TrimLeft(a, "-"))
		}
		name := hw + "/" + model + "/auto"
		if len(flags) > 0 {
			name = hw + "/" + model + "/" + strings.Join(flags, "-")
		}
		if backend != "" {
			name = backend + "-backend/" + name
		}
		cases = append(cases, goldenPlanCase{name: name, backend: backend, hw: hw, model: model, args: args})
	}
	add := func(hw, model string, args ...string) { addOn("", hw, model, args...) }
	// Every hardware class with every model, automatic context.
	for _, hw := range []string{"cpu", "gpu12", "gpu24", "mixed3"} {
		for _, model := range []string{"dense-small", "dense-large", "moe"} {
			add(hw, model)
		}
	}
	// A typed context.
	add("gpu12", "dense-small", "--ctx", "32768")
	add("mixed3", "dense-large", "--ctx", "32768")
	add("gpu24", "moe", "--ctx", "32768")
	// Claude Code, automatic context and typed.
	add("cpu", "dense-small", "--claude-code")
	add("gpu12", "dense-small", "--claude-code")
	add("gpu24", "dense-large", "--claude-code")
	add("mixed3", "moe", "--claude-code")
	add("mixed3", "moe", "--claude-code", "--ctx", "32768")
	// Explicit KV placement.
	add("gpu24", "dense-large", "--kv-placement", "cpu")
	// ik_llama: boolean --fit bound to the allocation guard by --fit-margin.
	addOn("ik", "gpu12", "dense-small", "--claude-code")
	addOn("ik", "gpu24", "dense-large")
	addOn("ik", "mixed3", "moe")
	// Known latent defect: DenseCPUOffload on a backend without --fit emits
	// -ngl 999 (every layer on the GPU) for a model larger than the card. A fix
	// is pending; this case changes when it lands.
	addOn("nofit", "gpu12", "dense-large")
	return cases
}

// goldenPlanUnsetenv removes keys for the rest of the test and restores them
// afterwards. t.Setenv alone would leave KEY= in the environment, which the
// plan's environment map would then carry.
func goldenPlanUnsetenv(t *testing.T, keys ...string) {
	t.Helper()
	for _, key := range keys {
		t.Setenv(key, "")
		if err := os.Unsetenv(key); err != nil {
			t.Fatal(err)
		}
	}
}

// goldenPlanDryRun plans one case the way cmdDryRun does, from
// parseLaunchArgs to the ggrun-server-plan-v1 object.
func goldenPlanDryRun(t *testing.T, c goldenPlanCase) goldenPlanRecord {
	t.Helper()
	record := goldenPlanRecord{Case: c.name, Args: append([]string{}, c.args...)}
	if record.Args == nil {
		record.Args = []string{}
	}
	// applyGPUVisibility exports these process-wide; start each run without them.
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES"} {
		if err := os.Unsetenv(key); err != nil {
			t.Fatal(err)
		}
	}
	cacheDir := t.TempDir()
	t.Setenv("LLM_CACHE_DIR", cacheDir)
	inventory := filepath.Join(t.TempDir(), "inventory.json")
	if err := os.WriteFile(inventory, []byte(goldenPlanInventories[c.hw]), 0o600); err != nil {
		t.Fatal(err)
	}
	model := goldenPlanModels[c.model]()
	args := append([]string{model.Path}, c.args...)
	args = append(args, "--inventory", inventory, "--emit-server-argv-json")

	req, err := parseLaunchArgs(args)
	if err != nil {
		record.Outcome, record.Error = "parse-error", err.Error()
		return record
	}
	caps, err := loadInventory(req.InventoryPath)
	if err != nil {
		t.Fatalf("%s: inventory: %v", c.name, err)
	}
	cfg, err := config.Load() // loadConfigOrExit
	if err != nil {
		t.Fatalf("%s: config: %v", c.name, err)
	}
	if cfg.CacheDir != cacheDir {
		t.Fatalf("%s: cache dir %q, want the case's own %q", c.name, cfg.CacheDir, cacheDir)
	}
	req.ModelPath = resolveModelPath(req.ModelPath, cfg.ModelDir)
	warnModelCompatibility(model) // model stands in for parseModel

	be := goldenPlanBackend(c.backend)
	// resolveLaunchBackend after selection; normalizeArchKVRequest is the part
	// of auto selection that edits the request.
	normalizeArchKVRequest(req, model)
	applyBackendFeatureCompatibility(req, model, be)
	preflightBackendArch(model, be, caps, req.AppHome)

	applyCachedBackendCapabilities(req, cfg.CacheDir, model, be)
	applyRememberedClaudeSelfClassify(req, cfg.CacheDir, model, caps, io.Discard)
	req.ReviewerReservation = claudeReviewerReservation(req, caps, cfg.CacheDir)

	placementOpts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
	strategy, err := placement.Compute(caps, model, placementOpts)
	if err != nil {
		record.Outcome, record.Error = "placement-error", err.Error()
		return record
	}
	claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	strategy = applyCalibrationDecision(req, cfg, model, be, caps, strategy)
	strategy = explainServingOptimization(req, cfg, model, be, caps, strategy, true)
	if refusal := dryRunFitRefusal(strategy); refusal != "" {
		record.Outcome, record.Error = "fit-refusal", refusal
		return record
	}

	serverArgs := buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
	envPrefix := applyGPUVisibility(req, backendDialect(be))
	noVulkan := func(string) ([]listedDevice, error) {
		t.Fatalf("%s: a CUDA fixture listed Vulkan devices", c.name)
		return nil, nil
	}
	if vkEnv, vkErr := vulkanDeviceOrderEnv(req, be, caps, noVulkan); vkErr != nil {
		t.Fatalf("%s: vulkan device order: %v", c.name, vkErr)
	} else if vkEnv != "" {
		envPrefix = vkEnv
	}
	record.Outcome = "plan"
	record.Plan = &goldenPlanJSON{
		Schema:               "ggrun-server-plan-v1",
		ModelPath:            req.ModelPath,
		BackendTag:           be.Tag,
		BackendID:            be.Identity,
		ClaudeProfile:        effectiveClaudeProfile(req),
		LaunchScope:          claudeLaunchLogScope(req, model, be, serverArgs),
		MemoryMaxMB:          backendMemoryMaxMB(req, caps),
		Environment:          launchPlanEnvironment(serverArgs, envPrefix, be.Path),
		ServerArgv:           serverArgs,
		Residency:            strategy.Residency,
		Bottleneck:           strategy.OptimizationBottleneck,
		ResourceLedger:       strategy.ResourceLedger,
		OptimizationBoundary: strategy.OptimizationBoundary,
		CachedOptimization:   req.AppliedCalibration,
	}
	return record
}

func TestGoldenPlanSyntheticMatrix(t *testing.T) {
	if runtime.GOOS != "linux" {
		// memory_max_mb and the host growth reserve are Linux-only
		// (backendMemoryMaxMB, hostGrowthReserveMB), so the golden is Linux's.
		t.Skip("golden launch plans are recorded on Linux")
	}
	home := t.TempDir()
	unset := []string{
		"LD_LIBRARY_PATH", "LLM_SERVER_LIB_HUB", "CUDA_DEVICE_ORDER", "CUDA_SCALE_LAUNCH_QUEUES",
		"CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES", "GGML_OP_OFFLOAD_MIN_BATCH",
		"LLAMA_ARG_LOG_VERBOSITY", "GGRUN_CLAUDE_AUTO_REVIEWER", "GGRUN_CLAUDE_REVIEWER_MODEL",
		"GGRUN_CLAUDE_REVIEWER_BIN", "GGRUN_TRACE_PLACEMENT", "LLM_SERVER_SKIP_MMPROJ_CHECK",
	}
	for _, key := range config.EnvKeys {
		unset = append(unset, key, config.GGRUNAlias(key))
	}
	goldenPlanUnsetenv(t, unset...)
	t.Setenv("HOME", home)
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, ".cache"))
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(home, ".config"))
	t.Setenv("LLM_CONFIG", filepath.Join(home, "missing-config"))
	t.Setenv("LLM_APP_HOME", filepath.Join(home, "app"))
	t.Setenv("LLM_SERVER_SKIP_DRAFT_DOWNLOAD", "1")
	restoreResident := residentReviewerVRAM
	residentReviewerVRAM = func(string) int { return 0 }
	t.Cleanup(func() { residentReviewerVRAM = restoreResident })

	// Every t.TempDir path of this test carries its name.
	tempMarker := []byte("TestGoldenPlanSyntheticMatrix")
	var records []goldenPlanRecord
	for _, c := range goldenPlanCases() {
		start := time.Now()
		var first, second []byte
		var record goldenPlanRecord
		// Twice, each in its own cache dir: the plan must not depend on the
		// run or on where the cache lives.
		t.Run(c.name, func(t *testing.T) {
			goldenPlanUnsetenv(t, "CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES")
			goldenPlanStubLiveVRAM(t)
			record = goldenPlanDryRun(t, c)
			first = goldenPlanMarshal(t, record)
			second = goldenPlanMarshal(t, goldenPlanDryRun(t, c))
		})
		if !bytes.Equal(first, second) {
			t.Errorf("%s: two identical dry runs disagree\nfirst:  %s\nsecond: %s", c.name, first, second)
		}
		if bytes.Contains(first, tempMarker) || bytes.Contains(first, []byte(home)) {
			t.Errorf("%s: plan carries a temporary path, so no golden can hold it: %s", c.name, first)
		}
		t.Logf("%s: %s in %v", c.name, record.Outcome, time.Since(start).Round(time.Millisecond))
		records = append(records, record)
	}
	if t.Failed() {
		return
	}

	got, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	if *updateGoldenPlans {
		if err := os.WriteFile(goldenPlanPath, got, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(goldenPlanPath)
	if os.IsNotExist(err) {
		_ = os.WriteFile(goldenPlanPath, got, 0o644)
		goldenPlanLogEncoded(t, got)
		t.Fatalf("%s was missing; wrote it from this run. Review and commit it, or regenerate with -update-launch-plans", goldenPlanPath)
	}
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(got, want) {
		return
	}
	var wantRecords []goldenPlanRecord
	if err := json.Unmarshal(want, &wantRecords); err != nil {
		goldenPlanLogEncoded(t, got)
		t.Fatalf("%s is not a golden plan list: %v", goldenPlanPath, err)
	}
	for i := 0; i < max(len(records), len(wantRecords)); i++ {
		var g, w []byte
		if i < len(records) {
			g = goldenPlanMarshal(t, records[i])
		}
		if i < len(wantRecords) {
			w = goldenPlanMarshal(t, wantRecords[i])
		}
		if !bytes.Equal(g, w) {
			t.Errorf("launch plan %d changed\nwant: %s\ngot:  %s", i, w, g)
		}
	}
	goldenPlanLogEncoded(t, got)
	t.Fatalf("launch plans differ from %s; if the change is intended, regenerate with -update-launch-plans", goldenPlanPath)
}

// goldenPlanStubLiveVRAM replaces the fit guard's live readings for one
// subtest: no live usage beyond the inventory, and driver free VRAM of
// total - 300 MiB per device (the inventory's vram_used_mb).
func goldenPlanStubLiveVRAM(t *testing.T) {
	t.Helper()
	restoreUsed, restoreFree := runtimeVRAMUsedMB, fitDriverFreeVRAM
	runtimeVRAMUsedMB = func(int) int { return 0 }
	fitDriverFreeVRAM = func(runtime *detect.Capabilities) map[int]int {
		if runtime == nil || len(runtime.GPUs) == 0 {
			return nil
		}
		free := map[int]int{}
		for _, g := range runtime.GPUs {
			free[g.Index] = g.VRAMTotalMB - 300
		}
		return free
	}
	t.Cleanup(func() { runtimeVRAMUsedMB, fitDriverFreeVRAM = restoreUsed, restoreFree })
}

func goldenPlanMarshal(t *testing.T, record goldenPlanRecord) []byte {
	t.Helper()
	b, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// goldenPlanLogEncoded logs data as numbered gzip+base64 lines with its sha256,
// so a CI log yields the exact file (see the comment at the top).
func goldenPlanLogEncoded(t *testing.T, data []byte) {
	t.Helper()
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	if _, err := zw.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	encoded := base64.StdEncoding.EncodeToString(buf.Bytes())
	const width = 1000
	chunks := (len(encoded) + width - 1) / width
	t.Logf("launch-plan-golden sha256 %x bytes %d chunks %d", sha256.Sum256(data), len(data), chunks)
	for i := 0; i < chunks; i++ {
		t.Logf("launch-plan-golden-b64 %04d %s", i, encoded[i*width:min((i+1)*width, len(encoded))])
	}
}
