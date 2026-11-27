package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
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
// Each case is planned by planLaunch, the function cmdDryRun
// --emit-server-argv-json calls, and records the same
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
// cmdDryRun cannot run in a test as is. Three of planLaunch's inputs are
// injected (goldenPlanInputs):
//   - hardware: an --inventory file, read by loadInventory as dry-run does; the
//     detect.Detect call before it is skipped (it would probe this host).
//   - model: a synthetic ModelProfile stands in for parseModel, which needs a
//     GGUF on disk and the Python parser.
//   - backend: a fixed backendInfo stands in for selectBackendForModel, which
//     discovers and runs backend binaries. What resolveLaunchBackend does after
//     selection (applyBackendFeatureCompatibility, preflightBackendArch) runs.
//
// Excluded because they need a live probe, network or nvidia-smi: backend
// discovery, GGUF parsing, live VRAM readings (an inventory plan derives the
// fit guard's driver free VRAM, live usage and resident reviewer from the
// inventory; this host's readers fail the test), the Vulkan device
// listing (the fixtures are CUDA backends; a call fails the test), draft-model discovery (speculation
// stays off) and the "Model does not fit" process hint of placementErrorMessage
// (the raw error is recorded). The console golden
// (dry_run_console_golden_test.go) runs cmdDryRun itself.

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

// Backend help fixtures. The planner keys behaviour on strings in a backend's
// --help, so each fixture carries every probed string its upstream advertises,
// with upstream's flag tokens, and none it lacks. Probed strings and where:
//
//	--samplers                    claudeSamplersSupported (mainline, ik)
//	--metrics                     claudeCodeProgressServerArgs (mainline, ik)
//	--cache-reuse                 claudeCodeCacheArgs (mainline)
//	-lv --verbosity --log-verbosity  backendVerbosityArgs (mainline; ik skips)
//	--swa-full                    applyBackendFeatureCompatibility (mainline)
//	--ctx-checkpoints             BackendSupportsCheckpointReuse (mainline, ik)
//	--checkpoint-min-step         backendCheckpointMinStepFlag (mainline)
//	--ctx-checkpoints-interval    backendCheckpointMinStepFlag (ik)
//	-fit, --fit [on|off]          BackendSupportsFit, backendFitTakesValue (mainline)
//	--fit                         BackendSupportsFit, backendSplitsLayersByBytes (ik)
//	--fit-margin                  bindFitToGuard, guardedFitArgs (ik only)
//	--kv-offload                  BackendSupportsKVOffload (mainline)
//	--no-context-shift            BackendSupportsNoContextShift (mainline, ik)
//	--cpu-range[-batch] --cpu-strict[-batch]  configureCPUAffinity (mainline)
//	--op-offload --no-op-offload  host weight offload (mainline)
//	--cache-type-k --cache-type-v --reasoning --reasoning-budget  (mainline, ik)
//	--load-mode                   backend identity, loader translation (mainline)
//	--lazy-mode                   backendAdvertisedLazyFlag (mainline)
//	--reuse-port                  backendBindStyle, launch only (mainline)
//	ikawrakow, split-mode-graph   ik detection in detectBackend (ik)
//	ctx-size-draft, spec-autotune newDraftConfig, inert with --spec off (ik)
//
// Mainline's speculative types (draft-mtp, draft-dflash, eagle3, ngram-*) are
// listed by --spec-type from the speculative module, not common/arg.cpp; they
// are only read with --spec on, which no case uses, so they are left out.

// goldenPlanMainlineHelp is mainline llama.cpp's help for the probed options
// (common/arg.cpp), without the fit lines.
const goldenPlanMainlineHelp = `-Cr,   --cpu-range lo-hi                range of CPUs for affinity. Complements --cpu-mask
--cpu-strict <0|1>                      use strict CPU placement (default: 0)
-Crb,  --cpu-range-batch lo-hi          ranges of CPUs for affinity. Complements --cpu-mask-batch
--cpu-strict-batch <0|1>                use strict CPU placement (default: same as --cpu-strict)
-c,    --ctx-size N                     size of the prompt context (default: 0, 0 = loaded from model)
--swa-full                              use full-size SWA cache (default: false)
-ctxcp, --ctx-checkpoints, --swa-checkpoints N  max number of context checkpoints to create per slot (default: 32)
-cms,  --checkpoint-min-step N          minimum spacing between context checkpoints in tokens (default: 0, 0 = no minimum)
-cram, --cache-ram N                    set the maximum cache size in MiB (default: 8192, -1 - no limit, 0 - disable)
-kvu,  --kv-unified, -no-kvu, --no-kv-unified  use single unified KV buffer shared across all sequences
--context-shift, --no-context-shift     whether to use context shift on infinite text generation (default: disabled)
-fa,   --flash-attn [on|off|auto]       set Flash Attention use ('on', 'off', or 'auto', default: 'auto')
--samplers SAMPLERS                     samplers that will be used for generation in the order, separated by ';'
-kvo,  --kv-offload, -nkvo, --no-kv-offload  whether to enable KV cache offloading (default: enabled)
-ctk,  --cache-type-k TYPE              KV cache data type for K (default: f16)
-ctv,  --cache-type-v TYPE              KV cache data type for V (default: f16)
-np,   --parallel N                     number of server slots (default: -1, -1 = auto)
-lm,   --load-mode MODE                 model loading mode (default: auto)
-lzm,  --lazy-mode MODE                 on-demand reading of certain tensors, for example per-layer embeddings (default: auto)
-ot,   --override-tensor <tensor name pattern>=<buffer type>,...  override tensor buffer type
-ncmoe, --n-cpu-moe N                   keep the Mixture of Experts (MoE) weights of the first N layers in the CPU
-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: auto)
-sm,   --split-mode {none,layer,row,tensor}  how to split the model across multiple GPUs
-ts,   --tensor-split N0,N1,N2,...      fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1
-mg,   --main-gpu INDEX                 the GPU to use for the model (with split-mode = none)
--op-offload, --no-op-offload           whether to offload host tensor operations to device (default: true)
--reuse-port                            allow multiple sockets to bind to the same port (default: disabled)
--cache-reuse N                         min chunk size to attempt reusing from the cache via KV shifting (default: 0)
--metrics                               enable prometheus compatible metrics endpoint (default: disabled)
--jinja, --no-jinja                     whether to use jinja template engine for chat (default: enabled)
-rea,  --reasoning [on|off|auto]        Use reasoning/thinking in the chat ('on', 'off', or 'auto', default: 'auto' (detect from template))
--reasoning-budget N                    token budget for thinking: -1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)
-lv,   --verbosity, --log-verbosity N   Set the verbosity threshold. Messages with a higher verbosity will be ignored.
`

// goldenPlanMainlineFitHelp is mainline's fit surface: a valued --fit and its
// target options, no --fit-margin.
const goldenPlanMainlineFitHelp = `-fit,  --fit [on|off]                   whether to adjust unset arguments to fit in device memory ('on' or 'off', default: 'on')
-fitt, --fit-target MiB0,MiB1,MiB2,...  target margin per device for --fit, comma-separated list of values, single value is broadcast across all devices, default: 1024
-fitc, --fit-ctx N                      minimum ctx size that can be set by --fit option, default: 4096
`

// goldenPlanIKHelp is ik_llama.cpp's help for the probed options
// (common/common.cpp gpt_params_print_usage), including its boolean --fit and
// --fit-margin.
const goldenPlanIKHelp = `-v,    --verbose                print verbose information
       --verbosity N            set specific verbosity level (default: 0)
-c,    --ctx-size N             size of the prompt context (default: 0, 0 = loaded from model)
-cd,   --ctx-size-draft N       size of the prompt context for the draft model (default: 0, 0 = loaded from model)
-ctx-ckpt N, --ctx-checkpoints N           max number of context checkpoints to create per slot (default: 8)
-ctx-ckpt-i N, --ctx-checkpoints-interval N  minimum number of tokens between each context checkpoint.  (default: 512, <=0 disable)
-cram, --cache-ram N            set the maximum cache size in MiB (default: 8192, -1 - no limit, 0 - disable)
-fa, --flash-attn (auto|on|off|0|1) set Flash Attention (default: on)
-smgs, --split-mode-graph-scheduling Force Split Mode Graph Scheduling (default: 0)
       --samplers SAMPLERS      samplers that will be used for generation in the order, separated by ';'
       --jinja                  use jinja template for chat (default: disabled)
-rea,  --reasoning              [on|off|auto]
       --reasoning-budget N     token budget for thinking: -1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)
-nkvo, --no-kv-offload          disable KV offload
-ctk,  --cache-type-k TYPE      KV cache data type for K (default: f16)
-ctv,  --cache-type-v TYPE      KV cache data type for V (default: f16)
-np,   --parallel N             number of parallel sequences to decode (default: 1)
       --no-context-shift       disable context-shift.
       --mlock                  force system to keep model in RAM rather than swapping or compressing
       --no-mmap                do not memory-map model (slower load but may reduce pageouts if not using mlock)
-ncmoe, --n-cpu-moe N           keep MoE weights of the first N layers in CPU memory
       --fit-margin N           safety margin in MiB when auto-fitting model offloading
       --fit                    automatically determine which tensors to offload to the GPU(s)
-ngl,  --gpu-layers N           number of layers to store in VRAM
-sm,   --split-mode SPLIT_MODE  how to split the model across multiple GPUs
-ts,   --tensor-split SPLIT     fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1
-mg,   --main-gpu i             the GPU to use for the model (with split-mode = none)
-ot,   --override-tensor NAME   override tensor buffer type as tensor_name=buft, comma-separated
--spec-autotune                 automatically tune speculative params to maximize tokens/sec
       --metrics                enable prometheus compatible metrics endpoint (default: disabled)
       --expiring-logit-bias-file  original PR: https://github.com/ikawrakow/ik_llama.cpp/pull/1731
`

// goldenPlanBackend is a fresh CUDA backend per case, with the fields
// detectBackend derives from that help (see goldenPlanProbedBackend). Paths
// are under /usr/bin, so no library directory is derived from them, and do not
// exist, so no architecture probe reads a file.
//   - "" (mainline): llama.cpp with its valued --fit.
//   - "ik": ik_llama.cpp with boolean --fit and --fit-margin.
//   - "nofit": mainline without the fit lines. Real backends all advertise
//     --fit; the case records a known latent defect (see goldenPlanCases).
func goldenPlanBackend(t *testing.T, kind string) *backendInfo {
	t.Helper()
	switch kind {
	case "ik":
		be := goldenPlanProbedBackend("/usr/bin/ggrun-golden-ik-llama-server", "golden-ik-build", goldenPlanIKHelp)
		if !be.IsIK || be.Tag != "ik_llama" {
			t.Fatalf("ik fixture help did not probe as ik_llama: %+v", be)
		}
		return be
	case "nofit":
		return goldenPlanProbedBackend("/usr/bin/ggrun-golden-nofit-llama-server", "golden-nofit-build", goldenPlanMainlineHelp)
	}
	return goldenPlanProbedBackend("/usr/bin/ggrun-golden-llama-server", "golden-mainline-build", goldenPlanMainlineHelp+goldenPlanMainlineFitHelp)
}

// goldenPlanProbedBackend applies detectBackend's help-derived fields to a
// fixed identity, without running a binary. Not mirrored: the --version text
// (identity is fixed) and server.RegisterLoadModeBackend, which only changes
// how a launch translates argv at process start, not the planned argv.
func goldenPlanProbedBackend(path, identity, help string) *backendInfo {
	info := &backendInfo{Path: path, Tag: "llama", Dialect: "llama", Help: help, Identity: identity}
	if strings.Contains(help, "ikawrakow") || strings.Contains(help, "split-mode-graph") {
		info.IsIK, info.Tag, info.Dialect = true, "ik_llama", "ik_llama"
	}
	if strings.Contains(help, "--reasoning") {
		info.SupportsReasoning = true
	}
	if backendParsesLoadMode(help) {
		info.Identity += "+" + loadModeTranslationVersion
	}
	info.CPUExpertMMapCapability, info.CPUExpertMMapEvidence = probedCPUExpertMMapCapability(info)
	return info
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
	// The launch exports these process-wide; start each run without them.
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
	cfg, err := config.Load() // the config planLaunch loads
	if err != nil {
		t.Fatalf("%s: config: %v", c.name, err)
	}
	if cfg.CacheDir != cacheDir {
		t.Fatalf("%s: cache dir %q, want the case's own %q", c.name, cfg.CacheDir, cacheDir)
	}
	plan, planErr := planLaunch(req, goldenPlanInputs(t, c.name, model, goldenPlanBackend(t, c.backend), nil))
	if planErr != nil {
		if !goldenPlanIsPlacementError(planErr) {
			t.Fatalf("%s: %s", c.name, planErr.Message)
		}
		record.Outcome, record.Error = "placement-error", planErr.Err.Error()
		return record
	}
	if plan.Refusal != nil {
		record.Outcome, record.Error = "fit-refusal", plan.Refusal.Message
		return record
	}
	record.Outcome = "plan"
	record.Plan = goldenPlanJSONOf(plan)
	return record
}

// goldenPlanInputs are planLaunch's inputs for a replay. Hardware is the
// case's --inventory (Detect must not run: it would probe this host), the model
// is the synthetic profile in place of parseModel, and the backend is the
// fixture in place of selectBackendForModel; what resolveLaunchBackend does
// after selection runs (normalizeArchKVRequest is the part of auto selection
// that edits the request). beforeBackend runs first, at the point where the
// dry run has the model and the hardware and has not resolved a backend. The
// fixtures are CUDA backends, so a Vulkan listing fails the test. Notes are
// dropped: the console golden pins them.
func goldenPlanInputs(t *testing.T, name string, model *placement.ModelProfile, be *backendInfo, beforeBackend func(req *launchRequest, model *placement.ModelProfile, caps *detect.Capabilities)) planInputs {
	t.Helper()
	return planInputs{
		Policy: dryRunPlanPolicy,
		Detect: func() (*detect.Capabilities, error) {
			t.Errorf("%s: an inventory plan detected this host's hardware", name)
			return &detect.Capabilities{}, nil
		},
		Inventory: loadInventory,
		Model:     func(string) (*placement.ModelProfile, error) { return model, nil },
		ResolveBackend: func(req *launchRequest, m *placement.ModelProfile, caps *detect.Capabilities) *backendInfo {
			if beforeBackend != nil {
				beforeBackend(req, m, caps)
			}
			normalizeArchKVRequest(req, m)
			applyBackendFeatureCompatibility(req, m, be)
			preflightBackendArch(m, be, caps, req.AppHome)
			return be
		},
		ListVulkan: func(string) ([]listedDevice, error) {
			t.Fatalf("%s: a CUDA fixture listed Vulkan devices", name)
			return nil, nil
		},
		Note: func(planNote) {},
	}
}

// goldenPlanIsPlacementError reports whether planLaunch stopped at
// placement.Compute; the replays record its raw error, without the
// running-server hint placementErrorMessage may append.
func goldenPlanIsPlacementError(e *planError) bool {
	return e != nil && e.Code == 1 && e.Err != nil && strings.HasPrefix(e.Message, "Error computing placement: ")
}

// goldenPlanJSONOf maps a plan to the ggrun-server-plan-v1 fields cmdDryRun
// encodes.
func goldenPlanJSONOf(plan *launchPlan) *goldenPlanJSON {
	return &goldenPlanJSON{
		Schema:               "ggrun-server-plan-v1",
		ModelPath:            plan.Req.ModelPath,
		BackendTag:           plan.Backend.Tag,
		BackendID:            plan.Backend.Identity,
		ClaudeProfile:        effectiveClaudeProfile(plan.Req),
		LaunchScope:          plan.LaunchScope,
		MemoryMaxMB:          plan.MemoryMaxMB,
		Environment:          plan.Environment,
		ServerArgv:           plan.ServerArgv,
		Residency:            plan.Strategy.Residency,
		Bottleneck:           plan.Strategy.OptimizationBottleneck,
		ResourceLedger:       plan.Strategy.ResourceLedger,
		OptimizationBoundary: plan.Strategy.OptimizationBoundary,
		CachedOptimization:   plan.Req.AppliedCalibration,
	}
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
			goldenPlanFailLiveReaders(t)
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

// goldenPlanFailLiveReaders makes this host's live readers fail the subtest.
// An inventory plan uses readings derived from the inventory instead
// (useInventoryReadings); they equal goldenPlanStubLiveVRAM's values and a
// resident reviewer of 0, which is why the golden did not change when the
// stubs were dropped.
func goldenPlanFailLiveReaders(t *testing.T) {
	t.Helper()
	restoreUsed, restoreFree, restoreResident := runtimeVRAMUsedMB, fitDriverFreeVRAM, residentReviewerVRAM
	runtimeVRAMUsedMB = func(int) int {
		t.Error("an inventory plan read this host's used VRAM")
		return 0
	}
	fitDriverFreeVRAM = func(*detect.Capabilities) map[int]int {
		t.Error("an inventory plan read this host's driver free VRAM")
		return nil
	}
	residentReviewerVRAM = func(string) int {
		t.Error("an inventory plan scanned this host for a resident reviewer")
		return 0
	}
	t.Cleanup(func() {
		runtimeVRAMUsedMB, fitDriverFreeVRAM, residentReviewerVRAM = restoreUsed, restoreFree, restoreResident
	})
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
