package main

import (
	"bytes"
	"errors"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/memprobe"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestFindFitParamsDoesNotCrossCustomForksViaPATH(t *testing.T) {
	dir := t.TempDir()
	pathBin := filepath.Join(dir, "path-bin")
	if err := os.MkdirAll(pathBin, 0o755); err != nil {
		t.Fatal(err)
	}
	fit := filepath.Join(pathBin, "llama-fit-params")
	if err := os.WriteFile(fit, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", pathBin)
	customServer := filepath.Join(dir, "custom-fork", "bin", "llama-server")
	if got := findFitParamsBin(customServer, ""); got != "" {
		t.Fatalf("custom fork must not use unrelated PATH fit-params: %s", got)
	}
}

func TestRunFitPreflightAddsSiblingLibraryDirectory(t *testing.T) {
	dir := t.TempDir()
	fit := filepath.Join(dir, "llama-fit-params")
	script := "#!/bin/sh\ncase \"$LD_LIBRARY_PATH\" in\n  \"" + dir + "\"*) echo 'CUDA0 100 20 30' ;;\n  *) exit 42 ;;\nesac\n"
	if err := os.WriteFile(fit, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	devs, _, err := runFitPreflight(fit, []string{"llama-server", "-m", "model.gguf"})
	if err != nil {
		t.Fatalf("fit preflight did not receive sibling LD_LIBRARY_PATH: %v", err)
	}
	want := []preflightDevice{{Name: "CUDA0", ModelMB: 100, ContextMB: 20, ComputeMB: 30}}
	if !reflect.DeepEqual(devs, want) {
		t.Fatalf("fit rows = %#v, want %#v", devs, want)
	}
}

func TestFitPreflightLibraryPathResolvesSymlinkTarget(t *testing.T) {
	root := t.TempDir()
	realDir := filepath.Join(root, "backend", "bin")
	linkDir := filepath.Join(root, "app", ".bin")
	if err := os.MkdirAll(realDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(linkDir, 0o755); err != nil {
		t.Fatal(err)
	}
	realBin := filepath.Join(realDir, "llama-fit-params")
	if err := os.WriteFile(realBin, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	linkBin := filepath.Join(linkDir, "llama-fit-params")
	if err := os.Symlink(realBin, linkBin); err != nil {
		t.Fatal(err)
	}
	got := strings.Split(fitPreflightLibraryPath(linkBin), string(os.PathListSeparator))
	want := []string{realDir, linkDir}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("library path = %#v, want %#v", got, want)
	}
}

func TestPreflightArgsKeepsOnlyMemoryShapingFlags(t *testing.T) {
	serverArgs := []string{
		"-m", "model.gguf",
		"--host", "0.0.0.0", "--port", "8081",
		"--ctx-size", "1048576",
		"--flash-attn", "on",
		"-b", "2048", "-ub", "512",
		"--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
		"--jinja",
		"--threads", "8", "--threads-batch", "8",
		"--parallel", "4",
		"-ngl", "999",
		"--fit", "off",
		"--tensor-split", "0.86,0.03,0.11",
		"--split-mode", "layer",
		"-ot", `blk\.(0|1)\.ffn_.*=CUDA0,exps=CPU`,
		"--n-cpu-moe", "36",
		"--no-mmap",
		"--alias", "local",
		"--presence-penalty", "1.0",
	}
	want := []string{
		"--fit-print", "on",
		"-m", "model.gguf",
		"--ctx-size", "1048576",
		"--flash-attn", "on",
		"-b", "2048", "-ub", "512",
		"--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
		"--parallel", "4",
		"-ngl", "999",
		"--tensor-split", "0.86,0.03,0.11",
		"--split-mode", "layer",
		"-ot", `blk\.(0|1)\.ffn_.*=CUDA0,exps=CPU`,
		"--n-cpu-moe", "36",
		"--no-mmap",
	}
	if got := preflightArgs(serverArgs); !reflect.DeepEqual(got, want) {
		t.Fatalf("preflightArgs:\n got  %q\n want %q", got, want)
	}
}

// The oracle must measure the same KV, graph and model shape as serving.
func TestPreflightArgsPreservesAllocationPolicy(t *testing.T) {
	for _, args := range [][]string{
		{"--no-kv-offload", "--swa-full", "--no-op-offload"},
		{"--swa-full=true", "--kv-unified=false"},
		{"-nkvo", "--kv-offload", "-kvo"},
		{"--kv-unified", "--no-kv-unified", "-kvu", "-no-kvu"},
		{"--mmap", "--no-mmap", "--mlock", "--no-repack"},
		{"--override-kv", "model.block_count=int:12", "--override-kv", "model.context_length=int:8192"},
	} {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			want := append([]string{"--fit-print", "on"}, args...)
			if got := preflightArgs(args); !reflect.DeepEqual(got, want) {
				t.Fatalf("oracle lost allocation policy: got %q, want %q", got, want)
			}
		})
	}
}

func TestPreflightArgsPreservesOverridesAndSignedValues(t *testing.T) {
	args := []string{"--ctx-size", "4096", "--ctx-size=8192", "--gpu-layers", "-1", "--override-kv=model.block_count=int:12", "--port=8081"}
	want := []string{"--fit-print", "on", "--ctx-size", "4096", "--ctx-size", "8192", "--gpu-layers", "-1", "--override-kv", "model.block_count=int:12"}
	if got := preflightArgs(args); !reflect.DeepEqual(got, want) {
		t.Fatalf("oracle changed last-wins arguments: got %q, want %q", got, want)
	}
}

func TestRunFitPreflightDoesNotHideUnsupportedMemoryPolicy(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell oracle fixture")
	}
	fit := filepath.Join(t.TempDir(), "llama-fit-params")
	script := `#!/bin/sh
for arg in "$@"; do
 if [ "$arg" = "--kv-unified" ]; then
  echo 'unsupported memory policy --kv-unified' >&2
  exit 2
 fi
done
echo 'CUDA0 100 20 30'
`
	if err := os.WriteFile(fit, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	devices, stderr, err := runFitPreflight(fit, []string{"llama-server", "--kv-unified"})
	if err == nil || len(devices) != 0 || !strings.Contains(stderr, "unsupported memory policy") {
		t.Fatalf("unsupported shape must select probe fallback, not oracle fit: devices=%v stderr=%q err=%v", devices, stderr, err)
	}
}

func TestPreflightArgsMissingValueDoesNotConsumeMemoryPolicy(t *testing.T) {
	got := preflightArgs([]string{"--ctx-size", "--no-kv-offload", "--tensor-split"})
	want := []string{"--fit-print", "on", "--ctx-size", "--no-kv-offload", "--tensor-split"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestDraftPreflightServerArgs(t *testing.T) {
	strategy := &placement.Strategy{
		BackendTag: "llama", ContextSize: 1048576, BatchSize: 2048, UBatchSize: 256,
		Parallel: 4, FlashAttention: true,
		Draft: &placement.DraftConfig{
			Type: placement.DraftDFlash, Path: "dspark.gguf", DraftGPU: 2,
			CTXSizeDraft: 1048576, KVTypeDraft: "q8_0", GPULayersDraft: "all",
		},
	}
	want := []string{
		"llama-fit-draft", "-m", "dspark.gguf", "-c", "1048576",
		"-b", "2048", "-ub", "256", "-ctk", "q8_0", "-ctv", "q8_0",
		"-np", "4", "-ngl", "all", "--device", "CUDA2", "--flash-attn", "on",
	}
	if got := draftPreflightServerArgs(strategy); !reflect.DeepEqual(got, want) {
		t.Fatalf("draft preflight args:\n got  %q\n want %q", got, want)
	}
}

func TestMergePreflightDevicesAddsCompanionMemory(t *testing.T) {
	target := []preflightDevice{
		{Name: "CUDA0", ModelMB: 15000, ContextMB: 3000, ComputeMB: 2000},
		{Name: "CUDA1", ModelMB: 9000, ContextMB: 200, ComputeMB: 600},
		{Name: "Host", ModelMB: 110000, ComputeMB: 20},
	}
	draft := []preflightDevice{
		{Name: "CUDA1", ModelMB: 11000, ContextMB: 400, ComputeMB: 900},
		{Name: "Host", ModelMB: 50, ContextMB: 10, ComputeMB: 5},
	}
	got := mergePreflightDevices(target, draft)
	want := []preflightDevice{
		{Name: "CUDA0", ModelMB: 15000, ContextMB: 3000, ComputeMB: 2000},
		{Name: "CUDA1", ModelMB: 20000, ContextMB: 600, ComputeMB: 1500},
		{Name: "Host", ModelMB: 110050, ContextMB: 10, ComputeMB: 25},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("merged preflight rows:\n got  %#v\n want %#v", got, want)
	}
}

func TestEmbeddedMTPPreflightReservationIsConservativePerGPU(t *testing.T) {
	model := &placement.ModelProfile{
		NumLayers: 33, NextNPredictLayers: 1, HasSSM: 1, FullAttnInterval: 4,
		HeadCountKV: 4, KeyLength: 256, ValueLength: 256,
	}
	strategy := &placement.Strategy{
		ContextSize: 262144, KVPlacement: "gpu",
		Draft: &placement.DraftConfig{Type: placement.DraftMTP, SpecType: "draft-mtp"},
	}
	target := []preflightDevice{
		{Name: "CUDA0", ComputeMB: 1600},
		{Name: "CUDA1", ComputeMB: 600},
		{Name: "Host", ComputeMB: 10},
	}
	got, err := embeddedMTPPreflightReservation(model, strategy, target)
	if err != nil {
		t.Fatal(err)
	}
	want := []preflightDevice{
		{Name: "CUDA0", ContextMB: 1024, ComputeMB: 1600},
		{Name: "CUDA1", ContextMB: 1024, ComputeMB: 1024},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("embedded MTP reservation:\n got  %#v\n want %#v", got, want)
	}
}

func TestEmbeddedMTPPreflightRejectsUnprovenCPUKV(t *testing.T) {
	model := &placement.ModelProfile{NextNPredictLayers: 1, HeadCountKV: 4, KeyLength: 128, ValueLength: 128}
	strategy := &placement.Strategy{
		ContextSize: 32768, KVPlacement: "cpu",
		Draft: &placement.DraftConfig{Type: placement.DraftMTP, SpecType: "draft-mtp"},
	}
	if _, err := embeddedMTPPreflightReservation(model, strategy, []preflightDevice{{Name: "CUDA0", ComputeMB: 1000}}); err == nil {
		t.Fatal("embedded MTP with unmeasured CPU KV must fail closed")
	}
}

func TestPreflightWorstDeficit(t *testing.T) {
	// Real shape from the 2026-07-07 DeepSeek-V4 launch: 3090Ti + 3060 + 4070,
	// fit-print rows in MiB (model, context, compute).
	devs := []preflightDevice{
		{Name: "CUDA0", ModelMB: 15648, ContextMB: 3238, ComputeMB: 2184},
		{Name: "CUDA1", ModelMB: 9070, ContextMB: 179, ComputeMB: 599},
		{Name: "CUDA2", ModelMB: 10248, ContextMB: 351, ComputeMB: 599},
		{Name: "Host", ModelMB: 114162, ContextMB: 0, ComputeMB: 17}, // ignored
	}
	gpus := []detect.GPU{
		{Index: 0, VRAMTotalMB: 24564},
		{Index: 1, VRAMTotalMB: 12288},
		{Index: 2, VRAMTotalMB: 12282},
	}

	// With the measured 678 MB overhead everything fits (CUDA2 has ~406 MiB slack).
	dev, deficit, _ := preflightWorstDeficit(devs, gpus, map[int]int{0: 678, 1: 678, 2: 678}, nil)
	if dev != -1 || deficit != 0 {
		t.Fatalf("expected fit, got device %d deficit %d", dev, deficit)
	}

	// Occupy 1 GB on the 4070: CUDA2 must be reported with the exact overshoot.
	gpus[2].VRAMUsedMB = 1024
	dev, deficit, summary := preflightWorstDeficit(devs, gpus, map[int]int{0: 678, 1: 678, 2: 678}, nil)
	if dev != 2 {
		t.Fatalf("expected CUDA2 deficit, got device %d (summary %s)", dev, summary)
	}
	want := (10248 + 351 + 599 + 678) - (12282 - 1024)
	if deficit != want {
		t.Fatalf("deficit = %d, want %d", deficit, want)
	}
}

func TestPreflightContextTotalIncludesHostAndGPU(t *testing.T) {
	devs := []preflightDevice{
		{Name: "CUDA0", ContextMB: 6252},
		{Name: "CUDA1", ContextMB: 0},
		{Name: "CUDA2", ContextMB: 649},
		{Name: "Host", ContextMB: 31},
		{Name: "ignored-negative", ContextMB: -10},
	}
	if got := preflightContextTotalMB(devs); got != 6932 {
		t.Fatalf("total context = %d MiB, want 6932", got)
	}
}

func TestParseIKAllocationDevicesSeparatesModelContextAndCompute(t *testing.T) {
	logData := `llm_load_tensors:      CUDA0 buffer size =  9285.25 MiB
llm_load_tensors:      CUDA1 buffer size = 10053.19 MiB
llm_load_tensors: CUDA_Host buffer size = 99957.50 MiB
llama_kv_cache_init:      CUDA0 KV buffer size =   962.25 MiB
llama_context:      CUDA0 compute buffer size =  7926.50 MiB
llama_context:      CUDA1 compute buffer size =   298.20 MiB`
	got := parseIKAllocationDevices(logData)
	want := []preflightDevice{
		{Name: "CUDA0", ModelMB: 9286, ContextMB: 963, ComputeMB: 7927},
		{Name: "CUDA1", ModelMB: 10054, ComputeMB: 299},
		{Name: "Host", ModelMB: 99958},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ik memory rows:\n got  %#v\n want %#v", got, want)
	}
}

func TestGuardPeakAddsOnlyUnaccountedAllocatorBytes(t *testing.T) {
	parsed := []preflightDevice{{Name: "CUDA0", ModelMB: 100, ContextMB: 20, ComputeMB: 30}}
	summary := memprobe.Summary{Devices: map[int]memprobe.DeviceMemory{
		0: {ID: "CUDA0", PeakBytes: 175 * 1024 * 1024},
		3: {ID: "CUDA3", PeakBytes: 64 * 1024 * 1024},
	}, Host: memprobe.HostMemory{CgroupPeakBytes: 512 * 1024 * 1024}}
	got := reconcileGuardedDevices(parsed, summary)
	want := []preflightDevice{
		{Name: "CUDA0", ModelMB: 100, ContextMB: 20, ComputeMB: 30, UnaccountedMB: 25},
		{Name: "CUDA3", UnaccountedMB: 64},
		{Name: "Host", UnaccountedMB: 512},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("guard reconciliation = %#v, want %#v", got, want)
	}
}

func TestGuardedPlanRetainsUnlabelledHostCgroupPeak(t *testing.T) {
	devices := []preflightDevice{{Name: "Host", ModelMB: 1024, ContextMB: 256, ComputeMB: 128}}
	summary := memprobe.Summary{Host: memprobe.HostMemory{CgroupPeakBytes: 4096 * 1024 * 1024}}
	_, host := guardedPlanDevices(devices, summary)
	if got, want := host.UnaccountedBytes, uint64(2688*1024*1024); got != want {
		t.Fatalf("unaccounted host bytes = %d, want %d", got, want)
	}
	if got := host.ModelBytes + host.ContextBytes + host.ComputeBytes + host.UnaccountedBytes; got != host.CgroupPeakBytes {
		t.Fatalf("host ledger total = %d, cgroup peak = %d", got, host.CgroupPeakBytes)
	}
}

func TestBackendAllocationDryRunMustBeAdvertisedExactly(t *testing.T) {
	if !backendSupportsAllocationDryRun(&backendInfo{Help: "  --dry-run   validate allocations"}) {
		t.Fatal("advertised --dry-run was not detected")
	}
	if backendSupportsAllocationDryRun(&backendInfo{Help: "--dry-run-mode experimental"}) {
		t.Fatal("a similarly named option must not authorize an automatic probe")
	}
}

func TestMemoryEvidenceKeyIncludesHostAllocationFlags(t *testing.T) {
	be := &backendInfo{Identity: "ik-build-a"}
	model := &placement.ModelProfile{Path: "model.gguf", SizeBytes: 1234}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}}
	resident := memoryEvidenceKey(be, model, caps, []string{"llama-server", "-m", "model.gguf", "--no-mmap"})
	mapped := memoryEvidenceKey(be, model, caps, []string{"llama-server", "-m", "model.gguf", "--mmap"})
	if resident == mapped {
		t.Fatal("resident and mmap launches shared allocation evidence key")
	}
}

func TestPersistFailedAllocationProbeKeepsBackendAndGuardEvidence(t *testing.T) {
	dir := t.TempDir()
	guard := filepath.Join(dir, "guard.jsonl")
	if err := os.WriteFile(guard, []byte("{\"type\":\"guard\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	path := persistFailedAllocationProbe(dir, "abc123", "backend failure\n", guard)
	if path == "" {
		t.Fatal("failed probe evidence was not persisted")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	if !strings.Contains(text, "backend failure") || !strings.Contains(text, "CUDA allocation firewall") || !strings.Contains(text, `"type":"guard"`) {
		t.Fatalf("incomplete failed probe evidence: %q", text)
	}
}

func TestBackendLoadFailureDiagnosticSkipsGenericTail(t *testing.T) {
	log := "mmap allocation failed: Cannot allocate memory\nAdding override blk.6=CUDA_Host\nunable to load model\n"
	if got := backendLoadFailureDiagnostic(log); !strings.Contains(got, "mmap allocation failed") {
		t.Fatalf("diagnostic = %q", got)
	}
}

// TestBackendUnclassifiedLogExcerptWindowsErrorLine checks that the excerpt
// helper keeps the actionable error line and some surrounding context instead of
// returning only the generic "unable to load model" tail, which is what the
// crash-diagnosis hook feeds the advisor. The error line can be hundreds of
// lines above the tail (the same point persistFailedAllocationProbe makes).
func TestBackendUnclassifiedLogExcerptWindowsErrorLine(t *testing.T) {
	var log strings.Builder
	log.WriteString(strings.Repeat("load progress tensor blk.X\n", 200))
	log.WriteString("GGML_ASSERT: quantized V cache not supported by this model\n")
	log.WriteString(strings.Repeat("unable to load model\n", 3))
	excerpt := backendUnclassifiedLogExcerpt(log.String())
	if excerpt == "" {
		t.Fatal("excerpt is empty")
	}
	if !strings.Contains(excerpt, "quantized V cache not supported") {
		t.Fatalf("excerpt lost the actionable error line: %q", excerpt)
	}
	if len(excerpt) > len(log.String()) {
		t.Fatalf("excerpt larger than source log")
	}
}

// TestBackendUnclassifiedLogExcerptFallsBackToTail covers a log with no
// classifyable error line: the excerpt must still be non-empty (the tail) rather
// than empty, so the advisor gets something to read.
func TestBackendUnclassifiedLogExcerptFallsBackToTail(t *testing.T) {
	var log strings.Builder
	log.WriteString(strings.Repeat("loading...\n", 600))
	excerpt := backendUnclassifiedLogExcerpt(log.String())
	if excerpt == "" {
		t.Fatal("excerpt is empty for an error-less log")
	}
	if len(excerpt) > 2048 {
		t.Fatalf("excerpt exceeds tail bound: %d bytes", len(excerpt))
	}
}

// TestBackendUnclassifiedProbeErrorWrapsCause verifies the typed error that
// marks the unclassified preflight path: it is advisory-only (wraps the real
// cause) and detectable via errors.As, so the launch handler can consult the
// advisor without changing launch behavior.
func TestBackendUnclassifiedProbeErrorWrapsCause(t *testing.T) {
	cause := errors.New("contained backend memory probe did not complete: GGML_ASSERT failed")
	unclassified := &backendUnclassifiedProbeError{LogExcerpt: "GGML_ASSERT: quantized V cache not supported", Cause: cause}
	if unclassified.Error() != cause.Error() {
		t.Fatalf("wrapped error message differs from cause: %q vs %q", unclassified.Error(), cause.Error())
	}
	var got *backendUnclassifiedProbeError
	if !errors.As(unclassified, &got) {
		t.Fatal("errors.As could not recover the unclassified error")
	}
	if !errors.Is(unclassified, cause) {
		t.Fatal("errors.Is on the cause failed")
	}
	if got.LogExcerpt == "" {
		t.Fatal("log excerpt was not carried on the wrapped error")
	}
}

func TestFailedAllocationPreflightMessageIncludesCauseAndNextStep(t *testing.T) {
	assertion := "GGML_ASSERT failed: n_embd_head_k % blck(type_k) == 0"
	message := failedAllocationPreflightMessage("loading model\n"+assertion+"\nunable to load model\n", "/tmp/probe.log")
	if !strings.Contains(message, assertion) {
		t.Fatalf("preflight message lost backend assertion: %q", message)
	}
	for _, flag := range []string{"--kv-quality f16", "--ctx-size", "--kv-placement cpu"} {
		if !strings.Contains(message, flag) {
			t.Errorf("preflight message lacks actionable flag %q: %q", flag, message)
		}
	}
	if !strings.Contains(message, "/tmp/probe.log") {
		t.Fatalf("preflight message lost evidence path: %q", message)
	}
}

func TestBackendAdjustmentFromLogDisablesUnsupportedKHadamard(t *testing.T) {
	log := "DeepSeek4 K-cache Hadamard is not supported; use an untransformed K-cache\n"
	got := backendAdjustmentFromLog(log)
	if got == nil || got.RemoveFlag != "-khad" {
		t.Fatalf("backend adjustment = %#v, want removal of -khad", got)
	}
}

func TestBackendAdjustmentFromLogPromotesUnsupportedDeepSeek4KV(t *testing.T) {
	log := "DeepSeek4 K-cache supports only F16, BF16, and Q8_0 (requested q4_0)\n"
	got := backendAdjustmentFromLog(log)
	if got == nil || got.KVQuality != "q8_0" || got.RemoveFlag != "" {
		t.Fatalf("backend adjustment = %#v, want q8_0 KV promotion", got)
	}
}

func TestBackendAdjustmentFromLogIgnoresHarmlessVCacheWarning(t *testing.T) {
	log := "DeepSeek4 has no independent V-cache; ignoring requested V-cache type q4_0\n"
	if got := backendAdjustmentFromLog(log); got != nil {
		t.Fatalf("harmless warning produced adjustment: %#v", got)
	}
}

// TestBackendAdjustmentFromLogPromotesUnsupportedQuantizedVCache reproduces the
// Inkling-Small launch failure: the backend rejects a compressed V cache
// outright, and the retry must promote only the V leg to f16 while leaving K at
// its requested quant.
func TestBackendAdjustmentFromLogPromotesUnsupportedQuantizedVCache(t *testing.T) {
	logs := []string{
		"model does not support a quantized V cache (q4_0)\n",
		"error: quantized V cache is not supported by this model\n",
		"GGML_ASSERT: quantized V cache unsupported\n",
	}
	for _, log := range logs {
		got := backendAdjustmentFromLog(log)
		if got == nil {
			t.Fatalf("quantized V cache error %q produced no adjustment", log)
		}
		if got.KVQualityV != "f16" {
			t.Fatalf("quantized V cache error %q = %#v, want KVQualityV=f16", log, got)
		}
		if got.KVQuality != "" || got.RemoveFlag != "" {
			t.Fatalf("quantized V cache adjustment must not touch K or remove flags: %#v", got)
		}
	}
}

func TestMemoryEvidenceCacheRejectsOtherScope(t *testing.T) {
	dir := t.TempDir()
	evidence := memoryPlanEvidence{
		Level: memoryEvidenceAllocated, Backend: "ik_llama",
		Devices: []preflightDevice{{Name: "CUDA0", ModelMB: 1000, ComputeMB: 200}},
	}
	if err := saveMemoryEvidence(dir, "scope-a", evidence); err != nil {
		t.Fatal(err)
	}
	if got, ok := loadMemoryEvidence(dir, "scope-a"); !ok || got.Level != evidence.Level || got.Backend != evidence.Backend || !reflect.DeepEqual(got.Devices, evidence.Devices) || !got.Coverage.Complete {
		t.Fatalf("saved evidence did not round-trip as complete evidence: ok=%v got=%#v", ok, got)
	}
	if _, ok := loadMemoryEvidence(dir, "scope-b"); ok {
		t.Fatal("memory evidence crossed launch scopes")
	}
}

func TestPreflightWorstDeficitIncludesMeasuredRuntimeGrowth(t *testing.T) {
	devs := []preflightDevice{
		{Name: "CUDA2", ModelMB: 10248, ContextMB: 351, ComputeMB: 599},
	}
	gpus := []detect.GPU{{Index: 2, VRAMTotalMB: 12282}}

	dev, deficit, summary := preflightWorstDeficit(devs, gpus, map[int]int{2: 678}, map[int]int{2: 1000})
	if dev != 2 {
		t.Fatalf("expected CUDA2 deficit, got device %d (summary %s)", dev, summary)
	}
	want := (10248 + 351 + 599 + 678 + 1000) - 12282
	if deficit != want {
		t.Fatalf("deficit = %d, want %d", deficit, want)
	}
	if !strings.Contains(summary, "fit=11198") ||
		!strings.Contains(summary, "overhead=678") ||
		!strings.Contains(summary, "runtime=1000") {
		t.Fatalf("summary missing exact terms: %s", summary)
	}
	// The failure path must decompose fit, not merely total it. A plan that does
	// not fit is the one case where knowing which component grew matters, and
	// printing only the total forced three wrong attributions on 2026-09-15.
	for _, want := range []string{"model=10248", "context=351", "compute=599"} {
		if !strings.Contains(summary, want) {
			t.Fatalf("summary missing component %q: %s", want, summary)
		}
	}
}

func TestPreflightWorstDeficitIgnoresUnknownDevices(t *testing.T) {
	devs := []preflightDevice{
		{Name: "CUDA5", ModelMB: 99999},
		{Name: "Vulkan0", ModelMB: 99999},
	}
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 8192}}
	if dev, deficit, _ := preflightWorstDeficit(devs, gpus, map[int]int{0: 600}, nil); dev != -1 || deficit != 0 {
		t.Fatalf("unknown devices must not produce deficits, got dev %d deficit %d", dev, deficit)
	}
}

// A host that cannot contain a live model load -- macOS, Windows, or Linux
// without cgroup v2 -- used to fail every launch with "requires a positive
// backend MemoryMax", because backendMemoryMaxMB returns 0 off Linux by
// construction. The measurement is an optimisation; a host with no GPU already
// skips it and launches on the planner's estimate.
func TestPreflightFallsBackToEstimateWhenProbeCannotRun(t *testing.T) {
	dir := t.TempDir()
	serverBin := filepath.Join(dir, "llama-server")
	if err := os.WriteFile(serverBin, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	be := &backendInfo{Path: serverBin, Tag: "llama", Dialect: "llama", Help: "--dry-run"}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "Test GPU", VRAMTotalMB: 24564}},
		RAM:  detect.RAMInfo{FreeMB: 0}, // forces backendMemoryMaxMB to 0, as off-Linux
	}
	model := &placement.ModelProfile{Name: "test", Path: filepath.Join(dir, "m.gguf")}
	strategy := &placement.Strategy{ContextSize: 4096, Parallel: 1}

	outcome := preflightPlacement(&launchRequest{}, be, &configForPreflight{CacheDir: dir},
		caps, model, strategy, []string{"--ctx-size", "4096"})

	if outcome.Err != nil {
		t.Fatalf("an unrunnable probe must not fail the launch: %v", outcome.Err)
	}
	if outcome.ProbeUnavailable == "" {
		t.Fatal("the skip must be reported so the launch can say it used an estimate")
	}
	if outcome.DoesNotFit {
		t.Fatal("no measurement was taken, so nothing can be known not to fit")
	}
	if outcome.Evidence.Level != memoryEvidenceNone {
		t.Fatalf("evidence level = %v, want none", outcome.Evidence.Level)
	}
}

// A restart races the predecessor's teardown: the port it held is the one
// signal that distinguishes "VRAM about to be released" from "VRAM genuinely
// in use", because the new server cannot bind until it exits either way.
func TestWaitForPredecessorPort(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	var warn bytes.Buffer
	if waitForPredecessorPort(port, 250*time.Millisecond, &warn) {
		t.Fatal("reported free while the predecessor still held the port")
	}
	if !strings.Contains(warn.String(), "still held") {
		t.Errorf("the wait must be explained: %q", warn.String())
	}
	_ = ln.Close()
	if !waitForPredecessorPort(port, time.Second, &warn) {
		t.Fatal("port was free but the wait never returned")
	}
	if !waitForPredecessorPort(0, time.Second, &warn) {
		t.Fatal("an unknown port must not block the launch")
	}
}

func TestAllocationProbeScopeSharesProductionReclaimBand(t *testing.T) {
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 114844}, CPU: detect.CPUInfo{Cores: 8}}
	req := &launchRequest{RAMLimitPercent: 95}
	mmapArgs := []string{"-m", "m3.gguf", "--n-cpu-moe", "44", "-ot", "exps=CPU"}
	probeArgs := append(append([]string{}, mmapArgs...), "--port", "8080", "--host", "127.0.0.1", "--dry-run")
	budget := backendMemoryMaxMB(req, caps)
	mainline := &backendInfo{CPUExpertMMapCapability: placement.CPUExpertMMapFileBacked}

	probeScope := backendStartOptions(req, caps, mainline, []string{"GGML_CUDA_NO_PINNED=1"}, probeArgs)

	// Regression assertion (this is what the old probe violated): the probe must
	// carry a reclaim band, not a hard cap at the budget.
	if probeScope.MemoryHighMB != budget {
		t.Errorf("probe reclaim threshold must be the plan budget, got high=%d budget=%d", probeScope.MemoryHighMB, budget)
	}
	if probeScope.MemoryMaxMB <= probeScope.MemoryHighMB {
		t.Errorf("mmap-backed probe has no reclaim band: high=%d max=%d (would OOM-kill on page cache prod survives)", probeScope.MemoryHighMB, probeScope.MemoryMaxMB)
	}
	if probeScope.MemoryMaxMB > caps.RAM.FreeMB {
		t.Errorf("probe hard ceiling %d exceeds free RAM %d", probeScope.MemoryMaxMB, caps.RAM.FreeMB)
	}

	// Probe must run under exactly the regime the launch will run in.
	prodScope := backendStartOptions(req, caps, mainline, nil, mmapArgs)
	if probeScope.MemoryHighMB != prodScope.MemoryHighMB || probeScope.MemoryMaxMB != prodScope.MemoryMaxMB {
		t.Errorf("probe scope (%d/%d) must match production scope (%d/%d)",
			probeScope.MemoryHighMB, probeScope.MemoryMaxMB, prodScope.MemoryHighMB, prodScope.MemoryMaxMB)
	}

	// A named --ram-budget stays a hard cap in the probe too (policy preserved).
	named := backendStartOptions(&launchRequest{RamBudgetMB: 90000}, caps, mainline, nil, probeArgs)
	if named.MemoryHighMB != named.MemoryMaxMB {
		t.Errorf("a named --ram-budget must stay a hard cap in the probe, got high=%d max=%d", named.MemoryHighMB, named.MemoryMaxMB)
	}

	// Resident plans still get a hard cap at the budget (nothing to reclaim).
	residentArgs := append(append([]string{}, mmapArgs...), "--no-mmap")
	resident := backendStartOptions(req, caps, mainline, nil, residentArgs)
	if resident.MemoryHighMB != resident.MemoryMaxMB || resident.MemoryMaxMB != budget {
		t.Errorf("resident probe must keep a hard cap at the budget, got high=%d max=%d budget=%d", resident.MemoryHighMB, resident.MemoryMaxMB, budget)
	}
}
