package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// Excerpts of the two observed warmup aborts: GLM-5.3-Flash at ubatch 512
// (2026-10-01) and Qwen3.8-Flash-Next (2026-09-21).
const (
	glmWarmupOOMLog = `5.15.169.593 I sched_reserve:      CUDA0 compute buffer size =  8127.01 MiB
5.15.169.598 I sched_reserve:      CUDA1 compute buffer size =  8147.88 MiB
5.15.169.911 I cmn  common_init_: warming up the model with an empty run - please wait ... (--no-warmup to disable)
5.15.180.471 E CUDA error: out of memory
5.15.180.773 E   current device: 0, in function ggml_cuda_kernel_can_use_pdl at /src/ggml/src/ggml-cuda/common.cuh:1637
5.15.180.774 E   cudaFuncGetAttributes(&attr, kernel)
/src/ggml/src/ggml-cuda/ggml-cuda.cu:107: CUDA error`
	qwenWarmupOOMLog = `3.00.590.492 I sched_reserve:      CUDA0 compute buffer size =  1645.75 MiB
3.00.590.961 I cmn  common_init_: warming up the model with an empty run - please wait ... (--no-warmup to disable)
3.00.676.288 E CUDA error: out of memory
3.00.676.294 E   current device: 1, in function ggml_cuda_kernel_can_use_pdl at /src/ggml/src/ggml-cuda/common.cuh:1637
3.00.676.294 E   cudaFuncGetAttributes(&attr, kernel)`
)

func TestWarmupCUDAOOMRecognizesOnlyPostAllocationAborts(t *testing.T) {
	cases := []struct {
		name   string
		log    string
		device int
		ok     bool
	}{
		{"GLM warmup on CUDA0", glmWarmupOOMLog, 0, true},
		{"Qwen warmup on CUDA1", qwenWarmupOOMLog, 1, true},
		{"still loading: no buffers allocated yet", "load_tensors: loading\nE CUDA error: out of memory\nE   current device: 0, in function x", 0, false},
		{"another CUDA error", "sched_reserve: CUDA0 compute buffer size = 10 MiB\nwarming up the model with an empty run\nE CUDA error: an illegal memory access was encountered\nE   current device: 0, in function x", 0, false},
		{"after the model loaded: runtime, not warmup", "sched_reserve: CUDA0 compute buffer size = 10 MiB\nwarming up the model with an empty run\nmain: model loaded\nE CUDA error: out of memory\nE   current device: 0, in function x", 0, false},
		{"no device named", "sched_reserve: CUDA0 compute buffer size = 10 MiB\nwarming up the model with an empty run\nE CUDA error: out of memory", 0, false},
		{"unrelated failure", "sched_reserve: CUDA0 compute buffer size = 10 MiB\nwarming up the model with an empty run\nerror: unknown model architecture", 0, false},
	}
	for _, c := range cases {
		device, ok := warmupCUDAOOM(c.log)
		if ok != c.ok || (ok && device != c.device) {
			t.Errorf("%s: device %d ok %v, want %d %v", c.name, device, ok, c.device, c.ok)
		}
		// A sizeless warmup abort is never read as a measured allocation.
		if _, _, _, sized := startupLogCUDAOOMDetailed(c.log); sized {
			t.Errorf("%s: parsed as a sized startup OOM", c.name)
		}
	}
}

// warmupFixture is a one-GPU MoE launch whose fake backend aborts in warmup
// whenever its microbatch is in failUB (or always, with "*"), and otherwise
// serves /health. Every load appends its argv to the returned log path.
func warmupFixture(t *testing.T, port, failUB string) (*launchRequest, *placement.ModelProfile, *placement.Strategy, *backendInfo, *detect.Capabilities, []string, string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("POSIX fakes")
	}
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 serves the fake backend's /health")
	}
	fakeBin := t.TempDir()
	if err := os.WriteFile(filepath.Join(fakeBin, "nvidia-smi"), []byte("#!/bin/sh\necho 300\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", fakeBin+string(os.PathListSeparator)+os.Getenv("PATH"))
	dir := t.TempDir()
	loads := filepath.Join(dir, "loads")
	server := filepath.Join(dir, "llama-server")
	script := `#!/bin/sh
ub=; port=; prev=
for a in "$@"; do
  { [ "$prev" = -ub ] || [ "$prev" = --ubatch-size ]; } && ub=$a
  [ "$prev" = --port ] && port=$a
  prev=$a
done
echo "$*" >> '` + loads + `'
echo "sched_reserve:      CUDA0 compute buffer size =  2100.00 MiB" >&2
case "` + failUB + `" in
  "*"|"$ub")
    echo "common_init_: warming up the model with an empty run - please wait ... (--no-warmup to disable)" >&2
    echo "E CUDA error: out of memory" >&2
    echo "E   current device: 0, in function ggml_cuda_kernel_can_use_pdl at common.cuh:1637" >&2
    echo "E   cudaFuncGetAttributes(&attr, kernel)" >&2
    exit 1 ;;
esac
exec python3 -c '
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{\"status\":\"ok\"}")
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
' "$port"
`
	if err := os.WriteFile(server, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	// The oracle admits ubatch 64 and 512 on this exact placement.
	oracle := `#!/bin/sh
ub=; prev=
for a in "$@"; do { [ "$prev" = -ub ] || [ "$prev" = --ubatch-size ]; } && ub=$a; prev=$a; done
case $ub in
  64) echo 'CUDA0 4000 2000 2100' ;;
  512) echo 'CUDA0 4000 2000 4000' ;;
  *) echo 'no such rung' >&2; exit 3 ;;
esac
`
	if err := os.WriteFile(filepath.Join(dir, "llama-fit-params"), []byte(oracle), 0o755); err != nil {
		t.Fatal(err)
	}
	model := &placement.ModelProfile{Path: filepath.Join(dir, "moe.gguf"), Basename: "moe.gguf", NumLayers: 48, IsMoE: true}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12000 + placement.UnmeasuredCUDAOverheadMB,
			VRAMUsedMB: 300, BandwidthMBps: 15760}},
		RAM: detect.RAMInfo{TotalMB: 262144, FreeMB: 200000},
		CPU: detect.CPUInfo{Cores: 24},
	}
	req := &launchRequest{SpecMode: "off", CtxFlag: "65536", Parallel: 1, ParallelSet: true, KVPlacement: "gpu",
		RAMLimitPercent: 95, Port: mustAtoi(t, port)}
	be := &backendInfo{Path: server, Dialect: "llama", Tag: "llama", Identity: "test",
		Help: "--op-offload, --no-op-offload  whether to offload host tensor operations to device (default: true)"}
	strategy := &placement.Strategy{Type: placement.MoEOffload, ContextSize: 65536, BatchSize: 2048, UBatchSize: 64,
		Parallel: 1, NCPUMoE: 47, KVQuality: "q8_0", KVPlacement: "gpu", PlannedHostFootprintMB: 1000}
	args := []string{server, "-m", model.Path, "-c", "65536", "-b", "2048", "-ub", "64", "--port", port}
	return req, model, strategy, be, caps, args, loads
}

func mustAtoi(t *testing.T, s string) int {
	t.Helper()
	n := 0
	for _, r := range s {
		n = n*10 + int(r-'0')
	}
	return n
}

func loadedArgvs(t *testing.T, path string) []string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return strings.Split(strings.TrimSpace(string(data)), "\n")
}

// The observed failure end to end: exact preflight admits a plan, the
// microbatch raise is admitted too, and the raised plan aborts in warmup with
// no size. The launch records a labelled estimate for that exact
// configuration, waits for the failed process's memory, and serves the plan
// exact preflight admitted before the raise, loaded once. The raised argv is
// never loaded again, and the next launch's raise keeps the estimate free.
func TestWarmupOOMOnRaisedPlanRestoresTheAdmittedBaseline(t *testing.T) {
	req, model, strategy, be, caps, args, loads := warmupFixture(t, "59981", "512")
	cacheDir := t.TempDir()
	recovery := newLaunchMemoryRecovery()
	recovery.plannerDisproved = true // keep the proven fit; no planner re-plan
	p, served, gotArgs, err := startLaunchWithCUDAOOMRecoveryState(req, &config.Config{CacheDir: cacheDir}, model, strategy, be, caps, args, 30*time.Second, recovery)
	if err != nil {
		t.Fatalf("launch failed instead of restoring the admitted plan: %v", err)
	}
	defer p.Stop()
	if formatCommand(gotArgs) != formatCommand(args) || served.UBatchSize != 64 {
		t.Fatalf("served %v (ubatch %d), want the admitted pre-raise argv", gotArgs, served.UBatchSize)
	}
	got := loadedArgvs(t, loads)
	if len(got) != 2 || !strings.Contains(got[0], "512") || strings.Contains(got[1], " 512") {
		t.Fatalf("loads = %q, want the raised plan once, then the admitted plan once", got)
	}
	if !recovery.isRejected(strings.Fields(got[0])) && !recovery.hasRejections() {
		t.Fatal("the failed raised argv was not rejected for this launch")
	}
	tag := scopedProbeBackendTagForStrategy(req, model, be, strategy)
	estimate := placement.EstimatedRuntimeGraphGrowthAtUBatch(cacheDir, model, 512, strategy.KVQuality, strategy.KVPlacement, tag, caps.GPUs, strategy.Parallel)
	if estimate[0] <= 0 {
		t.Fatalf("no estimate recorded for the failed rung: %v", estimate)
	}
	if measured := placement.RelatedMeasuredRuntimeGraphGrowth(cacheDir, model, caps.GPUs, strategy.Parallel); measured[0] > 0 {
		t.Fatalf("the guess was filed as a measurement: %v", measured)
	}
	productions := 0
	for _, rec := range readLaunchWork(t, cacheDir) {
		if rec.Kind == "production" {
			productions++
		}
	}
	if productions != 2 {
		t.Fatalf("%d production loads in the ledger, want 2", productions)
	}
	// The next launch's raise at the same rung keeps the estimate free, so it
	// does not return to the failed configuration.
	fit := filepath.Join(filepath.Dir(be.Path), "llama-fit-params")
	current := []preflightDevice{{Name: "CUDA0", ModelMB: 4000, ContextMB: 2000, ComputeMB: 2100}}
	if ub, _ := stagedPrefillUBatchRaise(fit, args, current, &configForPreflight{CacheDir: cacheDir}, caps, model, strategy, tag, []int{512}, false); ub == 512 {
		t.Fatalf("the next raise chose the rung that failed in warmup (estimate %v)", estimate)
	}
}

// A plan that was not raised has no earlier admitted plan to return to. Its
// warmup abort takes the ordinary bounded memory recovery: each retry is a
// different argv, the failed ones are never reloaded, and the error names the
// warmup failure once the budget is spent.
func TestWarmupOOMWithoutARaiseNeverReloadsAFailedArgv(t *testing.T) {
	for _, withGPUExperts := range []bool{false, true} {
		port := "59982"
		if withGPUExperts {
			port = "59983"
		}
		t.Run(map[bool]string{false: "no movable expert", true: "two expert layers on CUDA0"}[withGPUExperts], func(t *testing.T) {
			warmupWithoutRaise(t, port, withGPUExperts)
		})
	}
}

func warmupWithoutRaise(t *testing.T, port string, withGPUExperts bool) {
	req, model, strategy, be, caps, args, loads := warmupFixture(t, port, "*")
	req.UBatchSize, req.UBatchSizeSet = 64, true // explicit: no raise
	if withGPUExperts {
		model.ExpertBytes = 48 << 30 // 1 GiB per routed layer
		strategy.NCPUMoE = 46
		args = append(args, "--n-cpu-moe", "46", "-ot",
			`blk\.(46|47)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=CUDA0,exps=CPU`)
	}
	cacheDir := t.TempDir()
	recovery := newLaunchMemoryRecovery()
	recovery.plannerDisproved = true
	_, _, _, err := startLaunchWithCUDAOOMRecoveryState(req, &config.Config{CacheDir: cacheDir}, model, strategy, be, caps, args, 30*time.Second, recovery)
	if err == nil {
		t.Fatal("every load aborted in warmup, yet the launch reported success")
	}
	if !strings.Contains(err.Error(), "warmup") && !strings.Contains(err.Error(), "CUDA") {
		t.Fatalf("error does not name the failure: %v", err)
	}
	got := loadedArgvs(t, loads)
	seen := map[string]bool{}
	for _, argv := range got {
		if seen[argv] {
			t.Fatalf("argv reloaded after failing: %q (loads %q)", argv, got)
		}
		seen[argv] = true
	}
	if len(got) > 3 {
		t.Fatalf("%d loads, want at most the start plus two retries", len(got))
	}
	if withGPUExperts && len(got) < 2 {
		t.Fatalf("a movable expert layer was not moved off the failed device: loads %q", got)
	}
	tag := scopedProbeBackendTagForStrategy(req, model, be, strategy)
	if growth := placement.RuntimeGraphGrowthByGPU(cacheDir, model, 65536, 64, strategy.KVQuality, strategy.KVPlacement, tag, caps.GPUs, strategy.Parallel); growth[0] <= 0 {
		t.Fatalf("the warmup abort left no estimate for this exact configuration: %v", growth)
	}
}
