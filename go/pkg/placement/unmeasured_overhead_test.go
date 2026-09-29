package placement

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestUnmeasuredDevicesReserveTheBackendDefaultMargin(t *testing.T) {
	gpus := []detect.GPU{{Index: 0, Name: "a", VRAMTotalMB: 12282}, {Index: 1, Name: "b", VRAMTotalMB: 24564}}
	dir := t.TempDir()
	t.Setenv("HOME", t.TempDir()) // no legacy ~/.cache/ggrun probe to migrate
	if got := SystemCUDAOverheadByGPU(dir, gpus); got != nil {
		t.Fatalf("measurement view invented overhead: %v", got)
	}
	got := PlanningCUDAOverheadByGPU(dir, gpus)
	if got[0] != UnmeasuredCUDAOverheadMB || got[1] != UnmeasuredCUDAOverheadMB {
		t.Fatalf("unmeasured planning overhead = %v", got)
	}
}

// Two MiMo placements measured 608-611 MiB above the oracle's CUDA0 total, and
// the log-based source recorded 1775 MiB for the same launch. The oracle source
// must win, and it must also work for dense models (oracleOnly).
func TestPostLaunchOverheadPrefersLiveUsageAboveTheOracleTotal(t *testing.T) {
	bin := t.TempDir()
	script := "#!/bin/sh\ncase \"$*\" in *memory.used*) echo 10039 ;; *) exit 1 ;; esac\n"
	if err := os.WriteFile(filepath.Join(bin, "nvidia-smi"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin)
	t.Setenv("HOME", t.TempDir())
	dir := t.TempDir()
	gpus := []detect.GPU{{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12282}}
	log := "llama_model_load_from_file_impl: using device CUDA0\nload_tensors: CUDA0 model buffer size = 4002.00 MiB\n"
	RunPostLaunchProbe(dir, gpus, log, 0, nil, map[int]int{0: 9428}, true)
	if got := SystemCUDAOverheadByGPU(dir, gpus); got[0] != 611 {
		t.Fatalf("oracle-relative overhead = %v, want CUDA0 611", got)
	}
	if got := PlanningCUDAOverheadByGPU(dir, gpus); got[0] != 611 {
		t.Fatalf("measured overhead did not replace the default: %v", got)
	}
}

func TestPostLaunchGrowthIsMeasuredAgainstTheOracleTotal(t *testing.T) {
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 12282}}
	log := "load_tensors: CUDA0 model buffer size = 4459.00 MiB\n"
	// Qwen3.8-Flash-Next: log-itemized buffers miss the recurrent state, so the
	// log-only figure books it as growth; the oracle total accounts for it.
	got := runtimeGraphGrowthFromVRAMDelta(gpus, map[int]int{0: 1}, map[int]int{0: 11000}, map[int]int{0: 330}, log, map[int]int{0: 10449})
	if got[0] != 11000-1-330-10449 {
		t.Fatalf("growth = %v, want %d", got, 11000-1-330-10449)
	}
}

func TestVerifiedReuseKeepsTheKVQualitySpelling(t *testing.T) {
	s := &Strategy{Type: MoEOffload, KVType: "q8_0", KVQuality: "q8_0", ContextSize: 224256, UBatchSize: 128}
	vc := VerifiedConfigToRecord("k", "m.gguf", s, "b", "/bin/llama-server", "", "")
	restored := VerifiedToStrategy(&vc, Options{}, &detect.Capabilities{})
	if restored.KVQuality != "q8_0" {
		t.Fatalf("reused KV quality = %q, want the recorded q8_0", restored.KVQuality)
	}
	legacy := vc
	legacy.KVQuality = ""
	if got := VerifiedToStrategy(&legacy, Options{}, &detect.Capabilities{}).KVQuality; got == "" {
		t.Fatal("an older record lost its KV quality")
	}
}
