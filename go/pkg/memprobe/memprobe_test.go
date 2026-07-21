package memprobe

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseGuardLog(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.jsonl")
	data := strings.Join([]string{
		`{"type":"guard","event":"loaded","pid":10,"schema_version":1}`,
		`{"type":"allocation","phase":"result","api":"cudaMalloc","kind":"device","device":0,"bytes":100,"active_bytes":100,"peak_bytes":100,"limit_bytes":1000,"result":0}`,
		`{"type":"allocation","phase":"denied","api":"cudaHostAlloc","kind":"pinned","device":-1,"bytes":4096,"active_bytes":0,"peak_bytes":0,"limit_bytes":0,"result":2}`,
		`{"type":"allocation","phase":"denied","api":"cudaMalloc","kind":"device","device":0,"bytes":950,"active_bytes":100,"peak_bytes":100,"limit_bytes":1000,"result":2}`,
	}, "\n") + "\n"
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	summary, err := ParseGuardLog(path)
	if err != nil {
		t.Fatal(err)
	}
	if !summary.Loaded || !summary.DeviceEvents || !summary.PinnedEvents {
		t.Fatalf("incomplete summary: %#v", summary)
	}
	if summary.Devices[0].PeakBytes != 100 || summary.Devices[0].DeniedBytes != 950 {
		t.Fatalf("unexpected CUDA0: %#v", summary.Devices[0])
	}
	if summary.Denied == nil || summary.Denied.API != "cudaMalloc" {
		t.Fatalf("device denial did not take precedence over expected pinned fallback: %#v", summary.Denied)
	}
}

func TestDeviceSliceSortsSparseCUDAIndices(t *testing.T) {
	summary := Summary{Devices: map[int]DeviceMemory{
		17: {ID: "CUDA17"},
		2:  {ID: "CUDA2"},
	}}
	got := summary.DeviceSlice()
	if len(got) != 2 || got[0].ID != "CUDA2" || got[1].ID != "CUDA17" {
		t.Fatalf("sparse devices were not sorted: %#v", got)
	}
}

func TestGuardEnvironment(t *testing.T) {
	env := GuardEnvironment("/lib/guard.so", "/tmp/events", []int{100, 200}, 0, "/lib/existing.so")
	joined := strings.Join(env, "\n")
	for _, want := range []string{
		"LD_PRELOAD=/lib/guard.so" + string(os.PathListSeparator) + "/lib/existing.so",
		"GGRUN_MEMGUARD_GPU_LIMITS_MB=100,200",
		"GGRUN_MEMGUARD_PINNED_LIMIT_MB=0",
		"GGML_CUDA_NO_PINNED=1",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("missing %q in %q", want, joined)
		}
	}
}

func TestSaveLoadRequiresCompleteEvidence(t *testing.T) {
	dir := t.TempDir()
	plan := Plan{Key: "abc", Evidence: EvidenceGuardedAllocated, Coverage: Coverage{Complete: true}}
	if _, err := Save(dir, plan); err != nil {
		t.Fatal(err)
	}
	loaded, ok := Load(dir, "abc")
	if !ok || loaded.Evidence != EvidenceGuardedAllocated {
		t.Fatalf("plan did not round trip: %#v, %t", loaded, ok)
	}
}
