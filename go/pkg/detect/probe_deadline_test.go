package detect

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// W5/C-02: a hung nvidia-smi must not hang detection, and its silence must
// not read as "no GPU" (invariant 8: unknown stays unknown).
func TestHungNvidiaSmiLeavesGPUStateUnknownWithinTheDeadline(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture")
	}
	bin := t.TempDir()
	if err := os.WriteFile(filepath.Join(bin, "nvidia-smi"), []byte("#!/bin/sh\nsleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	// vulkaninfo on PATH would be a plausible fallback; it must not be consulted.
	if err := os.WriteFile(filepath.Join(bin, "vulkaninfo"), []byte("#!/bin/sh\necho 'deviceName = Fake GPU'\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+"/usr/bin"+string(os.PathListSeparator)+"/bin")
	old := nvidiaProbeTimeout
	nvidiaProbeTimeout = 300 * time.Millisecond
	t.Cleanup(func() { nvidiaProbeTimeout = old })

	start := time.Now()
	caps, err := Detect()
	if elapsed := time.Since(start); elapsed > 8*time.Second {
		t.Fatalf("Detect took %s with a hung nvidia-smi", elapsed)
	}
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(caps.GPUStateUnknown, "nvidia-smi did not answer within 300ms") {
		t.Fatalf("GPU state not marked unknown: %q", caps.GPUStateUnknown)
	}
	if len(caps.GPUs) != 0 {
		t.Fatalf("GPUs guessed on unknown state: %+v", caps.GPUs)
	}
}

// An nvidia-smi that fails quickly (no driver) is an ordinary absence, not an
// unknown state: detection falls through as before.
func TestFailingNvidiaSmiIsNotUnknown(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture")
	}
	bin := t.TempDir()
	if err := os.WriteFile(filepath.Join(bin, "nvidia-smi"), []byte("#!/bin/sh\nexit 9\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+"/usr/bin"+string(os.PathListSeparator)+"/bin")
	caps, err := Detect()
	if err != nil {
		t.Fatal(err)
	}
	if caps.GPUStateUnknown != "" {
		t.Fatalf("a failing tool was reported as unknown state: %q", caps.GPUStateUnknown)
	}
}
