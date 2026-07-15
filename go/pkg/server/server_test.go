package server

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestProcessIsRunning(t *testing.T) {
	// Start a dummy HTTP server to test readiness logic
	// We can't easily test subprocess here, but we can test the struct
	p := &Process{Port: 99999}
	if p.IsRunning() {
		t.Fatalf("expected not running for nil process")
	}
}

func TestStopTreatsOwnSignalExitAsSuccess(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses the POSIX sleep command")
	}
	cmd := exec.Command("sleep", "60")
	setSysProcAttr(cmd)
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	p := processForTest(cmd)
	if err := p.Stop(); err != nil {
		t.Fatalf("Stop() = %v, want successful requested termination", err)
	}
}

func TestStopIsSafeForConcurrentCallers(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses the POSIX sleep command")
	}
	cmd := exec.Command("sleep", "60")
	setSysProcAttr(cmd)
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	p := processForTest(cmd)
	var wg sync.WaitGroup
	errs := make(chan error, 4)
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- p.Stop()
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent Stop() = %v, want nil", err)
		}
	}
}

func processForTest(cmd *exec.Cmd) *Process {
	p := &Process{Cmd: cmd, cancel: func() {}, done: make(chan struct{}), stopDone: make(chan struct{})}
	go func() {
		err := cmd.Wait()
		p.waitMu.Lock()
		p.waitErr = err
		p.waitMu.Unlock()
		close(p.done)
	}()
	return p
}

func TestChildEnvEnablesScaledQueuesOnlyForMultiGPU(t *testing.T) {
	got := ChildEnv([]string{"PATH=/bin", "CUDA_DEVICE_ORDER=FASTEST_FIRST"}, []string{"llama-server", "--tensor-split", "1,0,0"})
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "CUDA_DEVICE_ORDER=PCI_BUS_ID") || !strings.Contains(joined, "CUDA_SCALE_LAUNCH_QUEUES=4x") {
		t.Fatalf("missing multi-GPU CUDA defaults: %v", got)
	}
	got = ChildEnv([]string{"CUDA_SCALE_LAUNCH_QUEUES=2x"}, []string{"llama-server", "-ts", "1,1"})
	joined = strings.Join(got, "\n")
	if !strings.Contains(joined, "CUDA_SCALE_LAUNCH_QUEUES=2x") || strings.Contains(joined, "CUDA_SCALE_LAUNCH_QUEUES=4x") {
		t.Fatalf("user queue override was not preserved: %v", got)
	}
	got = ChildEnv(nil, []string{"llama-server", "--parallel", "4"})
	if strings.Contains(strings.Join(got, "\n"), "CUDA_SCALE_LAUNCH_QUEUES=") {
		t.Fatalf("single-GPU launch should not receive scaled queues: %v", got)
	}
}

func TestOverrideEnvReplacesInheritedGPUVisibility(t *testing.T) {
	got := OverrideEnv(
		[]string{"PATH=/bin", "CUDA_VISIBLE_DEVICES=0,1", "OTHER=value"},
		[]string{"CUDA_VISIBLE_DEVICES=2"},
	)
	joined := strings.Join(got, "\n")
	if strings.Contains(joined, "CUDA_VISIBLE_DEVICES=0,1") || !strings.Contains(joined, "CUDA_VISIBLE_DEVICES=2") {
		t.Fatalf("GPU visibility override not applied exactly once: %v", got)
	}
}

func TestWaitReadyTimeout(t *testing.T) {
	p := &Process{Port: 59999} // no server here
	err := p.waitReady(100 * time.Millisecond)
	if err == nil {
		t.Fatalf("expected timeout")
	}
}

func TestStreamLogsFromStartKeepsTerminalGatedButStreamsFiles(t *testing.T) {
	if streamLogsFromStart(true, os.Stdout, os.Stderr) {
		t.Fatal("tty stdout/stderr should stay gated during startup")
	}
	if !streamLogsFromStart(false, os.Stdout, os.Stderr) {
		t.Fatal("non-tty stdout/stderr should stream from startup")
	}
	var out bytes.Buffer
	var err bytes.Buffer
	if !streamLogsFromStart(true, &out, &err) {
		t.Fatal("tty launch writing to log writers should stream from startup")
	}
}

func TestStartWithTimeoutReturnsCapturedLogOnFailure(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses a POSIX shell script")
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "backend.sh")
	content := "#!/bin/sh\necho 'common_memory_breakdown_print: |   - CUDA0 (GPU) | 100 = 90 + ( 80 = 70 + 1 + 9) + 0 |' >&2\nsleep 2\n"
	if err := os.WriteFile(script, []byte(content), 0o755); err != nil {
		t.Fatal(err)
	}
	p, err := StartWithTimeout([]string{script}, 59998, 100*time.Millisecond)
	if err == nil {
		if p != nil {
			_ = p.Stop()
		}
		t.Fatal("expected startup timeout")
	}
	if p == nil || p.LogBuf == nil {
		t.Fatalf("expected stopped process with captured log, got %#v", p)
	}
	if !strings.Contains(p.LogBuf.String(), "common_memory_breakdown_print") {
		t.Fatalf("captured log missing backend output: %q", p.LogBuf.String())
	}
	if p.IsRunning() {
		t.Fatal("failed startup process should already be stopped")
	}
}

func TestModelPathFromArgs(t *testing.T) {
	args := []string{"llama-server", "--host", "0.0.0.0", "-m", "/models/test.gguf"}
	if got := modelPathFromArgs(args); got != "/models/test.gguf" {
		t.Fatalf("model path mismatch: %q", got)
	}

	args = []string{"llama-server", "--model=/models/other.gguf"}
	if got := modelPathFromArgs(args); got != "/models/other.gguf" {
		t.Fatalf("equals model path mismatch: %q", got)
	}
}

func TestModelShardPathsSplitGGUF(t *testing.T) {
	dir := t.TempDir()
	sizes := []int{100, 200, 300}
	names := []string{
		"DeepSeek-V4-Flash-MXFP4-00001-of-00003.gguf",
		"DeepSeek-V4-Flash-MXFP4-00002-of-00003.gguf",
		"DeepSeek-V4-Flash-MXFP4-00003-of-00003.gguf",
	}
	for i, size := range sizes {
		path := filepath.Join(dir, names[i])
		if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	paths, total := modelShardPaths(filepath.Join(dir, "DeepSeek-V4-Flash-MXFP4-00001-of-00003.gguf"))
	if len(paths) != 3 {
		t.Fatalf("expected 3 shard paths, got %d", len(paths))
	}
	if total != 600 {
		t.Fatalf("total size mismatch: %d", total)
	}
}

func TestStartupStatusIncludesProgressAndLatestLine(t *testing.T) {
	logText := "main: loading model\nload_tensors: loading model tensors, this can take a while...\n"
	got := startupStatus(logText, 90*time.Second, 30*time.Minute, loadProgress{
		Done:  1 << 30,
		Total: 2 << 30,
	})
	for _, want := range []string{
		"[##########----------]  50%",
		"loading model weights",
		"1m30s/30m0s",
		"read 1.0GiB/2.0GiB",
		"load_tensors: loading model tensors",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("status %q missing %q", got, want)
		}
	}
}

func TestStartupStatusHidesUnknownZeroProgress(t *testing.T) {
	got := startupStatus("load_tensors: loading model", 5*time.Second, 30*time.Minute, loadProgress{Total: 128 << 30})
	if strings.Contains(got, "0%") || strings.Contains(got, "read 0.0GiB") {
		t.Fatalf("zero fd activity is unknown progress, not a truthful 0%% completion: %q", got)
	}
	if !strings.Contains(got, "loading model weights") {
		t.Fatalf("phase should remain visible without byte progress: %q", got)
	}
}

func TestStartupStatusDoesNotClaimReadyWhenWeightsAreOnlyRead(t *testing.T) {
	got := startupStatus("load_tensors: loading model", time.Minute, 30*time.Minute, loadProgress{
		Done: 2 << 30, Total: 2 << 30,
	})
	if !strings.Contains(got, " 99%") || !strings.Contains(got, "initializing model (weights read)") {
		t.Fatalf("completed reads should show truthful initialization state: %q", got)
	}
}

func TestLoadProgressRetainsClosedShardOffsets(t *testing.T) {
	tracker := &loadProgressTracker{
		paths: map[string]int64{"shard-1": 100, "shard-2": 200},
	}
	if got := tracker.recordPositions(map[string]int64{"shard-1": 100}); got != 100 {
		t.Fatalf("first shard progress = %d, want 100", got)
	}
	// shard-1 is now closed and absent. Its completed 100 bytes must remain.
	if got := tracker.recordPositions(map[string]int64{"shard-2": 25}); got != 125 {
		t.Fatalf("cross-shard progress = %d, want 125", got)
	}
	if got := tracker.recordPositions(map[string]int64{"shard-2": 200}); got != 300 {
		t.Fatalf("completed split progress = %d, want 300", got)
	}
}
