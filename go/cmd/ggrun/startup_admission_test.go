package main

import (
	"errors"
	"strings"
	"testing"
)

const startupGraphOOMFixture = `5.05.924 I sched_reserve: CUDA1 compute buffer size = 4873.03 MiB
5.05.925 I cmn common_init_: warming up the model with an empty run
5.06.633 E CUDA error: out of memory
5.06.633 E current device: 1, in function ggml_cuda_graph_evaluate_and_capture at ggml-cuda.cu:4216
5.06.633 E cudaGraphInstantiate(&graph->instance, graph->graph, 0, 0, 0)
Aborted (core dumped)`

func TestStartupExactAdmissionRejectsGraphOOMBeforeHealth(t *testing.T) {
	cause := errors.New("server exited during startup: exit status 134")
	err := startupExactAdmissionFailure(startupGraphOOMFixture, cause)
	if err == nil || !isStableExactAdmissionFailure(err) || !errors.Is(err, cause) {
		t.Fatalf("warmup graph OOM must be a typed rejection retaining the cause: %v", err)
	}
	class, reason := exactAdmissionFailureEvidence(err)
	if class != "cuda-oom" || !strings.Contains(reason, "device 1") || !strings.Contains(reason, "size unreported") {
		t.Fatalf("negative evidence must name the device without guessing bytes: %s / %s", class, reason)
	}
	if _, _, _, ok := runtimeLogCUDAOOM(startupGraphOOMFixture, nil, nil, nil); ok {
		t.Fatal("startup rejection must not become healthy runtime growth evidence")
	}
}

func TestStartupExactAdmissionLeavesUnknownFailuresRetryable(t *testing.T) {
	for _, log := range []string{"health check timed out", "current device: 1", "CUDA error: illegal memory access\ncurrent device: 1", "warming up the model", ""} {
		if err := startupExactAdmissionFailure(log, errors.New("startup failed")); err != nil {
			t.Fatalf("unproven OOM became stable negative evidence for %q: %v", log, err)
		}
	}
}

func TestStartupExactAdmissionPreservesReportedAllocation(t *testing.T) {
	log := "allocating 2206.07 MiB on device 0: cudaMalloc failed: out of memory"
	err := startupExactAdmissionFailure(log, errors.New("startup failed"))
	if err == nil || !strings.Contains(err.Error(), "allocating 2207 MiB") {
		t.Fatalf("lost reported allocation: %v", err)
	}
}

func TestStartupExactAdmissionUnknownSizeAndDeviceRemainUnknown(t *testing.T) {
	err := startupExactAdmissionFailure("CUDA error: out of memory", errors.New("startup failed"))
	if err == nil || !isStableExactAdmissionFailure(err) || strings.Contains(err.Error(), "device 0") || strings.Contains(err.Error(), "MiB") {
		t.Fatalf("OOM without device/size must reject without fabricating either: %v", err)
	}
}
