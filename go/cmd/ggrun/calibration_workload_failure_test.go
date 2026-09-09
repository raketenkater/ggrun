package main

import (
	"errors"
	"strings"
	"testing"
)

func TestCandidatePostStartCUDAOOMClassifiesOnlyLatestLoadedScope(t *testing.T) {
	cause := errors.New(`agent warm-up: Post "http://localhost:8081/v1/chat/completions": EOF`)
	log := `loading model
CUDA error: out of memory
current device: 0, in function load
llama_server: model loaded
request started
CUDA error: out of memory
current device: 1, in function ggml_cuda_graph_evaluate_and_capture
cudaGraphInstantiate(&graph->instance, graph->graph, 0, 0, 0)`
	class, reason, ok := candidatePostStartCUDAOOMFromLog(log, cause)
	if !ok || class != "cuda-oom" || !strings.Contains(reason, "device 1") || !strings.Contains(reason, "size unreported") {
		t.Fatalf("post-start OOM = %q, %q, %v", class, reason, ok)
	}
}

func TestCandidatePostStartCUDAOOMLeavesTransientAndPreloadFailuresInconclusive(t *testing.T) {
	cause := errors.New("benchmark request failed")
	for _, log := range []string{
		"llama_server: model loaded\nrequest cancelled",
		"CUDA error: out of memory\ncurrent device: 1\nllama_server: model loaded",
		"llama_server: model loaded\nhealth check ok\nrequest failed",
	} {
		if class, reason, ok := candidatePostStartCUDAOOMFromLog(log, cause); ok || class != "" || reason != "" {
			t.Fatalf("log was incorrectly classified: %q => %q, %q, %v", log, class, reason, ok)
		}
	}
	if class, reason, ok := candidatePostStartCUDAOOM(nil, cause); ok || class != "" || reason != "" {
		t.Fatalf("nil process was classified: %q, %q, %v", class, reason, ok)
	}
}
