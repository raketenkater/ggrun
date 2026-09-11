package benchmark

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRunAgentParallelExercisesConcurrentLongPromptAndMixedDecode(t *testing.T) {
	var mu sync.Mutex
	active := 0
	maxActive := 0
	sawNoCache := false
	sawThinkingOff := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
			MaxTokens          int             `json:"max_tokens"`
			CachePrompt        bool            `json:"cache_prompt"`
			ChatTemplateKwargs map[string]bool `json:"chat_template_kwargs"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		mu.Lock()
		active++
		if active > maxActive {
			maxActive = active
		}
		if !body.CachePrompt {
			sawNoCache = true
		}
		if enabled, ok := body.ChatTemplateKwargs["enable_thinking"]; ok && !enabled {
			sawThinkingOff = true
		}
		mu.Unlock()
		time.Sleep(15 * time.Millisecond)
		promptTokens := 10
		if len(body.Messages) > 0 {
			promptTokens = len(body.Messages[0].Content) / 4
		}
		completionTokens := body.MaxTokens
		cached := 0
		if body.CachePrompt {
			cached = promptTokens - 20
			if cached < 1 {
				cached = 1
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": "ok"}}},
			"usage":   map[string]int{"prompt_tokens": promptTokens, "completion_tokens": completionTokens},
			"timings": map[string]interface{}{"prompt_per_second": 100.0, "predicted_per_second": 20.0, "cache_n": cached},
		})
		mu.Lock()
		active--
		mu.Unlock()
	}))
	defer server.Close()

	result, err := (&Runner{
		BaseURL: server.URL, Model: "local", Timeout: time.Second,
		GPUUtilizationInterval: 2 * time.Millisecond,
		SampleGPUUtilization: func() []GPUUtilization {
			return []GPUUtilization{{GPU: 0, SMPercent: 92, MemPercent: 70}, {GPU: 2, SMPercent: 8, MemPercent: 60}}
		},
	}).RunAgentParallel(2)
	if err != nil {
		t.Fatalf("RunAgentParallel: %v", err)
	}
	if result.Parallel != 2 || result.PromptTokens < 3000 || result.GenTokens != 2*agentBenchmarkGenTokens {
		t.Fatalf("unexpected workload result: %+v", result)
	}
	if result.PromptTPS <= 0 || result.GenTPS <= 0 || result.MixedGenTPS <= 0 || result.MixedGenTokens != agentBenchmarkGenTokens {
		t.Fatalf("missing workload metrics: %+v", result)
	}
	if result.AgentSamples != agentBenchmarkTrials || result.AgentTurnTimeS <= 0 || result.AgentTurnMaxS < result.AgentTurnTimeS ||
		result.AgentCachedTokens <= 0 || result.AgentNewPromptTokens <= 0 {
		t.Fatalf("missing repeated cache-backed turn evidence: %+v", result)
	}
	if len(result.GPUUtilization) != 2 || result.GPUUtilization[0].GPU != 0 ||
		result.GPUUtilization[0].Observations < 2 || result.GPUUtilization[1].GPU != 2 {
		t.Fatalf("mixed-phase GPU observation was not retained deterministically: %+v", result.GPUUtilization)
	}
	if len(result.PhaseUtilization) != 4 {
		t.Fatalf("agent phases were not retained separately: %+v", result.PhaseUtilization)
	}
	for i, phase := range []AgentPhase{AgentPhasePrefill, AgentPhaseAppend, AgentPhaseDecode, AgentPhaseMixed} {
		if result.PhaseUtilization[i].Phase != phase || len(result.PhaseUtilization[i].GPUUtilization) != 2 {
			t.Fatalf("phase %d evidence=%+v", i, result.PhaseUtilization[i])
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if maxActive < 2 {
		t.Fatalf("benchmark never exercised concurrency: max active=%d", maxActive)
	}
	if !sawNoCache || !sawThinkingOff {
		t.Fatalf("benchmark did not disable cache/thinking: no-cache=%t thinking-off=%t", sawNoCache, sawThinkingOff)
	}
}

func TestGPUUtilizationObservationFailsClosedOnIdleFrames(t *testing.T) {
	runner := &Runner{
		GPUUtilizationInterval: time.Millisecond,
		SampleGPUUtilization: func() []GPUUtilization {
			return []GPUUtilization{{GPU: 0}, {GPU: 1}}
		},
	}
	stop := startGPUUtilizationObservation(runner)
	time.Sleep(5 * time.Millisecond)
	if got := stop(); got != nil {
		t.Fatalf("idle frames became topology evidence: %+v", got)
	}
}

func TestProbeAndSuggestedAgentPromptGeometry(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		promptTokens := 300
		if len(body.Messages) == 0 || len(body.Messages[0].Content) < agentBenchmarkProbeLen {
			t.Errorf("probe prompt was not generated at the requested scale")
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": "ok"}}},
			"usage":   map[string]int{"prompt_tokens": promptTokens, "completion_tokens": 1},
			"timings": map[string]interface{}{"prompt_per_second": 18.0, "predicted_per_second": 5.0},
		})
	}))
	defer server.Close()
	probe, err := (&Runner{BaseURL: server.URL, Model: "local", Timeout: time.Second}).ProbeAgentPrefill()
	if err != nil {
		t.Fatal(err)
	}
	if probe.PromptBytes < agentBenchmarkProbeLen || probe.PromptTokens != 300 || probe.PromptTPS != 18 {
		t.Fatalf("probe=%+v", probe)
	}
	got := SuggestedAgentPromptBytes(PrefillProbe{PromptBytes: 1200, PromptTokens: 300, PromptTPS: 18}, 2, 128, time.Minute)
	if got != 2304 {
		t.Fatalf("budget-scaled prompt=%d, want 2304", got)
	}
	if got := SuggestedAgentPromptBytes(PrefillProbe{PromptBytes: 1200, PromptTokens: 300, PromptTPS: 500}, 2, 128, time.Minute); got != agentBenchmarkPromptLen {
		t.Fatalf("fast baseline did not retain full workload: %d", got)
	}
	if got := SuggestedAgentPromptBytes(PrefillProbe{}, 2, 512, time.Minute); got != 2048 {
		t.Fatalf("missing pilot fallback=%d, want one capped ubatch (2048 bytes)", got)
	}
}

func TestBoundedAgentPromptRetainsEndToEndScenario(t *testing.T) {
	runner := &Runner{WorkloadID: "scope", AgentPromptBytes: 2048}
	prompt := runner.agentBenchmarkLongPrompt(0, 0)
	if len(prompt) < 2048 || len(prompt) > 2304 {
		t.Fatalf("bounded prompt length=%d", len(prompt))
	}
	trials := []*Result{
		{Model: "m", Parallel: 2, PromptTimeS: 6, AgentTurnTimeS: 4, AgentScenarioTimeS: 10, AgentPromptBytes: len(prompt), AgentCachedTokens: 400},
		{Model: "m", Parallel: 2, PromptTimeS: 8, AgentTurnTimeS: 3, AgentScenarioTimeS: 11, AgentPromptBytes: len(prompt), AgentCachedTokens: 390},
	}
	got := aggregateAgentTrials(trials)
	if got.AgentScenarioTimeS != 10.5 || got.AgentScenarioMaxS != 11 || got.AgentPromptBytes != len(prompt) {
		t.Fatalf("aggregated scenario=%+v", got)
	}
}

func TestSerialAgentTrialSamplesLongPrefillNotOnlyFinalDecode(t *testing.T) {
	var mu sync.Mutex
	longPromptActive := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
			MaxTokens int `json:"max_tokens"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		longPrompt := len(body.Messages) > 0 && len(body.Messages[0].Content) > 10000
		mu.Lock()
		longPromptActive = longPrompt
		mu.Unlock()
		if longPrompt {
			time.Sleep(12 * time.Millisecond)
		} else {
			time.Sleep(time.Millisecond)
		}
		mu.Lock()
		longPromptActive = false
		mu.Unlock()
		promptTokens := 100
		if longPrompt {
			promptTokens = 8000
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": "ok"}}},
			"usage":   map[string]int{"prompt_tokens": promptTokens, "completion_tokens": max(1, body.MaxTokens)},
			"timings": map[string]interface{}{"prompt_per_second": 100.0, "predicted_per_second": 20.0, "cache_n": max(1, promptTokens-20)},
		})
	}))
	defer server.Close()
	result, err := (&Runner{
		BaseURL: server.URL, Model: "local", Timeout: time.Second,
		GPUUtilizationInterval: time.Millisecond,
		SampleGPUUtilization: func() []GPUUtilization {
			mu.Lock()
			active := longPromptActive
			mu.Unlock()
			if active {
				return []GPUUtilization{{GPU: 0, SMPercent: 95}, {GPU: 1, SMPercent: 3}}
			}
			return []GPUUtilization{{GPU: 0}, {GPU: 1}}
		},
	}).RunAgentParallel(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.GPUUtilization) != 2 || result.GPUUtilization[0].SMPercent != 95 || result.GPUUtilization[1].SMPercent != 3 {
		t.Fatalf("serial prefill utilization was not observed: %+v", result.GPUUtilization)
	}
}

func TestMergeGPUUtilizationWeightsObservationCounts(t *testing.T) {
	got := mergeGPUUtilization(
		[]GPUUtilization{{GPU: 2, SMPercent: 90, MemPercent: 60, Observations: 3}},
		[]GPUUtilization{{GPU: 2, SMPercent: 10, MemPercent: 20, Observations: 1}, {GPU: 0, SMPercent: 50, Observations: 2}},
	)
	if len(got) != 2 || got[0].GPU != 0 || got[1].GPU != 2 ||
		got[1].SMPercent != 70 || got[1].MemPercent != 50 || got[1].Observations != 4 {
		t.Fatalf("weighted merge=%+v", got)
	}
}

func TestRunAgentParallelCapsSyntheticLoad(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": "token token token token"}}},
			"usage":   map[string]int{"prompt_tokens": 2000, "completion_tokens": agentBenchmarkGenTokens},
			"timings": map[string]interface{}{"prompt_per_second": 100.0, "predicted_per_second": 20.0, "cache_n": 1800},
		})
	}))
	defer server.Close()
	result, err := (&Runner{BaseURL: server.URL, Model: "local"}).RunAgentParallel(32)
	if err != nil {
		t.Fatal(err)
	}
	if result.Parallel != agentBenchmarkMaxLanes {
		t.Fatalf("parallel benchmark lanes=%d, want cap %d", result.Parallel, agentBenchmarkMaxLanes)
	}
}

// The one-slot baseline must receive every request sent to the wider candidate,
// including mixed-phase contenders, instead of extrapolating one agent's time.
func TestRunAgentWorkloadKeepsWorkIdenticalAcrossCapacity(t *testing.T) {
	var previous []string
	for _, slots := range []int{1, 2} {
		var mu sync.Mutex
		var requests []string
		capacity := make(chan struct{}, slots)
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			capacity <- struct{}{}
			defer func() { <-capacity }()
			var body map[string]interface{}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode: %v", err)
				http.Error(w, "bad request", 400)
				return
			}
			encoded, _ := json.Marshal(body)
			mu.Lock()
			requests = append(requests, string(encoded))
			mu.Unlock()
			time.Sleep(time.Millisecond)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": "ok"}}},
				"usage":   map[string]interface{}{"prompt_tokens": 256, "completion_tokens": body["max_tokens"]},
				"timings": map[string]interface{}{"prompt_per_second": 100.0, "predicted_per_second": 20.0, "cache_n": 200},
			})
		}))
		result, err := (&Runner{BaseURL: server.URL, Model: "local", WorkloadID: "matched", AgentPromptBytes: 1024, Timeout: time.Second}).RunAgentWorkload(slots, 2)
		server.Close()
		if err != nil {
			t.Fatal(err)
		}
		if result.Parallel != slots || result.AgentWorkloadLanes != 2 || result.GenTokens != 2*agentBenchmarkGenTokens {
			t.Fatalf("capacity changed measured work: %+v", result)
		}
		if result.AgentWorkloadTimeS != result.AgentScenarioTimeS || result.AgentWorkloadMaxS != result.AgentScenarioMaxS {
			t.Fatalf("workflow time was extrapolated: %+v", result)
		}
		sort.Strings(requests)
		if previous != nil && !reflect.DeepEqual(previous, requests) {
			t.Fatal("slot width changed benchmark requests")
		}
		previous = requests
	}
}

func TestRunAgentWorkloadRejectsUnsupportedDemand(t *testing.T) {
	for _, input := range [][2]int{{0, 1}, {1, 0}, {1, 9}} {
		if result, err := (&Runner{}).RunAgentWorkload(input[0], input[1]); err == nil || result != nil {
			t.Fatalf("invalid workload %v accepted: %+v, %v", input, result, err)
		}
	}
}

func TestRunAgentWorkloadFailsWhenQueuedLaneFails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
			MaxTokens int `json:"max_tokens"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if len(body.Messages) > 0 && strings.Contains(body.Messages[0].Content, "lane 1 repository") {
			http.Error(w, "queued request failed", 500)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": "ok"}}},
			"usage":   map[string]int{"prompt_tokens": 256, "completion_tokens": body.MaxTokens},
			"timings": map[string]interface{}{"prompt_per_second": 100.0, "predicted_per_second": 20.0, "cache_n": 200},
		})
	}))
	defer server.Close()
	result, err := (&Runner{BaseURL: server.URL, Model: "local", Timeout: time.Second}).RunAgentWorkload(1, 2)
	if err == nil || result != nil {
		t.Fatalf("partial workflow accepted: %+v, %v", result, err)
	}
}
