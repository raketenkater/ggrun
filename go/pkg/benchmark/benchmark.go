package benchmark

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Result holds benchmark metrics.
type Result struct {
	Model        string  `json:"model"`
	PromptTokens int     `json:"prompt_tokens"`
	PromptTimeS  float64 `json:"prompt_time_s"`
	PromptTPS    float64 `json:"prompt_tps"`
	GenTokens    int     `json:"gen_tokens"`
	GenTimeS     float64 `json:"gen_time_s"`
	GenTPS       float64 `json:"gen_tps"`
	// Parallel and MixedGenTPS are populated by RunAgentParallel. GenTPS is
	// aggregate decode throughput across the active lanes; MixedGenTPS is the
	// foreground lane's wall-clock throughput while the remaining lanes ingest
	// long prompts. The latter catches scheduler starvation that a serial
	// tokens/second benchmark cannot see.
	Parallel       int     `json:"parallel,omitempty"`
	MixedGenTokens int     `json:"mixed_gen_tokens,omitempty"`
	MixedTimeS     float64 `json:"mixed_time_s,omitempty"`
	MixedGenTPS    float64 `json:"mixed_gen_tps,omitempty"`
	// AgentTurnTimeS is the wall time of a cache-backed append turn across the
	// active lanes. It remains a diagnostic for cache behavior; the complete
	// cold-ingest-plus-append scenario below is the automatic calibration
	// objective. AgentTurnMaxS and AgentSamples retain repeatability evidence,
	// while AgentCachedTokens proves the measured turn reused a prefix instead of
	// winning by silently re-prefilling a different workload.
	AgentTurnTimeS float64 `json:"agent_turn_time_s,omitempty"`
	AgentTurnMaxS  float64 `json:"agent_turn_max_s,omitempty"`
	// AgentScenarioTimeS is the measured cold-ingest plus cache-backed append
	// scenario. AgentScenarioMaxS retains its slowest repeated sample. These are
	// the end-to-end automatic optimization objective; AgentTurnTimeS remains the
	// append-only diagnostic so cache regressions stay visible.
	AgentScenarioTimeS   float64 `json:"agent_scenario_time_s,omitempty"`
	AgentScenarioMaxS    float64 `json:"agent_scenario_max_s,omitempty"`
	AgentPromptBytes     int     `json:"agent_prompt_bytes,omitempty"`
	AgentSamples         int     `json:"agent_samples,omitempty"`
	AgentCachedTokens    int     `json:"agent_cached_tokens,omitempty"`
	AgentNewPromptTokens int     `json:"agent_new_prompt_tokens,omitempty"`
	// AgentWorkloadTimeS normalizes candidates with different server slot
	// counts to the same requested concurrency. A one-slot server serving four
	// agents pays four serial waves; aggregate tok/s alone cannot express that.
	AgentWorkloadLanes int     `json:"agent_workload_lanes,omitempty"`
	AgentWorkloadTimeS float64 `json:"agent_workload_time_s,omitempty"`
	AgentWorkloadMaxS  float64 `json:"agent_workload_max_s,omitempty"`
	// GPUUtilization is sampled across the complete active agent trial when a
	// sampler is installed. Empty means no observation, not a balanced topology.
	GPUUtilization []GPUUtilization `json:"gpu_utilization,omitempty"`
	// PhaseUtilization keeps prefill, cache-backed append, decode, and mixed
	// traffic separate. Aggregate GPUUtilization remains for compatibility and
	// topology ranking; phase evidence is what supports a bottleneck diagnosis.
	PhaseUtilization []PhaseUtilization `json:"phase_utilization,omitempty"`
	DraftTokens      int                `json:"draft_tokens,omitempty"`
	DraftAccepted    int                `json:"draft_accepted,omitempty"`
	DraftAcceptRate  float64            `json:"draft_accept_rate,omitempty"`
	PeakVRAMMB       int                `json:"peak_vram_mb,omitempty"`
	LoadTimeS        float64            `json:"load_time_s,omitempty"`
	Timestamp        int64              `json:"timestamp"`
}

// Runner executes a benchmark against a running server.
type Runner struct {
	BaseURL string
	Model   string
	Timeout time.Duration // per-request timeout (default 5 minutes)
	// ContextTokens is the per-slot context the server was actually launched
	// with. Zero means unknown. The cache canary uses it to size its prompt:
	// its segments are counted in words, and token expansion per word is a
	// property of the tokenizer, not a constant. A 512-entry vocabulary turned
	// the 1260-word canary into 15873 tokens, and the launch died against a
	// 2048-token context with HTTP 400 instead of serving.
	ContextTokens int
	// WorkloadID distinguishes repeated calibration samples inside one running
	// server. Candidates use the same IDs in separate processes, so they see the
	// same prompt lengths without a later sample inheriting an earlier prefix.
	WorkloadID string
	// AgentPromptBytes selects the per-lane deterministic prompt geometry used
	// by RunAgentParallel. Zero retains the full maintenance workload. Automatic
	// launch supplies one measured, budget-scaled value and reuses it unchanged
	// for the baseline and finalist.
	AgentPromptBytes int
	// SampleGPUUtilization, when set, is invoked throughout the active agent
	// trial. A nil sampler or empty result is not evidence of device balance.
	SampleGPUUtilization func() []GPUUtilization
	// GPUUtilizationInterval controls sampling cadence. Zero uses the production
	// default; tests may shorten it without changing global state.
	GPUUtilizationInterval time.Duration
	// SampleResources may add process and queue counters to the same bounded
	// phase observation. GPU samples from SampleGPUUtilization are merged in when
	// this callback leaves them empty, preserving existing callers.
	SampleResources func() ResourceSnapshot
	// ResourceSamplingInterval supersedes GPUUtilizationInterval when set. The
	// older field remains supported for API and test compatibility.
	ResourceSamplingInterval time.Duration
}

// GPUUtilization is one observed device during a bounded agent workload.
type GPUUtilization struct {
	GPU          int `json:"gpu"`
	SMPercent    int `json:"sm_percent"`
	MemPercent   int `json:"mem_percent"`
	PCIeRXMBps   int `json:"pcie_rx_mbps,omitempty"`
	PCIeTXMBps   int `json:"pcie_tx_mbps,omitempty"`
	Observations int `json:"observations,omitempty"`
}

func (r *Runner) client() *http.Client {
	timeout := r.Timeout
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}
	return &http.Client{Timeout: timeout}
}

// Run performs a warm-up + measurement prompt and returns metrics.
func (r *Runner) Run() (*Result, error) {
	warmUp := `Explain quantum computing in one sentence.`
	measurePrompt := `Write a practical local LLM inference runbook for an engineer tuning llama.cpp serving. Cover request batching, KV cache size, GPU layer placement, split mode, speculative decoding, and output quality checks. Use numbered sections and continue until the runbook is complete.`

	if _, err := r.chat(warmUp, 32); err != nil {
		return nil, fmt.Errorf("warm-up: %w", err)
	}

	start := time.Now()
	prefillResp, err := r.chat(measurePrompt, 1)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	prefillTime := time.Since(start).Seconds()

	start = time.Now()
	genResp, err := r.chat(measurePrompt, 256)
	if err != nil {
		return nil, fmt.Errorf("generation: %w", err)
	}
	genTime := time.Since(start).Seconds()

	promptTokens := prefillResp.PromptTokens
	if promptTokens <= 0 {
		promptTokens = estimateTokens(measurePrompt)
	}
	promptTPS := prefillResp.PromptTPS
	if promptTPS <= 0 && prefillTime > 0 {
		promptTPS = float64(promptTokens) / prefillTime
	}

	genTokens := genResp.CompletionTokens
	if genTokens <= 0 {
		genTokens = estimateTokens(genResp.Content)
	}
	genTPS := genResp.GenTPS
	if genTPS <= 0 && genTime > 0 {
		genTPS = float64(genTokens) / genTime
	}

	return &Result{
		Model:           r.Model,
		PromptTokens:    promptTokens,
		PromptTimeS:     prefillTime,
		PromptTPS:       promptTPS,
		GenTokens:       genTokens,
		GenTimeS:        genTime,
		GenTPS:          genTPS,
		DraftTokens:     genResp.DraftTokens,
		DraftAccepted:   genResp.DraftAccepted,
		DraftAcceptRate: draftAcceptRate(genResp.DraftTokens, genResp.DraftAccepted),
		Timestamp:       time.Now().Unix(),
	}, nil
}

type chatResult struct {
	Content          string
	PromptTokens     int
	CompletionTokens int
	PromptTPS        float64
	GenTPS           float64
	DraftTokens      int
	DraftAccepted    int
	CachedTokens     int
}

func (r *Runner) chat(prompt string, maxTokens int) (*chatResult, error) {
	return r.chatWithOptions(prompt, maxTokens, 0, true, 0)
}

// chatWithOptions is the common request path for serial and workload-aware
// benchmarks. minTokens keeps short benchmark generations from terminating at
// EOS before the concurrent scheduler has been exercised. Workload screens use
// distinct cold prompt IDs, then enable cache only for an exact append, so reuse
// is measured rather than inherited accidentally across samples.
func (r *Runner) chatWithOptions(prompt string, maxTokens, minTokens int, cachePrompt bool, seed int) (*chatResult, error) {
	body := map[string]interface{}{
		"model": r.Model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"max_tokens":   maxTokens,
		"cache_prompt": cachePrompt,
	}
	if minTokens > 0 {
		body["min_tokens"] = minTokens
	}
	if seed != 0 {
		body["seed"] = seed
	}
	body["temperature"] = 0
	body["chat_template_kwargs"] = map[string]bool{"enable_thinking": false}
	data, _ := json.Marshal(body)
	resp, err := r.client().Post(r.BaseURL+"/v1/chat/completions", "application/json", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			PromptDetails    struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
		Timings struct {
			PromptPerSecond    float64 `json:"prompt_per_second"`
			PredictedPerSecond float64 `json:"predicted_per_second"`
			DraftTokens        int     `json:"draft_n"`
			DraftAccepted      int     `json:"draft_n_accepted"`
			CacheN             int     `json:"cache_n"`
		} `json:"timings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Choices) == 0 {
		return nil, fmt.Errorf("no choices")
	}
	cached := out.Timings.CacheN
	if out.Usage.PromptDetails.CachedTokens > cached {
		cached = out.Usage.PromptDetails.CachedTokens
	}
	return &chatResult{
		Content:          out.Choices[0].Message.Content,
		PromptTokens:     out.Usage.PromptTokens,
		CompletionTokens: out.Usage.CompletionTokens,
		PromptTPS:        out.Timings.PromptPerSecond,
		GenTPS:           out.Timings.PredictedPerSecond,
		DraftTokens:      out.Timings.DraftTokens,
		DraftAccepted:    out.Timings.DraftAccepted,
		CachedTokens:     cached,
	}, nil
}

func draftAcceptRate(drafted, accepted int) float64 {
	if drafted <= 0 || accepted <= 0 {
		return 0
	}
	return float64(accepted) / float64(drafted)
}

func estimateTokens(text string) int {
	// Rough heuristic: ~4 chars per token for English
	return len(text) / 4
}
