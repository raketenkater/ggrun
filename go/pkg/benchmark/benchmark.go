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
	Model           string  `json:"model"`
	PromptTokens    int     `json:"prompt_tokens"`
	PromptTimeS     float64 `json:"prompt_time_s"`
	PromptTPS       float64 `json:"prompt_tps"`
	GenTokens       int     `json:"gen_tokens"`
	GenTimeS        float64 `json:"gen_time_s"`
	GenTPS          float64 `json:"gen_tps"`
	DraftTokens     int     `json:"draft_tokens,omitempty"`
	DraftAccepted   int     `json:"draft_accepted,omitempty"`
	DraftAcceptRate float64 `json:"draft_accept_rate,omitempty"`
	PeakVRAMMB      int     `json:"peak_vram_mb,omitempty"`
	LoadTimeS       float64 `json:"load_time_s,omitempty"`
	Timestamp       int64   `json:"timestamp"`
}

// Runner executes a benchmark against a running server.
type Runner struct {
	BaseURL string
	Model   string
	Timeout time.Duration // per-request timeout (default 5 minutes)
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

	// Warm up: run 3 times to stabilize JIT/allocator state
	for i := 0; i < 3; i++ {
		if _, err := r.chat(warmUp, 32); err != nil {
			return nil, fmt.Errorf("warm-up: %w", err)
		}
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
}

func (r *Runner) chat(prompt string, maxTokens int) (*chatResult, error) {
	body := map[string]interface{}{
		"model": r.Model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"max_tokens": maxTokens,
	}
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
		} `json:"usage"`
		Timings struct {
			PromptPerSecond    float64 `json:"prompt_per_second"`
			PredictedPerSecond float64 `json:"predicted_per_second"`
			DraftTokens        int     `json:"draft_n"`
			DraftAccepted      int     `json:"draft_n_accepted"`
		} `json:"timings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Choices) == 0 {
		return nil, fmt.Errorf("no choices")
	}
	return &chatResult{
		Content:          out.Choices[0].Message.Content,
		PromptTokens:     out.Usage.PromptTokens,
		CompletionTokens: out.Usage.CompletionTokens,
		PromptTPS:        out.Timings.PromptPerSecond,
		GenTPS:           out.Timings.PredictedPerSecond,
		DraftTokens:      out.Timings.DraftTokens,
		DraftAccepted:    out.Timings.DraftAccepted,
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
