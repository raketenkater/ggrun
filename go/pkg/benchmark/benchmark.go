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
	PeakVRAMMB   int     `json:"peak_vram_mb,omitempty"`
	LoadTimeS    float64 `json:"load_time_s,omitempty"`
	Timestamp    int64   `json:"timestamp"`
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
	measurePrompt := `Write a short story about a robot learning to paint. Include a beginning, middle, and end.`

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
	genResp, err := r.chat(measurePrompt, 128)
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
		Model:        r.Model,
		PromptTokens: promptTokens,
		PromptTimeS:  prefillTime,
		PromptTPS:    promptTPS,
		GenTokens:    genTokens,
		GenTimeS:     genTime,
		GenTPS:       genTPS,
		Timestamp:    time.Now().Unix(),
	}, nil
}

type chatResult struct {
	Content          string
	PromptTokens     int
	CompletionTokens int
	PromptTPS        float64
	GenTPS           float64
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
	}, nil
}

func estimateTokens(text string) int {
	// Rough heuristic: ~4 chars per token for English
	return len(text) / 4
}
