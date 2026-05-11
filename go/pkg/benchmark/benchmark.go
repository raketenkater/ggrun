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
}

// Run performs a warm-up + measurement prompt and returns metrics.
func (r *Runner) Run() (*Result, error) {
	warmUp := `Explain quantum computing in one sentence.`
	measurePrompt := `Write a short story about a robot learning to paint. Include a beginning, middle, and end.`

	// Warm-up
	if _, err := r.chat(warmUp, 32); err != nil {
		return nil, fmt.Errorf("warm-up: %w", err)
	}

	// Measurement: prefill
	start := time.Now()
	_, err := r.chat(measurePrompt, 1)
	if err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}
	prefillTime := time.Since(start).Seconds()

	// Measurement: generation
	start = time.Now()
	genResp, err := r.chat(measurePrompt, 128)
	if err != nil {
		return nil, fmt.Errorf("generation: %w", err)
	}
	genTime := time.Since(start).Seconds()

	res := &Result{
		Model:        r.Model,
		PromptTokens: estimateTokens(measurePrompt),
		PromptTimeS:  prefillTime,
		PromptTPS:    float64(estimateTokens(measurePrompt)) / prefillTime,
		GenTokens:    estimateTokens(genResp),
		GenTimeS:     genTime,
		GenTPS:       float64(estimateTokens(genResp)) / genTime,
		Timestamp:    time.Now().Unix(),
	}
	return res, nil
}

func (r *Runner) chat(prompt string, maxTokens int) (string, error) {
	body := map[string]interface{}{
		"model": r.Model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"max_tokens": maxTokens,
	}
	data, _ := json.Marshal(body)
	resp, err := http.Post(r.BaseURL+"/v1/chat/completions", "application/json", bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("no choices")
	}
	return out.Choices[0].Message.Content, nil
}

func estimateTokens(text string) int {
	// Rough heuristic: ~4 chars per token for English
	return len(text) / 4
}
