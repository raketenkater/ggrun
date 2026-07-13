package specbench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

type Prompt struct {
	Name      string
	Text      string
	MaxTokens int
	Expected  string
}

type Sample struct {
	Prompt          string  `json:"prompt"`
	PromptTokens    int     `json:"prompt_tokens"`
	PromptTPS       float64 `json:"prompt_tps"`
	GeneratedTokens int     `json:"generated_tokens"`
	GenerationTPS   float64 `json:"generation_tps"`
	WallSeconds     float64 `json:"wall_seconds"`
	DraftTokens     int     `json:"draft_tokens,omitempty"`
	DraftAccepted   int     `json:"draft_accepted,omitempty"`
	Correct         bool    `json:"correct"`
	Error           string  `json:"error,omitempty"`
}

type Result struct {
	Samples           []Sample `json:"samples"`
	MedianPromptTPS   float64  `json:"median_prompt_tps"`
	MedianGenerateTPS float64  `json:"median_generation_tps"`
	MeanGenerated     float64  `json:"mean_generated_tokens"`
	MeanWallSeconds   float64  `json:"mean_wall_seconds"`
	DraftTokens       int      `json:"draft_tokens"`
	DraftAccepted     int      `json:"draft_accepted"`
	DraftAcceptRate   float64  `json:"draft_accept_rate"`
	MaxPromptTokens   int      `json:"max_prompt_tokens"`
	CorrectnessPassed bool     `json:"correctness_passed"`
	StabilityPassed   bool     `json:"stability_passed"`
}

type Runner struct {
	BaseURL    string
	Model      string
	Timeout    time.Duration
	Rounds     int
	Parallel   int
	Include60K bool
}

func Prompts(include60K bool) []Prompt {
	prompts := []Prompt{
		verifiedPrompt("code", "Implement a bounded worker pool in Go with cancellation and tests. Explain the invariants.", 192),
		verifiedPrompt("explanation", "Explain how speculative decoding changes latency, throughput, and memory use for local LLM serving.", 192),
		verifiedPrompt("summary", "Summarize the tradeoffs of CPU-offloaded MoE inference in seven concise bullets.", 160),
		verifiedPrompt("factual", "What causes KV-cache memory growth? Separate model, context, batch, and parallelism effects.", 160),
		verifiedPrompt("translation", "Translate into German and preserve technical meaning: The scheduler must not mistake delayed telemetry for a failed backend.", 128),
		verifiedPrompt("creative", "Write a short scene in which three GPUs negotiate where to store a mixture-of-experts model.", 192),
		verifiedPrompt("math", "Derive the memory needed for a 32-layer GQA KV cache with 8 KV heads, head width 128, f16 K/V, and 65536 tokens.", 192),
		verifiedPrompt("tool-plan", "Plan a safe repository refactor: inspect changes, checkpoint, implement, test, and roll back on regression.", 192),
		verifiedPrompt("review", "Review a hypothetical inference launcher for race conditions, unsafe artifact selection, OOM recovery, and misleading progress reporting.", 224),
	}
	if include60K {
		last := &prompts[len(prompts)-1]
		last.Text = longReviewPrompt() + "\nEnd your answer with the exact marker " + last.Expected + "."
		prompts[len(prompts)-1].MaxTokens = 128
	}
	return prompts
}

func verifiedPrompt(name, text string, maxTokens int) Prompt {
	marker := "GGRUN-" + strings.ToUpper(name) + "-OK"
	return Prompt{Name: name, Text: text + " Keep the answer concise and end it with the exact marker " + marker + ".", MaxTokens: maxTokens, Expected: marker}
}

func longReviewPrompt() string {
	var b strings.Builder
	b.Grow(370000)
	b.WriteString("Review the following synthetic service trace. Identify stability and performance problems, then propose fixes.\n")
	// The leading-space form of this common word is one token in the target
	// tokenizer families used by the harness. Slightly exceed 60k so the
	// server-reported token count proves the long-context gate without getting
	// close to a 65,536-token Claude Code slot.
	for i := 0; i < 60500; i++ {
		b.WriteString(" event")
	}
	return b.String()
}

func (r *Runner) Run() Result {
	rounds := r.Rounds
	if rounds < 1 {
		rounds = 2
	}
	parallel := r.Parallel
	if parallel < 1 {
		parallel = 1
	}
	prompts := Prompts(r.Include60K)
	timeout := r.Timeout
	if timeout <= 0 {
		timeout = 30 * time.Minute
	}
	client := &http.Client{Timeout: timeout}
	// A short warmup is not included in measurements.
	_ = r.one(client, Prompt{Name: "warmup", Text: "Reply with the word ready.", MaxTokens: 8})

	result := Result{CorrectnessPassed: true, StabilityPassed: true}
	for round := 0; round < rounds; round++ {
		for start := 0; start < len(prompts); start += parallel {
			end := start + parallel
			if end > len(prompts) {
				end = len(prompts)
			}
			batch := make([]Sample, end-start)
			var wg sync.WaitGroup
			for i := start; i < end; i++ {
				wg.Add(1)
				go func(dst int, prompt Prompt) {
					defer wg.Done()
					batch[dst] = r.one(client, prompt)
				}(i-start, prompts[i])
			}
			wg.Wait()
			result.Samples = append(result.Samples, batch...)
		}
	}
	result.summarize(rounds, len(prompts))
	return result
}

func (r *Runner) one(client *http.Client, prompt Prompt) Sample {
	sample := Sample{Prompt: prompt.Name}
	body := map[string]any{
		"model": r.Model, "messages": []map[string]string{{"role": "user", "content": prompt.Text}},
		"max_tokens": prompt.MaxTokens,
	}
	data, _ := json.Marshal(body)
	start := time.Now()
	resp, err := client.Post(r.BaseURL+"/v1/chat/completions", "application/json", bytes.NewReader(data))
	sample.WallSeconds = time.Since(start).Seconds()
	if err != nil {
		sample.Error = err.Error()
		return sample
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		sample.Error = fmt.Sprintf("HTTP %d", resp.StatusCode)
		return sample
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
		sample.Error = err.Error()
		return sample
	}
	if len(out.Choices) == 0 {
		sample.Error = "no choices"
		return sample
	}
	sample.PromptTokens = out.Usage.PromptTokens
	sample.PromptTPS = out.Timings.PromptPerSecond
	sample.GeneratedTokens = out.Usage.CompletionTokens
	sample.GenerationTPS = out.Timings.PredictedPerSecond
	sample.DraftTokens = out.Timings.DraftTokens
	sample.DraftAccepted = out.Timings.DraftAccepted
	content := strings.TrimSpace(out.Choices[0].Message.Content)
	sample.Correct = content != "" && sample.GeneratedTokens > 0 && (prompt.Expected == "" || strings.Contains(content, prompt.Expected))
	if !sample.Correct {
		sample.Error = "empty, tokenless, or missing verification marker"
	}
	return sample
}

func (r *Result) summarize(rounds, promptCases int) {
	var promptRates, genRates []float64
	var wall float64
	generated := 0
	for _, sample := range r.Samples {
		if sample.PromptTPS > 0 {
			promptRates = append(promptRates, sample.PromptTPS)
		}
		if sample.GenerationTPS > 0 {
			genRates = append(genRates, sample.GenerationTPS)
		}
		wall += sample.WallSeconds
		generated += sample.GeneratedTokens
		r.DraftTokens += sample.DraftTokens
		r.DraftAccepted += sample.DraftAccepted
		if sample.PromptTokens > r.MaxPromptTokens {
			r.MaxPromptTokens = sample.PromptTokens
		}
		if !sample.Correct {
			r.CorrectnessPassed = false
		}
	}
	r.MedianPromptTPS = median(promptRates)
	r.MedianGenerateTPS = median(genRates)
	if len(r.Samples) > 0 {
		r.MeanWallSeconds = wall / float64(len(r.Samples))
		r.MeanGenerated = float64(generated) / float64(len(r.Samples))
	}
	if r.DraftTokens > 0 {
		r.DraftAcceptRate = float64(r.DraftAccepted) / float64(r.DraftTokens)
	}
	r.StabilityPassed = r.CorrectnessPassed && rounds >= 2 && len(r.Samples) == rounds*promptCases
}

func median(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sort.Float64s(values)
	mid := len(values) / 2
	if len(values)%2 == 0 {
		return (values[mid-1] + values[mid]) / 2
	}
	return values[mid]
}
