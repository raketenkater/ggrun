package tune

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/raketenkater/llm-server/pkg/benchmark"
	"github.com/raketenkater/llm-server/pkg/detect"
)

// Engine runs the AI-tune optimization loop.
type Engine struct {
	BaseURL    string
	Model      string
	Rounds     int
	Cache      *Cache
	Caps       *detect.Capabilities
	OnProgress func(msg string)
}

// Suggestion is the JSON format the tuning LLM returns.
type Suggestion struct {
	Name      string   `json:"name"`
	Flags     []string `json:"flags"`
	Reasoning string   `json:"reasoning"`
}

// Run executes the full tune loop for a given model + initial strategy.
func (e *Engine) Run(modelPath string, initialFlags []string) (*Entry, error) {
	if e.OnProgress != nil {
		e.OnProgress(fmt.Sprintf("AI-tune: starting %d rounds for %s", e.Rounds, modelPath))
	}

	// Round 0: baseline
	best, err := e.round(0, modelPath, initialFlags)
	if err != nil {
		return nil, fmt.Errorf("baseline benchmark failed: %w", err)
	}

	for round := 1; round <= e.Rounds; round++ {
		if e.OnProgress != nil {
			e.OnProgress(fmt.Sprintf("AI-tune: round %d/%d (best %.1f tok/s)", round, e.Rounds, best.Result.GenTPS))
		}

		// Query LLM for suggestions
		suggestion, err := e.queryLLM(modelPath, best)
		if err != nil {
			if e.OnProgress != nil {
				e.OnProgress(fmt.Sprintf("AI-tune: LLM query failed in round %d: %v", round, err))
			}
			continue
		}

		// Apply suggestion flags
		candidateFlags := applySuggestion(initialFlags, suggestion.Flags)

		// Benchmark candidate
		candidate, err := e.round(round, modelPath, candidateFlags)
		if err != nil {
			if e.OnProgress != nil {
				e.OnProgress(fmt.Sprintf("AI-tune: candidate benchmark failed: %v", err))
			}
			continue
		}

		if candidate.Result.GenTPS > best.Result.GenTPS {
			best = candidate
			if e.OnProgress != nil {
				e.OnProgress(fmt.Sprintf("AI-tune: new best %.1f tok/s (%s)", best.Result.GenTPS, suggestion.Name))
			}
		}
	}

	if e.OnProgress != nil {
		e.OnProgress(fmt.Sprintf("AI-tune: done. Best result: %.1f tok/s", best.Result.GenTPS))
	}

	return best, nil
}

func (e *Engine) round(round int, modelPath string, flags []string) (*Entry, error) {
	// Build and run benchmark
	runner := &benchmark.Runner{
		BaseURL: e.BaseURL,
		Model:   e.Model,
	}
	res, err := runner.Run()
	if err != nil {
		return nil, err
	}

	entry := &Entry{
		Timestamp:   Now(),
		ModelPath:   modelPath,
		ModelName:   e.Model,
		HardwareHash: HardwareHash(gpuNames(e.Caps), e.Caps.TotalVRAM()),
		Round:       round,
		Flags:       flagMap(flags),
		Result: BenchmarkResult{
			PromptTokens: res.PromptTokens,
			PromptTPS:    res.PromptTPS,
			GenTokens:    res.GenTokens,
			GenTPS:       res.GenTPS,
			PeakVRAMMB:   res.PeakVRAMMB,
		},
		Best: round == 0, // baseline starts as best
	}

	if e.Cache != nil {
		_ = e.Cache.Add(*entry)
	}

	return entry, nil
}

func (e *Engine) queryLLM(modelPath string, best *Entry) (*Suggestion, error) {
	prompt := buildTuningPrompt(modelPath, best, e.Caps)

	body := map[string]interface{}{
		"model":    e.Model,
		"messages": []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens": 4096,
		"temperature": 0.3,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
	}

	data, _ := json.Marshal(body)
	resp, err := http.Post(e.BaseURL+"/v1/chat/completions", "application/json", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Choices) == 0 {
		return nil, fmt.Errorf("no choices in LLM response")
	}

	content := out.Choices[0].Message.Content
	// Try to find JSON in the response
	jsonStart := strings.Index(content, "{")
	jsonEnd := strings.LastIndex(content, "}")
	if jsonStart >= 0 && jsonEnd > jsonStart {
		content = content[jsonStart : jsonEnd+1]
	}

	var suggestion Suggestion
	if err := json.Unmarshal([]byte(content), &suggestion); err != nil {
		return nil, fmt.Errorf("failed to parse suggestion JSON: %w", err)
	}

	return &suggestion, nil
}

func buildTuningPrompt(modelPath string, best *Entry, caps *detect.Capabilities) string {
	return fmt.Sprintf(`You are a performance optimization engineer for llama.cpp inference.

Current model: %s
Hardware: %d GPUs, %d MB total VRAM, %s CPU
Current performance: %.1f prefill tok/s, %.1f decode tok/s
Current flags: %v

Suggest llama-server command-line flag changes to improve throughput.
Return ONLY a JSON object with this exact format:
{"name":"short name","flags":["--flag1","--flag2"],"reasoning":"why"}

Rules:
- Only suggest flags that affect performance (batch, threads, cache type, split mode, etc.)
- Do not change model path or port
- Keep suggestions conservative (1-2 flag changes per round)
- If current performance is already good, say so with empty flags`,
		modelPath,
		len(caps.GPUs),
		caps.TotalVRAM(),
		caps.CPU.Model,
		best.Result.PromptTPS,
		best.Result.GenTPS,
		best.Flags,
	)
}

func applySuggestion(baseFlags, suggested []string) []string {
	// Start with base, override with suggested flags
	result := make([]string, len(baseFlags))
	copy(result, baseFlags)

	for i := 0; i < len(suggested); i++ {
		flag := suggested[i]
		// Remove conflicting flags
		result = removeConflicting(result, flag)
		result = append(result, flag)
		// If flag has a value, consume next element
		if i+1 < len(suggested) && !strings.HasPrefix(suggested[i+1], "-") {
			result = append(result, suggested[i+1])
			i++
		}
	}
	return result
}

func removeConflicting(flags []string, newFlag string) []string {
	// Simple prefix match for conflicting flags
	prefix := newFlag
	if idx := strings.Index(newFlag, "="); idx > 0 {
		prefix = newFlag[:idx]
	}
	var result []string
	for _, f := range flags {
		if !strings.HasPrefix(f, prefix) {
			result = append(result, f)
		}
	}
	return result
}

func flagMap(flags []string) map[string]string {
	m := make(map[string]string)
	for i := 0; i < len(flags); i++ {
		if i+1 < len(flags) && !strings.HasPrefix(flags[i+1], "-") {
			m[flags[i]] = flags[i+1]
			i++
		} else {
			m[flags[i]] = ""
		}
	}
	return m
}

func gpuNames(caps *detect.Capabilities) []string {
	var names []string
	for _, g := range caps.GPUs {
		names = append(names, g.Name)
	}
	return names
}
