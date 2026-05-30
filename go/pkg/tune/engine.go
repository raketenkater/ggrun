package tune

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/raketenkater/llm-server/pkg/benchmark"
	"github.com/raketenkater/llm-server/pkg/detect"
)

// Engine runs the AI-tune optimization loop.
type Engine struct {
	BaseURL           string
	Model             string
	Rounds            int
	Cache             *Cache
	Caps              *detect.Capabilities
	Backend           string
	Vision            bool
	MinImprovementPct float64
	OnProgress        func(msg string)
	StartServer       func(flags []string) (cleanup func(), err error)
}

// Suggestion is the JSON format the tuning LLM returns.
type Suggestion struct {
	Name       string                 `json:"name"`
	Flags      []string               `json:"-"`
	FlagValues map[string]interface{} `json:"-"`
	Reasoning  string                 `json:"reasoning"`
}

func (s *Suggestion) UnmarshalJSON(data []byte) error {
	var raw struct {
		Name      string          `json:"name"`
		Flags     json.RawMessage `json:"flags"`
		Reasoning string          `json:"reasoning"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	s.Name = raw.Name
	s.Reasoning = raw.Reasoning
	s.FlagValues = map[string]interface{}{}
	if len(raw.Flags) == 0 || string(raw.Flags) == "null" {
		return nil
	}
	var args []string
	if err := json.Unmarshal(raw.Flags, &args); err == nil {
		s.Flags = args
		s.FlagValues = flagArgsToValues(args)
		return nil
	}
	var values map[string]interface{}
	if err := json.Unmarshal(raw.Flags, &values); err != nil {
		return err
	}
	s.FlagValues = values
	s.Flags = flagValuesToArgs(values)
	return nil
}

// Run executes the full tune loop for a given model + initial strategy.
func (e *Engine) Run(modelPath string, initialFlags []string) (*Entry, error) {
	if e.Rounds < 0 {
		e.Rounds = 0
	}
	minImprovementPct := e.MinImprovementPct
	if minImprovementPct <= 0 {
		minImprovementPct = 1.0
	}
	if e.OnProgress != nil {
		e.OnProgress(fmt.Sprintf("AI-tune: starting %d rounds for %s", e.Rounds, modelPath))
	}

	var baselineCleanup func()
	startBaseline := func() error {
		if e.StartServer == nil {
			return nil
		}
		cleanup, err := e.StartServer(initialFlags)
		if err != nil {
			return err
		}
		baselineCleanup = cleanup
		return nil
	}
	stopBaseline := func() {
		if baselineCleanup != nil {
			baselineCleanup()
			baselineCleanup = nil
		}
	}
	defer stopBaseline()

	if err := startBaseline(); err != nil {
		return nil, fmt.Errorf("start baseline: %w", err)
	}

	// Round 0: benchmark the baseline server and seed the cache with the first best.
	best, err := e.roundRunning(0, modelPath, initialFlags)
	if err != nil {
		return nil, fmt.Errorf("baseline benchmark failed: %w", err)
	}
	baseline := best
	best.Name = "baseline"
	best.Status = "ok"
	best.Best = true
	e.addCache(best)
	entries := []Entry{*best}
	protected := DefaultProtectedFlags()

	for round := 1; round <= e.Rounds; round++ {
		if e.OnProgress != nil {
			e.OnProgress(fmt.Sprintf("AI-tune: round %d/%d (best %.1f tok/s)", round, e.Rounds, best.Result.GenTPS))
		}

		// Query while the known-good baseline server is alive. If the model cannot
		// produce valid tuning JSON, fall back to a deterministic safe candidate so
		// the run still yields measured data.
		suggestion, err := e.queryLLM(modelPath, best)
		if err != nil && e.OnProgress != nil {
			e.OnProgress(fmt.Sprintf("AI-tune: LLM query failed in round %d: %v; using deterministic candidate", round, err))
		}
		if err != nil || suggestion == nil || len(sanitizeFlagValues(suggestion.FlagValues, protected)) == 0 {
			suggestion = deterministicSuggestion(round, initialFlags)
			if suggestion == nil {
				if e.OnProgress != nil {
					e.OnProgress(fmt.Sprintf("AI-tune: no safe candidate left for round %d", round))
				}
				continue
			}
		}

		overrides := sanitizeFlagValues(suggestion.FlagValues, protected)
		candidateFlags := ApplyOverrides(initialFlags, overrides, protected)
		if equalFlags(initialFlags, candidateFlags) {
			if e.OnProgress != nil {
				e.OnProgress(fmt.Sprintf("AI-tune: candidate %q made no effective flag changes", suggestion.Name))
			}
			continue
		}

		// Candidate flags need their own backend process; otherwise every round
		// measures the same already-running baseline.
		stopBaseline()
		candidate, err := e.round(round, modelPath, candidateFlags)
		if err != nil {
			entries = append(entries, Entry{
				Timestamp:     Now(),
				ModelPath:     modelPath,
				ModelName:     e.Model,
				HardwareHash:  e.hardwareHash(),
				Backend:       e.Backend,
				Vision:        e.Vision,
				Round:         round,
				Name:          suggestion.Name,
				Flags:         flagMap(candidateFlags),
				OverrideFlags: overrides,
				Status:        "crashed",
			})
			if e.OnProgress != nil {
				e.OnProgress(fmt.Sprintf("AI-tune: candidate benchmark failed: %v", err))
			}
		} else {
			candidate.Name = suggestion.Name
			candidate.OverrideFlags = overrides
			candidate.Status = "ok"
			candidate.Backend = e.Backend
			candidate.Vision = e.Vision
			candidate.Best = meaningfulImprovement(candidate.Result.GenTPS, best.Result.GenTPS, minImprovementPct)
			e.addCache(candidate)
			entries = append(entries, *candidate)
			if candidate.Best {
				best = candidate
				if e.OnProgress != nil {
					e.OnProgress(fmt.Sprintf("AI-tune: new best %.1f tok/s (%s)", best.Result.GenTPS, suggestion.Name))
				}
			}
		}

		if round < e.Rounds {
			if err := startBaseline(); err != nil {
				return best, fmt.Errorf("restart baseline after round %d: %w", round, err)
			}
		}
	}

	if e.OnProgress != nil {
		e.OnProgress(fmt.Sprintf("AI-tune: done. Best result: %.1f tok/s", best.Result.GenTPS))
	}
	if e.Cache != nil {
		path, err := e.Cache.SaveTuneFile(modelPath, baseline, best, e.Rounds, e.Backend, e.Vision, minImprovementPct, gpuNames(e.Caps), entries)
		if err != nil && e.OnProgress != nil {
			e.OnProgress(fmt.Sprintf("AI-tune: failed to save tune cache: %v", err))
		} else if path != "" && e.OnProgress != nil {
			e.OnProgress(fmt.Sprintf("AI-tune: saved tune cache %s", path))
		}
	}

	return best, nil
}

func (e *Engine) round(round int, modelPath string, flags []string) (*Entry, error) {
	var cleanup func()
	if e.StartServer != nil {
		var err error
		cleanup, err = e.StartServer(flags)
		if err != nil {
			return nil, err
		}
		defer cleanup()
	}
	return e.roundRunning(round, modelPath, flags)
}

func (e *Engine) addCache(entry *Entry) {
	if e.Cache != nil && entry != nil {
		_ = e.Cache.Add(*entry)
	}
}

func (e *Engine) hardwareHash() string {
	if e.Caps == nil {
		return ""
	}
	return HardwareHash(gpuNames(e.Caps), e.Caps.TotalVRAM())
}

func (e *Engine) roundRunning(round int, modelPath string, flags []string) (*Entry, error) {
	runner := &benchmark.Runner{
		BaseURL: e.BaseURL,
		Model:   e.Model,
		Timeout: 5 * time.Minute,
	}
	res, err := runner.Run()
	if err != nil {
		return nil, err
	}

	entry := &Entry{
		Timestamp:    Now(),
		ModelPath:    modelPath,
		ModelName:    e.Model,
		HardwareHash: e.hardwareHash(),
		Backend:      e.Backend,
		Vision:       e.Vision,
		Round:        round,
		Flags:        flagMap(flags),
		Result: BenchmarkResult{
			PromptTokens: res.PromptTokens,
			PromptTPS:    res.PromptTPS,
			GenTokens:    res.GenTokens,
			GenTPS:       res.GenTPS,
			PeakVRAMMB:   res.PeakVRAMMB,
		},
		Best: false,
	}

	return entry, nil
}

func (e *Engine) queryLLM(modelPath string, best *Entry) (*Suggestion, error) {
	prompt := buildTuningPrompt(modelPath, best, e.Caps)

	body := map[string]interface{}{
		"model":                e.Model,
		"messages":             []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens":           512,
		"temperature":          0.3,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
	}

	data, _ := json.Marshal(body)
	client := &http.Client{Timeout: 3 * time.Minute}
	resp, err := client.Post(e.BaseURL+"/v1/chat/completions", "application/json", bytes.NewReader(data))
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
	// Try to find JSON in the response.
	jsonStart := strings.Index(content, "{")
	jsonEnd := strings.LastIndex(content, "}")
	if jsonStart >= 0 && jsonEnd > jsonStart {
		content = content[jsonStart : jsonEnd+1]
	}

	var suggestion Suggestion
	if err := json.Unmarshal([]byte(content), &suggestion); err != nil {
		return nil, fmt.Errorf("failed to parse suggestion JSON: %w", err)
	}
	if suggestion.Name == "" {
		suggestion.Name = "llm-suggestion"
	}

	return &suggestion, nil
}

func buildTuningPrompt(modelPath string, best *Entry, caps *detect.Capabilities) string {
	gpuCount := 0
	totalVRAM := 0
	cpuModel := "unknown"
	if caps != nil {
		gpuCount = len(caps.GPUs)
		totalVRAM = caps.TotalVRAM()
		cpuModel = caps.CPU.Model
	}

	return fmt.Sprintf(`You are a performance optimization engineer for llama.cpp inference.

Current model: %s
Hardware: %d GPUs, %d MB total VRAM, %s CPU
Current performance: %.1f prefill tok/s, %.1f decode tok/s
Current flags: %v

Suggest llama-server command-line flag changes to improve throughput.
Return ONLY a JSON object with this exact format:
{"name":"short name","flags":{"--flag":"value"},"reasoning":"why"}

Rules:
- Only suggest flags that affect performance (batch, microbatch, threads, parallel, cache type, flash attention, mmap/mlock)
- Do not change model path, port, host, context size, mmproj, tensor split, main GPU, device, n-gpu-layers, or override-tensor
- Keep suggestions conservative (1-2 flag changes per round)
- If current performance is already good, say so with empty flags`,
		modelPath,
		gpuCount,
		totalVRAM,
		cpuModel,
		best.Result.PromptTPS,
		best.Result.GenTPS,
		best.Flags,
	)
}

func applySuggestion(baseFlags, suggested []string) []string {
	return applySuggestionWithProtection(baseFlags, suggested, nil)
}

func applySuggestionWithProtection(baseFlags, suggested []string, protected map[string]bool) []string {
	result := make([]string, len(baseFlags))
	copy(result, baseFlags)

	for i := 0; i < len(suggested); i++ {
		flag := suggested[i]
		key := canonicalFlagName(flag)
		if protected != nil && protected[key] {
			if i+1 < len(suggested) && !strings.HasPrefix(suggested[i+1], "-") {
				i++
			}
			continue
		}
		// Remove conflicting flags.
		result = removeConflicting(result, flag)
		result = append(result, flag)
		// If flag has a value, consume next element.
		if i+1 < len(suggested) && !strings.HasPrefix(suggested[i+1], "-") {
			result = append(result, suggested[i+1])
			i++
		}
	}
	return result
}

func removeConflicting(flags []string, newFlag string) []string {
	want := canonicalFlagName(newFlag)
	result := make([]string, 0, len(flags))
	for i := 0; i < len(flags); i++ {
		if canonicalFlagName(flags[i]) == want {
			if !strings.Contains(flags[i], "=") && i+1 < len(flags) && !strings.HasPrefix(flags[i+1], "-") {
				i++
			}
			continue
		}
		result = append(result, flags[i])
	}
	return result
}

// DefaultProtectedFlags returns flags AI-tune should not override. Placement is
// owned by the placement engine; AI-tune focuses on performance knobs.
func DefaultProtectedFlags() map[string]bool {
	protected := map[string]bool{}
	for _, key := range []string{
		"-m", "--host", "--port", "--ctx-size", "--mmproj", "--jinja", "--reasoning",
		"--device", "--tensor-split", "--split-mode", "-mg", "-ngl", "--n-cpu-moe", "-ot",
	} {
		protected[canonicalFlagName(key)] = true
	}
	return protected
}

// ApplyOverrides applies a JSON-object tune override set on top of an argv.
func ApplyOverrides(baseFlags []string, overrides map[string]interface{}, protected map[string]bool) []string {
	return applySuggestionWithProtection(baseFlags, flagValuesToArgs(sanitizeFlagValues(overrides, protected)), protected)
}

func sanitizeFlagValues(values map[string]interface{}, protected map[string]bool) map[string]interface{} {
	out := map[string]interface{}{}
	for key, val := range values {
		canon := canonicalFlagName(key)
		if canon == "" || !allowedTuneFlag(canon) {
			continue
		}
		if protected != nil && protected[canon] {
			continue
		}
		if b, ok := val.(bool); ok {
			if !b {
				continue
			}
			if flagNeedsValue(canon) {
				if canon == "--flash-attn" {
					out[canon] = "on"
				}
				continue
			}
			out[canon] = true
			continue
		}
		out[canon] = normalizeFlagValue(val)
	}
	return out
}

func normalizeFlagValue(val interface{}) interface{} {
	switch v := val.(type) {
	case float64:
		if v == float64(int64(v)) {
			return fmt.Sprintf("%d", int64(v))
		}
		return fmt.Sprintf("%g", v)
	case int:
		return fmt.Sprintf("%d", v)
	default:
		return v
	}
}

func allowedTuneFlag(flag string) bool {
	switch canonicalFlagName(flag) {
	case "-b", "-ub", "--threads", "--threads-batch", "--parallel",
		"--cache-type-k", "--cache-type-v", "--flash-attn",
		"--mlock", "--no-mmap", "--defrag-thold":
		return true
	default:
		return false
	}
}

func flagNeedsValue(flag string) bool {
	switch canonicalFlagName(flag) {
	case "--mlock", "--no-mmap":
		return false
	default:
		return true
	}
}

func flagValuesToArgs(values map[string]interface{}) []string {
	if len(values) == 0 {
		return nil
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, canonicalFlagName(key))
	}
	sort.Strings(keys)
	args := make([]string, 0, len(keys)*2)
	for _, key := range keys {
		val := values[key]
		if b, ok := val.(bool); ok && b {
			args = append(args, key)
			continue
		}
		if val == nil {
			continue
		}
		s := fmt.Sprint(val)
		if s == "" {
			args = append(args, key)
			continue
		}
		args = append(args, key, s)
	}
	return args
}

func flagArgsToValues(args []string) map[string]interface{} {
	values := map[string]interface{}{}
	for i := 0; i < len(args); i++ {
		key := canonicalFlagName(args[i])
		if key == "" || !strings.HasPrefix(key, "-") {
			continue
		}
		if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
			values[key] = args[i+1]
			i++
		} else {
			values[key] = true
		}
	}
	return values
}

func deterministicSuggestion(round int, baseFlags []string) *Suggestion {
	base := flagMap(baseFlags)
	currentKV := base["--cache-type-k"]
	if currentKV == "" {
		currentKV = base["--cache-type-v"]
	}
	batch := atoiDefault(base["-b"], 4096)
	ubatch := atoiDefault(base["-ub"], 512)
	isMoEOffload := base["--n-cpu-moe"] != "" || base["-ot"] != ""
	candidates := []Suggestion{}
	if currentKV == "q4_0" && !isMoEOffload {
		candidates = append(candidates, Suggestion{
			Name:       "kv-q8-quality",
			FlagValues: map[string]interface{}{"--cache-type-k": "q8_0", "--cache-type-v": "q8_0"},
			Reasoning:  "test whether q8 KV improves quality/speed while still fitting memory",
		})
	} else if currentKV == "f16" {
		candidates = append(candidates, Suggestion{
			Name:       "kv-q8-memory",
			FlagValues: map[string]interface{}{"--cache-type-k": "q8_0", "--cache-type-v": "q8_0"},
			Reasoning:  "test q8 KV for lower memory pressure with minimal quality loss",
		})
	}
	if batch > 0 && batch < 8192 {
		candidates = append(candidates, Suggestion{
			Name:       "larger-batch",
			FlagValues: map[string]interface{}{"-b": fmt.Sprintf("%d", minInt(batch*2, 8192))},
			Reasoning:  "test a larger prompt batch for better prefill throughput",
		})
	}
	if ubatch > 0 && ubatch < 2048 {
		candidates = append(candidates, Suggestion{
			Name:       "larger-ubatch",
			FlagValues: map[string]interface{}{"-ub": fmt.Sprintf("%d", minInt(ubatch*2, 2048))},
			Reasoning:  "test a larger microbatch for GPU occupancy",
		})
	}
	if ubatch > 256 {
		candidates = append(candidates, Suggestion{
			Name:       "smaller-ubatch",
			FlagValues: map[string]interface{}{"-ub": fmt.Sprintf("%d", maxInt(ubatch/2, 256))},
			Reasoning:  "test a smaller microbatch in case compute buffers are limiting decode",
		})
	}
	if len(candidates) == 0 {
		return nil
	}
	c := candidates[(round-1)%len(candidates)]
	c.Flags = flagValuesToArgs(c.FlagValues)
	return &c
}

func meaningfulImprovement(candidate, incumbent, minPct float64) bool {
	if candidate <= 0 || incumbent <= 0 {
		return candidate > incumbent
	}
	return candidate >= incumbent*(1.0+minPct/100.0)
}

func equalFlags(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func atoiDefault(s string, fallback int) int {
	var n int
	_, err := fmt.Sscanf(s, "%d", &n)
	if err != nil || n <= 0 {
		return fallback
	}
	return n
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func canonicalFlagName(flag string) string {
	if idx := strings.Index(flag, "="); idx > 0 {
		flag = flag[:idx]
	}
	switch flag {
	case "-c", "--ctx", "--ctx-size":
		return "--ctx-size"
	case "-b", "--batch-size":
		return "-b"
	case "-ub", "--ubatch-size":
		return "-ub"
	case "-t", "--threads":
		return "--threads"
	case "-tb", "--threads-batch":
		return "--threads-batch"
	case "-ngl", "--ngl", "--n-gpu-layers":
		return "-ngl"
	case "-np", "--parallel":
		return "--parallel"
	case "-fa", "--flash-attn":
		return "--flash-attn"
	case "--mg", "--main-gpu":
		return "-mg"
	case "-ot", "--override-tensor":
		return "-ot"
	case "-m", "--model":
		return "-m"
	default:
		return flag
	}
}

func flagMap(flags []string) map[string]string {
	m := make(map[string]string)
	for i := 0; i < len(flags); i++ {
		key := canonicalFlagName(flags[i])
		if rawKey, val, ok := strings.Cut(flags[i], "="); ok && strings.HasPrefix(rawKey, "-") {
			m[canonicalFlagName(rawKey)] = val
			continue
		}
		if i+1 < len(flags) && !strings.HasPrefix(flags[i+1], "-") {
			m[key] = flags[i+1]
			i++
		} else {
			m[key] = ""
		}
	}
	return m
}

func gpuNames(caps *detect.Capabilities) []string {
	var names []string
	if caps == nil {
		return names
	}
	for _, g := range caps.GPUs {
		names = append(names, g.Name)
	}
	return names
}
