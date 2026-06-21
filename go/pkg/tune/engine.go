package tune

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/benchmark"
	"github.com/raketenkater/ggrun/pkg/detect"
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
	BenchmarkTimeout  time.Duration
	BackendHelp       string
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
		if e.StartServer == nil || baselineCleanup != nil {
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
	e.saveTuneProgress(modelPath, baseline, best, entries, minImprovementPct, false)
	protected := DefaultProtectedFlags()
	plan := deterministicPlan(initialFlags, e.Backend, e.Caps, e.BackendHelp)
	triedCandidates := map[string]bool{}

	for round := 1; round <= e.Rounds; round++ {
		if e.OnProgress != nil {
			e.OnProgress(fmt.Sprintf("AI-tune: round %d/%d (best %.1f tok/s)", round, e.Rounds, best.Result.GenTPS))
		}

		var suggestion *Suggestion
		if round <= len(plan) {
			c := plan[round-1]
			suggestion = &c
		} else {
			// Query only when deterministic candidates are exhausted. Large MoE models
			// are expensive to reload, so avoid keeping the baseline alive between
			// deterministic candidate rounds.
			if err := startBaseline(); err != nil {
				return best, fmt.Errorf("restart baseline for round %d query: %w", round, err)
			}
			var err error
			suggestion, err = e.queryLLM(modelPath, best)
			if err != nil && e.OnProgress != nil {
				e.OnProgress(fmt.Sprintf("AI-tune: LLM query failed in round %d: %v; using deterministic candidate", round, err))
			}
			if err != nil || suggestion == nil || len(sanitizeFlagValues(suggestion.FlagValues, protected)) == 0 {
				suggestion = deterministicSuggestionFor(round, initialFlags, e.Backend, e.Caps, e.BackendHelp)
			}
		}
		if suggestion == nil {
			if e.OnProgress != nil {
				e.OnProgress(fmt.Sprintf("AI-tune: no safe candidate left for round %d", round))
			}
			continue
		}

		overrides := sanitizeFlagValues(suggestion.FlagValues, protected)
		overrides = guardRiskyMoEOverrides(overrides, initialFlags)
		candidateKey := suggestionKey(overrides)
		if triedCandidates[candidateKey] {
			if e.OnProgress != nil {
				e.OnProgress(fmt.Sprintf("AI-tune: candidate %q duplicates an earlier flag set", suggestion.Name))
			}
			continue
		}
		triedCandidates[candidateKey] = true
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
			crashed := Entry{
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
			}
			e.addCache(&crashed)
			entries = append(entries, crashed)
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
		e.saveTuneProgress(modelPath, baseline, best, entries, minImprovementPct, false)
	}

	if e.OnProgress != nil {
		e.OnProgress(fmt.Sprintf("AI-tune: done. Best result: %.1f tok/s", best.Result.GenTPS))
	}
	if e.Cache != nil {
		path, err := e.saveTuneProgress(modelPath, baseline, best, entries, minImprovementPct, true)
		if err != nil && e.OnProgress != nil {
			e.OnProgress(fmt.Sprintf("AI-tune: failed to save tune cache: %v", err))
		} else if path != "" && e.OnProgress != nil {
			e.OnProgress(fmt.Sprintf("AI-tune: saved tune cache %s", path))
		}
	}

	return best, nil
}

func (e *Engine) saveTuneProgress(modelPath string, baseline, best *Entry, entries []Entry, minImprovementPct float64, final bool) (string, error) {
	if e.Cache == nil || baseline == nil {
		return "", nil
	}
	path, err := e.Cache.SaveTuneFile(modelPath, baseline, best, e.Rounds, e.Backend, e.Vision, minImprovementPct, gpuNames(e.Caps), entries, final)
	if err != nil {
		return path, err
	}
	if !final && e.OnProgress != nil {
		if _, statErr := os.Stat(path); statErr == nil {
			e.OnProgress(fmt.Sprintf("AI-tune: progress saved %s", path))
		}
	}
	return path, nil
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
		Timeout: e.benchmarkTimeout(),
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
			PromptTokens:    res.PromptTokens,
			PromptTPS:       res.PromptTPS,
			GenTokens:       res.GenTokens,
			GenTPS:          res.GenTPS,
			PeakVRAMMB:      res.PeakVRAMMB,
			DraftTokens:     res.DraftTokens,
			DraftAccepted:   res.DraftAccepted,
			DraftAcceptRate: res.DraftAcceptRate,
		},
		Best: false,
	}

	return entry, nil
}

func (e *Engine) benchmarkTimeout() time.Duration {
	if e.BenchmarkTimeout > 0 {
		return e.BenchmarkTimeout
	}
	return 5 * time.Minute
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
- Only suggest flags that affect performance: batch, microbatch, threads, parallel, cache type, flash attention, mmap/mlock, defrag threshold, or ik_llama MoE runtime flags
- Do not change model path, port, host, context size, mmproj, tensor split, main GPU, device, n-gpu-layers, or override-tensor
- Keep suggestions conservative (1-2 flag changes per round)
- Use false for a currently-present boolean flag when you want to test removing it
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
		if flagHasSeparateValue(suggested, i) {
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
			if flagHasSeparateValue(flags, i) {
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
	values := sanitizeFlagValues(overrides, protected)
	if len(values) == 0 {
		return baseFlags
	}
	result := make([]string, len(baseFlags))
	copy(result, baseFlags)
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, canonicalFlagName(key))
	}
	sort.Strings(keys)
	for _, key := range keys {
		val := values[key]
		if b, ok := val.(bool); ok && !b {
			result = removeConflicting(result, key)
			continue
		}
		result = applySuggestionWithProtection(result, flagValuesToArgs(map[string]interface{}{key: val}), protected)
	}
	return result
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
			if flagNeedsValue(canon) {
				if canon == "--flash-attn" {
					if b {
						out[canon] = "on"
					} else {
						out[canon] = "off"
					}
					continue
				}
				if !b {
					out[canon] = false
				}
				continue
			}
			out[canon] = b
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
		"--mlock", "--no-mmap", "--defrag-thold", "--ctx-checkpoints",
		"--run-time-repack", "-khad", "-ger", "-mqkv", "-muge",
		"--defer-experts", "--cont-batching",
		"--spec-type", "--spec-ngram-size-n", "--spec-ngram-size-m", "--spec-ngram-min-hits",
		"--spec-ngram-map-k-size-n", "--spec-ngram-map-k-size-m", "--spec-ngram-map-k-min-hits",
		"--spec-ngram-map-k4v-size-n", "--spec-ngram-map-k4v-size-m", "--spec-ngram-map-k4v-min-hits",
		"--spec-ngram-mod-n-match", "--spec-ngram-mod-n-min", "--spec-ngram-mod-n-max",
		"--spec-draft-n-max", "--spec-draft-n-min", "--draft-max", "--draft-min",
		"--spec-autotune", "--multi-token-prediction":
		return true
	default:
		return false
	}
}

func flagNeedsValue(flag string) bool {
	switch canonicalFlagName(flag) {
	case "--mlock", "--no-mmap", "--jinja", "--no-jinja",
		"--kv-offload", "--no-kv-offload", "--no-context-shift",
		"--run-time-repack", "-khad", "-ger", "-mqkv", "-muge",
		"--defer-experts", "--cont-batching", "--spec-autotune",
		"--multi-token-prediction":
		return false
	default:
		return true
	}
}

func flagHasSeparateValue(args []string, i int) bool {
	if i+1 >= len(args) || strings.Contains(args[i], "=") {
		return false
	}
	key := canonicalFlagName(args[i])
	if !flagNeedsValue(key) {
		return false
	}
	if strings.HasPrefix(args[i+1], "-") && !flagValueMayStartWithDash(key) {
		return false
	}
	return true
}

func flagValueMayStartWithDash(flag string) bool {
	switch canonicalFlagName(flag) {
	case "--defrag-thold":
		return true
	default:
		return false
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
		if flagHasSeparateValue(args, i) {
			values[key] = args[i+1]
			i++
		} else {
			values[key] = true
		}
	}
	return values
}

func deterministicSuggestion(round int, baseFlags []string) *Suggestion {
	return deterministicSuggestionFor(round, baseFlags, "", nil, "")
}

func deterministicSuggestionFor(round int, baseFlags []string, backend string, caps *detect.Capabilities, backendHelp string) *Suggestion {
	candidates := deterministicPlan(baseFlags, backend, caps, backendHelp)
	if len(candidates) == 0 {
		return nil
	}
	c := candidates[(round-1)%len(candidates)]
	c.Flags = flagValuesToArgs(c.FlagValues)
	return &c
}

func deterministicPlan(baseFlags []string, backend string, caps *detect.Capabilities, backendHelp string) []Suggestion {
	base := flagMap(baseFlags)
	currentKV := base["--cache-type-k"]
	if currentKV == "" {
		currentKV = base["--cache-type-v"]
	}
	batch := atoiDefault(base["-b"], 4096)
	ubatch := atoiDefault(base["-ub"], 512)
	isMoEOffload := isMoEOffloadFlags(base)
	isIK := backendIsIK(backend) || hasIKRuntimeFlags(base)
	currentSpec := strings.TrimSpace(base["--spec-type"])
	specAutoTune := isIK || tuneBackendHelpSupports(backendHelp, "spec-autotune")
	candidates := []Suggestion{}
	seen := map[string]bool{}
	add := func(name string, values map[string]interface{}, reasoning string) {
		if !specAutoTune {
			delete(values, "--spec-autotune")
		}
		values = sanitizeFlagValues(values, nil)
		if len(values) == 0 {
			return
		}
		key := suggestionKey(values)
		if seen[key] {
			return
		}
		seen[key] = true
		candidates = append(candidates, Suggestion{
			Name:       name,
			FlagValues: values,
			Flags:      flagValuesToArgs(values),
			Reasoning:  reasoning,
		})
	}

	if currentSpec == "ngram-mod" {
		add("spec-ngram-mod-lower-depth",
			map[string]interface{}{"--spec-ngram-mod-n-min": "16", "--spec-ngram-mod-n-max": "48", "--spec-autotune": true},
			"test lower ngram-mod draft depth for latency-sensitive prompts")
	} else if strings.Contains(currentSpec, "ngram") {
		add("spec-disable-autotune",
			map[string]interface{}{"--spec-autotune": false},
			"test whether backend speculative autotune overhead hurts this workload")
	}

	if currentKV == "q4_0" && !isMoEOffload {
		add("kv-q8-quality",
			map[string]interface{}{"--cache-type-k": "q8_0", "--cache-type-v": "q8_0"},
			"test whether q8 KV improves quality/speed while still fitting memory")
	} else if currentKV == "f16" {
		add("kv-q8-memory",
			map[string]interface{}{"--cache-type-k": "q8_0", "--cache-type-v": "q8_0"},
			"test q8 KV for lower memory pressure with minimal quality loss")
	}
	if !isMoEOffload && batch > 0 && batch < 8192 {
		add("larger-batch",
			map[string]interface{}{"-b": fmt.Sprintf("%d", minInt(batch*2, 8192))},
			"test a larger prompt batch for better prefill throughput")
	}
	if !isMoEOffload && batch > 0 && batch < 16384 {
		add("max-prefill-batch",
			map[string]interface{}{"-b": "16384"},
			"test an aggressive prefill batch on dense models")
	}
	if isMoEOffload && batch > 1024 {
		add("smaller-moe-batch",
			map[string]interface{}{"-b": fmt.Sprintf("%d", maxInt(batch/2, 1024))},
			"test lower batch pressure for CPU/GPU expert handoff without changing context")
		if batch > 1536 {
			add("moe-batch-1536",
				map[string]interface{}{"-b": "1536"},
				"test a middle MoE batch size to reduce expert handoff pressure while preserving prefill")
		}
	}
	if !isMoEOffload && ubatch > 0 && ubatch < 2048 {
		limit := 2048
		if ubatch < limit {
			add("larger-ubatch",
				map[string]interface{}{"-ub": fmt.Sprintf("%d", minInt(ubatch*2, limit))},
				"test a larger microbatch for GPU occupancy")
		}
	}
	if isMoEOffload && ubatch > 384 {
		add("moe-ubatch-384",
			map[string]interface{}{"-ub": "384"},
			"test a moderate MoE microbatch to smooth CPU/GPU expert traffic")
	}
	if ubatch > 256 {
		add("smaller-ubatch",
			map[string]interface{}{"-ub": fmt.Sprintf("%d", maxInt(ubatch/2, 256))},
			"test a smaller microbatch in case compute buffers are limiting decode")
	}

	if caps != nil {
		if caps.CPU.Cores > 0 && atoiDefault(base["--threads"], 0) != caps.CPU.Cores {
			add("threads-physical",
				map[string]interface{}{"--threads": fmt.Sprintf("%d", caps.CPU.Cores)},
				"pin generation threads to detected physical CPU cores")
		}
		if caps.CPU.Threads > caps.CPU.Cores && atoiDefault(base["--threads-batch"], 0) != caps.CPU.Threads {
			add("threads-batch-logical",
				map[string]interface{}{"--threads-batch": fmt.Sprintf("%d", caps.CPU.Threads)},
				"test logical CPU threads for prompt processing")
		}
	}

	if isMoEOffload {
		if !flagPresent(base, "--defer-experts") && tuneMoEFlagSupported(base, backendHelp, isIK, "--defer-experts") {
			add("moe-defer-experts",
				map[string]interface{}{"--defer-experts": true},
				"test deferred expert residency to reduce startup and host memory pressure")
		}
		if flagPresent(base, "--no-mmap") {
			add("moe-mmap-pagecache",
				map[string]interface{}{"--no-mmap": false},
				"test mmap page-cache expert loading for large MoE stability")
		}
		if !flagPresent(base, "--cont-batching") && tuneMoEFlagSupported(base, backendHelp, isIK, "--cont-batching") {
			add("moe-cont-batching",
				map[string]interface{}{"--cont-batching": true},
				"test continuous batching for MoE serving throughput without changing context")
		}
	}

	if isMoEOffload && isIK {
		add("moe-disable-repack",
			map[string]interface{}{"--run-time-repack": false},
			"test whether runtime repack overhead hurts this MoE placement")
		add("moe-disable-khad",
			map[string]interface{}{"-khad": false},
			"test whether K-cache hadamard helps or hurts this MoE workload")
		add("moe-defrag-off",
			map[string]interface{}{"--defrag-thold": "-1"},
			"test disabling KV defrag for steadier decode throughput")
		add("moe-no-muge",
			map[string]interface{}{"-muge": false},
			"test disabling merged up/gate experts for this quant/backend combination")
		add("moe-no-ger",
			map[string]interface{}{"-ger": false},
			"test disabling grouped expert routing for this model")
		add("moe-no-mqkv",
			map[string]interface{}{"-mqkv": false},
			"test disabling merged QKV on this backend")
		add("moe-checkpoints-8",
			map[string]interface{}{"--ctx-checkpoints": "8"},
			"test fewer context checkpoints to reduce per-slot overhead")
		add("moe-checkpoints-0",
			map[string]interface{}{"--ctx-checkpoints": "0"},
			"test disabling context checkpoints when cache RAM is not helping")
	}

	return candidates
}

func guardRiskyMoEOverrides(overrides map[string]interface{}, baseFlags []string) map[string]interface{} {
	if len(overrides) == 0 {
		return overrides
	}
	base := flagMap(baseFlags)
	if !isMoEOffloadFlags(base) {
		return overrides
	}

	out := map[string]interface{}{}
	currentBatch := atoiDefault(base["-b"], 2048)
	currentUBatch := atoiDefault(base["-ub"], 512)
	currentK := base["--cache-type-k"]
	currentV := base["--cache-type-v"]
	for key, val := range overrides {
		canon := canonicalFlagName(key)
		switch {
		case canon == "--parallel" || canon == "--ctx-size":
			continue
		case canon == "--spec-type" || strings.HasPrefix(canon, "--spec-") || canon == "--multi-token-prediction":
			continue
		case canon == "--flash-attn" && strings.EqualFold(fmt.Sprint(val), "off"):
			continue
		case canon == "-ub":
			if atoiFlagValue(val, currentUBatch) > currentUBatch {
				continue
			}
		case canon == "-b":
			next := atoiFlagValue(val, currentBatch)
			if next > maxInt(currentBatch, 4096) {
				continue
			}
		case canon == "--cache-type-k":
			if currentK != "" && fmt.Sprint(val) != currentK {
				continue
			}
		case canon == "--cache-type-v":
			if currentV != "" && fmt.Sprint(val) != currentV {
				continue
			}
		}
		out[canon] = val
	}
	return out
}

func atoiFlagValue(val interface{}, fallback int) int {
	switch v := val.(type) {
	case int:
		if v > 0 {
			return v
		}
	case float64:
		if v > 0 {
			return int(v)
		}
	case string:
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			return n
		}
	default:
		if n, err := strconv.Atoi(strings.TrimSpace(fmt.Sprint(v))); err == nil && n > 0 {
			return n
		}
	}
	return fallback
}

func suggestionKey(values map[string]interface{}) string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, canonicalFlagName(key))
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, key := range keys {
		b.WriteString(key)
		b.WriteByte('=')
		b.WriteString(fmt.Sprint(values[key]))
		b.WriteByte(';')
	}
	return b.String()
}

func isMoEOffloadFlags(flags map[string]string) bool {
	return flags["--n-cpu-moe"] != "" || flags["-ot"] != ""
}

func flagPresent(flags map[string]string, flag string) bool {
	_, ok := flags[canonicalFlagName(flag)]
	return ok
}

func tuneMoEFlagSupported(base map[string]string, backendHelp string, isIK bool, flags ...string) bool {
	if isIK {
		return true
	}
	for _, flag := range flags {
		if flagPresent(base, flag) || tuneBackendHelpSupports(backendHelp, strings.TrimLeft(flag, "-")) {
			return true
		}
	}
	return false
}

func tuneBackendHelpSupports(help, token string) bool {
	if help == "" || token == "" {
		return false
	}
	return strings.Contains(strings.ToLower(help), strings.ToLower(token))
}

func backendIsIK(backend string) bool {
	backend = strings.ToLower(strings.TrimSpace(backend))
	return backend == "ik" || backend == "ik_llama" || strings.Contains(backend, "ik_llama")
}

func hasIKRuntimeFlags(flags map[string]string) bool {
	return flags["--run-time-repack"] != "" ||
		flags["-khad"] != "" ||
		flags["-ger"] != "" ||
		flags["-mqkv"] != "" ||
		flags["-muge"] != ""
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
	case "-dt", "--defrag-thold":
		return "--defrag-thold"
	case "-ctxcp", "--ctx-checkpoints", "--swa-checkpoints":
		return "--ctx-checkpoints"
	case "-khad", "--k-cache-hadamard":
		return "-khad"
	case "-ger", "--grouped-expert-routing":
		return "-ger"
	case "-mqkv", "--merge-qkv":
		return "-mqkv"
	case "-muge", "--merge-up-gate-experts":
		return "-muge"
	case "-cb", "--cont-batching":
		return "--cont-batching"
	case "-mtp", "--multi-token-prediction":
		return "--multi-token-prediction"
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
		if flagHasSeparateValue(flags, i) {
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
