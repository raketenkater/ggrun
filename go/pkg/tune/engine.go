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
    ServerStartupTimeout time.Duration
    BackendHelp       string
    OnProgress        func(msg string)
    StartServer       func(flags []string) (cleanup func(), err error)

    // benchmarkFn, when set, replaces the live HTTP benchmark. Test seam only.
    benchmarkFn func() (*benchmark.Result, error)
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
        cleanup, err := e.startServerWithTimeout(initialFlags)
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
    protected := QualityProtectedFlags()
    plan := deterministicPlan(initialFlags, e.Backend, e.Caps, e.BackendHelp)
    triedCandidates := map[string]bool{}
    
    // Track crashed flag combinations for OOM-aware pruning
    crashedFlagSets := []map[string]interface{}{}

    planIndex := 0
    for round := 1; round <= e.Rounds; {
        if e.OnProgress != nil {
            e.OnProgress(fmt.Sprintf("AI-tune: round %d/%d (best %.1f tok/s)", round, e.Rounds, best.Result.GenTPS))
        }

        var suggestion *Suggestion
        var overrides map[string]interface{}
        var candidateFlags []string

        // Try deterministic candidates first, skipping deduped/OOM ones without consuming a round.
        for planIndex < len(plan) {
            c := plan[planIndex]
            planIndex++
            o := sanitizeFlagValues(c.FlagValues, protected)
            o = guardRiskyMoEOverrides(o, initialFlags)
            key := suggestionKey(o)
            if triedCandidates[key] {
                if e.OnProgress != nil {
                    e.OnProgress(fmt.Sprintf("AI-tune: candidate %q duplicates an earlier flag set", c.Name))
                }
                continue
            }
            if isSkippedDueToOOM(o, crashedFlagSets, initialFlags) {
                if e.OnProgress != nil {
                    e.OnProgress(fmt.Sprintf("AI-tune: skipping candidate %q due to predicted OOM", c.Name))
                }
                continue
            }
            f := ApplyOverrides(initialFlags, o, protected)
            if equalFlags(initialFlags, f) {
                if e.OnProgress != nil {
                    e.OnProgress(fmt.Sprintf("AI-tune: candidate %q made no effective flag changes", c.Name))
                }
                continue
            }
            triedCandidates[key] = true
            suggestion = &c
            overrides = o
            candidateFlags = f
            break
        }

        if suggestion == nil {
            // Deterministic candidates exhausted; fall back to LLM.
            if err := startBaseline(); err != nil {
                return best, fmt.Errorf("restart baseline for round %d query: %w", round, err)
            }
            var err error
            var crashedNames []string
            for _, c := range crashedFlagSets {
                for k, v := range c {
                    if v != nil && v != true {
                        crashedNames = append(crashedNames, fmt.Sprintf("%s=%v", k, v))
                    } else {
                        crashedNames = append(crashedNames, k)
                    }
                }
            }
            suggestion, err = e.queryLLM(modelPath, best, crashedNames)
            if err != nil && e.OnProgress != nil {
                e.OnProgress(fmt.Sprintf("AI-tune: LLM query failed in round %d: %v; using deterministic candidate", round, err))
            }
            if err != nil || suggestion == nil || len(sanitizeFlagValues(suggestion.FlagValues, protected)) == 0 {
                suggestion = deterministicSuggestionFor(round, initialFlags, e.Backend, e.Caps, e.BackendHelp)
            }
            if suggestion == nil {
                if e.OnProgress != nil {
                    e.OnProgress("AI-tune: no safe candidates left, stopping early")
                }
                break
            }
            overrides = sanitizeFlagValues(suggestion.FlagValues, protected)
            overrides = guardRiskyMoEOverrides(overrides, initialFlags)
            candidateKey := suggestionKey(overrides)
            if triedCandidates[candidateKey] {
                if e.OnProgress != nil {
                    e.OnProgress(fmt.Sprintf("AI-tune: candidate %q duplicates an earlier flag set", suggestion.Name))
                }
                round++
                continue
            }
            if isSkippedDueToOOM(overrides, crashedFlagSets, initialFlags) {
                if e.OnProgress != nil {
                    e.OnProgress(fmt.Sprintf("AI-tune: skipping candidate %q due to predicted OOM", suggestion.Name))
                }
                round++
                continue
            }
            candidateFlags = ApplyOverrides(initialFlags, overrides, protected)
            if equalFlags(initialFlags, candidateFlags) {
                if e.OnProgress != nil {
                    e.OnProgress(fmt.Sprintf("AI-tune: candidate %q made no effective flag changes", suggestion.Name))
                }
                round++
                continue
            }
            triedCandidates[candidateKey] = true
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
            // Record the flags that caused the crash to prune future candidates
            crashedFlagSets = append(crashedFlagSets, overrides)
            if e.OnProgress != nil {
                e.OnProgress(fmt.Sprintf("AI-tune: candidate benchmark failed: %v", err))
            }
        } else {
            candidate.Name = suggestion.Name
            candidate.OverrideFlags = overrides
            candidate.Status = "ok"
            candidate.Backend = e.Backend
            candidate.Vision = e.Vision
            // VRAM penalty: if peak VRAM exceeds 80% of total, penalize TPS
            // to avoid configurations that are fast but dangerously close to OOM.
            totalVRAM := 0
            if e.Caps != nil {
                totalVRAM = e.Caps.TotalVRAM()
                if totalVRAM > 0 && candidate.Result.PeakVRAMMB > totalVRAM*80/100 {
                    vramPenalty := 1.0 - float64(candidate.Result.PeakVRAMMB-totalVRAM*80/100)/float64(totalVRAM)*0.5
                    if vramPenalty < 0.5 {
                        vramPenalty = 0.5
                    }
                    candidate.Result.GenTPS = candidate.Result.GenTPS * vramPenalty
                }
            }
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
        round++
    }

    // Confirmation pass. Each candidate is judged on a single benchmark sample, so
    // a noisy or cold-GPU reading can crown a config that is not actually faster
    // than the baseline. Before caching a non-baseline winner, re-measure the
    // baseline and the winner back-to-back (same warm GPU state) and keep the
    // winner only if it still beats the baseline by the margin. Otherwise fall
    // back to baseline, so AI-tune never caches — and the launcher never silently
    // applies — a config that is slower than the default on every future launch.
    if best != nil && baseline != nil && best != baseline && len(best.OverrideFlags) > 0 {
        if e.OnProgress != nil {
            e.OnProgress(fmt.Sprintf("AI-tune: confirming %s against baseline...", best.Name))
        }
        stopBaseline()
        bestFlags := ApplyOverrides(initialFlags, best.OverrideFlags, protected)
        confBase, errBase := e.round(e.Rounds+1, modelPath, initialFlags)
        confBest, errBest := e.round(e.Rounds+2, modelPath, bestFlags)
        switch {
        case errBase != nil || errBest != nil:
            if e.OnProgress != nil {
                e.OnProgress(fmt.Sprintf("AI-tune: confirmation benchmark failed; keeping baseline instead of %s", best.Name))
            }
            best.Best = false
            baseline.Best = true
            best = baseline
        case meaningfulImprovement(confBest.Result.GenTPS, confBase.Result.GenTPS, minImprovementPct):
            // Reproduced under equal conditions: trust the confirmed numbers.
            baseline.Result = confBase.Result
            best.Result = confBest.Result
            if e.OnProgress != nil {
                e.OnProgress(fmt.Sprintf("AI-tune: confirmed %.1f tok/s vs baseline %.1f", best.Result.GenTPS, baseline.Result.GenTPS))
            }
        default:
            if e.OnProgress != nil {
                e.OnProgress(fmt.Sprintf("AI-tune: %s did not hold up on re-measure (%.1f vs baseline %.1f); keeping baseline", best.Name, confBest.Result.GenTPS, confBase.Result.GenTPS))
            }
            best.Best = false
            baseline.Best = true
            best = baseline
        }
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
    cleanup, err := e.startServerWithTimeout(flags)
    if err != nil {
        return nil, err
    }
    if cleanup != nil {
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
    var (
        res *benchmark.Result
        err error
    )
    if e.benchmarkFn != nil {
        res, err = e.benchmarkFn()
    } else {
        runner := &benchmark.Runner{
            BaseURL: e.BaseURL,
            Model:   e.Model,
            Timeout: e.benchmarkTimeout(),
        }
        res, err = runner.Run()
    }
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

func (e *Engine) startServerWithTimeout(flags []string) (cleanup func(), err error) {
    if e.StartServer == nil {
        return nil, nil
    }
    type result struct {
        cleanup func()
        err     error
    }
    done := make(chan result, 1)
    go func() {
        c, err := e.StartServer(flags)
        done <- result{c, err}
    }()
    timeout := 5 * time.Minute
    if e.ServerStartupTimeout > 0 {
        timeout = e.ServerStartupTimeout
    }
    select {
    case res := <-done:
        return res.cleanup, res.err
    case <-time.After(timeout):
        return nil, fmt.Errorf("server startup timed out after %v", timeout)
    }
}

func (e *Engine) queryLLM(modelPath string, best *Entry, crashedNames []string) (*Suggestion, error) {
    prompt := buildTuningPrompt(modelPath, best, e.Caps, crashedNames)

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

func buildTuningPrompt(modelPath string, best *Entry, caps *detect.Capabilities, crashedNames []string) string {
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
- Recent candidates that crashed with OOM: %v\n
Suggest llama-server command-line flag changes to improve throughput.
Return ONLY a JSON object with this exact format:
{"name":"short name","flags":{"--flag":"value"},"reasoning":"why"}

Rules:
- Only suggest flags that affect performance: batch, microbatch, threads, flash attention, mmap/mlock, defrag threshold, or ik_llama MoE runtime flags
- Do not change model path, port, host, context size, parallel, mmproj, tensor split, main GPU, device, n-gpu-layers, or override-tensor. Context size and parallel sequence slots are user-owned.
- You may suggest cache type changes if they enable a better placement (e.g., GPU KV with a smaller cache type to fit within VRAM).
- Keep suggestions conservative (1-2 flag changes per round)
- Use false for a currently-present boolean flag when you want to test removing it
- Previous crashes indicate VRAM pressure. Try reducing batch size, using smaller KV cache types, or offloading more to CPU.
- If current performance is already good, say so with empty flags`,
        modelPath,
        gpuCount,
        totalVRAM,
        cpuModel,
        best.Result.PromptTPS,
        best.Result.GenTPS,
        best.Flags,
        crashedNames,
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

// QualityProtectedFlags extends the placement-protected set with flags that
// change the model's OUTPUT QUALITY or effective context — knobs AI-tune must
// never set on the user's behalf, in any path: the autonomous loop, a cached
// tune file, or a community-shared config. KV cache quantization
// (--cache-type-k/-v) is a user-owned quality/memory tradeoff, and --parallel
// divides --ctx-size across sequence slots, shrinking the usable per-request
// context. The user can still set any of these directly on the command line;
// the tune machinery just never overrides them.
func QualityProtectedFlags() map[string]bool {
    protected := DefaultProtectedFlags()
    protected[canonicalFlagName("--parallel")] = true
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
		"--kv-offload", "--no-kv-offload",
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

    // Combination candidates: test synergistic flag sets that isolated 1-hop
    // changes miss. These are placed first so they are tested before the
    // individual axis candidates.
    if flagPresent(base, "--no-kv-offload") && strings.Contains(backendHelp, "turbo") {
        add("gpu-kv-turbo4",
            map[string]interface{}{"--no-kv-offload": false, "--cache-type-k": "turbo4", "--cache-type-v": "turbo3"},
            "test GPU KV combined with turbo4 cache type to reclaim GPU speed on constrained VRAM")
        if !flagPresent(base, "--no-mmap") {
            add("gpu-kv-no-mmap",
                map[string]interface{}{"--no-kv-offload": false, "--no-mmap": true},
                "test GPU KV combined with no-mmap to free page-cache overhead for MoE weights")
        }
    }
    if !flagPresent(base, "--no-kv-offload") && strings.Contains(backendHelp, "turbo") {
        if !flagPresent(base, "--no-mmap") {
            add("turbo4-no-mmap",
                map[string]interface{}{"--cache-type-k": "turbo4", "--cache-type-v": "turbo3", "--no-mmap": true},
                "test turbo4 cache type combined with no-mmap for maximum VRAM headroom")
        }
    }

    // VRAM Descent Ladder: if GPU KV is the goal but it OOMs on the baseline,
    // generate progressively smaller compensating configurations.
    if flagPresent(base, "--no-kv-offload") && strings.Contains(backendHelp, "turbo") {
        if batch > 1024 {
            add("gpu-kv-turbo4-small-batch",
                map[string]interface{}{"--no-kv-offload": false, "--cache-type-k": "turbo4", "--cache-type-v": "turbo3", "-b": fmt.Sprintf("%d", maxInt(batch/2, 1024))},
                "test GPU KV + turbo4 with halved batch to fit constrained VRAM")
        }
        if ubatch > 256 {
            add("gpu-kv-turbo4-small-ubatch",
                map[string]interface{}{"--no-kv-offload": false, "--cache-type-k": "turbo4", "--cache-type-v": "turbo3", "-ub": fmt.Sprintf("%d", maxInt(ubatch/2, 256))},
                "test GPU KV + turbo4 with halved ubatch to fit constrained VRAM")
        }
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
    if isIK && !isMoEOffload && base["--defrag-thold"] != "0.5" {
        add("dense-defrag-0.5",
            map[string]interface{}{"--defrag-thold": "0.5"},
            "retest the historically faster dense-model KV defrag threshold")
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

    // KV Type Grid Search
    add("kv-type-f16", map[string]interface{}{"--cache-type-k": "f16", "--cache-type-v": "f16"}, "test f16 KV cache for max quality")
    add("kv-type-q8_0", map[string]interface{}{"--cache-type-k": "q8_0", "--cache-type-v": "q8_0"}, "test q8_0 KV cache")
    add("kv-type-asym-f16-q8", map[string]interface{}{"--cache-type-k": "f16", "--cache-type-v": "q8_0"}, "test asymmetric f16 keys, q8_0 values")
    // Turboquant Turbo Cache Types Grid Search
    // Only emit turbo cache candidates when the backend actually supports turbo
    if strings.Contains(backendHelp, "turbo") {
        add("kv-type-turbo4-turbo3", map[string]interface{}{"--cache-type-k": "turbo4", "--cache-type-v": "turbo3"}, "test turboquant turbo4/turbo3 KV cache")
        add("kv-type-turbo3-turbo3", map[string]interface{}{"--cache-type-k": "turbo3", "--cache-type-v": "turbo3"}, "test turboquant turbo3 KV cache")
    }

    // Dense Model Tensor Split Grid Search
    if !isMoEOffload && strings.TrimSpace(base["--tensor-split"]) != "" {
        add("tensor-split-50-50", map[string]interface{}{"--tensor-split": "0.50,0.50"}, "test even 50/50 tensor split")
        add("tensor-split-60-40", map[string]interface{}{"--tensor-split": "0.60,0.40"}, "test 60/40 tensor split")
    }

    // MoE Mmap Grid Search
    if isMoEOffload {
        if !flagPresent(base, "--no-mmap") {
            add("moe-no-mmap", map[string]interface{}{"--no-mmap": true}, "test disabling mmap for MoE")
        } else {
            add("moe-mmap", map[string]interface{}{"--no-mmap": false}, "test enabling mmap for MoE")
        }
        if base["--n-cpu-moe"] != "" {
            currentMoe, _ := strconv.Atoi(base["--n-cpu-moe"])
            if currentMoe > 0 {
                add("moe-more-cpu", map[string]interface{}{"--n-cpu-moe": fmt.Sprintf("%d", currentMoe+2)}, "test moving 2 more MoE layers to CPU")
            }
        }
    }


    // Native MoE Grid Search (Test removing -ot to let llama.cpp handle it)
    if isMoEOffload && base["-ot"] != "" {
        add("native-moe-gpu-kv", map[string]interface{}{"-ot": "", "--no-kv-offload": false, "--n-cpu-moe": base["--n-cpu-moe"]}, "test native dynamic MoE offload with GPU KV cache")
    }

    // KV Cache Placement Grid Search
    if flagPresent(base, "--no-kv-offload") {
        add("gpu-kv-cache", map[string]interface{}{"--no-kv-offload": false}, "test moving KV cache back to GPU for tg speed")
    } else {
        add("cpu-kv-cache", map[string]interface{}{"--no-kv-offload": true}, "test moving KV cache to CPU to free VRAM")
    }

    // Dynamic Ubatch Grid Search
    ub, _ := strconv.Atoi(base["-ub"])
    if ub > 0 {
        add("larger-ubatch-2x", map[string]interface{}{"-ub": fmt.Sprintf("%d", minInt(ub*2, 4096))}, "test 2x ubatch for pp speed")
        add("larger-ubatch-4x", map[string]interface{}{"-ub": fmt.Sprintf("%d", minInt(ub*4, 4096))}, "test 4x ubatch for pp speed")
    }

    return candidates
}

func guardRiskyMoEOverrides(overrides map[string]interface{}, baseFlags []string) map[string]interface{} {
    // Native offload (-ncmoe) dynamically manages VRAM, so it is safe to test 
    // batch sizes and KV cache types without the strict guards. 
    return overrides
}

// isStrictVRAMSuperset returns true if the new overrides will definitely crash
// based on a previous crash, because the resulting full candidate flags add
// VRAM pressure without introducing any known VRAM-saving flags.
func isStrictVRAMSuperset(newOverrides, crashedSet map[string]interface{}, baseFlags []string) bool {
	newFlags := ApplyOverrides(baseFlags, newOverrides, QualityProtectedFlags())
	crashedFlags := ApplyOverrides(baseFlags, crashedSet, QualityProtectedFlags())

	newMap := flagMap(newFlags)
	crashedMap := flagMap(crashedFlags)

	// Does the new candidate contain all flags of the crashed candidate?
	for k, v := range crashedMap {
		if newMap[k] != v {
			return false // Not a superset
		}
	}

	// If it is a superset, check if it adds any known VRAM-saving flags.
	vramSavers := map[string]bool{
		"--cache-type-k": true, // changing cache type might shrink it
		"--cache-type-v": true,
		"-b":             true, // changing batch size might shrink it
		"-ub":            true,
		"--n-cpu-moe":    true, // moving more layers to CPU saves VRAM
	}
	for k := range newMap {
		if _, inCrashed := crashedMap[k]; inCrashed {
			continue
		}
		if _, isSaver := vramSavers[k]; isSaver {
			// It changed a cache/batch/moe size. We can't be sure it will crash.
			return false
		}
	}

	// It contains all the crashed flags, and only added non-saving flags. It will crash.
	return true
}

func isSkippedDueToOOM(overrides map[string]interface{}, crashedFlagSets []map[string]interface{}, baseFlags []string) bool {
	for _, crashedSet := range crashedFlagSets {
		if isStrictVRAMSuperset(overrides, crashedSet, baseFlags) {
			return true
		}
	}
	return false
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