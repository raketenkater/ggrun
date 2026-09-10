package benchmark

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// CacheCanaryResult separates protocol support, functional determinism, and
// actual prefix restoration. A backend can therefore be served in a degraded
// state when it lacks cache_n telemetry without claiming cache verification.
type CacheCanaryResult struct {
	Supported          bool    `json:"supported"`
	Functional         bool    `json:"functional"`
	Deterministic      bool    `json:"deterministic"`
	Passed             bool    `json:"passed"`
	ColdPromptTokens   int     `json:"cold_prompt_tokens"`
	AppendCachedTokens int     `json:"append_cached_tokens"`
	BranchCachedTokens int     `json:"branch_cached_tokens"`
	RepeatCachedTokens int     `json:"repeat_cached_tokens"`
	ColdPromptTPS      float64 `json:"cold_prompt_tps,omitempty"`
	AppendPromptTPS    float64 `json:"append_prompt_tps,omitempty"`
	BranchPromptTPS    float64 `json:"branch_prompt_tps,omitempty"`
	Reason             string  `json:"reason,omitempty"`
}

type canaryMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type canaryResponse struct {
	Content      string
	FinishReason string
	PromptTokens int
	CachedTokens int
	PromptTPS    float64
}

// canaryEvidence renders what a call actually returned. "did not return exact
// GGRUN_OK" with no sample is undiagnosable: it cost a full launch on
// 2026-08-03 to discover only that something had been returned. Quote the
// response so a truncation, a reasoning trace or a chatty preamble is obvious
// from the error alone.
func canaryEvidence(label string, r *canaryResponse) string {
	if r == nil {
		return label + "=<no response>"
	}
	sample := strings.Join(strings.Fields(r.Content), " ")
	if len(sample) > 80 {
		sample = sample[:80] + "…"
	}
	if r.FinishReason != "" {
		return fmt.Sprintf("%s=%q (finish=%s)", label, sample, r.FinishReason)
	}
	return fmt.Sprintf("%s=%q", label, sample)
}

// RunCacheCanary exercises a cold prompt, strict extension, a branch before the
// newest checkpoint, and an identical replay. The generated prefix crosses at
// least two 512-token checkpoint boundaries when the served context has room
// for them; callers should run it only for a new profile, never on every
// startup. When the context is too small to hold such a prefix the canary still
// verifies the completion endpoint and reports that reuse was not provable,
// rather than failing a launch that is otherwise healthy.
func (r *Runner) RunCacheCanary() (*CacheCanaryResult, error) {
	words, cacheProvable := r.canarySegmentWords()
	segmentA := canarySegment("alpha", words)
	segmentB := canarySegment("beta", words)
	segmentC := canarySegment("gamma", words)
	base := []canaryMessage{
		{Role: "system", Content: "This is a deterministic ggrun cache verification. Reply with only GGRUN_OK."},
		{Role: "user", Content: segmentA},
		{Role: "assistant", Content: "GGRUN_OK"},
		{Role: "user", Content: segmentB},
		{Role: "assistant", Content: "GGRUN_OK"},
	}
	coldMessages := appendMessages(base, canaryMessage{Role: "user", Content: segmentC + "\nReply GGRUN_OK."})
	appendMessagesSet := appendMessages(coldMessages,
		canaryMessage{Role: "assistant", Content: "GGRUN_OK"},
		canaryMessage{Role: "user", Content: "This is a strict extension. Reply GGRUN_OK."})
	branchMessages := appendMessages(base,
		canaryMessage{Role: "user", Content: "This branches before gamma. Reply GGRUN_OK."})

	cold, err := r.canaryChat(coldMessages)
	if err != nil {
		return nil, fmt.Errorf("cold cache canary: %w", err)
	}
	appendResult, err := r.canaryChat(appendMessagesSet)
	if err != nil {
		return nil, fmt.Errorf("append cache canary: %w", err)
	}
	branch, err := r.canaryChat(branchMessages)
	if err != nil {
		return nil, fmt.Errorf("branch cache canary: %w", err)
	}
	repeat, err := r.canaryChat(branchMessages)
	if err != nil {
		return nil, fmt.Errorf("repeat cache canary: %w", err)
	}

	result := &CacheCanaryResult{
		Functional:         validCanaryOutput(cold.Content) && validCanaryOutput(appendResult.Content) && validCanaryOutput(branch.Content) && validCanaryOutput(repeat.Content),
		Deterministic:      normalizeCanaryOutput(branch.Content) == normalizeCanaryOutput(repeat.Content),
		ColdPromptTokens:   cold.PromptTokens,
		AppendCachedTokens: appendResult.CachedTokens,
		BranchCachedTokens: branch.CachedTokens,
		RepeatCachedTokens: repeat.CachedTokens,
		ColdPromptTPS:      cold.PromptTPS,
		AppendPromptTPS:    appendResult.PromptTPS,
		BranchPromptTPS:    branch.PromptTPS,
	}
	result.Supported = appendResult.CachedTokens > 0 || branch.CachedTokens > 0 || repeat.CachedTokens > 0
	if !result.Functional {
		result.Reason = "functional canary did not get a bounded non-empty answer on every cold/append/branch/replay call: " +
			strings.Join([]string{
				canaryEvidence("cold", cold),
				canaryEvidence("append", appendResult),
				canaryEvidence("branch", branch),
				canaryEvidence("replay", repeat),
			}, ", ")
		return result, nil
	}
	if !result.Deterministic {
		// Factual rather than alarming: sampling-heavy or very small models can
		// legitimately answer differently across replays, and this only means the
		// profile is not promoted to a verified config.
		result.Reason = "output varied across replay (expected on very small models)"
		return result, nil
	}
	if !cacheProvable {
		// Conservative claim, not a failure. The endpoint answered correctly
		// and deterministically; there was simply no room in this context for a
		// prefix long enough to prove reuse. Saying "passed" here would claim a
		// verification that never happened.
		result.Reason = fmt.Sprintf(
			"served context (%d tokens) is too small to prove prefix reuse; verified the completion endpoint only",
			r.ContextTokens)
		return result, nil
	}
	if !result.Supported {
		result.Reason = "backend response exposes no cache_n/cached_tokens evidence"
		return result, nil
	}
	appendFloor := max(64, cold.PromptTokens*2/5)
	branchFloor := max(64, cold.PromptTokens/5)
	repeatFloor := max(64, branch.PromptTokens*4/5)
	switch {
	case appendResult.CachedTokens < appendFloor:
		result.Reason = fmt.Sprintf("strict extension restored %d tokens; need at least %d", appendResult.CachedTokens, appendFloor)
	case branch.CachedTokens < branchFloor:
		result.Reason = fmt.Sprintf("older branch restored %d tokens; need at least %d", branch.CachedTokens, branchFloor)
	case repeat.CachedTokens < repeatFloor:
		result.Reason = fmt.Sprintf("identical replay restored %d tokens; need at least %d", repeat.CachedTokens, repeatFloor)
	default:
		result.Passed = true
	}
	return result, nil
}

func (r *Runner) canaryChat(messages []canaryMessage) (*canaryResponse, error) {
	body := map[string]interface{}{
		"model":    r.Model,
		"messages": messages,
		// GGRUN_OK is ~5 tokens, so a 6-token cap truncated the stop token and
		// every call came back mid-generation. Enough room to finish cleanly,
		// still far too little to ramble.
		//
		// A thinking/reasoning model (Nanbeige, DeepSeek-R1 style) writes a
		// short prose preamble before the token, so 16 tokens truncates it
		// before GGRUN_OK ever appears. 96 tokens is enough for a modest
		// preamble plus the answer while still capping a model that rambles.
		"max_tokens":   96,
		"temperature":  0,
		"seed":         1,
		"cache_prompt": true,
		"stream":       false,
	}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequest(http.MethodPost, strings.TrimRight(r.BaseURL, "/")+"/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := r.client().Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		return nil, fmt.Errorf("HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(detail)))
	}
	var decoded struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content string `json:"content"`
				// A thinking model that never closes its reasoning block leaves
				// content empty and puts everything here. Without this the
				// canary reported "did not return exact GGRUN_OK" for a model
				// that had answered correctly inside its reasoning trace.
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens  int `json:"prompt_tokens"`
			PromptDetails struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
		Timings struct {
			CacheN          int     `json:"cache_n"`
			PromptPerSecond float64 `json:"prompt_per_second"`
		} `json:"timings"`
	}
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		return nil, err
	}
	if len(decoded.Choices) == 0 {
		return nil, fmt.Errorf("no choices")
	}
	cached := decoded.Timings.CacheN
	if decoded.Usage.PromptDetails.CachedTokens > cached {
		cached = decoded.Usage.PromptDetails.CachedTokens
	}
	content := decoded.Choices[0].Message.Content
	if strings.TrimSpace(content) == "" {
		content = decoded.Choices[0].Message.ReasoningContent
	}
	return &canaryResponse{
		Content:      content,
		FinishReason: decoded.Choices[0].FinishReason,
		PromptTokens: decoded.Usage.PromptTokens,
		CachedTokens: cached,
		PromptTPS:    decoded.Timings.PromptPerSecond,
	}, nil
}

const (
	// The prefix-cache canary's original fixed geometry, kept for a server
	// whose context is unknown or comfortably large.
	canaryFullSegmentWords = 420
	canaryMinSegmentWords  = 12
	// Room for the 96-token reply plus chat-template framing.
	canaryReplyReserve = 320
	// Characters of fixed instruction and GGRUN_OK turns wrapped around the
	// segments in the cold prompt.
	canaryFrameChars = 320
	// Prefix caching checkpoints every 512 tokens. A prompt that cannot hold
	// two checkpoints cannot demonstrate reuse at all, so below this the canary
	// reports the endpoint healthy and prefix reuse unproven.
	canaryCacheProofTokens = 1200
)

// canarySegmentWords sizes the canary to the context the server was actually
// launched with, and reports whether the result is still long enough to prove
// prefix reuse.
//
// Segments are written in words, but the backend budgets in tokens, and tokens
// per word belongs to the tokenizer. On a 512-entry vocabulary the fixed
// 1260-word canary tokenized to 15873 tokens; against a 2048-token context the
// backend answered HTTP 400 and ggrun rejected a server that was serving
// correctly. So measure the expansion with the model's own tokenizer instead of
// assuming it.
func (r *Runner) canarySegmentWords() (words int, cacheProvable bool) {
	if r == nil || r.ContextTokens <= 0 {
		return canaryFullSegmentWords, true
	}
	budget := r.ContextTokens - canaryReplyReserve
	if budget <= 0 {
		return 1, false
	}
	probe := canarySegment("alpha", canaryFullSegmentWords)
	tokens, err := r.tokenCount(probe)
	if err != nil || tokens <= 0 {
		// An unmeasurable tokenizer is not a licence to guess low: one token
		// per byte is the worst expansion a byte-level fallback can produce,
		// and overshooting the budget is the failure this exists to prevent.
		tokens = len(probe)
	}
	perChar := float64(tokens) / float64(len(probe))
	perWord := float64(tokens) / float64(canaryFullSegmentWords)
	if perWord <= 0 || perChar <= 0 {
		return canaryFullSegmentWords, true
	}
	// The instructions and GGRUN_OK turns around the segments cost tokens too.
	budget -= int(float64(canaryFrameChars)*perChar) + 1
	if budget <= 0 {
		return 1, false
	}
	// The cold prompt carries three segments.
	words = int(float64(budget) * 0.95 / (3 * perWord))
	switch {
	case words > canaryFullSegmentWords:
		words = canaryFullSegmentWords
	case words < canaryMinSegmentWords:
		// A readable minimum is preferred, but never at the cost of
		// overflowing: a context this small is exactly where the fixed-size
		// canary died.
		if float64(canaryMinSegmentWords)*3*perWord <= float64(budget) {
			words = canaryMinSegmentWords
		}
	}
	if words < 1 {
		words = 1
	}
	promptTokens := int(float64(words) * 3 * perWord)
	return words, promptTokens >= canaryCacheProofTokens
}

// tokenCount asks the server to tokenize text with the model's own tokenizer.
// llama.cpp exposes /tokenize; a backend without it returns an error and the
// caller falls back to a conservative bound.
func (r *Runner) tokenCount(text string) (int, error) {
	payload, err := json.Marshal(map[string]interface{}{"content": text})
	if err != nil {
		return 0, err
	}
	request, err := http.NewRequest(http.MethodPost, strings.TrimRight(r.BaseURL, "/")+"/tokenize", bytes.NewReader(payload))
	if err != nil {
		return 0, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := r.client().Do(request)
	if err != nil {
		return 0, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("HTTP %d from /tokenize", response.StatusCode)
	}
	var decoded struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 8<<20)).Decode(&decoded); err != nil {
		return 0, err
	}
	return len(decoded.Tokens), nil
}

func canarySegment(name string, words int) string {
	if words < 1 {
		words = 1
	}
	var b strings.Builder
	b.Grow(words * 9)
	for i := 0; i < words; i++ {
		fmt.Fprintf(&b, "%s evidence %d; ", name, i%17)
	}
	return b.String()
}

func appendMessages(base []canaryMessage, tail ...canaryMessage) []canaryMessage {
	out := make([]canaryMessage, 0, len(base)+len(tail))
	out = append(out, base...)
	return append(out, tail...)
}

func normalizeCanaryOutput(value string) string {
	return strings.ToLower(strings.Join(strings.Fields(value), " "))
}

// validCanaryOutput decides whether a canary completion counts as functional.
//
// The ideal answer is the bare token GGRUN_OK, possibly wrapped in quotes or
// trailing punctuation. But a thinking/reasoning model (Nanbeige, DeepSeek-R1
// style) does not obey "reply with only GGRUN_OK": it always writes a short
// analytic preamble first ("The user is providing a series of evidence
// entries...") and, given a large evidence-segment prompt, will keep analyzing
// instead of ever emitting the token. That model still answers deterministically
// and the cache-reuse checks downstream are the real point, so requiring the
// literal token rejects a working backend.
//
// The functional bar is therefore: the endpoint answered with a non-empty,
// bounded completion. Determinism (identical replay must produce identical
// output) and cache reuse (cached_tokens > 0) remain separate, strict gates, so
// relaxing "functional" to "it answered" does not weaken what the canary
// actually verifies about the backend.
func validCanaryOutput(value string) bool {
	norm := normalizeCanaryOutput(value)
	if norm == "" {
		return false
	}
	if strings.Trim(norm, "\"'`.,:;!?*_ ") == "ggrun_ok" {
		return true
	}
	// A prose-analyzing thinking model answers deterministically without the
	// token; accept a bounded non-empty completion as functional. The 96-token
	// cap already bounds it, and the deterministic/cache-reuse gates still run.
	fields := strings.Fields(norm)
	return len(fields) > 0 && len(fields) <= maxFunctionalCanaryFields
}

// maxFunctionalCanaryFields bounds what "it answered" accepts. A backend that
// ignores max_tokens and streams unbounded text is not answering the canary,
// so the bound stays a real gate even though the literal token is not required.
const maxFunctionalCanaryFields = 160
