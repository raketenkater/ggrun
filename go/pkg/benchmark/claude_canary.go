package benchmark

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/raketenkater/ggrun/pkg/claudeauto"
)

// RunClaudeRouterCanary verifies the actual Anthropic-compatible workload path
// used by Claude Code. The OpenAI cache canary remains separate evidence for
// backend checkpoint reuse; this call covers /v1/messages routing, admission,
// request transforms, tool-schema transport, and Anthropic response decoding.
func (r *Runner) RunClaudeRouterCanary() error {
	body := map[string]interface{}{
		"model":       r.Model,
		"max_tokens":  96, // thinking models write a preamble before the token; 8 truncates before GGRUN_OK
		"temperature": 0,
		"stream":      false,
		"system":      "This is a deterministic ggrun Claude gateway verification. Reply with only GGRUN_OK and do not call tools.",
		"messages": []map[string]interface{}{{
			"role": "user", "content": []map[string]string{{"type": "text", "text": "Reply with only GGRUN_OK."}},
		}},
		"tools": []map[string]interface{}{{
			"name": "ggrun_canary_noop", "description": "Never call this tool during the canary.",
			"input_schema": map[string]interface{}{"type": "object", "properties": map[string]interface{}{}},
		}},
	}
	answer, err := r.runBoundedClaudeCanary(body, "Claude router")
	if err != nil {
		return err
	}
	if !validCanaryOutput(answer) {
		return fmt.Errorf("Claude router functional canary returned %q instead of a bounded non-empty answer", strings.TrimSpace(answer))
	}
	return nil
}

// RunClaudeReviewerCanary exercises the classifier route Claude Code actually
// uses in Auto permission mode. It deliberately carries no tool schema: a
// security reviewer must return Claude's XML verdict as text, while presenting
// a synthetic tool can tempt a small model to call it instead of classifying.
func (r *Runner) RunClaudeReviewerCanary() error {
	body := map[string]interface{}{
		"model":       r.Model,
		"max_tokens":  96,
		"temperature": 0,
		"stream":      false,
		"system": []map[string]string{{
			"type": "text",
			"text": claudeauto.ClassifierMarker + " This is a deterministic health check. " +
				"If the action is safe, output exactly <block>no</block>; otherwise output <block>yes</block>. Do not use tools.",
		}},
		"messages": []map[string]interface{}{{
			"role": "user", "content": []map[string]string{{"type": "text", "text": "The action is a read-only health check."}},
		}},
	}
	answer, err := r.runBoundedClaudeCanary(body, "Claude reviewer route")
	if err != nil {
		return err
	}
	if strings.TrimSpace(answer) != "<block>no</block>" {
		return fmt.Errorf("Claude reviewer functional canary returned %q instead of <block>no</block>", strings.TrimSpace(answer))
	}
	return nil
}

// canaryReasoningMaxTokens is the one larger output cap a canary may retry with.
// A small reasoning model can spend the ordinary 96 tokens thinking and return
// no text at all (Qwen3.5-4B did, on a server that otherwise worked). The retry
// happens only when the backend reports that the cap, not the model, ended the
// answer, so a non-reasoning model never pays for it.
const canaryReasoningMaxTokens = 1024

func (r *Runner) runBoundedClaudeCanary(body map[string]interface{}, lane string) (string, error) {
	answer, stopReason, err := r.runClaudeMessagesCanary(body, lane)
	if err != nil || strings.TrimSpace(answer) != "" || stopReason != "max_tokens" {
		return answer, err
	}
	retry := make(map[string]interface{}, len(body))
	for key, value := range body {
		retry[key] = value
	}
	retry["max_tokens"] = canaryReasoningMaxTokens
	answer, _, err = r.runClaudeMessagesCanary(retry, lane)
	return answer, err
}

func (r *Runner) runClaudeMessagesCanary(body map[string]interface{}, lane string) (string, string, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return "", "", err
	}
	request, err := http.NewRequest(http.MethodPost, strings.TrimRight(r.BaseURL, "/")+"/v1/messages", bytes.NewReader(data))
	if err != nil {
		return "", "", err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := r.client().Do(request)
	if err != nil {
		return "", "", fmt.Errorf("%s request: %w", lane, err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		return "", "", fmt.Errorf("%s HTTP %d: %s", lane, response.StatusCode, strings.TrimSpace(string(detail)))
	}
	var decoded struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		Completion string `json:"completion"`
		StopReason string `json:"stop_reason"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&decoded); err != nil {
		return "", "", fmt.Errorf("decode %s response: %w", strings.ToLower(lane), err)
	}
	texts := make([]string, 0, len(decoded.Content)+1)
	for _, block := range decoded.Content {
		if block.Type == "text" || block.Type == "" {
			texts = append(texts, block.Text)
		}
	}
	if decoded.Completion != "" {
		texts = append(texts, decoded.Completion)
	}
	return strings.Join(texts, " "), decoded.StopReason, nil
}
