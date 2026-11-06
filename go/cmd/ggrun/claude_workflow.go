package main

import (
	"encoding/json"
	"fmt"
	"github.com/raketenkater/ggrun/clientkit"
	"os"
)

// JavaScript timers are signed 32-bit millisecond values. This is the largest
// safe value Claude Code accepts: about 24.8 days, effectively no inference
// deadline while still avoiding the immediate-failure overflow above it.
const claudeNoTimeoutMS = clientkit.MaxTimeoutMS

const claudeWorkflowSystemPrompt = clientkit.WorkflowSystemPrompt

type claudeWorkflowHookInput struct {
	ToolName       string                 `json:"tool_name"`
	ToolInput      map[string]interface{} `json:"tool_input"`
	TranscriptPath string                 `json:"transcript_path"`
	PermissionMode string                 `json:"permission_mode"`
}

// claudeWorkflowDecision is the PreToolUse decision for a patched Workflow
// call. The hook exists to rewrite timeouts, not to grant permission: "allow"
// skips the prompt in every mode. Only modes that already run Workflow without
// asking keep the silent approval -- bypassPermissions, and Auto, where each
// agent's own tool calls still go through the Auto reviewer. Every other mode,
// and a Claude Code too old to report its mode, gets "ask" with the patched
// input so the user approves what will actually run.
func claudeWorkflowDecision(permissionMode string) string {
	switch permissionMode {
	case "bypassPermissions", "auto":
		return "allow"
	}
	return "ask"
}

// cmdClaudeWorkflowHook is invoked by Claude Code's PreToolUse hook. Workflow
// has a private 180-second default that is independent of API_TIMEOUT_MS and
// CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS. Rewriting the generated script at the
// tool boundary makes the policy deterministic instead of trusting the model
// to remember an instruction in a long conversation.
func cmdClaudeWorkflowHook(_ []string) {
	var input claudeWorkflowHookInput
	if err := json.NewDecoder(os.Stdin).Decode(&input); err != nil {
		fmt.Fprintf(os.Stderr, "ggrun Claude Workflow hook: %v\n", err)
		fmt.Println(`{}`)
		return
	}
	if input.ToolName != "Workflow" || input.ToolInput == nil {
		fmt.Println(`{}`)
		return
	}
	if err := claudeWorkflowPatchInput(input.ToolInput, input.TranscriptPath); err != nil {
		// An opaque named workflow is exactly where the real 600-second aborts
		// survived: there is no script for this hook to rewrite, so allowing it
		// made the no-timeout promise false. Fail closed before any agents start
		// and tell Claude how to retry in a form ggrun can make deterministic.
		reason := fmt.Sprintf("ggrun could not enforce the local-model Workflow timeout policy: %v. Retry with an inline script or scriptPath so every agent() call can receive stallMs: %d", err, claudeNoTimeoutMS)
		fmt.Fprintf(os.Stderr, "[claude-code] %s\n", reason)
		output := map[string]interface{}{
			"hookSpecificOutput": map[string]interface{}{
				"hookEventName":            "PreToolUse",
				"permissionDecision":       "deny",
				"permissionDecisionReason": reason,
			},
		}
		_ = json.NewEncoder(os.Stdout).Encode(output)
		return
	}
	output := map[string]interface{}{
		"hookSpecificOutput": map[string]interface{}{
			"hookEventName":            "PreToolUse",
			"permissionDecision":       claudeWorkflowDecision(input.PermissionMode),
			"permissionDecisionReason": fmt.Sprintf("ggrun local Workflow policy: every agent() call gets stallMs: %d", claudeNoTimeoutMS),
			"updatedInput":             input.ToolInput,
		},
	}
	_ = json.NewEncoder(os.Stdout).Encode(output)
}

func claudeWorkflowPatchInput(in map[string]interface{}, transcript string) error {
	return clientkit.PatchWorkflowInput(in, transcript)
}
func claudeWorkflowPatchedScriptPath(path string) string {
	return clientkit.WorkflowPatchedScriptPath(path)
}
func claudeWorkflowNoTimeoutScript(script string) string {
	return clientkit.WorkflowNoTimeoutScript(script)
}
func claudeCodeWorkflowPromptArgs(args []string) []string { return clientkit.WorkflowPromptArgs(args) }
