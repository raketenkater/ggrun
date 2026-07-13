package main

import (
	"fmt"
	"strings"
	"testing"
)

func TestClaudeWorkflowNoTimeoutScript(t *testing.T) {
	script := "const first = await agent(`prompt, with ${value}, commas`, {\n" +
		"  label: 'first',\n  phase: 'Research',\n})\n" +
		"const second = await agent(\"plain prompt\")\n" +
		"const third = await agent(prompt, options)\n" +
		"object.agent('not the Workflow primitive')\n" +
		"const text = \"agent('also not syntax')\"\n"

	got := claudeWorkflowNoTimeoutScript(script)
	wantStall := fmt.Sprintf("stallMs: %d", claudeNoTimeoutMS)
	if count := strings.Count(got, wantStall); count != 3 {
		t.Fatalf("added stall policy %d times, want 3:\n%s", count, got)
	}
	if !strings.Contains(got, "Object.assign({}, ({\n  label: 'first'") {
		t.Fatalf("object options were not safely wrapped:\n%s", got)
	}
	if !strings.Contains(got, `agent("plain prompt", { stallMs: 2147483647 })`) {
		t.Fatalf("one-argument call was not extended:\n%s", got)
	}
	if !strings.Contains(got, "Object.assign({}, (options), { stallMs: 2147483647 })") {
		t.Fatalf("expression options were not safely wrapped:\n%s", got)
	}

	withoutTrailingComma := `await agent(prompt, { label: "worker", phase: "Test" })`
	got = claudeWorkflowNoTimeoutScript(withoutTrailingComma)
	if !strings.Contains(got, `Object.assign({}, ({ label: "worker", phase: "Test" }), { stallMs: 2147483647 })`) {
		t.Fatalf("non-trailing-comma options became invalid: %s", got)
	}
}

func TestClaudeWorkflowNoTimeoutScriptIsIdempotentAtMaximum(t *testing.T) {
	script := `await agent(prompt, { label: "x", stallMs: 2147483647 })`
	if got := claudeWorkflowNoTimeoutScript(script); got != script {
		t.Fatalf("maximum stall policy should not be duplicated: %s", got)
	}
}

func TestClaudeCodeWorkflowPromptArgsPreservesUserPrompt(t *testing.T) {
	args := claudeCodeWorkflowPromptArgs([]string{"--append-system-prompt", "user policy", "--model", "local"})
	if len(args) != 4 || !strings.Contains(args[1], "user policy") || !strings.Contains(args[1], "stallMs: 2147483647") {
		t.Fatalf("user and ggrun prompts were not merged: %v", args)
	}

	args = claudeCodeWorkflowPromptArgs([]string{"--model", "local"})
	if len(args) < 2 || args[0] != "--append-system-prompt" || !strings.Contains(args[1], "stallMs: 2147483647") {
		t.Fatalf("ggrun Workflow prompt missing: %v", args)
	}
}
