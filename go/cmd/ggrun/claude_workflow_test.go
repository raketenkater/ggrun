package main

import (
	"fmt"
	"os"
	"path/filepath"
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

func TestClaudeWorkflowPatchInputScriptPath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "research.js")
	if err := os.WriteFile(path, []byte(`await agent("research")`), 0o600); err != nil {
		t.Fatal(err)
	}
	input := map[string]interface{}{"scriptPath": path, "resumeFromRunId": "wf-1"}
	if err := claudeWorkflowPatchInput(input, ""); err != nil {
		t.Fatal(err)
	}
	patchedPath := input["scriptPath"].(string)
	patched, err := os.ReadFile(patchedPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(patched), "stallMs: 2147483647") || input["resumeFromRunId"] != "wf-1" {
		t.Fatalf("scriptPath policy not applied: input=%v script=%s", input, patched)
	}
}

func TestClaudeWorkflowPatchInputNamedWorkflow(t *testing.T) {
	project := t.TempDir()
	session := filepath.Join(project, "session-id")
	scripts := filepath.Join(session, "workflows", "scripts")
	if err := os.MkdirAll(scripts, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(scripts, "deep-research-wf_test.js")
	script := "export const meta = { name: 'deep-research' }\nawait agent('search')\n"
	if err := os.WriteFile(path, []byte(script), 0o600); err != nil {
		t.Fatal(err)
	}
	input := map[string]interface{}{"name": "deep-research", "args": "question"}
	transcript := filepath.Join(project, "session-id.jsonl")
	if err := claudeWorkflowPatchInput(input, transcript); err != nil {
		t.Fatal(err)
	}
	if _, exists := input["name"]; exists {
		t.Fatalf("named workflow was not converted: %v", input)
	}
	patched, err := os.ReadFile(input["scriptPath"].(string))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(patched), "stallMs: 2147483647") || input["args"] != "question" {
		t.Fatalf("named workflow policy not applied: input=%v script=%s", input, patched)
	}
}

func TestClaudeWorkflowPatchInputRejectsUnresolvedName(t *testing.T) {
	input := map[string]interface{}{"name": "never-seen"}
	if err := claudeWorkflowPatchInput(input, filepath.Join(t.TempDir(), "session.jsonl")); err == nil {
		t.Fatal("unresolved named workflow must fail before the 180-second default can run")
	}
}
