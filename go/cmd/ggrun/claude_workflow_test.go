package main

import (
	"bytes"
	"encoding/json"
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

func TestClaudeWorkflowPatchInputRejectsAmbiguousSources(t *testing.T) {
	for _, input := range []map[string]interface{}{
		{"name": "saved", "script": `await agent("inline")`},
		{"scriptPath": "/tmp/workflow.js", "script": `await agent("inline")`},
		{"name": "saved", "scriptPath": "/tmp/workflow.js"},
	} {
		if err := claudeWorkflowPatchInput(input, ""); err == nil || !strings.Contains(err.Error(), "ambiguous") {
			t.Fatalf("ambiguous input %v returned %v, want an actionable rejection", input, err)
		}
	}
}

func TestClaudeWorkflowPatchedScriptPathIsIdempotent(t *testing.T) {
	first := claudeWorkflowPatchedScriptPath("/w/scripts/deep-research-wf_894b.js")
	if first != "/w/scripts/deep-research-wf_894b.ggrun.js" {
		t.Fatalf("first patch = %q", first)
	}
	// Resuming feeds the patched path back in. Before this was idempotent each
	// resume added another suffix and wrote another copy of the script.
	if again := claudeWorkflowPatchedScriptPath(first); again != first {
		t.Errorf("re-patching %q gave %q, want the same path", first, again)
	}
	// Already-accumulated paths from earlier runs must stop growing too.
	grown := "/w/scripts/deep-research-wf_894b.ggrun.ggrun.ggrun.js"
	if got := claudeWorkflowPatchedScriptPath(grown); got != grown {
		t.Errorf("re-patching %q gave %q", grown, got)
	}
	// A script whose name merely contains "ggrun" is not already patched.
	other := claudeWorkflowPatchedScriptPath("/w/scripts/ggrun-bench.js")
	if other != "/w/scripts/ggrun-bench.ggrun.js" {
		t.Errorf("unpatched name = %q", other)
	}
}

// An opaque named workflow cannot be rewritten, so allowing it would silently
// restore the private Workflow deadline. Reject it before work starts and make
// the retry shape explicit instead of losing a live run after ten minutes.
func TestClaudeWorkflowHookRejectsAnUnpatchableNamedWorkflow(t *testing.T) {
	inR, inW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	outR, outW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	errR, errW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	oldIn, oldOut, oldErr := os.Stdin, os.Stdout, os.Stderr
	os.Stdin, os.Stdout, os.Stderr = inR, outW, errW
	defer func() { os.Stdin, os.Stdout, os.Stderr = oldIn, oldOut, oldErr }()

	go func() {
		_, _ = inW.WriteString(`{"tool_name":"Workflow","tool_input":{"name":"deep-research"},"transcript_path":""}`)
		_ = inW.Close()
	}()
	cmdClaudeWorkflowHook(nil)
	_ = outW.Close()
	_ = errW.Close()

	var out, stderr bytes.Buffer
	_, _ = out.ReadFrom(outR)
	_, _ = stderr.ReadFrom(errR)
	os.Stdin, os.Stdout, os.Stderr = oldIn, oldOut, oldErr

	if !strings.Contains(out.String(), `"permissionDecision":"deny"`) {
		t.Fatalf("hook allowed an unenforceable workflow: %s", out.String())
	}
	if !strings.Contains(out.String(), "inline script or scriptPath") || !strings.Contains(stderr.String(), "could not enforce") {
		t.Fatalf("denial was not actionable: out=%q stderr=%q", out.String(), stderr.String())
	}
}

// The hook rewrites timeouts; it must not grant permission the session's mode
// would not. Modes that prompt get "ask" with the patched input, so the user
// approves the script that will actually run.
func TestClaudeWorkflowHookAsksOutsideAutonomousModes(t *testing.T) {
	for _, tc := range []struct {
		mode, want string
	}{
		{"default", "ask"},
		{"acceptEdits", "ask"},
		{"plan", "ask"},
		{"dontAsk", "ask"},
		{"", "ask"}, // older Claude Code without permission_mode
		{"auto", "allow"},
		{"bypassPermissions", "allow"},
	} {
		inR, inW, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		outR, outW, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		oldIn, oldOut := os.Stdin, os.Stdout
		os.Stdin, os.Stdout = inR, outW
		go func() {
			_, _ = fmt.Fprintf(inW, `{"tool_name":"Workflow","tool_input":{"script":"await agent('x')"},"permission_mode":%q}`, tc.mode)
			_ = inW.Close()
		}()
		cmdClaudeWorkflowHook(nil)
		_ = outW.Close()
		os.Stdin, os.Stdout = oldIn, oldOut
		var out bytes.Buffer
		_, _ = out.ReadFrom(outR)

		var got struct {
			HookSpecificOutput struct {
				PermissionDecision string                 `json:"permissionDecision"`
				UpdatedInput       map[string]interface{} `json:"updatedInput"`
			} `json:"hookSpecificOutput"`
		}
		if err := json.Unmarshal(out.Bytes(), &got); err != nil {
			t.Fatalf("mode %q: bad hook output %q: %v", tc.mode, out.String(), err)
		}
		if got.HookSpecificOutput.PermissionDecision != tc.want {
			t.Errorf("mode %q: decision %q, want %q", tc.mode, got.HookSpecificOutput.PermissionDecision, tc.want)
		}
		script, _ := got.HookSpecificOutput.UpdatedInput["script"].(string)
		if !strings.Contains(script, fmt.Sprintf("stallMs: %d", claudeNoTimeoutMS)) {
			t.Errorf("mode %q: patched input missing from %q decision: %q", tc.mode, tc.want, out.String())
		}
	}
}
