package clientkit

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const WorkflowSystemPrompt = `You are using a slow local model through ggrun. Never abort a model request or agent merely because it is quiet while processing a prompt. For every Workflow tool script, set stallMs: 2147483647 in every agent() options object. This is required even for foreground agents and retries. Keep large protocols and source material in files and pass their paths to agents instead of embedding duplicate copies in prompts.`

// PatchWorkflowInput applies the timeout policy to every Workflow input
// shape Claude Code exposes. Named workflows are materialized outside the tool
// input, so resolve a prior materialization and convert the call to scriptPath.
func PatchWorkflowInput(toolInput map[string]interface{}, transcriptPath string) error {
	// Workflow accepts three mutually exclusive script sources. Claude Code has
	// historically applied precedence when more than one is present, which can
	// silently run a saved workflow instead of the inline script the model just
	// generated. Refuse ambiguity before any agents start.
	var sources []string
	for _, key := range []string{"script", "scriptPath", "name"} {
		if value, ok := toolInput[key].(string); ok && strings.TrimSpace(value) != "" {
			sources = append(sources, key)
		}
	}
	if len(sources) > 1 {
		return fmt.Errorf("ggrun Workflow policy: script source is ambiguous (%s); provide exactly one of script, scriptPath, or name", strings.Join(sources, ", "))
	}
	if script, ok := toolInput["script"].(string); ok {
		toolInput["script"] = WorkflowNoTimeoutScript(script)
		return nil
	}

	path, _ := toolInput["scriptPath"].(string)
	if path == "" {
		name, _ := toolInput["name"].(string)
		if name == "" {
			return fmt.Errorf("ggrun Workflow policy: missing script, scriptPath, or name")
		}
		var err error
		path, err = claudeWorkflowResolveNamedScript(transcriptPath, name)
		if err != nil {
			return err
		}
		delete(toolInput, "name")
	}

	patchedPath, err := claudeWorkflowPatchedScriptFile(path)
	if err != nil {
		return fmt.Errorf("ggrun Workflow policy: %w", err)
	}
	toolInput["scriptPath"] = patchedPath
	return nil
}

// WorkflowPatchedScriptPath names the patched copy of a workflow script.
//
// Resuming a run feeds the previous scriptPath straight back in, and that path
// is already a patched one, so appending unconditionally grew a suffix per
// resume: one session on this machine accumulated
// deep-research-....ggrun.ggrun.ggrun.ggrun.ggrun.js, a fresh copy of the script
// each time. Patching is idempotent -- the content is regenerated from the same
// source either way -- so an already-patched path patches back onto itself.
func WorkflowPatchedScriptPath(path string) string {
	ext := filepath.Ext(path)
	base := strings.TrimSuffix(path, ext)
	if strings.HasSuffix(base, ".ggrun") {
		return path
	}
	return base + ".ggrun" + ext
}

func claudeWorkflowPatchedScriptFile(path string) (string, error) {
	script, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", path, err)
	}
	patched := WorkflowNoTimeoutScript(string(script))
	patchedPath := WorkflowPatchedScriptPath(path)
	if err := os.WriteFile(patchedPath, []byte(patched), 0o600); err != nil {
		return "", fmt.Errorf("write %s: %w", patchedPath, err)
	}
	// The generated file is the executable artifact handed to Workflow. Read it
	// back before approval so a short/partial materialization cannot be accepted
	// silently and discovered only after a long model run.
	readback, err := os.ReadFile(patchedPath)
	if err != nil {
		return "", fmt.Errorf("verify %s: %w", patchedPath, err)
	}
	if !bytes.Equal(readback, []byte(patched)) {
		return "", fmt.Errorf("verify %s: written script differs from patched input", patchedPath)
	}
	return patchedPath, nil
}

func claudeWorkflowResolveNamedScript(transcriptPath, name string) (string, error) {
	if transcriptPath == "" {
		return "", fmt.Errorf("ggrun Workflow policy: cannot safely run named workflow %q without transcript path; invoke it with script or scriptPath", name)
	}
	projectDir := filepath.Dir(transcriptPath)
	patterns := []string{
		filepath.Join(strings.TrimSuffix(transcriptPath, filepath.Ext(transcriptPath)), "workflows", "scripts", "*.js"),
		filepath.Join(projectDir, "*", "workflows", "scripts", "*.js"),
	}
	var candidates []string
	for _, pattern := range patterns {
		matches, _ := filepath.Glob(pattern)
		candidates = append(candidates, matches...)
	}
	var newest string
	var newestMod int64
	for _, candidate := range candidates {
		if strings.Contains(candidate, ".ggrun.js") {
			continue
		}
		body, err := os.ReadFile(candidate)
		if err != nil || (!strings.Contains(string(body), "name: '"+name+"'") && !strings.Contains(string(body), `name: "`+name+`"`)) {
			continue
		}
		info, err := os.Stat(candidate)
		if err == nil && (newest == "" || info.ModTime().UnixNano() > newestMod) {
			newest, newestMod = candidate, info.ModTime().UnixNano()
		}
	}
	if newest == "" {
		return "", fmt.Errorf("ggrun Workflow policy: named workflow %q has no materialized script to patch; invoke it once as inline script or pass scriptPath", name)
	}
	return newest, nil
}

type workflowScriptEdit struct {
	start, end int
	text       string
}

// WorkflowNoTimeoutScript adds the maximum safe stallMs to every direct
// agent(...) call. It understands strings, template literals, comments and
// nested delimiters, so commas in a long agent prompt cannot fool the rewrite.
func WorkflowNoTimeoutScript(script string) string {
	var edits []workflowScriptEdit
	for i := 0; i < len(script); {
		switch {
		case script[i] == '\'' || script[i] == '"':
			i = skipJSQuoted(script, i, script[i])
		case script[i] == '`':
			i = skipJSTemplate(script, i)
		case i+1 < len(script) && script[i:i+2] == "//":
			i = skipJSLineComment(script, i)
		case i+1 < len(script) && script[i:i+2] == "/*":
			i = skipJSBlockComment(script, i)
		case isJSIdentStart(script[i]):
			start := i
			i++
			for i < len(script) && isJSIdentPart(script[i]) {
				i++
			}
			if script[start:i] != "agent" || previousNonSpace(script, start) == '.' {
				continue
			}
			open := skipJSSpace(script, i)
			if open >= len(script) || script[open] != '(' {
				continue
			}
			close, commas, ok := scanJSCall(script, open)
			if !ok {
				continue
			}
			if len(commas) == 0 {
				edits = append(edits, workflowScriptEdit{start: close, end: close, text: fmt.Sprintf(", { stallMs: %d }", MaxTimeoutMS)})
				i = close + 1
				continue
			}

			secondStart := skipJSSpace(script, commas[0]+1)
			secondEnd := close
			if len(commas) > 1 {
				secondEnd = commas[1]
			}
			trimmedEnd := trimJSSpaceBack(script, secondEnd)
			if secondStart >= trimmedEnd {
				edits = append(edits, workflowScriptEdit{start: secondStart, end: secondStart, text: fmt.Sprintf("{ stallMs: %d }", MaxTimeoutMS)})
			} else {
				expr := strings.TrimSpace(script[secondStart:trimmedEnd])
				compact := strings.NewReplacer(" ", "", "\t", "", "\r", "", "\n", "").Replace(expr)
				if !strings.Contains(compact, fmt.Sprintf("stallMs:%d", MaxTimeoutMS)) {
					// Wrapping is safer than inserting a property into an object literal:
					// comments, spreads and an existing lower stallMs all remain valid,
					// while the final Object.assign operand deterministically wins.
					replacement := fmt.Sprintf("Object.assign({}, (%s), { stallMs: %d })", expr, MaxTimeoutMS)
					edits = append(edits, workflowScriptEdit{start: secondStart, end: trimmedEnd, text: replacement})
				}
			}
			i = close + 1
		default:
			i++
		}
	}
	if len(edits) == 0 {
		return script
	}
	sort.Slice(edits, func(i, j int) bool { return edits[i].start > edits[j].start })
	for _, edit := range edits {
		script = script[:edit.start] + edit.text + script[edit.end:]
	}
	return script
}

func scanJSCall(s string, open int) (int, []int, bool) {
	paren, brace, bracket := 1, 0, 0
	var commas []int
	for i := open + 1; i < len(s); {
		switch {
		case s[i] == '\'' || s[i] == '"':
			i = skipJSQuoted(s, i, s[i])
		case s[i] == '`':
			i = skipJSTemplate(s, i)
		case i+1 < len(s) && s[i:i+2] == "//":
			i = skipJSLineComment(s, i)
		case i+1 < len(s) && s[i:i+2] == "/*":
			i = skipJSBlockComment(s, i)
		default:
			switch s[i] {
			case '(':
				paren++
			case ')':
				paren--
				if paren == 0 {
					return i, commas, true
				}
			case '{':
				brace++
			case '}':
				if brace > 0 {
					brace--
				}
			case '[':
				bracket++
			case ']':
				if bracket > 0 {
					bracket--
				}
			case ',':
				if paren == 1 && brace == 0 && bracket == 0 {
					commas = append(commas, i)
				}
			}
			i++
		}
	}
	return 0, nil, false
}

func skipJSQuoted(s string, i int, quote byte) int {
	for i++; i < len(s); i++ {
		if s[i] == '\\' {
			i++
			continue
		}
		if s[i] == quote {
			return i + 1
		}
	}
	return len(s)
}

func skipJSTemplate(s string, i int) int {
	// A Workflow prompt can contain ${...}; skipping the complete template is
	// sufficient because commas and agent text inside it are not call syntax.
	for i++; i < len(s); i++ {
		if s[i] == '\\' {
			i++
			continue
		}
		if s[i] == '`' {
			return i + 1
		}
	}
	return len(s)
}

func skipJSLineComment(s string, i int) int {
	if nl := strings.IndexByte(s[i+2:], '\n'); nl >= 0 {
		return i + 2 + nl + 1
	}
	return len(s)
}

func skipJSBlockComment(s string, i int) int {
	if end := strings.Index(s[i+2:], "*/"); end >= 0 {
		return i + 2 + end + 2
	}
	return len(s)
}

func skipJSSpace(s string, i int) int {
	for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n') {
		i++
	}
	return i
}

func trimJSSpaceBack(s string, i int) int {
	for i > 0 && (s[i-1] == ' ' || s[i-1] == '\t' || s[i-1] == '\r' || s[i-1] == '\n') {
		i--
	}
	return i
}

func previousNonSpace(s string, i int) byte {
	for i--; i >= 0; i-- {
		if s[i] != ' ' && s[i] != '\t' && s[i] != '\r' && s[i] != '\n' {
			return s[i]
		}
	}
	return 0
}

func isJSIdentStart(b byte) bool {
	return b == '_' || b == '$' || b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z'
}

func isJSIdentPart(b byte) bool {
	return isJSIdentStart(b) || b >= '0' && b <= '9'
}

// WorkflowPromptArgs is a belt-and-suspenders instruction for Claude
// versions or custom --settings configurations that suppress hooks.
func WorkflowPromptArgs(args []string) []string {
	out := append([]string(nil), args...)
	for i, arg := range out {
		if arg == "--append-system-prompt" && i+1 < len(out) {
			out[i+1] += "\n\n" + WorkflowSystemPrompt
			return out
		}
		if strings.HasPrefix(arg, "--append-system-prompt=") {
			out[i] = arg + "\n\n" + WorkflowSystemPrompt
			return out
		}
	}
	return append([]string{"--append-system-prompt", WorkflowSystemPrompt}, out...)
}
