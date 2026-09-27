package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"unicode/utf16"
)

var pinnedIdentityArgs = []string{
	`C:\Users\u\ggrun\.bin\llama-server.exe`, "-m", `C:\m\Qwen.gguf`, "-ngl", "999",
	"--tensor-split", "0,1",
	"-ot", `blk\.(0|1)\.ffn_((gate|up)_(ch|)exps|(gate_inp|gate)_shexp).*=CUDA0,exps=CPU`,
	"--alias", "it's", "-lv", "4",
}

const issue74OT = `blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13)\.ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b).*=CUDA0,exps=CPU`

// powerShellRows are the arguments every PowerShell rendering must round-trip.
var powerShellRows = []string{
	`C:\Program Files\ggrun\.bin\llama-server.exe`, `C:\Users\Jürgen\m.gguf`,
	"a b", "it's", `say "hi"`, "a|b", "(x)", "x)", "^caret", "%PATH%", "50%",
	"a&b", "$env:X", "`tick", `{"mcpServers":{"d":{"command":"uvx"}}}`,
	"@file", `~\x`, "--%", "-foo.bar", "-test.run=X", "-a:b", "#c", "x;y",
	"<in", "a,b", "left’right", "‘q‛", "", "--", "-", "“dq”", "–dash", "a=b,c",
	"999", "-1", "1e-5", "0x10", "+1", "1kb", "1.50", "007", "127.0.0.1",
	issue74OT,
}

// pinnedArgvHash mirrors the persisted argv hash: hex of the first 8 bytes of
// sha256(formatCommand(args)).
func pinnedArgvHash(args []string) string {
	sum := sha256.Sum256([]byte(formatCommand(args)))
	return hex.EncodeToString(sum[:8])
}

func TestFormatCommandIdentityIsPinned(t *testing.T) {
	want := `'C:\Users\u\ggrun\.bin\llama-server.exe' -m 'C:\m\Qwen.gguf' -ngl 999 --tensor-split 0,1 -ot 'blk\.(0|1)\.ffn_((gate|up)_(ch|)exps|(gate_inp|gate)_shexp).*=CUDA0,exps=CPU' --alias 'it'\''s' -lv 4`
	if got := formatCommand(pinnedIdentityArgs); got != want {
		t.Fatalf("formatCommand identity moved:\n got %s\nwant %s", got, want)
	}
	if got := pinnedArgvHash(pinnedIdentityArgs); got != "659fdadd0e1da0ad" {
		t.Fatalf("argv hash moved: %s", got)
	}
}

func TestDisplayCommandPOSIXMatchesIdentity(t *testing.T) {
	fixtures := [][]string{
		pinnedIdentityArgs,
		{"llama-server", "-m", "model.gguf", "-b", "128", "-ub", "64", "--tensor-split", "0,1"},
		{""},
		{"a b", "@x", "%y"},
	}
	for _, a := range fixtures {
		for _, goos := range []string{"linux", "darwin"} {
			if got := displayCommandFor(goos, "", nil, a); got != formatCommand(a) {
				t.Fatalf("%s: %q != %q", goos, got, formatCommand(a))
			}
		}
		// Git Bash/MSYS on Windows keeps the POSIX rendering.
		if got := displayCommandFor("windows", "MINGW64", nil, a); got != formatCommand(a) {
			t.Fatalf("windows+MSYSTEM: %q != %q", got, formatCommand(a))
		}
		if got, want := displayCommandFor("linux", "", []string{"CUDA_VISIBLE_DEVICES=0,1"}, a), "CUDA_VISIBLE_DEVICES=0,1 "+formatCommand(a); got != want {
			t.Fatalf("env prefix: %q != %q", got, want)
		}
		if got, want := displayCommandFor("linux", "", []string{"CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=1"}, a), "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 "+formatCommand(a); got != want {
			t.Fatalf("free-token prefix: %q != %q", got, want)
		}
	}
}

func isPowerShellBareSafe(r rune, dash bool) bool {
	if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' {
		return true
	}
	return !dash && strings.ContainsRune(`_.+=:/\`, r)
}

// parsePowerShellNativeLine accepts only the PowerShell subset the renderer may
// emit, and is at least as strict as powerShellArg about bare words.
func parsePowerShellNativeLine(t *testing.T, line string) (env, argv []string) {
	t.Helper()
	rs := []rune(line)
	i := 0
	literal := func() string {
		if i >= len(rs) || !isPowerShellSingleQuote(rs[i]) {
			t.Fatalf("expected a quoted string at %d in %q", i, line)
		}
		i++
		var b strings.Builder
		for {
			if i >= len(rs) {
				t.Fatalf("unterminated string in %q", line)
			}
			r := rs[i]
			i++
			if !isPowerShellSingleQuote(r) {
				b.WriteRune(r)
				continue
			}
			if i < len(rs) && rs[i] == r {
				b.WriteRune(r)
				i++
				continue
			}
			return b.String()
		}
	}
	for strings.HasPrefix(string(rs[i:]), "$env:") {
		i += len("$env:")
		start := i
		for i < len(rs) && (rs[i] == '_' || rs[i] >= 'A' && rs[i] <= 'Z' || rs[i] >= 'a' && rs[i] <= 'z' || rs[i] >= '0' && rs[i] <= '9') {
			i++
		}
		if i == start || i >= len(rs) || rs[i] != '=' {
			t.Fatalf("bad env assignment in %q", line)
		}
		name := string(rs[start:i])
		i++
		env = append(env, name+"="+literal())
		if !strings.HasPrefix(string(rs[i:]), "; ") {
			t.Fatalf("env assignment not followed by '; ' in %q", line)
		}
		i += 2
	}
	if !strings.HasPrefix(string(rs[i:]), "& ") {
		t.Fatalf("missing call operator in %q", line)
	}
	i += 2
	for {
		if strings.HasPrefix(string(rs[i:]), "@(") {
			// The only array form: the stop-parsing marker as a value.
			i += 2
			if v := literal(); v != "--%" || i >= len(rs) || rs[i] != ')' {
				t.Fatalf("unexpected array argument in %q", line)
			}
			i++
			argv = append(argv, "--%")
		} else if i < len(rs) && isPowerShellSingleQuote(rs[i]) {
			v := literal()
			if v == "--%" {
				t.Fatalf("quoted --%% is dropped by PowerShell in %q", line)
			}
			argv = append(argv, v)
		} else {
			start := i
			for i < len(rs) && rs[i] != ' ' {
				i++
			}
			word := string(rs[start:i])
			if word == "" || word == "-" || word == "--" {
				t.Fatalf("unsafe bare word %q in %q", word, line)
			}
			dash := word[0] == '-'
			for _, r := range word {
				if !isPowerShellBareSafe(r, dash) {
					t.Fatalf("unsafe bare word %q in %q", word, line)
				}
			}
			argv = append(argv, word)
		}
		if i == len(rs) {
			return env, argv
		}
		if rs[i] != ' ' {
			t.Fatalf("expected a space at %d in %q", i, line)
		}
		i++
	}
}

func TestDisplayCommandPowerShellRoundTrip(t *testing.T) {
	exe := `C:\bin\llama-server.exe`
	for _, r := range powerShellRows {
		in := []string{exe, "-x", r}
		line := displayCommandFor("windows", "", nil, in)
		if !strings.HasPrefix(line, "& ") {
			t.Fatalf("%q: no call operator: %s", r, line)
		}
		if _, got := parsePowerShellNativeLine(t, line); !reflect.DeepEqual(got, in) {
			t.Fatalf("%q: round trip %q != %q (line %s)", r, got, in, line)
		}
	}
	all := append([]string{exe}, powerShellRows...)
	env := []string{"A=it's ~ C:\\x y", "B=‘q’"}
	gotEnv, got := parsePowerShellNativeLine(t, displayCommandFor("windows", "", env, all))
	if !reflect.DeepEqual(got, all) || !reflect.DeepEqual(gotEnv, env) {
		t.Fatalf("all rows: env %q argv %q", gotEnv, got)
	}
}

func TestDisplayCommandPowerShellRendersIssue74(t *testing.T) {
	args := []string{
		`C:\Users\xxx\ggrun\.bin\llama-server.exe`,
		"-m", `C:\Users\xxx\.lmstudio\models\AtomicChat\Qwen3.8-Flash-Next-GGUF\Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64-00001-of-00028.gguf`,
		"--host", "127.0.0.1", "--port", "8081", "--ctx-size", "262144",
		"--tensor-split", "0,1", "-ngl", "999", "-ot", issue74OT,
		"--chat-template-file", `C:\Users\xxx\ggrun\.cache\chat-templates\qwen3.8-27b.jinja`, "-lv", "4",
	}
	out := displayCommandFor("windows", "", nil, args)
	for _, want := range []string{
		` -ot 'blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13)\.ffn_(`,
		" --host 127.0.0.1 ", " -ngl 999 ", " --tensor-split '0,1' ",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in %s", want, out)
		}
	}
	if !strings.HasPrefix(out, `& C:\Users\xxx\ggrun\.bin\llama-server.exe -m `) {
		t.Fatalf("unexpected prefix: %s", out)
	}
	if strings.Contains(out, `'\''`) {
		t.Fatalf("POSIX escape leaked into PowerShell output: %s", out)
	}
	if !regexp.MustCompile(`(?m)^\[launch\] .* -m `).MatchString("[launch] " + out) {
		t.Fatalf("verify-installed-serving.py regex no longer matches: %s", out)
	}
}

func TestDisplayCommandPowerShellEnvPrefix(t *testing.T) {
	got := displayCommandFor("windows", "", []string{"CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=0,1"}, []string{`C:\x\llama-server.exe`, "-m", "m.gguf"})
	if want := `$env:CUDA_DEVICE_ORDER='PCI_BUS_ID'; $env:CUDA_VISIBLE_DEVICES='0,1'; & C:\x\llama-server.exe -m m.gguf`; got != want {
		t.Fatalf("got %s\nwant %s", got, want)
	}
	got = displayCommandFor("windows", "", nil, []string{"claude", "--mcp-config", `{"a":"b"}`})
	if want := `& claude --mcp-config '{"a":"b"}'`; got != want {
		t.Fatalf("got %s\nwant %s", got, want)
	}
}

func packageSourceFiles(t *testing.T) (*token.FileSet, map[string]*ast.File) {
	t.Helper()
	paths, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	files := map[string]*ast.File{}
	for _, p := range paths {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, p, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		files[p] = f
	}
	return fset, files
}

var fmtPrintFuncs = map[string]bool{
	"Print": true, "Printf": true, "Println": true,
	"Fprint": true, "Fprintf": true, "Fprintln": true,
}

func isFmtPrintCall(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || !fmtPrintFuncs[sel.Sel.Name] {
		return false
	}
	x, ok := sel.X.(*ast.Ident)
	return ok && x.Name == "fmt"
}

func calledName(n ast.Node) string {
	call, ok := n.(*ast.CallExpr)
	if !ok {
		return ""
	}
	if id, ok := call.Fun.(*ast.Ident); ok {
		return id.Name
	}
	return ""
}

// Printed commands must go through the display renderer (issue #74). Only
// the backend log line, which records the argv identity, and the claude recipe,
// whose inline JSON Windows PowerShell 5.1 would mangle, stay on formatCommand.
func TestPrintedCommandsUseDisplayRenderer(t *testing.T) {
	const backendLogFormat = "[ggrun] launch: %s\n"
	const recipeFunc = "printClaudeCodeRecipe"
	fset, files := packageSourceFiles(t)
	var backendLogSeen, recipeSeen int
	for _, f := range files {
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok || !isFmtPrintCall(call) {
					return true
				}
				usesIdentity := false
				for _, arg := range call.Args {
					ast.Inspect(arg, func(m ast.Node) bool {
						if calledName(m) == "formatCommand" {
							usesIdentity = true
						}
						return true
					})
				}
				if !usesIdentity {
					return true
				}
				for _, arg := range call.Args {
					if lit, ok := arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {
						if s, err := strconv.Unquote(lit.Value); err == nil && s == backendLogFormat {
							backendLogSeen++
							return true
						}
					}
				}
				if fn.Name.Name == recipeFunc {
					recipeSeen++
					return true
				}
				t.Errorf("%s: printed command uses formatCommand; use displayCommand", fset.Position(call.Pos()))
				return true
			})
		}
	}
	if backendLogSeen != 1 || recipeSeen == 0 {
		t.Fatalf("allowlist is stale: backend log sites %d, recipe sites %d", backendLogSeen, recipeSeen)
	}
}

// The display renderer differs per OS, so it must never feed a comparison, a
// map key, a stored argv identity or a hash. Outside its own file it may only
// appear as a direct argument of a fmt print call.
func TestDisplayCommandNeverUsedForIdentity(t *testing.T) {
	fset, files := packageSourceFiles(t)
	display := map[string]bool{"displayCommand": true, "displayCommandWithEnv": true, "displayCommandFor": true}
	uses := 0
	for p, f := range files {
		if p == "command_display.go" {
			continue
		}
		allowed := map[ast.Node]bool{}
		ast.Inspect(f, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok && isFmtPrintCall(call) {
				for _, arg := range call.Args {
					allowed[arg] = true
				}
			}
			return true
		})
		ast.Inspect(f, func(n ast.Node) bool {
			if name := calledName(n); display[name] {
				uses++
				if !allowed[n] {
					t.Errorf("%s: %s used outside a print call", fset.Position(n.Pos()), name)
				}
			}
			return true
		})
	}
	if uses == 0 {
		t.Fatal("found no display renderer uses; the guard is not looking at the package")
	}
}

// TestHelperEchoArgv is the child process of the Windows replay test: it
// writes the arguments after the sentinel as JSON to a file, not to stdout,
// because PowerShell re-decodes captured native stdout with the OEM code page.
func TestHelperEchoArgv(t *testing.T) {
	if os.Getenv("GGRUN_TEST_ECHO_ARGV") != "1" {
		return
	}
	for i, a := range os.Args {
		if a == echoArgvSentinel {
			data, err := json.Marshal(os.Args[i+1:])
			if err != nil {
				os.Exit(3)
			}
			if err := os.WriteFile(os.Getenv("GGRUN_TEST_ECHO_ARGV_OUT"), data, 0o644); err != nil {
				os.Exit(4)
			}
			os.Exit(0)
		}
	}
	os.Exit(2)
}

const echoArgvSentinel = "ggrun-echo-argv-begin"

func TestDisplayCommandPowerShellReplayOnWindows(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("replays the rendered line in a real PowerShell; Windows only")
	}
	t.Setenv("MSYSTEM", "")
	found := false
	for _, shell := range []string{"powershell", "pwsh"} {
		path, err := exec.LookPath(shell)
		if err != nil {
			continue
		}
		found = true
		t.Run(shell, func(t *testing.T) {
			rows := powerShellRows
			if !powerShellPassesEmbeddedQuotes(t, path) {
				// Legacy native argument passing (Windows PowerShell 5.1,
				// pwsh < 7.3) strips embedded double quotes and drops empty
				// arguments however they are quoted: a documented limitation.
				rows = nil
				for _, r := range powerShellRows {
					if r != "" && !strings.Contains(r, `"`) {
						rows = append(rows, r)
					}
				}
			}
			dir := filepath.Join(t.TempDir(), "ggrun ~ echo")
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			out := filepath.Join(dir, "argv.json")
			line := displayCommandWithEnv(
				[]string{"GGRUN_TEST_ECHO_ARGV=1", "GGRUN_TEST_ECHO_ARGV_OUT=" + out},
				append([]string{os.Args[0], "-test.run=^TestHelperEchoArgv$", echoArgvSentinel}, rows...),
			)
			output, err := exec.Command(path, "-NoProfile", "-NonInteractive", "-EncodedCommand",
				encodePowerShellCommand(line+"; exit $LASTEXITCODE")).CombinedOutput()
			if err != nil {
				t.Fatalf("%s exited with %v\nline: %s\noutput: %s", shell, err, line, output)
			}
			data, err := os.ReadFile(out)
			if err != nil {
				t.Fatalf("helper wrote no argv file: %v\nline: %s\noutput: %s", err, line, output)
			}
			var got []string
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, rows) {
				for i := range rows {
					if i >= len(got) || got[i] != rows[i] {
						t.Errorf("row %d: sent %q", i, rows[i])
					}
				}
				t.Fatalf("%s replay changed argv:\n got %q\nwant %q\nline: %s", shell, got, rows, line)
			}
		})
	}
	if !found {
		t.Skip("no powershell or pwsh on PATH")
	}
}

// powerShellPassesEmbeddedQuotes reports whether the shell is pwsh 7.3 or
// newer, where native arguments are passed with standard quoting.
func powerShellPassesEmbeddedQuotes(t *testing.T, shell string) bool {
	out, err := exec.Command(shell, "-NoProfile", "-NonInteractive", "-Command",
		"$v = $PSVersionTable.PSVersion; '{0}.{1}' -f $v.Major, $v.Minor").Output()
	if err != nil {
		t.Fatalf("%s version: %v", shell, err)
	}
	major, minor, _ := strings.Cut(strings.TrimSpace(string(out)), ".")
	ma, _ := strconv.Atoi(major)
	mi, _ := strconv.Atoi(minor)
	return ma > 7 || ma == 7 && mi >= 3
}

func encodePowerShellCommand(s string) string {
	u := utf16.Encode([]rune(s))
	b := make([]byte, 2*len(u))
	for i, c := range u {
		binary.LittleEndian.PutUint16(b[2*i:], c)
	}
	return base64.StdEncoding.EncodeToString(b)
}
