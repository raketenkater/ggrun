package main

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
)

func requestWithoutArgs(t *testing.T, args []string) (map[string]any, error) {
	t.Helper()
	req, err := parseLaunchArgs(args)
	if err != nil {
		return nil, err
	}
	m := requestData(reflect.ValueOf(req)).(map[string]any)
	delete(m, "OriginalArgs") // the spelling itself, by design
	b, _ := json.Marshal(m)
	var out map[string]any
	_ = json.Unmarshal(b, &out)
	return out, nil
}

// C-14: every ggrun flag means the same in --flag value and --flag=value.
func TestEveryLaunchFlagHasTheSameMeaningInBothForms(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	b, err := os.ReadFile(filepath.Join("testdata", "launch_flags_golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	var rows []goldenRow
	if err := json.Unmarshal(b, &rows); err != nil {
		t.Fatal(err)
	}
	for _, row := range rows {
		if row.Request == "rejected" {
			continue
		}
		space := []string{row.Flag, "m.gguf"}
		equals := []string{row.Flag + "=true", "m.gguf"}
		if row.Sample != "" {
			space = []string{row.Flag, row.Sample, "m.gguf"}
			equals = []string{row.Flag + "=" + row.Sample, "m.gguf"}
		}
		a, errA := requestWithoutArgs(t, space)
		e, errE := requestWithoutArgs(t, equals)
		if errA != nil || errE != nil {
			t.Errorf("%s: space err %v, equals err %v", row.Flag, errA, errE)
			continue
		}
		if !reflect.DeepEqual(a, e) {
			t.Errorf("%s: %v and %v parse differently", row.Flag, space, equals)
		}
		extra, _ := e["ExtraArgs"].([]any)
		spaceExtra, _ := a["ExtraArgs"].([]any)
		for _, x := range extra {
			if s, _ := x.(string); strings.HasPrefix(s, row.Flag) && len(spaceExtra) == 0 {
				t.Errorf("%s=... reached the backend: %v", row.Flag, e["ExtraArgs"])
			}
		}
	}
}

// A switch spelled --flag=false is the same as leaving it out.
func TestSwitchFalseInEqualsFormIsOff(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	plain, _ := requestWithoutArgs(t, []string{"m.gguf"})
	for name, spec := range launchFlags {
		if spec.value || spec.equalsCase {
			continue
		}
		got, err := requestWithoutArgs(t, []string{name + "=false", "m.gguf"})
		if err != nil {
			t.Errorf("%s=false: %v", name, err)
			continue
		}
		if !reflect.DeepEqual(got, plain) {
			t.Errorf("%s=false is not off", name)
		}
	}
	if _, err := parseLaunchArgs([]string{"--dry-run=maybe", "m.gguf"}); err == nil {
		t.Error("--dry-run=maybe accepted")
	}
}

func TestUnknownFlagNeverTakesTheOnlyModelPath(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	req, err := parseLaunchArgs([]string{"--no-such-bool", "model.gguf", "--dry-run"})
	if err != nil || req.ModelPath != "model.gguf" || !reflect.DeepEqual(req.ExtraArgs, []string{"--no-such-bool"}) {
		t.Fatalf("model %q extra %v err %v", req.ModelPath, req.ExtraArgs, err)
	}
	req, _ = parseLaunchArgs([]string{"--lora", "adapter.gguf", "model.gguf"})
	if req.ModelPath != "model.gguf" || !reflect.DeepEqual(req.ExtraArgs, []string{"--lora", "adapter.gguf"}) {
		t.Fatalf("value-taking backend flag: model %q extra %v", req.ModelPath, req.ExtraArgs)
	}
	req, _ = parseLaunchArgs([]string{"model.gguf", "--temp", "0.7"})
	if req.ModelPath != "model.gguf" || !reflect.DeepEqual(req.ExtraArgs, []string{"--temp", "0.7"}) {
		t.Fatalf("model first: model %q extra %v", req.ModelPath, req.ExtraArgs)
	}
	req, _ = parseLaunchArgs([]string{"--temp", "0.7", "model.gguf"})
	if req.ModelPath != "model.gguf" || !reflect.DeepEqual(req.ExtraArgs, []string{"--temp", "0.7"}) {
		t.Fatalf("value before model: model %q extra %v", req.ModelPath, req.ExtraArgs)
	}
}

// The table and parseLaunchArgs's case labels must list the same flags.
func TestLaunchFlagTableMatchesParser(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "main.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var fn *ast.FuncDecl
	for _, d := range file.Decls {
		if f, ok := d.(*ast.FuncDecl); ok && f.Name.Name == "parseLaunchArgs" {
			fn = f
		}
	}
	if fn == nil {
		t.Fatal("parseLaunchArgs not found")
	}
	equals, all := map[string]bool{}, map[string]bool{}
	ast.Inspect(fn, func(n ast.Node) bool {
		sw, ok := n.(*ast.SwitchStmt)
		if !ok {
			return true
		}
		tag, _ := sw.Tag.(*ast.Ident)
		if tag == nil || (tag.Name != "key" && tag.Name != "a") {
			return true
		}
		for _, st := range sw.Body.List {
			for _, e := range st.(*ast.CaseClause).List {
				lit, ok := e.(*ast.BasicLit)
				if !ok {
					continue
				}
				name, _ := strconv.Unquote(lit.Value)
				if name == "--" || !strings.HasPrefix(name, "-") {
					continue
				}
				all[name] = true
				if tag.Name == "key" {
					equals[name] = true
				}
			}
		}
		return true
	})
	var missing, extra, eqWrong []string
	for name := range all {
		if _, ok := launchFlags[name]; !ok {
			missing = append(missing, name)
		}
	}
	for name, spec := range launchFlags {
		if !all[name] {
			extra = append(extra, name)
		}
		if spec.equalsCase != equals[name] {
			eqWrong = append(eqWrong, name)
		}
	}
	sort.Strings(missing)
	sort.Strings(extra)
	sort.Strings(eqWrong)
	if len(missing)+len(extra)+len(eqWrong) > 0 {
		t.Fatalf("flag table out of step with parseLaunchArgs: missing %v, stale %v, equalsCase wrong %v", missing, extra, eqWrong)
	}
}
