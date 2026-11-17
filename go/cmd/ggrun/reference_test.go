package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
)

var updateReference = flag.Bool("update-reference", false, "rewrite docs/reference.md")

// envVarsReadByCode lists every literal name passed to os.Getenv or
// os.LookupEnv in the module (tests excluded).
func envVarsReadByCode(t *testing.T) []string {
	t.Helper()
	seen := map[string]bool{}
	root := filepath.Join("..", "..")
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return err
		}
		f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			return err
		}
		ast.Inspect(f, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok || len(call.Args) == 0 {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || (sel.Sel.Name != "Getenv" && sel.Sel.Name != "LookupEnv") {
				return true
			}
			if lit, ok := call.Args[0].(*ast.BasicLit); ok && lit.Kind == token.STRING {
				if name, err := strconv.Unquote(lit.Value); err == nil && name != "" {
					seen[name] = true
				}
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range config.EnvKeys {
		seen[k], seen[config.GGRUNAlias(k)] = true, true
	}
	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func renderReference(t *testing.T) string {
	var b strings.Builder
	b.WriteString("# Flag and environment reference\n\n")
	b.WriteString("Generated from the code by `go test ./cmd/ggrun -run TestReferenceIsCurrent -args -update-reference`;\n")
	b.WriteString("do not edit by hand. See [usage.md](usage.md) for what the options do.\n\n")
	b.WriteString("## Launch flags\n\n`--flag value` and `--flag=value` mean the same. A switch takes `=true`/`=false`.\n")
	b.WriteString("Flags ggrun does not know are passed to the backend.\n\n| Flag | Takes a value |\n|---|---|\n")
	names := make([]string, 0, len(launchFlags))
	for n := range launchFlags {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		v := "no (switch)"
		if launchFlags[n].value {
			v = "yes"
		}
		fmt.Fprintf(&b, "| `%s` | %s |\n", n, v)
	}
	b.WriteString("\n## Configuration variables\n\nEach sets a config key; the `GGRUN_` spelling wins over the `LLM_` one when both are set.\n\n| Variable | Alias |\n|---|---|\n")
	for _, k := range config.EnvKeys {
		fmt.Fprintf(&b, "| `%s` | `%s` |\n", k, config.GGRUNAlias(k))
	}
	b.WriteString("\n## Every environment variable the code reads\n\n")
	for _, k := range envVarsReadByCode(t) {
		fmt.Fprintf(&b, "- `%s`\n", k)
	}
	return b.String()
}

// M-02: the reference lists every flag and variable; it fails when one is
// added or removed without regenerating it.
func TestReferenceIsCurrent(t *testing.T) {
	path := filepath.Join("..", "..", "..", "docs", "reference.md")
	got := renderReference(t)
	if *updateReference {
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(want) != got {
		t.Fatal("docs/reference.md is out of date; run: go test ./cmd/ggrun -run TestReferenceIsCurrent -args -update-reference")
	}
}
