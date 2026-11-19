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

// Parse every platform's Go files. A build on one OS would miss lookups in
// files guarded by build tags for another OS.
func envVarsReadByCode(t *testing.T) []string {
	t.Helper()
	files := make(map[string][]*ast.File)
	root := filepath.Join("..", "..")
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return err
		}
		f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			return err
		}
		dir := filepath.Dir(path)
		files[dir] = append(files[dir], f)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for dir, pkg := range files {
		cli := dir == filepath.Join(root, "cmd", "ggrun")
		constants := packageStringConstants(pkg)
		for _, file := range pkg {
			ast.Inspect(file, func(node ast.Node) bool {
				call, ok := node.(*ast.CallExpr)
				if !ok || len(call.Args) == 0 || !isEnvLookup(call.Fun, cli) {
					return true
				}
				name := stringLiteral(call.Args[0])
				if id, ok := call.Args[0].(*ast.Ident); ok {
					name = constants[id.Name]
				}
				if name != "" {
					seen[name] = true
				}
				return true
			})
		}
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

func packageStringConstants(files []*ast.File) map[string]string {
	constants := make(map[string]string)
	for _, file := range files {
		for _, decl := range file.Decls {
			group, ok := decl.(*ast.GenDecl)
			if !ok || group.Tok != token.CONST {
				continue
			}
			for _, spec := range group.Specs {
				values := spec.(*ast.ValueSpec)
				for i, value := range values.Values {
					if text := stringLiteral(value); text != "" {
						constants[values.Names[i].Name] = text
					}
				}
			}
		}
	}
	return constants
}

func stringLiteral(expr ast.Expr) string {
	if literal, ok := expr.(*ast.BasicLit); ok && literal.Kind == token.STRING {
		value, _ := strconv.Unquote(literal.Value)
		return value
	}
	return ""
}

func isEnvLookup(expr ast.Expr, cli bool) bool {
	switch fn := expr.(type) {
	case *ast.SelectorExpr:
		pkg, ok := fn.X.(*ast.Ident)
		return ok && pkg.Name == "os" && (fn.Sel.Name == "Getenv" || fn.Sel.Name == "LookupEnv")
	case *ast.Ident:
		// CLI wrappers read the first argument through os.Getenv.
		return cli && (fn.Name == "disabledEnv" || fn.Name == "getenv")
	}
	return false
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
	b.WriteString("\n## Environment variables read by the Go code\n\n")
	for _, k := range envVarsReadByCode(t) {
		fmt.Fprintf(&b, "- `%s`\n", k)
	}
	return b.String()
}

func TestEnvironmentReferenceIncludesConstantsAndWrappers(t *testing.T) {
	names := envVarsReadByCode(t)
	for _, want := range []string{
		"GGRUN_BACKEND_FORMAT_RETRY", "GGRUN_LEARNED_REPLAN", "GGRUN_VERIFY_OOM_REPLAN",
		"GGRUN_SCOPE_MODE", "GGRUN_CLAUDE_AUTO_REVIEWER", "LLAMA_API_KEY",
	} {
		i := sort.SearchStrings(names, want)
		if i == len(names) || names[i] != want {
			t.Errorf("environment reference is missing %s", want)
		}
	}
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
