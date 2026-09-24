package backends

import (
	"os"
	"path/filepath"
	"testing"
)

func TestBuildDefinesTargetFindsConfiguredTarget(t *testing.T) {
	dir := t.TempDir()
	if BuildDefinesTarget(dir, FitParamsTool) || len(OptionalBuildTargets(dir)) != 0 {
		t.Fatal("empty tree reported a target")
	}
	if err := os.MkdirAll(filepath.Join(dir, "tools", "fit-params", "CMakeFiles", "llama-fit-params-impl.dir"), 0o755); err != nil {
		t.Fatal(err)
	}
	if BuildDefinesTarget(dir, FitParamsTool) {
		t.Fatal("the -impl library was mistaken for the executable target")
	}
	if err := os.MkdirAll(filepath.Join(dir, "tools", "fit-params", "CMakeFiles", "llama-fit-params.dir"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := OptionalBuildTargets(dir); len(got) != 1 || got[0] != FitParamsTool {
		t.Fatalf("targets = %v", got)
	}
	if got := MissingBuildTools(dir); len(got) != 1 {
		t.Fatalf("defined but unbuilt oracle not reported: %v", got)
	}
}

func TestCheckFitParamsToolRequiresFitPrint(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) string {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
			t.Fatal(err)
		}
		return path
	}
	if err := CheckFitParamsTool(write("good", "#!/bin/sh\necho '-fitp, --fit-print [on|off]'\n")); err != nil {
		t.Fatalf("good oracle rejected: %v", err)
	}
	if err := CheckFitParamsTool(write("old", "#!/bin/sh\necho usage\n")); err == nil {
		t.Fatal("oracle without --fit-print accepted")
	}
	if err := CheckFitParamsTool(write("crash", "#!/bin/sh\necho 'error while loading shared libraries' >&2; exit 127\n")); err == nil {
		t.Fatal("oracle that cannot start accepted")
	}
	if err := CheckFitParamsTool(filepath.Join(dir, "absent")); err == nil {
		t.Fatal("missing oracle accepted")
	}
}
