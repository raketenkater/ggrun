package main

import (
	"os"
	"path/filepath"
	"testing"
)

// writeArchFixture creates a file the arch probe can scan. Genuine architecture
// literals in a llama.cpp binary are NUL-terminated and follow the previous
// string's terminator, so the fixture reproduces that framing exactly.
func writeArchFixture(t *testing.T, name string, arches ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	buf := []byte("\x00some.unrelated.symbol\x00")
	for _, a := range arches {
		buf = append(buf, 0)
		buf = append(buf, a...)
		buf = append(buf, 0)
	}
	if err := os.WriteFile(path, buf, 0o755); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

// The invariant: a backend that demonstrably carries the architecture must not
// be vetoed by the static ik_llama-only name table.
//
// Measured case this guards (2026-09-16): minimax-m3 is listed as ik_llama-only,
// but the user-registered fork-llama.cpp-minimax-m3 build carries the
// `minimax-m3` literal in its libllama.so. ggrun routed to that backend and then
// refused it, stranding a 152 GB MoE whose backend was present and correct.
func TestBackendCarryingArchIsNotVetoedByTheStaticTable(t *testing.T) {
	be := &backendInfo{Path: writeArchFixture(t, "llama-server", "minimax-m2", "minimax-m3")}
	if !backendProvenToCarryArch(be, "minimax-m3") {
		t.Fatal("backend carrying the arch literal was not recognised; the static table would veto a working backend")
	}
}

// The complement: absence of the literal must NOT override the table, or the
// guard would wave through genuinely unsupported architectures.
func TestBackendMissingArchDoesNotOverrideTheTable(t *testing.T) {
	be := &backendInfo{Path: writeArchFixture(t, "llama-server", "llama", "qwen2moe")}
	if backendProvenToCarryArch(be, "minimax-m3") {
		t.Fatal("backend without the arch literal was treated as proven; the ik_llama guard would be bypassed")
	}
}

// Fail-safe, and the reason this is an override rather than a replacement: when
// the probe cannot read the binary at all it reports probed == false, and the
// conservative hard failure must be preserved. A Windows build keeping its
// architectures in a DLL probes as unreadable while working perfectly, so
// "unknown" must never be read as "supported".
func TestUnprobableBackendPreservesTheConservativeFailure(t *testing.T) {
	be := &backendInfo{Path: filepath.Join(t.TempDir(), "does-not-exist")}
	if backendProvenToCarryArch(be, "minimax-m3") {
		t.Fatal("unreadable backend was treated as proven; missing evidence must not authorise a launch")
	}
}

func TestArchProbeGuardsAgainstEmptyInputs(t *testing.T) {
	good := writeArchFixture(t, "llama-server", "minimax-m3")
	cases := []struct {
		name string
		be   *backendInfo
		arch string
	}{
		{"nil backend", nil, "minimax-m3"},
		{"empty path", &backendInfo{Path: ""}, "minimax-m3"},
		{"empty arch", &backendInfo{Path: good}, ""},
		{"whitespace arch", &backendInfo{Path: good}, "   "},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if backendProvenToCarryArch(tc.be, tc.arch) {
				t.Fatalf("%s was treated as proven support", tc.name)
			}
		})
	}
}

// A substring must not count. "minimax-m3" appearing inside a longer literal or
// a file path is not evidence that the backend implements that architecture.
func TestArchProbeRejectsSubstringMatches(t *testing.T) {
	be := &backendInfo{Path: writeArchFixture(t, "llama-server", "minimax-m30", "not-minimax-m3-either")}
	if backendProvenToCarryArch(be, "minimax-m3") {
		t.Fatal("a substring of a longer literal was accepted as arch support")
	}
}
