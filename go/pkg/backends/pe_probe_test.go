package backends

import (
	"os"
	"path/filepath"
	"testing"
)

// Windows resolves DLLs case-insensitively from the executable's directory; only
// llama libraries are followed.
func TestResolveSiblingLlamaLibsMatchesLikeWindows(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"llama.dll", "llama-server-impl.dll", "ggml.dll"} {
		if err := os.WriteFile(filepath.Join(dir, name), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got := resolveSiblingLlamaLibs(dir, []string{"LLAMA-SERVER-IMPL.DLL", "KERNEL32.dll", "ggml.dll", "llama-missing.dll"})
	if len(got) != 1 || filepath.Base(got[0]) != "llama-server-impl.dll" {
		t.Fatalf("want only the present llama import, resolved case-insensitively; got %v", got)
	}
}

// Opt-in check against a real installed backend (CI sets it on Windows).
// Before PE imports were followed, the published Windows llama-server.exe
// probed even "llama" as proven unsupported: the literals live in llama.dll.
func TestArchProbeOnInstalledBackend(t *testing.T) {
	path := os.Getenv("GGRUN_ARCH_PROBE_BACKEND")
	if path == "" {
		t.Skip("set GGRUN_ARCH_PROBE_BACKEND to an installed llama-server to run")
	}
	if supported, probed := BackendSupportsArch(path, "llama"); !probed || !supported {
		t.Fatalf("%s: architecture \"llama\" probed=%v supported=%v; files scanned: %v",
			path, probed, supported, archProbeFiles(path))
	}
	if supported, _ := BackendSupportsArch(path, "ggrun-no-such-arch"); supported {
		t.Fatalf("%s: an invented architecture probed as supported", path)
	}
}
