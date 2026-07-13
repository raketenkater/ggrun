package libhub

import (
	"os"
	"path/filepath"
	"testing"
)

func touch(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
}

// A flat release bundle: thin binary + co-located .so, not under a build/ dir.
func TestSetupFlatBundle(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "llama-server")
	touch(t, bin)
	touch(t, filepath.Join(dir, "libllama.so"))
	touch(t, filepath.Join(dir, "libggml-base.so.0"))

	hub, ok, err := Setup(bin)
	if err != nil || !ok {
		t.Fatalf("flat bundle: ok=%v err=%v", ok, err)
	}
	defer Cleanup(hub)
	for _, want := range []string{"libllama.so", "libggml-base.so.0"} {
		if _, err := os.Stat(filepath.Join(hub, want)); err != nil {
			t.Fatalf("hub missing %s: %v", want, err)
		}
	}
}

// The real ggrun layout: .bin/llama-server-cuda is a SYMLINK into a build tree
// whose libs sit beside the binary (build/bin). Setup must resolve the symlink.
func TestSetupResolvesSymlinkToColocatedLibs(t *testing.T) {
	root := t.TempDir()
	realBinDir := filepath.Join(root, "src", "build-cuda", "bin")
	realBin := filepath.Join(realBinDir, "llama-server")
	touch(t, realBin)
	touch(t, filepath.Join(realBinDir, "libllama.so"))

	linkDir := filepath.Join(root, ".bin")
	if err := os.MkdirAll(linkDir, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(linkDir, "llama-server-cuda")
	if err := os.Symlink(realBin, link); err != nil {
		t.Fatal(err)
	}

	hub, ok, err := Setup(link)
	if err != nil || !ok {
		t.Fatalf("symlinked bundle: ok=%v err=%v", ok, err)
	}
	defer Cleanup(hub)
	if _, err := os.Stat(filepath.Join(hub, "libllama.so")); err != nil {
		t.Fatalf("hub missing libllama.so after symlink resolve: %v", err)
	}
}

func TestSetupFindsScatteredLibrariesInNamedBuildDir(t *testing.T) {
	root := t.TempDir()
	bin := filepath.Join(root, "src", "build-cuda", "bin", "llama-server")
	lib := filepath.Join(root, "src", "build-cuda", "common", "libmtmd.so")
	touch(t, bin)
	touch(t, lib)

	hub, ok, err := Setup(bin)
	if err != nil || !ok {
		t.Fatalf("named build directory: ok=%v err=%v", ok, err)
	}
	defer Cleanup(hub)
	if _, err := os.Stat(filepath.Join(hub, "libmtmd.so")); err != nil {
		t.Fatalf("hub missing scattered libmtmd.so: %v", err)
	}
}

// A static binary (no libraries anywhere) needs no hub.
func TestSetupStaticBinaryNoHub(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "llama-server")
	touch(t, bin)
	if _, ok, _ := Setup(bin); ok {
		t.Fatal("static binary should not produce a hub")
	}
}

func TestApplyToChildEnv(t *testing.T) {
	t.Setenv("LLM_SERVER_LIB_HUB", "/hub")

	// no existing LD_LIBRARY_PATH → appended
	got := ApplyToChildEnv([]string{"FOO=1"})
	if got[len(got)-1] != "LD_LIBRARY_PATH=/hub" {
		t.Fatalf("append case: %v", got)
	}
	// existing LD_LIBRARY_PATH → prepended, preserving the old value
	got = ApplyToChildEnv([]string{"LD_LIBRARY_PATH=/a:/b", "FOO=1"})
	if got[0] != "LD_LIBRARY_PATH=/hub:/a:/b" {
		t.Fatalf("prepend case: %v", got)
	}
	// empty existing value → just the hub
	got = ApplyToChildEnv([]string{"LD_LIBRARY_PATH="})
	if got[0] != "LD_LIBRARY_PATH=/hub" {
		t.Fatalf("empty case: %v", got)
	}

	// no hub configured → env unchanged
	t.Setenv("LLM_SERVER_LIB_HUB", "")
	in := []string{"FOO=1"}
	if got := ApplyToChildEnv(in); len(got) != 1 || got[0] != "FOO=1" {
		t.Fatalf("no-hub case altered env: %v", got)
	}
}

func TestApplyHubToChildEnv(t *testing.T) {
	got := ApplyHubToChildEnv([]string{"LD_LIBRARY_PATH=/system", "FOO=1"}, "/probe")
	if got[0] != "LD_LIBRARY_PATH=/probe:/system" {
		t.Fatalf("explicit hub: %v", got)
	}
}
