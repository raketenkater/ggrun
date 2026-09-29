package main

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// The mainline build installed on 2026-09-24 prints a timestamped log line
// before its version. Its identity must not change between two probes, or
// every identity-keyed cache (calibration, memory evidence, verified configs)
// misses on every launch.
func TestBackendIdentityIgnoresTimestampedLogLines(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX fake backend")
	}
	dir := t.TempDir()
	counter := filepath.Join(dir, "n")
	server := filepath.Join(dir, "llama-server")
	body := "#!/bin/sh\nn=$(cat " + counter + " 2>/dev/null || echo 0); n=$((n+1)); echo $n > " + counter + "\n" +
		"printf '0.00.000.%03d I srv  llama_server: initializing ...\\n' $n\n" +
		"echo 'version: 0.5.0-dev (build 11159, commit 6b790a9c2)'\necho 'built with GNU 13.3.0 for Linux x86_64'\n"
	if err := os.WriteFile(server, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	first, second := backendBuildIdentity(server), backendBuildIdentity(server)
	if first != second {
		t.Fatalf("identity changed between probes: %s vs %s", first, second)
	}
	if err := os.WriteFile(server, []byte(body+"echo 'version: other'\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if backendBuildIdentity(server) == first {
		t.Fatal("a different build kept the same identity")
	}
}

func TestStableBackendProbeOutputKeepsLoaderErrors(t *testing.T) {
	out := "0.00.000.398 I srv  llama_server: initializing ...\n" +
		"llama-server: error while loading shared libraries: libggml.so.0\n  --model FNAME\n"
	got := stableBackendProbeOutput(out)
	if got != "llama-server: error while loading shared libraries: libggml.so.0\n  --model FNAME\n" {
		t.Fatalf("normalized output = %q", got)
	}
	if !backendLoaderFailed(got) {
		t.Fatal("loader failure no longer detectable after normalization")
	}
}
