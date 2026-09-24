package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A MiMo-V2.6 GGUF adds three MTP layers to the mimo2 architecture. A llama.cpp
// build from before that change knows "mimo2" but rejects the file's metadata.
// The launch reported this as a failed memory probe and suggested memory
// controls, none of which can make an older loader read a newer file.
const mimoOldLoaderLog = `0.01.041.117 I llama_prepare_model_devices: using device CUDA2 (NVIDIA GeForce RTX 3060) (0000:b3:00.0) - 11796 MiB free
0.01.043.412 E llama_model_load: error loading model: error loading model hyperparameters: key mimo2.attention.sliding_window_pattern has wrong array length; expected 48, got 51
0.01.043.416 E llama_model_load_from_file_impl: failed to load model
0.01.043.428 E srv    load_model: failed to load model, '/models/MiMo-V2.6-00001-of-00008.gguf'
0.01.046.205 E srv  llama_server: exiting due to model loading error
`

func TestModelFormatRejectionIsNotReportedAsMemory(t *testing.T) {
	detail := modelFormatRejection(mimoOldLoaderLog)
	if !strings.HasPrefix(detail, "error loading model hyperparameters") || !strings.Contains(detail, "expected 48, got 51") {
		t.Fatalf("loader rejection not recognized: %q", detail)
	}
	msg := (&backendModelFormatError{Backend: "/app/.bin/llama-server-cuda", Detail: detail}).Error()
	if strings.Contains(msg, "--kv-quality") || strings.Contains(msg, "--ctx-size") || strings.Contains(msg, "memory") {
		t.Fatalf("format rejection must not suggest memory controls: %s", msg)
	}
	if !strings.Contains(msg, "ggrun backend update") {
		t.Fatalf("format rejection must name the backend update: %s", msg)
	}
}

func TestAllocationFailureIsNotAModelFormatRejection(t *testing.T) {
	for _, log := range []string{
		"E llama_model_load: error loading model: unable to allocate Vulkan1 buffer\n",
		"E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2181.00 MiB on device 0: cudaMalloc failed: out of memory\n",
		"E srv load_model: failed to load model\n",
	} {
		if got := modelFormatRejection(log); got != "" {
			t.Fatalf("memory or generic failure classified as a format rejection: %q from %q", got, log)
		}
	}
}

func mainlineBackendFixture(t *testing.T) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), ".src", "llama.cpp", "build-cuda", "bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(bin, "llama-server")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestModelFormatRejectionUpdatesMainlineOnceAndOnlyOnNewCommit(t *testing.T) {
	path := mainlineBackendFixture(t)
	formatErr := &backendModelFormatError{Backend: path, Detail: "error loading model hyperparameters: wrong array length"}
	commit := "89fe242"
	commitOf := func(string) string { return commit }
	updates := 0
	advance := func() error { updates++; commit = "d2462f8"; return nil }

	var out bytes.Buffer
	if !offerBackendUpdateForModelFormatWith(&launchRequest{}, &backendInfo{Path: path}, formatErr, false, false,
		strings.NewReader("y\n"), &out, true, commitOf, advance) {
		t.Fatalf("accepted update to a new commit must relaunch; output: %s", out.String())
	}
	if updates != 1 {
		t.Fatalf("updates = %d, want 1", updates)
	}

	// The relaunched process must not update again if the new build still
	// rejects the file: that would loop through 20-minute builds forever.
	if offerBackendUpdateForModelFormatWith(&launchRequest{}, &backendInfo{Path: path}, formatErr, true, true,
		strings.NewReader(""), &out, true, commitOf, advance) {
		t.Fatal("a retried launch must not update again")
	}

	// An update that leaves the commit unchanged cannot fix the load.
	unchanged := func() error { updates++; return nil }
	if offerBackendUpdateForModelFormatWith(&launchRequest{}, &backendInfo{Path: path}, formatErr, true, false,
		strings.NewReader(""), &out, true, commitOf, unchanged) {
		t.Fatal("an update that did not change the commit must not relaunch")
	}

	failing := func() error { return errors.New("build failed") }
	if offerBackendUpdateForModelFormatWith(&launchRequest{}, &backendInfo{Path: path}, formatErr, true, false,
		strings.NewReader(""), &out, true, commitOf, failing) {
		t.Fatal("a failed update must not relaunch")
	}
}

func TestModelFormatUpdateRespectsChoiceAndBackendFamily(t *testing.T) {
	path := mainlineBackendFixture(t)
	formatErr := &backendModelFormatError{Backend: path, Detail: "missing tensor 'blk.48.nextn.eh_proj.weight'"}
	commitOf := func(string) string { return "" }
	update := func() error { t.Fatal("update must not run"); return nil }
	var out bytes.Buffer

	if offerBackendUpdateForModelFormatWith(&launchRequest{BackendExplicit: true}, &backendInfo{Path: path}, formatErr, true, false,
		strings.NewReader(""), &out, true, commitOf, update) {
		t.Fatal("an explicitly chosen backend must not be updated automatically")
	}
	fork := filepath.Join(t.TempDir(), ".src", "fork-llama.cpp-x", "build-cuda", "bin", "llama-server")
	if offerBackendUpdateForModelFormatWith(&launchRequest{}, &backendInfo{Path: fork}, formatErr, true, false,
		strings.NewReader(""), &out, true, commitOf, update) {
		t.Fatal("a mainline update cannot fix a fork backend")
	}
	if offerBackendUpdateForModelFormatWith(&launchRequest{}, &backendInfo{Path: path}, formatErr, false, false,
		strings.NewReader("n\n"), &out, true, commitOf, update) {
		t.Fatal("a declined prompt must not update")
	}
	out.Reset()
	if offerBackendUpdateForModelFormatWith(&launchRequest{}, &backendInfo{Path: path}, formatErr, false, false,
		strings.NewReader(""), &out, false, commitOf, update) {
		t.Fatal("a non-terminal launch without consent must not update")
	}
	if !strings.Contains(out.String(), "ggrun backend update") {
		t.Fatalf("non-terminal launch must print the fix: %q", out.String())
	}
}

// Upstream llama.cpp changed its version line. Reading only the old form made
// every new build look commit-less, so an update that succeeded was reported
// as "already at the newest llama.cpp" and the launch never relaunched.
func TestBackendCommitReadsOldAndNewVersionLines(t *testing.T) {
	for out, want := range map[string]string{
		"version: 10954 (89fe24240)\nbuilt with GNU 13.3.0 for Linux x86_64\n":                                     "89fe24240",
		"0.00.000.388 I srv  llama_server: initializing ...\nversion: 0.5.0-dev (build 11159, commit 6b790a9c2)\n": "6b790a9c2",
		"version: 1 (d2462f8)\n": "d2462f8",
		"no version here\n":      "",
	} {
		if got := parseBackendVersionCommit(out); got != want {
			t.Fatalf("parseBackendVersionCommit(%q) = %q, want %q", out, got, want)
		}
	}
}
