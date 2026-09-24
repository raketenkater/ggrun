package main

import (
	"bufio"
	"fmt"
	"io"
	"path/filepath"
	"strings"
)

// backendFormatRetryEnv marks a launch that already updated the backend once
// for a model-format rejection, so a still-incompatible backend fails instead
// of updating and relaunching in a loop.
const backendFormatRetryEnv = "GGRUN_BACKEND_FORMAT_RETRY"

// isMainlineBackendPath reports whether a backend binary is a build of the
// mainline llama.cpp checkout (.src/llama.cpp/build-*/bin/llama-server), the
// only backend the mainline update advances.
func isMainlineBackendPath(path string) bool {
	if path == "" {
		return false
	}
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		path = resolved
	}
	repo := filepath.Dir(filepath.Dir(filepath.Dir(path)))
	return filepath.Base(repo) == "llama.cpp"
}

// offerBackendUpdateForModelFormatWith updates the mainline llama.cpp backend
// when it rejected the model file itself. It returns true only when the update
// actually moved the backend to a different commit, so the caller relaunches
// on the new build. An explicit backend choice, a fork backend, a prior retry,
// a declined prompt or an unchanged commit all return false and keep the error.
func offerBackendUpdateForModelFormatWith(req *launchRequest, be *backendInfo, formatErr *backendModelFormatError,
	assumeYes, retried bool, in io.Reader, out io.Writer, terminal bool,
	commitOf func(string) string, update func() error,
) bool {
	if req == nil || be == nil || formatErr == nil || retried || backendChoiceExplicit(req) || !isMainlineBackendPath(be.Path) {
		return false
	}
	if !assumeYes {
		if !terminal {
			fmt.Fprintf(out, "[launch] the mainline llama.cpp backend cannot read this model; update it with: ggrun backend update\n")
			return false
		}
		fmt.Fprintf(out, "The installed llama.cpp backend cannot read this model file:\n  %s\nUpdate mainline llama.cpp and relaunch (about 20-40 min)? [y/N] ", formatErr.Detail)
		line, _ := bufio.NewReader(in).ReadString('\n')
		if answer := strings.ToLower(strings.TrimSpace(line)); answer != "y" && answer != "yes" {
			return false
		}
	}
	before := commitOf(be.Path)
	fmt.Fprintln(out, "[launch] updating mainline llama.cpp for this model")
	if err := update(); err != nil {
		fmt.Fprintf(out, "[launch] backend update failed; the previous build is kept: %v\n", err)
		return false
	}
	if after := commitOf(be.Path); after == "" || after == before {
		fmt.Fprintln(out, "[launch] the backend is already at the newest llama.cpp; this model needs a newer upstream release or a fork that supports it")
		return false
	}
	return true
}
