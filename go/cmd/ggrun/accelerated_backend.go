package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strings"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

const mainlineLlamaCppGit = "https://github.com/ggml-org/llama.cpp.git"

var backendVersionCommit = regexp.MustCompile(`version:\s*\d+\s*\(([0-9a-f]{7,40})\)`)

// backendCommit reads the source commit a llama.cpp build reports.
func backendCommit(path string) string {
	out, _ := exec.Command(path, "--version").CombinedOutput()
	if m := backendVersionCommit.FindStringSubmatch(string(out)); m != nil {
		return m[1]
	}
	return ""
}

func hostHasNVIDIAGPU(caps *detect.Capabilities) bool {
	if caps == nil {
		return false
	}
	for _, g := range caps.GPUs {
		if strings.Contains(strings.ToUpper(g.Name), "NVIDIA") {
			return true
		}
	}
	return false
}

// acceleratedMainlineRecipe is the CUDA build to offer when automatic selection
// could only serve an architecture from the Vulkan build on an NVIDIA host.
//
// A fresh install ships ik_llama for CUDA and mainline llama.cpp for Vulkan.
// An architecture only mainline knows (MiMo-V2.6's mimo2) then runs on Vulkan
// across NVIDIA cards: slower, and a different placement path than the rest of
// the product is tuned for. The existing mainline update only refreshes a
// mainline *source* build, which a release install does not have, so nothing
// offered the obvious fix. The recipe pins the Vulkan build's own commit: the
// probe already proved that source knows the architecture, and a pin keeps the
// build reproducible.
func acceleratedMainlineRecipe(arch string, be *backendInfo, caps *detect.Capabilities, commitOf func(string) string) (*backends.Recipe, bool) {
	arch = strings.TrimSpace(arch)
	if arch == "" || be == nil || !strings.EqualFold(be.Dialect, "vulkan") || !hostHasNVIDIAGPU(caps) {
		return nil, false
	}
	commit := commitOf(be.Path)
	if commit == "" {
		return nil, false
	}
	short := commit
	if len(short) > 7 {
		short = short[:7]
	}
	tag := "llama-cuda-" + short
	return &backends.Recipe{
		Name:      tag,
		Tag:       tag,
		GitURL:    mainlineLlamaCppGit,
		Branch:    "master",
		Commit:    commit,
		RouteArch: arch,
		Accel:     "cuda",
	}, true
}

func acceleratedMainlineArgs(r *backends.Recipe) []string {
	return []string{r.GitURL, "--tag", r.Tag, "--checkout-name", r.Tag, "--branch", r.Branch,
		"--commit", r.Commit, "--route-arch", r.RouteArch, "--accel", r.Accel}
}

// offerAcceleratedMainlineWith asks (or, headless, explains) before building a
// CUDA mainline backend for arch. Declining or a failed build keeps the Vulkan
// backend that already works: this is an upgrade offer, never a dead end.
func offerAcceleratedMainlineWith(req *launchRequest, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities,
	assumeYes bool, in io.Reader, out io.Writer, terminal bool,
	commitOf func(string) string, install func([]string, *backends.Recipe) error,
) bool {
	if backendChoiceExplicit(req) || model == nil {
		return false
	}
	recipe, ok := acceleratedMainlineRecipe(model.ModelArch, be, caps, commitOf)
	if !ok {
		return false
	}
	args := acceleratedMainlineArgs(recipe)
	if !assumeYes {
		if !terminal {
			fmt.Fprintf(out, "[launch] %s is served here only by the Vulkan build. For CUDA, build mainline llama.cpp at the same commit with: ggrun backend add %s\n",
				model.ModelArch, strings.Join(args, " "))
			return false
		}
		fmt.Fprintf(out, "%s runs here only on the Vulkan build. Build mainline llama.cpp %s with CUDA as a separate backend %q (about 20-40 min)? [y/N] ",
			model.ModelArch, recipe.Commit, recipe.Tag)
		line, _ := bufio.NewReader(in).ReadString('\n')
		if answer := strings.ToLower(strings.TrimSpace(line)); answer != "y" && answer != "yes" {
			return false
		}
	}
	fmt.Fprintf(out, "[launch] building mainline llama.cpp %s with CUDA as backend %q for %s\n", recipe.Commit, recipe.Tag, model.ModelArch)
	if err := install(args, recipe); err != nil {
		fmt.Fprintf(out, "[launch] CUDA build failed, continuing on the Vulkan backend: %v\n", err)
		return false
	}
	return true
}

func offerAcceleratedMainline(req *launchRequest, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, assumeYes bool) bool {
	return offerAcceleratedMainlineWith(req, model, be, caps, assumeYes, os.Stdin, os.Stderr, stdinIsTerminal(),
		backendCommit, addBackendRecipe)
}
