package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func vulkanMiMo() (*placement.ModelProfile, *backendInfo) {
	return &placement.ModelProfile{ModelArch: "mimo2"},
		&backendInfo{Path: "/x/llama-server-vulkan", Dialect: "vulkan"}
}

func fixedCommit(string) string { return "d2462f8f7ac6d80070a587ffebf6cd73730f4280" }

// On an NVIDIA host an architecture only the Vulkan build knows must lead to an
// offer of a CUDA mainline build pinned to the Vulkan build's commit.
func TestVulkanOnlyArchitectureOffersAPinnedCUDABuild(t *testing.T) {
	model, be := vulkanMiMo()
	var gotArgs []string
	var gotRecipe *backends.Recipe
	install := func(args []string, r *backends.Recipe) error { gotArgs, gotRecipe = args, r; return nil }
	var out bytes.Buffer
	if !offerAcceleratedMainlineWith(&launchRequest{}, model, be, rigGPUs(), true, strings.NewReader(""), &out, false, fixedCommit, install) {
		t.Fatalf("assume-yes did not install the CUDA build: %s", out.String())
	}
	joined := strings.Join(gotArgs, " ")
	for _, want := range []string{mainlineLlamaCppGit, "--commit d2462f8f7ac6d80070a587ffebf6cd73730f4280", "--route-arch mimo2", "--accel cuda", "--tag llama-cuda-d2462f8"} {
		if !strings.Contains(joined, want) {
			t.Errorf("install args %q lack %q", joined, want)
		}
	}
	if gotRecipe == nil || gotRecipe.RouteArch != "mimo2" {
		t.Fatalf("recipe does not route the architecture: %+v", gotRecipe)
	}
}

func TestCUDABuildOfferRespectsConsentAndExplicitChoices(t *testing.T) {
	model, be := vulkanMiMo()
	called := false
	install := func([]string, *backends.Recipe) error { called = true; return nil }

	// Headless without consent: explain with the exact command, keep Vulkan.
	var out bytes.Buffer
	if offerAcceleratedMainlineWith(&launchRequest{}, model, be, rigGPUs(), false, strings.NewReader(""), &out, false, fixedCommit, install) || called {
		t.Fatal("a headless run without consent built a backend")
	}
	if !strings.Contains(out.String(), "ggrun backend add "+mainlineLlamaCppGit) {
		t.Fatalf("headless run did not print the command: %q", out.String())
	}
	// A terminal answer of no keeps Vulkan.
	if offerAcceleratedMainlineWith(&launchRequest{}, model, be, rigGPUs(), false, strings.NewReader("n\n"), &out, true, fixedCommit, install) || called {
		t.Fatal("declining still built a backend")
	}
	// An explicit backend is a constraint.
	if offerAcceleratedMainlineWith(&launchRequest{Backend: "vulkan", BackendExplicit: true}, model, be, rigGPUs(), true, strings.NewReader(""), &out, false, fixedCommit, install) || called {
		t.Fatal("an explicit backend choice was overridden")
	}
}

func TestCUDABuildOfferOnlyForVulkanOnNVIDIA(t *testing.T) {
	model, _ := vulkanMiMo()
	install := func([]string, *backends.Recipe) error { t.Fatal("unexpected install"); return nil }
	var out bytes.Buffer
	cuda := &backendInfo{Path: "/x/llama-server", Dialect: "ik_llama"}
	if offerAcceleratedMainlineWith(&launchRequest{}, model, cuda, rigGPUs(), true, strings.NewReader(""), &out, false, fixedCommit, install) {
		t.Fatal("offered a CUDA build although the chosen backend is already CUDA")
	}
	amd := rigGPUs()
	for i := range amd.GPUs {
		amd.GPUs[i].Name = "AMD Radeon RX 7900 XTX"
	}
	_, be := vulkanMiMo()
	if offerAcceleratedMainlineWith(&launchRequest{}, model, be, amd, true, strings.NewReader(""), &out, false, fixedCommit, install) {
		t.Fatal("offered CUDA on a host without NVIDIA GPUs")
	}
	if offerAcceleratedMainlineWith(&launchRequest{}, model, be, rigGPUs(), true, strings.NewReader(""), &out, false, func(string) string { return "" }, install) {
		t.Fatal("offered a build without knowing which commit supports the architecture")
	}
}

// A failed build keeps the Vulkan backend that already works.
func TestFailedCUDABuildFallsBackToVulkan(t *testing.T) {
	model, be := vulkanMiMo()
	var out bytes.Buffer
	failing := func([]string, *backends.Recipe) error { return errors.New("nvcc not found") }
	if offerAcceleratedMainlineWith(&launchRequest{}, model, be, rigGPUs(), true, strings.NewReader(""), &out, false, fixedCommit, failing) {
		t.Fatal("a failed build reported success")
	}
	if !strings.Contains(out.String(), "continuing on the Vulkan backend") {
		t.Fatalf("failure not explained: %q", out.String())
	}
}

// Nanbeige4.2 on a fresh install: neither bundled backend knows "nanbeige",
// the only recipe is the CPU-only helper, and the mainline-update offer could
// only say "nothing to update". The launch must offer an accelerated build of
// the reviewed source instead, without touching the helper build.
func TestUnsupportedArchOffersAcceleratedBuildOfReviewedSource(t *testing.T) {
	nvidia := &detect.Capabilities{GPUs: []detect.GPU{{Name: "NVIDIA GeForce RTX 3090 Ti"}}}
	r := acceleratedArchBuildRecipe("nanbeige", nvidia, "cpu")
	if r == nil || r.HelperOnly || r.Accel != "cuda" || r.Tag != "nanbeige42-cuda" || r.Commit == "" || len(r.Patches) == 0 {
		t.Fatalf("reviewed helper source not offered as an accelerated main build: %+v", r)
	}
	if backends.RecipeByName(r.Tag) != nil {
		t.Fatal("the accelerated build shares a tag with a catalog recipe and would inherit its helper policy")
	}
	novel := acceleratedArchBuildRecipe("brandnew", nil, "cpu")
	if novel == nil || novel.GitURL != mainlineLlamaCppGit || novel.Commit != "" || novel.Accel != "cpu" || novel.RouteArch != "brandnew" {
		t.Fatalf("novel architecture did not fall back to current mainline: %+v", novel)
	}
	if args := strings.Join(acceleratedMainlineArgs(novel), " "); strings.Contains(args, "--commit") {
		t.Fatalf("an unpinned build passed an empty commit: %q", args)
	}

	model := &placement.ModelProfile{ModelArch: "nanbeige"}
	var installed []string
	install := func(args []string, _ *backends.Recipe) error { installed = args; return nil }
	var out strings.Builder
	if offerAcceleratedArchBuildWith(&launchRequest{}, model, nvidia, false, strings.NewReader(""), &out, false, "cpu", install) || installed != nil {
		t.Fatal("headless launch built without consent")
	}
	if !strings.Contains(out.String(), "ggrun backend add") {
		t.Fatalf("headless launch did not print the build command: %q", out.String())
	}
	if offerAcceleratedArchBuildWith(&launchRequest{}, model, nvidia, false, strings.NewReader("n\n"), &out, true, "cpu", install) || installed != nil {
		t.Fatal("declined build still ran")
	}
	if !offerAcceleratedArchBuildWith(&launchRequest{}, model, nvidia, false, strings.NewReader("y\n"), &out, true, "cpu", install) || installed == nil {
		t.Fatal("accepted build did not run")
	}
	installed = nil
	if offerAcceleratedArchBuildWith(&launchRequest{BackendExplicit: true}, model, nvidia, true, nil, &out, true, "cpu", install) || installed != nil {
		t.Fatal("an explicit backend choice was overridden by a build offer")
	}
	failed := func([]string, *backends.Recipe) error { return errors.New("cmake failed") }
	if offerAcceleratedArchBuildWith(&launchRequest{}, model, nvidia, true, nil, &out, true, "cpu", failed) {
		t.Fatal("a failed build reported success")
	}
}
