package main

import (
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/recommend"
)

// installedBackendPaths lists the backend binaries a default launch can pick:
// APP_HOME builds, backends found on PATH, and registered non-helper forks.
// It only stats files; probing them is the arch check's job.
func installedBackendPaths(caps *detect.Capabilities) []string {
	seen := map[string]bool{}
	var paths []string
	add := func(path string) {
		path = strings.TrimSpace(path)
		if path == "" || seen[path] {
			return
		}
		seen[path] = true
		if st, err := os.Stat(path); err == nil && !st.IsDir() {
			paths = append(paths, path)
		}
	}
	for _, path := range backendSearchPaths(backends.AppHome()) {
		add(path)
	}
	if caps != nil {
		for _, b := range caps.Backends {
			add(b.Path)
		}
	}
	for _, fork := range backends.Load() {
		if !backends.IsHelperOnly(fork) {
			add(fork.Path)
		}
	}
	return paths
}

// archLoadedBy reports whether any of paths loads arch. known is true only
// when every path was probed, so one unreadable backend never turns into a
// "needs a build" claim.
func archLoadedBy(paths []string, arch string, probe backendArchProbe) (loads, known bool) {
	if len(paths) == 0 {
		return false, false
	}
	known = true
	for _, path := range paths {
		supported, probed := probe(path, arch)
		if probed && supported {
			return true, true
		}
		if !probed {
			known = false
		}
	}
	return false, known
}

// backendBuildRoute reports whether a launch builds a backend for an
// architecture no installed backend loads, following unsupportedArchRoutes: a
// reviewed recipe, or a current-mainline build when upstream registers the
// architecture. A configured backend name turns those offers off, and each is
// a source build. Open-PR discovery is not counted: whether it finds a fork
// is unknown until launch, and the code is unreviewed. A mainline build does
// not serve an architecture that requires ik_llama (chooseAutoBackend scores
// it 0), so upstream alone is no route there.
func backendBuildRoute(arch string, upstream, pinned, buildable bool) bool {
	if pinned || !buildable {
		return false
	}
	if len(backends.RecipesForArch(arch)) > 0 || reviewedArchBuildAvailable(arch) {
		return true
	}
	return upstream && backends.RequiredBackendForArch(arch) == ""
}

// buildToolsPresent reports whether the source builds every route runs can
// start here. Windows has none: buildLlamaForkAt expects bin/llama-server,
// which a Windows build names llama-server.exe.
func buildToolsPresent(goos string, onPath func(string) bool) bool {
	if goos == "windows" {
		return false
	}
	return onPath("git") && onPath("cmake") && (onPath("c++") || onPath("g++") || onPath("clang++"))
}

// enableInstalledArchSupport lets recommendations flag models that no
// installed backend loads. Results are cached per architecture for the
// process; installing a backend means a new ggrun process anyway.
func enableInstalledArchSupport() {
	var mu sync.Mutex
	type answer struct{ loads, known bool }
	cache := map[string]answer{}
	recommend.SetInstalledArchSupport(func(caps *detect.Capabilities, arch string) (bool, bool) {
		arch = strings.ToLower(strings.TrimSpace(arch))
		mu.Lock()
		defer mu.Unlock()
		if a, ok := cache[arch]; ok {
			return a.loads, a.known
		}
		loads, known := archLoadedBy(installedBackendPaths(caps), arch, backends.BackendSupportsArch)
		cache[arch] = answer{loads, known}
		return loads, known
	})
	pinned := false
	if cfg, err := config.Load(); err == nil {
		pinned = configuredBackendExplicit(cfg.Backend)
	}
	tools := buildToolsPresent(runtime.GOOS, func(name string) bool {
		_, err := exec.LookPath(name)
		return err == nil
	})
	recommend.SetBackendRoute(func(caps *detect.Capabilities, arch string, upstream bool) bool {
		// An NVIDIA host builds with CUDA (acceleratedArchBuildRecipe), which
		// needs nvcc; defaultAccel looks for it.
		cuda := !hostHasNVIDIAGPU(caps) || defaultAccel() == "cuda"
		return backendBuildRoute(arch, upstream, pinned, tools && cuda)
	})
	var candidates []autoBackendCandidate
	candidatesLoaded := false
	recommend.SetVulkanOnCUDA(func(caps *detect.Capabilities, arch string, moe bool, sizeMB int) bool {
		if !hostHasNVIDIAGPU(caps) {
			return false
		}
		mu.Lock()
		defer mu.Unlock()
		if !candidatesLoaded {
			candidates = autoBackendCandidates(caps, &launchRequest{AppHome: backends.AppHome()})
			candidatesLoaded = true
		}
		return defaultLaunchUsesVulkan(candidates, arch, moe, sizeMB, backends.BackendSupportsArch)
	})
}

// defaultLaunchUsesVulkan mirrors selectBackendForModel for a recommendation
// row: a registered route or sole helper wins first, then chooseAutoBackend
// ranks the installed backends with the row's size (the large-MoE file-backed
// tie-break depends on it).
func defaultLaunchUsesVulkan(candidates []autoBackendCandidate, arch string, moe bool, sizeMB int, probe backendArchProbe) bool {
	arch = strings.ToLower(strings.TrimSpace(arch))
	if arch == "" || backends.ForArch(arch) != nil || backends.SoleHelperForArch(arch) != nil {
		return false
	}
	model := &placement.ModelProfile{ModelArch: arch, IsMoE: moe, TotalSizeMB: sizeMB}
	chosen, _ := chooseAutoBackend(candidates, arch, probe, model)
	return chosen != nil && strings.EqualFold(chosen.Dialect, "vulkan")
}
