package main

import (
	"os"
	"strings"
	"sync"

	"github.com/raketenkater/ggrun/pkg/backends"
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
