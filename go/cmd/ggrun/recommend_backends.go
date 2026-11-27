package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/gguf"
	"github.com/raketenkater/ggrun/pkg/models"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/recommend"
	"github.com/raketenkater/ggrun/pkg/tui"
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
// reviewed recipe, or a current-mainline build when mainline registers the
// architecture. Every build is mainline or a reviewed recipe, so an
// architecture only ik_llama registers has no route. A configured backend
// name turns those offers off, and each is a source build. Open-PR discovery
// is not counted: whether it finds a fork is unknown until launch, and the
// code is unreviewed. A mainline build does not serve an architecture that
// requires ik_llama (chooseAutoBackend scores it 0), so mainline alone is no
// route there.
func backendBuildRoute(arch string, mainline, pinned, buildable bool) bool {
	if pinned || !buildable {
		return false
	}
	if len(backends.RecipesForArch(arch)) > 0 || reviewedArchBuildAvailable(arch) {
		return true
	}
	return mainline && backends.RequiredBackendForArch(arch) == ""
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

// buildCanStart reports whether the launch's build can run: the tools, and on
// an NVIDIA host the CUDA toolkit, since acceleratedArchBuildRecipe builds
// with CUDA there and defaultAccel only says "cuda" when nvcc is found.
func buildCanStart(tools, nvidiaHost bool, accel string) bool {
	return tools && (!nvidiaHost || accel == "cuda")
}

// enableInstalledArchSupport lets recommendations flag models that no
// installed backend loads. Results are cached per architecture for the
// process; installing a backend means a new ggrun process anyway.
func enableInstalledArchSupport() {
	// Every caller that probes backends for recommendations also wants the
	// exact precision of catalog files already on disk.
	enableLocalQuantProfiles()
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
	accel := defaultAccel()
	recommend.SetBackendRoute(func(caps *detect.Capabilities, arch string, mainline bool) bool {
		return backendBuildRoute(arch, mainline, pinned, buildCanStart(tools, hostHasNVIDIAGPU(caps), accel))
	})
	enableSharedBackendSupport(pinned, tools, accel)
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

// installedBackendsFor answers the arch probe for the installed backend paths,
// in order, named by their registered tag or detected name (never a path;
// "" when neither exists). It reads only the arch probe (file reads cached on
// disk by file identity); it never starts a process or an encoding probe, so
// every Encodings is nil until such a cache exists. With stopAtLoader it stops
// at the first backend that registers arch, as archLoadedBy does: that one
// already settles the state when no catalog encoding entry needs every loader.
func installedBackendsFor(paths []string, arch string, probe backendArchProbe, names map[string]string, stopAtLoader bool) []backends.InstalledBackend {
	out := make([]backends.InstalledBackend, 0, len(paths))
	for _, path := range paths {
		loads, probed := probe(path, arch)
		out = append(out, backends.InstalledBackend{Name: names[path], Path: path, LoadsArch: loads, Probed: probed})
		if stopAtLoader && loads && probed {
			break
		}
	}
	return out
}

// backendNames maps installed backend paths to a display name: the registered
// tag, else the name detection gave the binary.
func backendNames(caps *detect.Capabilities) map[string]string {
	names := map[string]string{}
	if caps != nil {
		for _, b := range caps.Backends {
			if n := strings.TrimSpace(b.Name); n != "" && !strings.ContainsAny(n, `/\`) {
				names[strings.TrimSpace(b.Path)] = n
			}
		}
	}
	for _, b := range backends.Load() {
		names[strings.TrimSpace(b.Path)] = b.Tag
	}
	return names
}

// rowSupportInput is the resolver input for a recommendation row.
func rowSupportInput(r recommend.Recommendation, installed []backends.InstalledBackend, blocked string) backends.SupportInput {
	return backends.SupportInput{
		Arch:              r.Arch,
		Installed:         installed,
		MainlineRegisters: r.MainlineRegisters(),
		MainlineKnown:     r.RunnableMainline != nil || r.Runnable != nil,
		BuildBlocked:      blocked,
	}
}

// buildBlockedReason says why the first launch cannot start a source build,
// "" exactly when !pinned && buildCanStart(tools, nvidiaHost, accel) on a
// platform where buildToolsPresent can be true.
func buildBlockedReason(goos string, pinned, tools, nvidiaHost bool, accel string) string {
	switch {
	case pinned:
		return "a backend is pinned in the config (set backend to auto to get the build offers back)"
	case goos == "windows":
		return "source builds are not available on Windows"
	case !tools:
		return "a source build needs git, cmake and a C++ compiler on PATH"
	case nvidiaHost && accel != "cuda":
		return "an NVIDIA build needs the CUDA toolkit (nvcc)"
	}
	return ""
}

// enableSharedBackendSupport installs backends.ResolveSupport as the one answer
// for recommendation rows and the TUI model list. Both hooks are read only by
// recommend.TopCategories and the TUI list; no launch path reads them. Answers
// are memoized per architecture for the process (one cmdGUI pass); what they
// read is files only (backends.json, the arch-token cache and the backend
// files it has not seen, the model catalog JSON), no process is started.
func enableSharedBackendSupport(pinned, tools bool, accel string) {
	var mu sync.Mutex
	installed := map[string][]backends.InstalledBackend{}
	mainline := map[string][2]bool{}
	installedFor := func(caps *detect.Capabilities, arch string) []backends.InstalledBackend {
		arch = strings.ToLower(strings.TrimSpace(arch))
		mu.Lock()
		defer mu.Unlock()
		if got, ok := installed[arch]; ok {
			return got
		}
		req := backends.ResolveRequirement(arch, nil)
		stop := req.Recipe == nil && len(req.Unchecked) == 0
		got := installedBackendsFor(installedBackendPaths(caps), arch, backends.BackendSupportsArch, backendNames(caps), stop)
		installed[arch] = got
		return got
	}
	blocked := func(caps *detect.Capabilities) string {
		return buildBlockedReason(runtime.GOOS, pinned, tools, hostHasNVIDIAGPU(caps), accel)
	}
	localTypes := localRowTypes()
	recommend.SetBackendSupport(func(caps *detect.Capabilities, r recommend.Recommendation) backends.Support {
		in := rowSupportInput(r, installedFor(caps, r.Arch), blocked(caps))
		// A remote row's types are unknown. Read a downloaded copy's only
		// when the catalog has an encoding entry for the architecture.
		if len(backends.ResolveRequirement(r.Arch, nil).Unchecked) > 0 {
			in.Types = localTypes(r)
		}
		return backends.ResolveSupport(in)
	})
	tui.SetBackendSupport(func(caps *detect.Capabilities, arch string, types []gguf.TypeUsage) backends.Support {
		key := strings.ToLower(strings.TrimSpace(arch))
		mu.Lock()
		m, ok := mainline[key]
		mu.Unlock()
		if !ok {
			reg, known := recommend.MainlineRegistersArch(key)
			m = [2]bool{reg, known}
			mu.Lock()
			mainline[key] = m
			mu.Unlock()
		}
		return backends.ResolveSupport(backends.SupportInput{
			Arch:              arch,
			Types:             types,
			Installed:         installedFor(caps, arch),
			MainlineRegisters: m[0],
			MainlineKnown:     m[1],
			BuildBlocked:      blocked(caps),
		})
	})
}

// localRowTypes returns the tensor type histogram of a recommendation row's
// downloaded file (matched as enableLocalQuantProfiles matches it), nil when
// none is on disk or it cannot be read.
func localRowTypes() func(recommend.Recommendation) []gguf.TypeUsage {
	cfg, err := config.Load()
	if err != nil || strings.TrimSpace(cfg.ModelDir) == "" {
		return func(recommend.Recommendation) []gguf.TypeUsage { return nil }
	}
	root := cfg.ModelDir
	var (
		once   sync.Once
		listed []models.Model
	)
	return func(r recommend.Recommendation) []gguf.TypeUsage {
		var q recommend.QuantOption
		for _, opt := range r.Quants {
			if strings.EqualFold(opt.Name, r.QuantName) {
				q = opt
				break
			}
		}
		if q.SizeBytes <= 0 {
			return nil
		}
		once.Do(func() {
			done := make(chan []models.Model, 1)
			go func() {
				list, _ := models.List(root)
				done <- list
			}()
			select {
			case listed = <-done:
			case <-time.After(localModelListTimeout):
			}
		})
		path, ok := matchLocalModel(listed, root, q)
		if !ok {
			return nil
		}
		types, err := gguf.ReadTypeHistogramModel(filepath.Clean(path))
		if err != nil {
			return nil
		}
		return types
	}
}
