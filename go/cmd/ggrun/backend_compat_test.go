package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// fakeArchBackend is a launchable fake llama-server whose binary carries the
// given architecture literals the way a real loader table does (NUL-delimited),
// so BackendSupportsArch probes it as supporting exactly those architectures.
func fakeArchBackend(t *testing.T, dir, name string, archs ...string) string {
	t.Helper()
	body := "#!/bin/sh\necho 'llama server help'\nexit 0\n"
	for _, arch := range archs {
		body += "\x00" + arch + "\x00"
	}
	path := filepath.Join(dir, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// compatHome isolates config and APP_HOME and installs a mainline build at the
// canonical app-home path that carries mainlineArchs only.
func compatHome(t *testing.T, mainlineArchs ...string) (home, mainline string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake backends use shell scripts")
	}
	isolateConfig(t)
	home = t.TempDir()
	t.Setenv("LLM_APP_HOME", home)
	mainline = fakeArchBackend(t, filepath.Join(home, ".bin"), "llama-server", mainlineArchs...)
	return home, mainline
}

// Any spelling of "auto" requests model-aware routing. configuredBackendExplicit
// already compared case-insensitively; the flag parser compared exactly, so
// `--backend Auto` became an explicit pin that skipped the architecture route.
func TestBackendAutoFlagIsCaseInsensitive(t *testing.T) {
	_, _ = compatHome(t)
	fork := fakeArchBackend(t, t.TempDir(), "route-fork", "auditarch")
	if err := backends.Save([]backends.Backend{{Tag: "route-fork", Path: fork, RouteArch: "auditarch"}}); err != nil {
		t.Fatal(err)
	}
	model := &placement.ModelProfile{ModelArch: "auditarch"}
	for _, flags := range [][]string{{"--backend", "Auto"}, {"--backend=AUTO"}, {"--backend", " auto "}} {
		req, err := parseLaunchArgs(append([]string{"model.gguf"}, flags...))
		if err != nil {
			t.Fatal(err)
		}
		if req.BackendExplicit || req.Backend != "auto" {
			t.Fatalf("%q must request automatic selection, got backend=%q explicit=%v", flags, req.Backend, req.BackendExplicit)
		}
		if got := selectBackendForModel(&detect.Capabilities{}, req, model); got == nil || got.Path != fork {
			t.Fatalf("%q did not follow the architecture route: %#v", flags, got)
		}
	}
}

// An explicit --server-bin that does not exist is a failed constraint. It used
// to warn and fall through to findBackend, which also skipped the architecture
// route and launched a binary proven not to load the model.
func TestMissingExplicitServerBinNeverFallsBack(t *testing.T) {
	_, mainline := compatHome(t) // mainline lacks auditarch
	fork := fakeArchBackend(t, t.TempDir(), "route-fork", "auditarch")
	if err := backends.Save([]backends.Backend{{Tag: "route-fork", Path: fork, RouteArch: "auditarch"}}); err != nil {
		t.Fatal(err)
	}
	missing := filepath.Join(t.TempDir(), "gone", "llama-server")
	req, err := parseLaunchArgs([]string{"model.gguf", "--server-bin", missing})
	if err != nil {
		t.Fatal(err)
	}
	if got := selectBackendForModel(&detect.Capabilities{}, req, &placement.ModelProfile{ModelArch: "auditarch"}); got != nil {
		t.Fatalf("missing --server-bin silently switched to %s (mainline is %s)", got.Path, mainline)
	}
	message := backendUnavailableMessage(req)
	if !strings.Contains(message, missing) || !strings.Contains(message, "--server-bin") {
		t.Fatalf("missing --server-bin diagnostic is not actionable: %q", message)
	}
	// It must not look like an automatic refusal: that reason is what arms the
	// fork-search and mainline-update install offers.
	if req.BackendUnavailableReason != "" {
		t.Fatalf("a missing explicit binary armed the automatic install offers: %q", req.BackendUnavailableReason)
	}
}

// A reviewed-install offer is pointless for an explicit choice: the launch keeps
// the pinned backend after the install, so it built a fork and then ignored it.
func TestExplicitBackendIsNotOfferedAReviewedInstall(t *testing.T) {
	_, mainline := compatHome(t)
	var arch string
	for _, recipe := range backends.Recipes() {
		if recipe.RouteArch != "" && !recipe.HelperOnly && backends.RequiredBackendForArch(recipe.RouteArch) == "" {
			arch = recipe.RouteArch
			break
		}
	}
	if arch == "" {
		t.Skip("no recipe-backed architecture without a required backend family")
	}
	be := detectBackend(mainline) // probed: lacks arch
	if supported, probed := backends.BackendSupportsArch(be.Path, arch); !probed || supported {
		t.Fatalf("fixture must be probe-proven unsupported for %s (probed=%v supported=%v)", arch, probed, supported)
	}

	auto, err := parseLaunchArgs([]string{"model.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	if reviewedInstallOffer(auto, arch, be) == nil {
		t.Fatalf("automatic selection on a backend lacking %s must still offer the reviewed recipe", arch)
	}
	for _, flags := range [][]string{{"--backend", "pinned"}, {"--server-bin", mainline}} {
		req, err := parseLaunchArgs(append([]string{"model.gguf"}, flags...))
		if err != nil {
			t.Fatal(err)
		}
		if recipe := reviewedInstallOffer(req, arch, be); recipe != nil {
			t.Fatalf("%v was offered %q although the explicit choice would ignore it", flags, recipe.Name)
		}
	}
}

// A fork registered without --route-arch is still an installed backend. Auto
// selection never looked at the manifest, so a model only that fork loads was
// refused as unsupported.
func TestAutoSelectionConsidersUnroutedInstalledForks(t *testing.T) {
	_, mainline := compatHome(t, "tiearch")
	forkDir := t.TempDir()
	fork := fakeArchBackend(t, forkDir, "fork-server", "auditarch", "tiearch")
	helper := fakeArchBackend(t, forkDir, "helper-server", "helperarch")
	if err := backends.Save([]backends.Backend{
		{Tag: "unrouted-fork", Path: fork},
		{Tag: "helper", Path: helper, HelperOnly: true},
	}); err != nil {
		t.Fatal(err)
	}
	pick := func(arch string) *backendInfo {
		req, err := parseLaunchArgs([]string{"model.gguf"})
		if err != nil {
			t.Fatal(err)
		}
		return selectBackendForModel(&detect.Capabilities{}, req, &placement.ModelProfile{ModelArch: arch})
	}

	if got := pick("auditarch"); got == nil || got.Path != fork {
		t.Fatalf("the only installed backend that loads auditarch was not chosen: %#v", got)
	} else if got.Tag != "unrouted-fork" {
		t.Fatalf("fork selected without its registered tag %q: %q", "unrouted-fork", got.Tag)
	}
	// Equal support: mainline keeps the tie, so an installed fork never displaces
	// it for an architecture mainline already loads.
	if got := pick("tiearch"); got == nil || got.Path != mainline {
		t.Fatalf("a fork displaced mainline on an architecture both load: %#v", got)
	}
	// Helper-only forks keep their metadata-only role.
	if got := pick("helperarch"); got != nil && got.Path == helper {
		t.Fatalf("a helper-only fork became a main-model candidate")
	}
}

// A failed clone or build must come back as an error. addBackendRecipe used to
// os.Exit, so the launch never reached the mainline-update fallback and the TUI
// install action ended ggrun instead of returning to the model.
func TestFailedForkInstallReturnsInsteadOfExiting(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a file:// clone")
	}
	isolateConfig(t)
	t.Setenv("LLM_APP_HOME", t.TempDir())
	bad := backends.Recipe{
		Name: "bad", Tag: "bad-fork", RouteArch: "auditarch",
		GitURL: "file://" + filepath.Join(t.TempDir(), "no-such-repo.git"),
	}
	err := installDiscoveredArchFork(bad)
	if err == nil || !strings.Contains(err.Error(), "clone failed") {
		t.Fatalf("a failed fork clone must return its error, got %v", err)
	}
	if list := backends.Load(); len(list) != 0 {
		t.Fatalf("a failed install registered a backend: %#v", list)
	}

	if err := runTUIBackendAction([]string{"install", "no-such-recipe"}); err == nil {
		t.Fatal("an unknown recipe chosen in the TUI must return an error")
	} else if _, usage := err.(backendUsageError); !usage {
		t.Fatalf("an unknown recipe is a usage error, got %T %v", err, err)
	}
}

// stdin redirected from /dev/null (CI, nohup, cron, service units) is not a
// terminal. The ModeCharDevice test accepted it, so consent prompts printed
// [y/N], read EOF and declined, and a headless launch never printed the exact
// install command it had for an unsupported architecture.
func TestStdinFromDevNullIsNotATerminal(t *testing.T) {
	devNull, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatal(err)
	}
	defer devNull.Close()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()
	saved := os.Stdin
	defer func() { os.Stdin = saved }()
	for name, f := range map[string]*os.File{"/dev/null": devNull, "a pipe": r} {
		os.Stdin = f
		if stdinIsTerminal() {
			t.Errorf("stdin from %s was treated as an interactive terminal", name)
		}
	}
}

// A dry-run allocation probe must get a size-scaled budget: ik_llama's dry run
// still reads and repacks expert weights, and a fixed two minutes failed a
// fresh 148.5 GiB MoE launch closed before the probe finished.
func TestAllocationDryRunProbeScalesWithModelSize(t *testing.T) {
	large := &placement.ModelProfile{SizeBytes: 148 << 30, IsMoE: true}
	if got, want := allocationProbeTimeout(large), autoStartupTimeout(large); got != want || got < 10*time.Minute {
		t.Fatalf("dry-run probe timeout %v for a 148 GiB MoE, want the startup budget %v", got, want)
	}
}

// ggrun's own post-launch clients must reach the backend where it is bound.
func TestBackendBaseURLFollowsTheBoundHost(t *testing.T) {
	for host, want := range map[string]string{
		"":               "http://localhost:18080",
		"0.0.0.0":        "http://localhost:18080",
		"192.168.178.97": "http://192.168.178.97:18080",
		"fd00::5":        "http://[fd00::5]:18080",
	} {
		if got := backendBaseURL(&launchRequest{Host: host, Port: 18080}); got != want {
			t.Errorf("host %q: got %s, want %s", host, got, want)
		}
	}
}
