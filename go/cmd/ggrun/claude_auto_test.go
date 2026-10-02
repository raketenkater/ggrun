package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/advisor"
	"github.com/raketenkater/ggrun/pkg/claudeauto"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestClaudeReviewerGPUCandidatesPreservesLargestGPU(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24564, BandwidthMBps: 15754},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 985},
		{Index: 2, VRAMTotalMB: 12282, BandwidthMBps: 3938},
	}}
	got := claudeReviewerGPUCandidates(caps, &launchRequest{})
	want := []int{1, 2, 0}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestClaudeMainMaxActiveSerializesHostOffload(t *testing.T) {
	req := &launchRequest{ClaudeCode: true}
	for _, strategyType := range []placement.StrategyType{placement.MoEOffload, placement.DenseCPUOffload} {
		strategy := &placement.Strategy{Type: strategyType, Parallel: 4}
		if got := claudeMainMaxActive(req, strategy); got != 1 {
			t.Fatalf("strategy %s max active=%d, want 1", strategyType, got)
		}
	}
}

func TestClaudeMainMaxActiveLeavesGPUResidentParallel(t *testing.T) {
	for _, tc := range []struct {
		req      *launchRequest
		strategy *placement.Strategy
	}{
		{&launchRequest{ClaudeCode: true}, &placement.Strategy{Type: placement.MultiGPUDense, Parallel: 4}},
		{&launchRequest{}, &placement.Strategy{Type: placement.MoEOffload, Parallel: 4}},
	} {
		if got := claudeMainMaxActive(tc.req, tc.strategy); got != 0 {
			t.Fatalf("unexpected admission cap %d for req=%+v strategy=%+v", got, tc.req, tc.strategy)
		}
	}
}

// A single slot used to return 0, and 0 means "build no scheduler at all" --
// so the one configuration that most needs ordering got none: lane priority,
// affinity and aging all off, with llama.cpp queueing the fan-out FIFO. One
// slot serves one request either way; the limit is what keeps a permission
// review from waiting behind bulk work.
func TestClaudeMainMaxActiveStillSchedulesAtOneSlot(t *testing.T) {
	for _, strategyType := range []placement.StrategyType{placement.MoEOffload, placement.MultiGPUDense} {
		strategy := &placement.Strategy{Type: strategyType, Parallel: 1}
		if got := claudeMainMaxActive(&launchRequest{ClaudeCode: true}, strategy); got != 1 {
			t.Errorf("strategy %s at one slot: max active=%d, want 1", strategyType, got)
		}
	}
}

// The flag is how the serialized default gets tested against real concurrency,
// so it has to actually override -- and it must never exceed the slot count,
// because admitting more than there are slots moves the queue into llama.cpp
// rather than removing it.
func TestClaudeMaxActiveOverrideIsClampedToSlots(t *testing.T) {
	moe := func(parallel int) *placement.Strategy {
		return &placement.Strategy{Type: placement.MoEOffload, Parallel: parallel}
	}
	req := func(limit int) *launchRequest {
		return &launchRequest{ClaudeCode: true, ClaudeMaxActive: limit, ClaudeMaxActiveSet: true}
	}
	for _, tc := range []struct {
		name     string
		req      *launchRequest
		strategy *placement.Strategy
		want     int
	}{
		{"override raises the host-offload default", req(4), moe(4), 4},
		{"clamped to the available slots", req(8), moe(4), 4},
		{"zero is an explicit opt out", req(0), moe(4), 0},
		{"override lowers below the default", req(1), moe(4), 1},
		{"unset keeps the measured default", &launchRequest{ClaudeCode: true}, moe(4), 1},
	} {
		if got := claudeMainMaxActive(tc.req, tc.strategy); got != tc.want {
			t.Errorf("%s: max active=%d, want %d", tc.name, got, tc.want)
		}
	}
}

func TestClaudeReviewerGPUCandidatesKeepSparsePhysicalSelection(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0}, {Index: 1}, {Index: 2}}}
	got := claudeReviewerGPUCandidates(caps, &launchRequest{GPUsFlag: "2,1,2,9"})
	want := []int{2, 1}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("got %v, want physical selection %v", got, want)
	}
}

func TestClaudeReviewerArgsUsesIsolatedDeviceAsLocalMain(t *testing.T) {
	args := claudeReviewerArgs("server", "reviewer.gguf", 1234, "CUDA7", "--reasoning ARG --cache-type-k TYPE --cache-type-v TYPE")
	for _, want := range []string{"--device", "CUDA7", "-mg", "0", "--reasoning", "off", "--ctx-size", "131072", "--cache-type-k", "q8_0", "--cache-type-v"} {
		if !hasArg(args, want) {
			t.Fatalf("missing %q in %v", want, args)
		}
	}
	for _, flag := range []string{"--cache-type-k", "--cache-type-v"} {
		if !hasArgValue(args, flag, "q8_0") {
			t.Fatalf("expected %s q8_0 in %v", flag, args)
		}
	}
}

func TestReviewerProfilesScopeCapacityToTheirContext(t *testing.T) {
	qwen := resolveClaudeCompanionProfile(&launchRequest{
		AppHome: t.TempDir(), ClaudeReviewerOverride: claudeReviewerQwen,
	}, t.TempDir())
	if qwen.ContextTokens != 131072 || qwen.ReservationVRAMMB != claudeReviewerReservationVRAMMB ||
		!strings.Contains(qwen.companionMeasurementKey(), "ctx131072") {
		t.Fatalf("Qwen4B reviewer capacity is not context-scoped: %+v", qwen)
	}

	small := resolveClaudeCompanionProfile(&launchRequest{
		AppHome: t.TempDir(), ClaudeReviewerOverride: claudeReviewerQwen2B,
	}, t.TempDir())
	if small.ContextTokens != 262144 || small.ReservationVRAMMB != claudeSmallReviewerReservationVRAMMB ||
		!strings.Contains(small.companionMeasurementKey(), "ctx262144") {
		t.Fatalf("Qwen2B review-only capacity is not full-context scoped: %+v", small)
	}
	args := claudeReviewerArgsWithContextAndKV(
		"server", "reviewer.gguf", 1234, "CUDA0",
		"--cache-type-k TYPE --cache-type-v TYPE", "q8_0", "", small.ContextTokens,
	)
	if !hasArgValue(args, "--ctx-size", "262144") {
		t.Fatalf("small reviewer did not receive its planned context: %v", args)
	}
}

func TestClaudeNanoReviewerArgsUsesQ4KV(t *testing.T) {
	args := claudeReviewerArgsWithKV(
		"server", "nanbeige.gguf", 1234, "CUDA0",
		"--cache-type-k TYPE --cache-type-v TYPE", "q4_0", "",
	)
	for _, flag := range []string{"--cache-type-k", "--cache-type-v"} {
		if !hasArgValue(args, flag, "q4_0") {
			t.Fatalf("expected %s q4_0 in %v", flag, args)
		}
	}
}

func TestClaudeReviewerArgsKeepsOlderBackendCompatibility(t *testing.T) {
	args := claudeReviewerArgs("server", "reviewer.gguf", 1234, "", "--reasoning ARG")
	for _, unsupported := range []string{"--cache-type-k", "--cache-type-v"} {
		if hasArg(args, unsupported) {
			t.Fatalf("unexpected unsupported %q in %v", unsupported, args)
		}
	}
}

func TestClaudeNanoReviewerArgsOverridesNanbeigeTemplate(t *testing.T) {
	cacheDir := t.TempDir()
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "Nanbeige4.2-3B-Q4_K_M.gguf")
	writeGGUFWithTemplate(t, modelPath, "nanbeige", "<broken raise_exception('System message must be at the beginning.')>")
	bin := filepath.Join(dir, "build-cuda", "bin", "llama-server")
	if err := os.MkdirAll(filepath.Dir(bin), 0o755); err != nil {
		t.Fatal(err)
	}
	args := claudeReviewerArgsWithKV(
		bin, modelPath, 1234, "CUDA0",
		"--cache-type-k TYPE --cache-type-v TYPE --reasoning ARG", "q4_0", cacheDir,
	)
	got := valueAfter(args, "--chat-template-file")
	if got == "" || !strings.HasSuffix(got, ".jinja") {
		t.Fatalf("nanbeige reviewer args must include --chat-template-file with a .jinja path, got %v", args)
	}
	data, err := os.ReadFile(got)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		t.Fatal("materialized reviewer template is empty")
	}
	// A reviewer with no catalog entry (unknown basename, no arch probe) must not
	// receive the override.
	otherPath := filepath.Join(dir, "reviewer.gguf")
	writeGGUFWithTemplate(t, otherPath, "some-arch", "<broken raise_exception>")
	plain := claudeReviewerArgs(bin, otherPath, 1234, "", "--reasoning ARG")
	if hasArg(plain, "--chat-template-file") {
		t.Fatalf("non-catalog reviewer must not receive template override, got %v", plain)
	}
}

func TestClaudeReviewerGPUDeviceUsesAdvertisedName(t *testing.T) {
	dir := t.TempDir()
	binary := filepath.Join(dir, "llama-server")
	script := "#!/bin/sh\nprintf 'Available devices:\\n  CUDA3: Test GPU\\n'\n"
	if err := os.WriteFile(binary, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	got, err := claudeReviewerGPUDevice(binary, []string{"CUDA_VISIBLE_DEVICES=2"})
	if err != nil {
		t.Fatal(err)
	}
	if got != "CUDA3" {
		t.Fatalf("got %q, want backend-advertised CUDA3", got)
	}
}

func TestClaudeReviewerGPUDeviceRejectsBackendWithoutCUDA(t *testing.T) {
	dir := t.TempDir()
	binary := filepath.Join(dir, "llama-server")
	script := "#!/bin/sh\nprintf 'Available devices:\\n  Vulkan0: Test GPU\\n'\n"
	if err := os.WriteFile(binary, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	if _, err := claudeReviewerGPUDevice(binary, nil); err == nil || !strings.Contains(err.Error(), "no CUDA device") {
		t.Fatalf("expected clear missing-CUDA error, got %v", err)
	}
}

// A fresh NVIDIA install runs the reviewer on the mainline Vulkan server, which
// ignores CUDA_VISIBLE_DEVICES and orders cards its own way.
func TestClaudeReviewerDeviceForGPUMatchesVulkanByName(t *testing.T) {
	dir := t.TempDir()
	binary := filepath.Join(dir, "llama-server")
	script := "#!/bin/sh\nprintf 'Available devices:\\n  Vulkan0: NVIDIA GeForce RTX 3090 Ti (24810 MiB, 24352 MiB free)\\n  Vulkan1: NVIDIA GeForce RTX 4070 (12528 MiB, 12100 MiB free)\\n  Vulkan2: NVIDIA GeForce RTX 3060 (12534 MiB, 12149 MiB free)\\n'\n"
	if err := os.WriteFile(binary, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	env := []string{"CUDA_VISIBLE_DEVICES=2"}
	got, err := claudeReviewerDeviceForGPU(binary, env, &detect.GPU{Index: 2, Name: "NVIDIA GeForce RTX 3060"})
	if err != nil || got != "Vulkan2" {
		t.Fatalf("got %q, %v; want Vulkan2", got, err)
	}
	if got, _ := claudeReviewerDeviceForGPU(binary, env, &detect.GPU{Index: 0, Name: "NVIDIA GeForce RTX 4070"}); got != "Vulkan1" {
		t.Fatalf("physical GPU 0 mapped to %q, want Vulkan1", got)
	}
	if _, err := claudeReviewerDeviceForGPU(binary, env, &detect.GPU{Index: 1, Name: "Other GPU"}); err == nil {
		t.Fatal("an unmatched GPU must not get a Vulkan device")
	}
	twin := filepath.Join(dir, "twin")
	script = "#!/bin/sh\nprintf 'Available devices:\\n  Vulkan0: RTX 3090 (24576 MiB)\\n  Vulkan1: RTX 3090 (24576 MiB)\\n'\n"
	if err := os.WriteFile(twin, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	if _, err := claudeReviewerDeviceForGPU(twin, env, &detect.GPU{Index: 1, Name: "RTX 3090"}); err == nil || !strings.Contains(err.Error(), "cannot tell") {
		t.Fatalf("identical cards must be refused, got %v", err)
	}
}

func TestFindClaudeReviewerBackendSkipsVulkanForCUDA(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("LLM_APP_HOME", "")
	for _, tc := range []struct {
		path    string
		devices string
	}{
		{filepath.Join(home, "llama.cpp", "build-vulkan", "bin", "llama-server"), "Vulkan0: Test GPU"},
		{filepath.Join(home, "llama.cpp", "build", "bin", "llama-server"), "CUDA0: Test GPU"},
	} {
		if err := os.MkdirAll(filepath.Dir(tc.path), 0755); err != nil {
			t.Fatal(err)
		}
		script := "#!/bin/sh\nif [ \"$1\" = --help ]; then printf '%s\\n' '--reasoning ARG'; else printf 'Available devices:\\n  %s\\n' '" + tc.devices + "'; fi\n"
		if err := os.WriteFile(tc.path, []byte(script), 0755); err != nil {
			t.Fatal(err)
		}
	}
	got := findClaudeReviewerBackend(nil)
	want := filepath.Join(home, "llama.cpp", "build", "bin", "llama-server")
	if got == nil || got.Path != want {
		t.Fatalf("got %#v, want CUDA backend %q", got, want)
	}
}

func TestClaudeReviewerCPUFallbackHidesAccelerators(t *testing.T) {
	got := claudeReviewerCPUEnv()
	for _, want := range []string{"CUDA_VISIBLE_DEVICES=-1", "HIP_VISIBLE_DEVICES=-1", "ROCR_VISIBLE_DEVICES=-1"} {
		if !hasArg(got, want) {
			t.Fatalf("missing %q in %v", want, got)
		}
	}
}

func TestClaudeReviewerBackendEnvAddsResolvedLibraryPath(t *testing.T) {
	root := t.TempDir()
	binDir := filepath.Join(root, "build-cuda", "bin")
	linkDir := filepath.Join(root, ".bin")
	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(linkDir, 0755); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(binDir, "llama-server")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\n"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(binDir, "libllama-server-impl.so"), []byte("lib"), 0644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(linkDir, "llama-server-cuda")
	if err := os.Symlink(binary, link); err != nil {
		t.Fatal(err)
	}
	got := claudeReviewerBackendEnv(link, []string{"CUDA_VISIBLE_DEVICES=2"})
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "CUDA_VISIBLE_DEVICES=2") {
		t.Fatalf("reviewer env lost GPU isolation: %v", got)
	}
	if !strings.Contains(joined, "LD_LIBRARY_PATH="+binDir) {
		t.Fatalf("reviewer env missing resolved backend lib dir %q: %v", binDir, got)
	}
}

func TestClaudeAutoReviewerNeededDefaultsOnForAuto(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "")
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	if !claudeAutoReviewerNeeded(nil) {
		t.Fatal("default local Auto launch must start its reviewer")
	}
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "acceptEdits")
	if claudeAutoReviewerNeeded(nil) {
		t.Fatal("non-Auto permission mode should not spend memory on a reviewer")
	}
	if !claudeCompanionNeeded(nil) {
		t.Fatal("non-Auto permission mode still needs the cheap-tier worker")
	}
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "off")
	if claudeCompanionNeeded(nil) {
		t.Fatal("the explicit companion disable switch must remain authoritative")
	}
}

func TestClaudeReviewerReservationBuildsCompanion(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "")
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24564, BandwidthMBps: 15754},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 985},
	}}
	res := claudeReviewerReservation(&launchRequest{ClaudeCode: true}, caps, "")
	if res == nil {
		t.Fatal("Claude Code launch with GPUs must reserve the reviewer")
	}
	if res.Name != claudeReviewerCompanionName {
		t.Fatalf("companion name = %q, want %q", res.Name, claudeReviewerCompanionName)
	}
	if res.VRAMMB <= 0 {
		t.Fatalf("reservation must carry a positive VRAM footprint, got %d", res.VRAMMB)
	}
	if res.AllowCPU {
		t.Fatal("companion CPU fallback must not bypass the placement RAM ledger")
	}
	// Preference order mirrors the legacy walk: slow GPU first, main last.
	if len(res.GPUPreference) != 2 || res.GPUPreference[0] != 1 || res.GPUPreference[1] != 0 {
		t.Fatalf("GPU preference = %v, want [1 0]", res.GPUPreference)
	}
}

// TestClaudeCompanionPrefersVerifiedNanoAndFreezesChoice verifies the explicit
// --claude-reviewer nanbeige opt-in: with a verified NanoBeige artifact and a
// GPU-capable backend, the forced selection returns the NanoBeige worker profile
// (Q4 KV, its own measurement key), and the choice freezes once made.
func TestClaudeCompanionPrefersVerifiedNanoAndFreezesChoice(t *testing.T) {
	cacheDir := t.TempDir()
	restoreReady, restoreBackend, restoreGPU := claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable
	defer func() {
		claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable = restoreReady, restoreBackend, restoreGPU
	}()
	claudeNanoArtifactReady = func(path string) bool {
		return path == advisor.DefaultModelPath(cacheDir)
	}
	claudeNanoBackend = func() (string, error) { return "/reviewed/nanbeige/llama-server", nil }
	claudeNanoGPUCapable = func(string) bool { return true }
	req := &launchRequest{AppHome: t.TempDir(), ClaudeReviewerOverride: claudeReviewerNanbeige}
	profile := resolveClaudeCompanionProfile(req, cacheDir)
	if !profile.NanoBeige || profile.Name != claudeNanoCompanionName || profile.BackendPath == "" || profile.KVType != "q4_0" {
		t.Fatalf("verified NanoBeige pair was not selected: %+v", profile)
	}
	if profile.companionMeasurementKey() == profile.Name {
		t.Fatal("NanoBeige Q4 profile reused the historical Q8 VRAM measurement key")
	}
	// State appearing or disappearing after placement must not change the seat.
	claudeNanoArtifactReady = func(string) bool { return false }
	if again := resolveClaudeCompanionProfile(req, cacheDir); again != profile {
		t.Fatalf("companion changed after launch choice: before=%p after=%p", profile, again)
	}
	// Auto (no explicit override) must NOT pick NanoBeige even when it is ready:
	// Nano is a --claude-reviewer nanbeige opt-in, never the automatic choice.
	autoReq := &launchRequest{AppHome: t.TempDir(), ClaudeReviewerOverride: claudeReviewerAuto}
	autoProfile := resolveClaudeCompanionProfile(autoReq, cacheDir)
	if autoProfile.NanoBeige || autoProfile.Name != claudeReviewerCompanionName {
		t.Fatalf("auto must resolve to the Qwen4B worker/reviewer, not NanoBeige: %+v", autoProfile)
	}
}

func TestClaudeCompanionRejectsCPUOnlyNanoBackend(t *testing.T) {
	restoreReady, restoreBackend, restoreGPU := claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable
	defer func() {
		claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable = restoreReady, restoreBackend, restoreGPU
	}()
	claudeNanoArtifactReady = func(string) bool { return true }
	claudeNanoBackend = func() (string, error) { return "/reviewed/nanbeige/llama-server", nil }
	claudeNanoGPUCapable = func(string) bool { return false }
	profile := resolveClaudeCompanionProfile(&launchRequest{AppHome: t.TempDir()}, t.TempDir())
	if profile.NanoBeige || profile.Name != claudeReviewerCompanionName {
		t.Fatalf("CPU-only helper backend received a GPU worker profile: %+v", profile)
	}
}

func TestClaudeCompanionExplicitModelOverrideWinsNano(t *testing.T) {
	custom := filepath.Join(t.TempDir(), "custom.gguf")
	t.Setenv("GGRUN_CLAUDE_REVIEWER_MODEL", custom)
	restoreReady, restoreBackend := claudeNanoArtifactReady, claudeNanoBackend
	defer func() {
		claudeNanoArtifactReady, claudeNanoBackend = restoreReady, restoreBackend
	}()
	claudeNanoArtifactReady = func(string) bool { return true }
	claudeNanoBackend = func() (string, error) { return "/reviewed/nanbeige/llama-server", nil }
	profile := resolveClaudeCompanionProfile(&launchRequest{AppHome: t.TempDir()}, t.TempDir())
	if profile.NanoBeige || profile.ModelPath != custom || profile.Name != claudeReviewerCompanionName {
		t.Fatalf("explicit reviewer override lost to automatic Nano selection: %+v", profile)
	}
	if profile.ServesWorkers {
		t.Fatal("an arbitrary reviewer model override advertised unverified worker capability")
	}
}

func TestClaudeCompanionExplicitBackendOverrideWinsNano(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_REVIEWER_BIN", filepath.Join(t.TempDir(), "custom-server"))
	restoreReady, restoreBackend := claudeNanoArtifactReady, claudeNanoBackend
	defer func() {
		claudeNanoArtifactReady, claudeNanoBackend = restoreReady, restoreBackend
	}()
	claudeNanoArtifactReady = func(string) bool { return true }
	claudeNanoBackend = func() (string, error) { return "/reviewed/nanbeige/llama-server", nil }
	profile := resolveClaudeCompanionProfile(&launchRequest{AppHome: t.TempDir()}, t.TempDir())
	if profile.NanoBeige || profile.Name != claudeReviewerCompanionName {
		t.Fatalf("explicit reviewer backend lost to automatic Nano selection: %+v", profile)
	}
	if !profile.ServesWorkers {
		t.Fatal("a backend-only override disabled the pinned 4B model's worker capability")
	}
}

func TestClaudeReviewerReservationSkipsNonClaudeAndCPU(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "")
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}}
	if res := claudeReviewerReservation(&launchRequest{}, caps, ""); res != nil {
		t.Fatal("non-Claude launch must not reserve a reviewer")
	}
	if res := claudeReviewerReservation(&launchRequest{ClaudeCode: true, CPUMode: true}, caps, ""); res != nil {
		t.Fatal("CPU-mode launch must not reserve GPU VRAM for the reviewer")
	}
	if res := claudeReviewerReservation(&launchRequest{ClaudeCode: true}, &detect.Capabilities{}, ""); res != nil {
		t.Fatal("GPU-less host must not reserve GPU VRAM for the reviewer")
	}
	if res := claudeReviewerReservation(&launchRequest{ClaudeCode: true, ClaudeReviewerDisabled: true}, caps, ""); res != nil {
		t.Fatal("main-model review fallback must not reserve a separate reviewer")
	}
}

func TestStartClaudeReviewerSkipsMainModelFallback(t *testing.T) {
	runtime, err := startClaudeAutoReviewer(&launchRequest{ClaudeCode: true, ClaudeReviewerDisabled: true}, nil, nil, nil)
	if err != nil || runtime != nil {
		t.Fatalf("main-model review fallback started a separate reviewer: runtime=%#v err=%v", runtime, err)
	}
}

// A reviewer left running by a previous ggrun is already inside the VRAM the
// hardware scan reports as used. Adding the full reservation on top charges one
// process twice: measured here as 2096 MiB resident plus a 2600 MiB reservation,
// ~4.7 GB withheld for a 2.1 GB helper. On the 12 GB card that seat is on, and
// at 1371 MiB per expert layer, it cost two layers -- the card took 4 where an
// unoccupied one took 6.
func TestReviewerReservationIsNotChargedTwice(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24564, BandwidthMBps: 15754},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 985},
	}}
	req := &launchRequest{ClaudeCode: true}
	restore := residentReviewerVRAM
	defer func() { residentReviewerVRAM = restore }()

	// Nothing resident: the full bound is reserved.
	residentReviewerVRAM = func(string) int { return 0 }
	got := claudeReviewerReservation(req, caps, "")
	if got == nil || got.VRAMMB != claudeReviewerReservationVRAMMB {
		t.Fatalf("with no reviewer running, reservation = %+v, want %d MiB", got, claudeReviewerReservationVRAMMB)
	}

	// A stored measurement supersedes the constant. The Qwen3.5-4B profile
	// records under its own MeasurementKey so it never inherits the 2B footprint
	// stored under the legacy companion name.
	dir := t.TempDir()
	key := resolveClaudeCompanionProfile(req, dir).companionMeasurementKey()
	if key == claudeReviewerCompanionName {
		t.Fatal("Qwen4B profile must not measure under the legacy 2B key")
	}
	if err := placement.RecordCompanionVRAM(dir, key, 2096); err != nil {
		t.Fatal(err)
	}
	if got := claudeReviewerReservation(req, caps, dir); got == nil || got.VRAMMB != 2096 {
		t.Errorf("measured reservation = %+v, want 2096 MiB", got)
	}

	// A leftover reviewer already occupies that VRAM, so nothing more is owed.
	residentReviewerVRAM = func(string) int { return 2096 }
	if got := claudeReviewerReservation(req, caps, dir); got != nil {
		t.Errorf("reservation = %+v, want none: the seat is already occupied", got)
	}
	// A partially covered seat reserves only the difference.
	residentReviewerVRAM = func(string) int { return 1500 }
	if got := claudeReviewerReservation(req, caps, dir); got == nil || got.VRAMMB != 596 {
		t.Errorf("reservation = %+v, want the uncovered 596 MiB", got)
	}
}

// --parallel is an instruction. Every slot costs context whether or not it is
// ever used, so allocating the slots and then admitting one request is the one
// outcome nobody asked for: observed live as two 131072 slots, one permanently
// idle, at the throughput a single 262144 slot already delivered.
func TestExplicitParallelRaisesAdmissionOnHostOffload(t *testing.T) {
	req := &launchRequest{ClaudeCode: true, ParallelSet: true}
	for _, strategyType := range []placement.StrategyType{placement.MoEOffload, placement.DenseCPUOffload} {
		strategy := &placement.Strategy{Type: strategyType, Parallel: 2}
		if got := claudeMainMaxActive(req, strategy); got != 2 {
			t.Fatalf("strategy %s with explicit --parallel 2: max active=%d, want 2", strategyType, got)
		}
	}
}

// The conservative default still stands when ggrun picked the slot count.
func TestImplicitParallelKeepsHostOffloadSerialized(t *testing.T) {
	req := &launchRequest{ClaudeCode: true}
	strategy := &placement.Strategy{Type: placement.MoEOffload, Parallel: 4}
	if got := claudeMainMaxActive(req, strategy); got != 1 {
		t.Fatalf("max active=%d, want the conservative default of 1", got)
	}
}

func TestMeasuredHostOffloadAdmitsOnlyValidatedWorkloadDemand(t *testing.T) {
	req := &launchRequest{ClaudeCode: true, Parallel: 1}
	strategy := &placement.Strategy{Type: placement.MoEOffload, Parallel: 4, PerformanceTuned: true}
	if got := claudeMainMaxActive(req, strategy); got != 2 {
		t.Fatalf("measured four-slot strategy admitted %d requests, want the validated two-agent workload", got)
	}
}

func TestPendingWorkflowEvidenceAppliesMeasuredAdmissionBeforePersistence(t *testing.T) {
	req := &launchRequest{ClaudeCode: true, Parallel: 1}
	strategy := &placement.Strategy{Type: placement.MoEOffload, Parallel: 2}
	decision := &placement.CalibrationDecision{
		Finalist: "parallel-2", FinalistOutcome: "promoted",
		DefaultAgentSamples: 2, WinnerAgentSamples: 2,
		DefaultTurnTimeS: 10, WinnerTurnTimeS: 8,
		DefaultTurnMaxS: 11, WinnerTurnMaxS: 9,
		AgentPromptBytes:    4096,
		DefaultCachedTokens: 100, WinnerCachedTokens: 100,
		DefaultNewPromptTokens: 10, WinnerNewPromptTokens: 10,
		DefaultMixedTPS: 1, WinnerMixedTPS: 1,
	}
	if got := claudeMainMaxActive(req, strategy, decision); got != 2 {
		t.Fatalf("pending measured p2 admitted %d requests, want 2", got)
	}
	decision.FinalistOutcome = "unavailable"
	if got := claudeMainMaxActive(req, strategy, decision); got != 1 {
		t.Fatalf("admission-only evidence admitted %d requests, want conservative p1", got)
	}
}

// An explicit --claude-max-active is more specific than --parallel and wins.
func TestExplicitMaxActiveOverridesExplicitParallel(t *testing.T) {
	req := &launchRequest{ClaudeCode: true, ParallelSet: true, ClaudeMaxActive: 1, ClaudeMaxActiveSet: true}
	strategy := &placement.Strategy{Type: placement.MoEOffload, Parallel: 4}
	if got := claudeMainMaxActive(req, strategy); got != 1 {
		t.Fatalf("max active=%d, want the explicit 1", got)
	}
}

// "--reasoning" must not be satisfied by "--reasoning-format": mainline rejects
// `--reasoning off` and an unknown flag kills the reviewer at startup. Recent
// mainline disables thinking with --reasoning-budget 0 instead.
func TestClaudeReviewerArgsMatchReasoningFlagsExactly(t *testing.T) {
	legacy := strings.Join(claudeReviewerArgs("bin", "m.gguf", 1, "", "--reasoning FMT --cache-type-k T"), " ")
	if !strings.Contains(legacy, "--reasoning off") {
		t.Errorf("binary with exact --reasoning must get `--reasoning off`: %s", legacy)
	}
	mainline := strings.Join(claudeReviewerArgs("bin", "m.gguf", 1, "", "--reasoning-format FMT --reasoning-budget N"), " ")
	if strings.Contains(mainline, "--reasoning off") {
		t.Errorf("substring match sent a fatal flag to a mainline binary: %s", mainline)
	}
	if !strings.Contains(mainline, "--reasoning-budget 0") {
		t.Errorf("mainline binary must disable thinking via --reasoning-budget 0: %s", mainline)
	}
}

// A reviewer model only a fork can load needs the binary named alongside it.
func TestReviewerBinaryOverrideWinsResolution(t *testing.T) {
	dir := t.TempDir()
	stub := filepath.Join(dir, "fork-server")
	script := "#!/bin/sh\ncase \"$1\" in\n--help) echo '--reasoning-budget N --cache-type-k T' ;;\n--list-devices) echo 'CUDA0: stub' ;;\nesac\nexit 0\n"
	if err := os.WriteFile(stub, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GGRUN_CLAUDE_REVIEWER_BIN", stub)
	be := findClaudeReviewerBackend(&detect.Capabilities{})
	if be == nil || be.Path != stub {
		t.Fatalf("GGRUN_CLAUDE_REVIEWER_BIN was not honoured: %+v", be)
	}
}

func TestClaudeReviewerFlagQwenForcesQwenProfile(t *testing.T) {
	restoreReady, restoreBackend, restoreGPU := claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable
	defer func() {
		claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable = restoreReady, restoreBackend, restoreGPU
	}()
	// NanoBeige is fully available — auto would pick it — but --claude-reviewer
	// qwen must force the historical Qwen profile regardless.
	claudeNanoArtifactReady = func(string) bool { return true }
	claudeNanoBackend = func() (string, error) { return "/reviewed/nanbeige/llama-server", nil }
	claudeNanoGPUCapable = func(string) bool { return true }
	req := &launchRequest{AppHome: t.TempDir(), ClaudeReviewerOverride: claudeReviewerQwen}
	profile := resolveClaudeCompanionProfile(req, t.TempDir())
	if profile.NanoBeige || profile.Name != claudeReviewerCompanionName {
		t.Fatalf("--claude-reviewer qwen did not force the Qwen profile: %+v", profile)
	}
	if profile.KVType != "q8_0" {
		t.Fatalf("--claude-reviewer qwen must keep the Qwen Q8 KV profile, got %q", profile.KVType)
	}
}

func TestClaudeReviewerFlagNanbeigeForcesNanoProfile(t *testing.T) {
	restoreReady, restoreBackend, restoreGPU := claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable
	defer func() {
		claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable = restoreReady, restoreBackend, restoreGPU
	}()
	claudeNanoArtifactReady = func(string) bool { return true }
	claudeNanoBackend = func() (string, error) { return "/reviewed/nanbeige/llama-server", nil }
	claudeNanoGPUCapable = func(string) bool { return true }
	req := &launchRequest{AppHome: t.TempDir(), ClaudeReviewerOverride: claudeReviewerNanbeige}
	profile := resolveClaudeCompanionProfile(req, t.TempDir())
	if !profile.NanoBeige || profile.Name != claudeNanoCompanionName {
		t.Fatalf("--claude-reviewer nanbeige did not force the NanoBeige profile: %+v", profile)
	}
}

func TestClaudeReviewerFlagNanbeigeFallsBackWhenArtifactMissing(t *testing.T) {
	restoreReady, restoreBackend, restoreGPU := claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable
	defer func() {
		claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable = restoreReady, restoreBackend, restoreGPU
	}()
	claudeNanoArtifactReady = func(string) bool { return false }
	claudeNanoBackend = func() (string, error) { return "/reviewed/nanbeige/llama-server", nil }
	claudeNanoGPUCapable = func(string) bool { return true }
	req := &launchRequest{AppHome: t.TempDir(), ClaudeReviewerOverride: claudeReviewerNanbeige}
	profile := resolveClaudeCompanionProfile(req, t.TempDir())
	if profile.NanoBeige || profile.Name != claudeReviewerCompanionName {
		t.Fatalf("forced NanoBeige without an installed artifact must degrade to Qwen: %+v", profile)
	}
}

// TestClaudeReviewerQwenProfileResolvesQwen4B verifies the forced-Qwen profile
// carries the Qwen3.5-4B display name and a model path that resolves to the 4B
// artifact (the pinned reviewer cache path, and the local model directory when
// the artifact is installed there).
func TestClaudeReviewerQwenProfileResolvesQwen4B(t *testing.T) {
	restoreReady, restoreBackend, restoreGPU := claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable
	defer func() {
		claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable = restoreReady, restoreBackend, restoreGPU
	}()
	claudeNanoArtifactReady = func(string) bool { return false }
	req := &launchRequest{AppHome: t.TempDir(), ClaudeReviewerOverride: claudeReviewerQwen}
	profile := resolveClaudeCompanionProfile(req, t.TempDir())
	if profile.NanoBeige || profile.Name != claudeReviewerCompanionName {
		t.Fatalf("qwen override must force the Qwen profile: %+v", profile)
	}
	if profile.DisplayName != claudeauto.DefaultReviewerDisplayName {
		t.Fatalf("Qwen reviewer display name = %q, want %q", profile.DisplayName, claudeauto.DefaultReviewerDisplayName)
	}
	if profile.DisplayName != "Qwen3.5-4B" {
		t.Fatalf("Qwen reviewer display name = %q, want Qwen3.5-4B", profile.DisplayName)
	}
	// The reviewer model path resolves to the pinned 4B cache artifact.
	wantPath := filepath.Join(req.AppHome, ".cache", "claude-reviewer", claudeauto.DefaultReviewerFile)
	if profile.ModelPath != wantPath {
		t.Fatalf("Qwen reviewer path = %q, want %q", profile.ModelPath, wantPath)
	}
	// When the artifact is already installed in the model directory, the local
	// model dir lookup resolves it without a download.
	modelDir := t.TempDir()
	sub := filepath.Join(modelDir, claudeauto.DefaultReviewerLocalDir)
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	localPath := filepath.Join(sub, claudeauto.DefaultReviewerFile)
	if err := os.WriteFile(localPath, []byte("GGUF fake"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, ok := claudeauto.LocalReviewerModelPath(modelDir)
	if !ok || got != localPath {
		t.Fatalf("local 4B lookup = %q ok=%v, want %q", got, ok, localPath)
	}
}

// TestClaudeReviewerFlagAutoResolvesQwen4B pins the corrected default: auto
// (and empty) must resolve to the Qwen3.5-4B worker/reviewer — the big/fast
// dense companion that handles BOTH safety reviews and cheap-tier worker calls —
// even when a verified NanoBeige artifact and backend are present. NanoBeige is
// now an explicit --claude-reviewer nanbeige opt-in, never the automatic choice.
func TestClaudeReviewerFlagAutoResolvesQwen4B(t *testing.T) {
	restoreReady, restoreBackend, restoreGPU := claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable
	defer func() {
		claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable = restoreReady, restoreBackend, restoreGPU
	}()
	claudeNanoArtifactReady = func(string) bool { return true }
	claudeNanoBackend = func() (string, error) { return "/reviewed/nanbeige/llama-server", nil }
	claudeNanoGPUCapable = func(string) bool { return true }
	// auto must resolve to the Qwen4B worker/reviewer even when Nano is ready.
	autoReq := &launchRequest{AppHome: t.TempDir(), ClaudeReviewerOverride: claudeReviewerAuto}
	profile := resolveClaudeCompanionProfile(autoReq, t.TempDir())
	if profile.NanoBeige || profile.Name != claudeReviewerCompanionName {
		t.Fatalf("auto must resolve to the Qwen4B worker/reviewer: %+v", profile)
	}
	if profile.DisplayName != "Qwen3.5-4B" {
		t.Fatalf("auto reviewer display name = %q, want Qwen3.5-4B", profile.DisplayName)
	}
	if !profile.ServesWorkers {
		t.Fatal("auto Qwen3.5-4B profile lost its cheap-tier worker capability")
	}
	// Empty override (the historical default) picks the same profile.
	emptyReq := &launchRequest{AppHome: t.TempDir()}
	if again := resolveClaudeCompanionProfile(emptyReq, t.TempDir()); again.Name != profile.Name {
		t.Fatalf("auto and empty override diverged: auto=%q empty=%q", profile.Name, again.Name)
	}
	// With the artifact missing, auto stays on Qwen (unchanged behavior).
	claudeNanoArtifactReady = func(string) bool { return false }
	fallback := resolveClaudeCompanionProfile(&launchRequest{AppHome: t.TempDir(), ClaudeReviewerOverride: claudeReviewerAuto}, t.TempDir())
	if fallback.NanoBeige || fallback.Name != claudeReviewerCompanionName {
		t.Fatalf("auto without an artifact must stay on Qwen: %+v", fallback)
	}
}

// TestClaudeReviewerFlagQwen2BResolvesSmallReviewer pins the small/light
// review-only profile: --claude-reviewer qwen2b must force the Qwen3.5-2B
// artifact (the cheap safety-review lane) even when NanoBeige is ready.
func TestClaudeReviewerFlagQwen2BResolvesSmallReviewer(t *testing.T) {
	restoreReady, restoreBackend, restoreGPU := claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable
	defer func() {
		claudeNanoArtifactReady, claudeNanoBackend, claudeNanoGPUCapable = restoreReady, restoreBackend, restoreGPU
	}()
	claudeNanoArtifactReady = func(string) bool { return true }
	claudeNanoBackend = func() (string, error) { return "/reviewed/nanbeige/llama-server", nil }
	claudeNanoGPUCapable = func(string) bool { return true }
	req := &launchRequest{AppHome: t.TempDir(), ClaudeReviewerOverride: claudeReviewerQwen2B}
	profile := resolveClaudeCompanionProfile(req, t.TempDir())
	if profile.NanoBeige || profile.Name != claudeSmallReviewerCompanionName {
		t.Fatalf("--claude-reviewer qwen2b must force the small reviewer: %+v", profile)
	}
	if profile.DisplayName != "Qwen3.5-2B" {
		t.Fatalf("small reviewer display name = %q, want Qwen3.5-2B", profile.DisplayName)
	}
	wantPath := filepath.Join(req.AppHome, ".cache", "claude-reviewer", claudeauto.DefaultSmallReviewerFile)
	if profile.ModelPath != wantPath {
		t.Fatalf("small reviewer path = %q, want %q", profile.ModelPath, wantPath)
	}
	if profile.ServesWorkers {
		t.Fatal("Qwen3.5-2B review-only profile advertised cheap-tier worker capability")
	}
	if (&claudeAutoRuntime{reviewerPort: 1, companion: profile}).workerRouteEnabled() {
		t.Fatal("a seated Qwen3.5-2B reviewer enabled the local-fast worker route")
	}
}

func TestClaudeWorkerRouteRequiresAWorkerCapableSeatedCompanion(t *testing.T) {
	worker := &claudeCompanionProfile{ServesWorkers: true}
	for name, runtime := range map[string]*claudeAutoRuntime{
		"no runtime":          nil,
		"no backend":          {reviewerPort: 0, companion: worker},
		"no profile":          {reviewerPort: 1},
		"review-only profile": {reviewerPort: 1, companion: &claudeCompanionProfile{}},
	} {
		if runtime.workerRouteEnabled() {
			t.Errorf("%s enabled the worker route", name)
		}
	}
	if !(&claudeAutoRuntime{reviewerPort: 1, companion: worker}).workerRouteEnabled() {
		t.Fatal("seated worker-capable companion did not enable the worker route")
	}
}
