package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

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
		{&launchRequest{ClaudeCode: true}, &placement.Strategy{Type: placement.MoEOffload, Parallel: 1}},
		{&launchRequest{}, &placement.Strategy{Type: placement.MoEOffload, Parallel: 4}},
	} {
		if got := claudeMainMaxActive(tc.req, tc.strategy); got != 0 {
			t.Fatalf("unexpected admission cap %d for req=%+v strategy=%+v", got, tc.req, tc.strategy)
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
	for _, want := range []string{"--device", "CUDA7", "-mg", "0", "--reasoning", "off", "--ctx-size", "65536", "--cache-type-k", "q8_0", "--cache-type-v"} {
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

func TestClaudeReviewerArgsKeepsOlderBackendCompatibility(t *testing.T) {
	args := claudeReviewerArgs("server", "reviewer.gguf", 1234, "", "--reasoning ARG")
	for _, unsupported := range []string{"--cache-type-k", "--cache-type-v"} {
		if hasArg(args, unsupported) {
			t.Fatalf("unexpected unsupported %q in %v", unsupported, args)
		}
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
}

func TestClaudeReviewerReservationBuildsCompanion(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "")
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24564, BandwidthMBps: 15754},
		{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 985},
	}}
	res := claudeReviewerReservation(&launchRequest{ClaudeCode: true}, caps)
	if res == nil {
		t.Fatal("Claude Code launch with GPUs must reserve the reviewer")
	}
	if res.Name != claudeReviewerCompanionName {
		t.Fatalf("companion name = %q, want %q", res.Name, claudeReviewerCompanionName)
	}
	if res.VRAMMB <= 0 {
		t.Fatalf("reservation must carry a positive VRAM footprint, got %d", res.VRAMMB)
	}
	if !res.AllowCPU {
		t.Fatal("a full-GPU host must keep fail-closed Auto working via CPU")
	}
	// Preference order mirrors the legacy walk: slow GPU first, main last.
	if len(res.GPUPreference) != 2 || res.GPUPreference[0] != 1 || res.GPUPreference[1] != 0 {
		t.Fatalf("GPU preference = %v, want [1 0]", res.GPUPreference)
	}
}

func TestClaudeReviewerReservationSkipsNonClaudeAndCPU(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "")
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}}
	if res := claudeReviewerReservation(&launchRequest{}, caps); res != nil {
		t.Fatal("non-Claude launch must not reserve a reviewer")
	}
	if res := claudeReviewerReservation(&launchRequest{ClaudeCode: true, CPUMode: true}, caps); res != nil {
		t.Fatal("CPU-mode launch must not reserve GPU VRAM for the reviewer")
	}
	if res := claudeReviewerReservation(&launchRequest{ClaudeCode: true}, &detect.Capabilities{}); res != nil {
		t.Fatal("GPU-less host must not reserve GPU VRAM for the reviewer")
	}
}
