package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/controller"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
	"github.com/raketenkater/ggrun/pkg/tui"
)

func writeFakeBackend(t *testing.T, name, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+body), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestParseLaunchArgsMMapPolicy(t *testing.T) {
	req, err := parseLaunchArgs([]string{"model.gguf", "--mmap"})
	if err != nil {
		t.Fatal(err)
	}
	if !req.ForceMMap || req.NoMMap {
		t.Fatalf("--mmap policy not preserved: %#v", req)
	}
	req, err = parseLaunchArgs([]string{"model.gguf", "--mmap", "--no-mmap"})
	if err != nil {
		t.Fatal(err)
	}
	if !req.NoMMap || req.ForceMMap {
		t.Fatalf("last mmap policy must win: %#v", req)
	}
}

func TestParseLaunchArgsWorkerBenchmark(t *testing.T) {
	req, err := parseLaunchArgs([]string{"model.gguf", "--worker-benchmark"})
	if err != nil {
		t.Fatal(err)
	}
	if !req.WorkerBenchmark || req.Benchmark {
		t.Fatalf("worker benchmark mode not preserved: %#v", req)
	}
}

func TestMeasuredLaunchVRAMIgnoresNegativeDeltas(t *testing.T) {
	// No GPUs is the deterministic unit seam; physical nvidia-smi sampling is
	// covered by the live one-shot benchmark.
	if got := measuredLaunchVRAMMB(&detect.Capabilities{}, nil, map[int]int{0: 100}); got != 0 {
		t.Fatalf("empty hardware reported %d MiB", got)
	}
}

func TestConfirmRequiredMMap(t *testing.T) {
	strategy := &placement.Strategy{MMap: true, MMapRequired: true}
	req := &launchRequest{}
	var output bytes.Buffer
	if err := confirmRequiredMMap(req, strategy, strings.NewReader("yes\n"), &output, true); err != nil {
		t.Fatal(err)
	}
	if !req.ForceMMap || !strings.Contains(output.String(), "Use mmap?") {
		t.Fatalf("confirmation did not approve mmap: req=%#v output=%q", req, output.String())
	}
	if err := confirmRequiredMMap(&launchRequest{}, strategy, strings.NewReader(""), &output, false); err == nil || !strings.Contains(err.Error(), "--mmap") {
		t.Fatalf("non-interactive launch must require explicit --mmap, got %v", err)
	}
	declined := &launchRequest{}
	if err := confirmRequiredMMap(declined, strategy, strings.NewReader("no\n"), &output, true); err != errMMapDeclined {
		t.Fatalf("negative answer = %v, want mmap-declined replan signal", err)
	}
	if !declined.NoMMap || declined.ForceMMap {
		t.Fatalf("negative answer did not select strict resident policy: %#v", declined)
	}
}

func TestResidentReviewerFallbackRequiresFitAndConsent(t *testing.T) {
	originalErr := fmt.Errorf("resident placement with reviewer does not fit")
	reservation := &placement.CompanionReservation{Name: claudeReviewerCompanionName, VRAMMB: 2600}
	req := &launchRequest{ClaudeCode: true, NoMMap: true, ReviewerReservation: reservation}
	want := &placement.Strategy{Type: placement.MoEOffload, MMap: false}
	called := 0
	compute := func(candidateReq *launchRequest) (*placement.Strategy, error) {
		called++
		if candidateReq.ReviewerReservation != nil || !candidateReq.ClaudeReviewerDisabled || !candidateReq.NoMMap {
			t.Fatalf("fallback candidate did not isolate only the reviewer: %#v", candidateReq)
		}
		return want, nil
	}
	var output bytes.Buffer
	got, err := tryResidentWithoutClaudeReviewer(req, originalErr, strings.NewReader("yes\n"), &output, true, compute)
	if err != nil {
		t.Fatal(err)
	}
	if got != want || called != 1 {
		t.Fatalf("fallback result=%#v calls=%d, want %#v and one compute", got, called, want)
	}
	if req.ReviewerReservation != nil || !req.ClaudeReviewerDisabled {
		t.Fatalf("accepted fallback was not retained on request: %#v", req)
	}
	if !strings.Contains(output.String(), "routed through the main model") || !strings.Contains(output.String(), "Auto reviews will use the main model") {
		t.Fatalf("fallback prompt/result did not explain routing: %q", output.String())
	}
}

func TestResidentReviewerFallbackDeclinePreservesReviewer(t *testing.T) {
	originalErr := fmt.Errorf("resident placement with reviewer does not fit")
	reservation := &placement.CompanionReservation{Name: claudeReviewerCompanionName, VRAMMB: 2600}
	req := &launchRequest{ClaudeCode: true, NoMMap: true, ReviewerReservation: reservation}
	compute := func(*launchRequest) (*placement.Strategy, error) {
		return &placement.Strategy{Type: placement.MoEOffload}, nil
	}
	if _, err := tryResidentWithoutClaudeReviewer(req, originalErr, strings.NewReader("no\n"), io.Discard, true, compute); err != originalErr {
		t.Fatalf("declined fallback error=%v, want original placement error", err)
	}
	if req.ReviewerReservation != reservation || req.ClaudeReviewerDisabled {
		t.Fatalf("declined fallback mutated request: %#v", req)
	}
}

func TestResidentReviewerFallbackIsClaudeOnlyAndExplicitWhenNonInteractive(t *testing.T) {
	originalErr := fmt.Errorf("resident placement does not fit")
	reservation := &placement.CompanionReservation{Name: claudeReviewerCompanionName, VRAMMB: 2600}
	called := 0
	compute := func(*launchRequest) (*placement.Strategy, error) {
		called++
		return &placement.Strategy{Type: placement.MoEOffload}, nil
	}
	ordinary := &launchRequest{NoMMap: true, ReviewerReservation: reservation}
	if _, err := tryResidentWithoutClaudeReviewer(ordinary, originalErr, strings.NewReader("yes\n"), io.Discard, true, compute); err != originalErr || called != 0 {
		t.Fatalf("ordinary launch entered Claude fallback: err=%v calls=%d", err, called)
	}
	claude := &launchRequest{ClaudeCode: true, NoMMap: true, ReviewerReservation: reservation}
	_, err := tryResidentWithoutClaudeReviewer(claude, originalErr, strings.NewReader(""), io.Discard, false, compute)
	if err == nil || !strings.Contains(err.Error(), "GGRUN_CLAUDE_AUTO_REVIEWER=off") {
		t.Fatalf("non-interactive fallback lacks explicit opt-in guidance: %v", err)
	}
	if claude.ReviewerReservation != reservation || claude.ClaudeReviewerDisabled {
		t.Fatalf("non-interactive fallback mutated request: %#v", claude)
	}
}

// TestProactivelyDropReviewerKeepsWhenReviewerStillFits confirms the proactive
// gate does NOT drop the separate Auto reviewer when the main model's plan still
// leaves room for a co-resident reviewer. Dropping here would silently route
// Auto review through the main model (self-review, insecure and unwanted); a
// separate model must keep judging the main whenever it can sit alongside it.
func TestProactivelyDropReviewerKeepsWhenReviewerStillFits(t *testing.T) {
	reservation := &placement.CompanionReservation{Name: claudeReviewerCompanionName, VRAMMB: 4600}
	req := &launchRequest{ClaudeCode: true, ReviewerReservation: reservation}
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 24564, VRAMUsedMB: 0, BandwidthMBps: 15754},
	}}
	// A small dense model on a 24 GB card: model + KV + overhead + compute floor
	// leave thousands of MiB of spare VRAM — far more than the reviewer's own
	// 4600 MiB reservation — so a separate reviewer fits comfortably alongside.
	// The old gate treated that spare seat as "pure overhead" and dropped the
	// reviewer, making the main model review itself. The gate must keep it.
	model := &placement.ModelProfile{
		Path:        "/models/qwen3.5-4b-q4.gguf",
		Basename:    "qwen3.5-4b-q4.gguf",
		TotalSizeMB: 2741,
		SizeBytes:   2741 * 1024 * 1024,
		NumLayers:   36,
	}
	strategy := &placement.Strategy{Type: placement.SingleGPU, MainGPU: 0, ContextSize: 65536, KVType: "q8_0", KVPlacement: "gpu", IsMoE: false}
	called := 0
	compute := func(candidateReq *launchRequest) (*placement.Strategy, error) {
		called++
		if candidateReq.ReviewerReservation != nil || !candidateReq.ClaudeReviewerDisabled {
			t.Fatalf("gate candidate did not isolate only the reviewer: %#v", candidateReq)
		}
		return strategy, nil
	}
	var output bytes.Buffer
	got := proactivelyDropReviewerForVRAMModel(req, caps, model, strategy, compute, strings.NewReader("n\n"), &output, true, "")
	if got != strategy || called != 1 {
		t.Fatalf("gate result=%#v calls=%d, want original strategy and one recompute", got, called)
	}
	if req.ReviewerReservation != reservation || req.ClaudeReviewerDisabled {
		t.Fatalf("gate dropped a reviewer that could stay co-resident: %#v", req)
	}
	if strings.Contains(output.String(), "separate Auto reviewer disabled") {
		t.Fatalf("gate dropped a reviewer that could stay co-resident, yet printed a drop notice: %q", output.String())
	}
}

// TestProactivelyDropReviewerSkipsWhenRecomputeOOMs confirms the gate does NOT
// drop the reviewer when recomputing without it would OOM (recompute fails or
// degrades to an offload/mmap plan).
func TestProactivelyDropReviewerSkipsWhenRecomputeOOMs(t *testing.T) {
	reservation := &placement.CompanionReservation{Name: claudeReviewerCompanionName, VRAMMB: 4600}
	req := &launchRequest{ClaudeCode: true, ReviewerReservation: reservation}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}}
	model := &placement.ModelProfile{Path: "/models/big.gguf", TotalSizeMB: 20000, SizeBytes: 20000 * 1024 * 1024}
	strategy := &placement.Strategy{Type: placement.SingleGPU, MainGPU: 0, ContextSize: 8192, KVType: "q8_0", KVPlacement: "gpu", IsMoE: false}

	// Recompute without reviewer fails outright: keep the reviewer.
	recomputeFails := func(*launchRequest) (*placement.Strategy, error) {
		return nil, fmt.Errorf("does not fit without reviewer either")
	}
	if got := proactivelyDropReviewerForVRAMModel(req, caps, model, strategy, recomputeFails, strings.NewReader("n\n"), io.Discard, true, ""); got != strategy {
		t.Fatalf("gate dropped reviewer on a failed recompute; result=%#v", got)
	}
	if req.ReviewerReservation != reservation || req.ClaudeReviewerDisabled {
		t.Fatalf("failed recompute mutated request: %#v", req)
	}

	// Recompute succeeds but is not a fully-resident dense plan (mmap required):
	// dropping the reviewer would jeopardize residency, so the reviewer stays.
	req.ReviewerReservation = reservation
	req.ClaudeReviewerDisabled = false
	mmapPlan := &placement.Strategy{Type: placement.MoEOffload, MMapRequired: true, IsMoE: true}
	if got := proactivelyDropReviewerForVRAMModel(req, caps, model, strategy, func(*launchRequest) (*placement.Strategy, error) {
		return mmapPlan, nil
	}, strings.NewReader("n\n"), io.Discard, true, ""); got != strategy {
		t.Fatalf("gate dropped reviewer into an mmap/offload plan; result=%#v", got)
	}
	if req.ReviewerReservation != reservation || req.ClaudeReviewerDisabled {
		t.Fatalf("mmap-plan recompute mutated request: %#v", req)
	}

	// Recompute succeeds, dense, but the main model leaves NO room for a separate
	// co-resident reviewer (headroom below the reviewer's own reservation). The
	// drop would mean self-review, which stays fail-closed: an interactive
	// decline keeps the reviewer.
	req.ReviewerReservation = reservation
	req.ClaudeReviewerDisabled = false
	// 22 GB weights on a 24 GB card: free 24564 - (22000 + 1024 floor + 0 kv) =
	// 1540 MiB headroom, well below the 4600 MiB reviewer reservation, so a
	// separate reviewer no longer fits alongside.
	tightModel := &placement.ModelProfile{Path: "/models/tight.gguf", TotalSizeMB: 22000, SizeBytes: 22000 * 1024 * 1024}
	tightPlan := &placement.Strategy{Type: placement.SingleGPU, MainGPU: 0, ContextSize: 65536, KVType: "q4_0", KVPlacement: "gpu", IsMoE: false}
	if got := proactivelyDropReviewerForVRAMModel(req, caps, tightModel, strategy, func(*launchRequest) (*placement.Strategy, error) {
		return tightPlan, nil
	}, strings.NewReader("n\n"), io.Discard, true, ""); got != strategy {
		t.Fatalf("gate dropped reviewer into self-review without consent; result=%#v", got)
	}
	if req.ReviewerReservation != reservation || req.ClaudeReviewerDisabled {
		t.Fatalf("tight-plan recompute mutated request: %#v", req)
	}
}

// TestProactivelyDropReviewerSkipsMoEAndOffload confirms the gate never fires for
// MoE or offload strategies — reserved for dense models that fit fully on GPU.
func TestProactivelyDropReviewerSkipsMoEAndOffload(t *testing.T) {
	reservation := &placement.CompanionReservation{Name: claudeReviewerCompanionName, VRAMMB: 4600}
	model := &placement.ModelProfile{Path: "/models/moe.gguf", TotalSizeMB: 40000, SizeBytes: 40000 * 1024 * 1024, IsMoE: true}
	for name, strategy := range map[string]*placement.Strategy{
		"moeOffload": {Type: placement.MoEOffload, IsMoE: true},
		"denseCPU":   {Type: placement.DenseCPUOffload, IsMoE: false},
		"mmapDense":  {Type: placement.SingleGPU, MainGPU: 0, MMapRequired: true, IsMoE: false},
	} {
		req := &launchRequest{ClaudeCode: true, ReviewerReservation: reservation}
		compute := func(*launchRequest) (*placement.Strategy, error) {
			t.Fatalf("%s: gate must not recompute for an ineligible strategy", name)
			return nil, nil
		}
		if got := proactivelyDropReviewerForVRAMModel(req, &detect.Capabilities{}, model, strategy, compute, strings.NewReader("n\n"), io.Discard, true, ""); got != strategy {
			t.Fatalf("%s: gate dropped reviewer for an ineligible strategy; result=%#v", name, got)
		}
		if req.ReviewerReservation != reservation || req.ClaudeReviewerDisabled {
			t.Fatalf("%s: gate mutated request: %#v", name, req)
		}
	}
}

// TestProactiveGateKeepsReactiveFallbackIntact confirms that the reactive
// fallback path (placement WITH reviewer fails) still works independently of the
// proactive gate, and that the gate itself is a no-op when there is no reviewer
// reservation to reclaim.
func TestProactiveGateKeepsReactiveFallbackIntact(t *testing.T) {
	// Reactive fallback (existing behavior) still fires when placement with the
	// reviewer fails: no proactive gate involved, consent still required.
	originalErr := fmt.Errorf("resident placement with reviewer does not fit")
	reservation := &placement.CompanionReservation{Name: claudeReviewerCompanionName, VRAMMB: 2600}
	req := &launchRequest{ClaudeCode: true, NoMMap: true, ReviewerReservation: reservation}
	want := &placement.Strategy{Type: placement.MoEOffload, MMap: false}
	compute := func(candidateReq *launchRequest) (*placement.Strategy, error) {
		if candidateReq.ReviewerReservation != nil || !candidateReq.ClaudeReviewerDisabled || !candidateReq.NoMMap {
			t.Fatalf("fallback candidate did not isolate only the reviewer: %#v", candidateReq)
		}
		return want, nil
	}
	var output bytes.Buffer
	got, err := tryResidentWithoutClaudeReviewer(req, originalErr, strings.NewReader("yes\n"), &output, true, compute)
	if err != nil {
		t.Fatal(err)
	}
	if got != want || req.ReviewerReservation != nil || !req.ClaudeReviewerDisabled {
		t.Fatalf("reactive fallback result=%#v req=%#v, want plan and disabled reviewer", got, req)
	}

	// Proactive gate with no reservation is a no-op: nothing to reclaim.
	noReservation := &launchRequest{ClaudeCode: true}
	strategy := &placement.Strategy{Type: placement.SingleGPU, MainGPU: 0}
	if got := proactivelyDropReviewerForVRAMModel(noReservation, nil, nil, strategy, func(*launchRequest) (*placement.Strategy, error) {
		t.Fatal("gate must not recompute when there is no reviewer reservation")
		return nil, nil
	}, strings.NewReader("n\n"), io.Discard, true, ""); got != strategy {
		t.Fatalf("gate fired without a reviewer reservation; result=%#v", got)
	}
}

// TestProactivelyDropReviewerSelfReviewRequiresConsent confirms that when the
// main model leaves no room for a separate co-resident reviewer, dropping the
// reviewer into self-review is NEVER silent: an interactive "yes" drops it with a
// clear self-review notice, an interactive decline keeps it, and a non-interactive
// launch fail-closes unless GGRUN_CLAUDE_AUTO_SELF_REVIEW=on is set.
func TestProactivelyDropReviewerSelfReviewRequiresConsent(t *testing.T) {
	reservation := &placement.CompanionReservation{Name: claudeReviewerCompanionName, VRAMMB: 4600}
	req := &launchRequest{ClaudeCode: true, ReviewerReservation: reservation}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}}
	// 22 GB weights on a 24 GB card leave ~1540 MiB headroom — below the 4600 MiB
	// reviewer reservation, so no separate reviewer can sit alongside.
	tightModel := &placement.ModelProfile{Path: "/models/tight.gguf", TotalSizeMB: 22000, SizeBytes: 22000 * 1024 * 1024}
	tightPlan := &placement.Strategy{Type: placement.SingleGPU, MainGPU: 0, ContextSize: 65536, KVType: "q4_0", KVPlacement: "gpu", IsMoE: false}
	compute := func(candidateReq *launchRequest) (*placement.Strategy, error) {
		if candidateReq.ReviewerReservation != nil || !candidateReq.ClaudeReviewerDisabled {
			t.Fatalf("gate candidate did not isolate only the reviewer: %#v", candidateReq)
		}
		return tightPlan, nil
	}

	// Interactive decline keeps the reviewer.
	req.ReviewerReservation = reservation
	req.ClaudeReviewerDisabled = false
	if got := proactivelyDropReviewerForVRAMModel(req, caps, tightModel, tightPlan, compute, strings.NewReader("n\n"), io.Discard, true, ""); got != tightPlan {
		t.Fatalf("interactive decline dropped the reviewer; result=%#v", got)
	}
	if req.ReviewerReservation != reservation || req.ClaudeReviewerDisabled {
		t.Fatalf("interactive decline mutated request into self-review: %#v", req)
	}

	// Interactive "yes" drops the reviewer, retains the drop, and prints a clear
	// self-review notice.
	req.ReviewerReservation = reservation
	req.ClaudeReviewerDisabled = false
	var output bytes.Buffer
	if got := proactivelyDropReviewerForVRAMModel(req, caps, tightModel, tightPlan, compute, strings.NewReader("yes\n"), &output, true, ""); got != tightPlan {
		t.Fatalf("consented drop did not adopt the recomputed plan; result=%#v", got)
	}
	if req.ReviewerReservation != nil || !req.ClaudeReviewerDisabled {
		t.Fatalf("consented drop was not retained on request: %#v", req)
	}
	if !strings.Contains(output.String(), "self-review") {
		t.Fatalf("consented drop did not surface a self-review notice: %q", output.String())
	}

	// Non-interactive without the explicit opt-in fail-closes and keeps the
	// reviewer (no silent self-review).
	req.ReviewerReservation = reservation
	req.ClaudeReviewerDisabled = false
	t.Setenv("GGRUN_CLAUDE_AUTO_SELF_REVIEW", "")
	if got := proactivelyDropReviewerForVRAMModel(req, caps, tightModel, tightPlan, compute, strings.NewReader(""), io.Discard, false, ""); got != tightPlan {
		t.Fatalf("non-interactive without opt-in dropped the reviewer; result=%#v", got)
	}
	if req.ReviewerReservation != reservation || req.ClaudeReviewerDisabled {
		t.Fatalf("non-interactive without opt-in mutated request into self-review: %#v", req)
	}

	// Non-interactive with the explicit GGRUN_CLAUDE_AUTO_SELF_REVIEW=on opt-in
	// is the one non-interactive way to permit self-review.
	req.ReviewerReservation = reservation
	req.ClaudeReviewerDisabled = false
	t.Setenv("GGRUN_CLAUDE_AUTO_SELF_REVIEW", "on")
	if got := proactivelyDropReviewerForVRAMModel(req, caps, tightModel, tightPlan, compute, strings.NewReader(""), io.Discard, false, ""); got != tightPlan {
		t.Fatalf("explicit opt-in did not adopt the recomputed plan; result=%#v", got)
	}
	if req.ReviewerReservation != nil || !req.ClaudeReviewerDisabled {
		t.Fatalf("explicit opt-in was not retained on request: %#v", req)
	}
}

func TestConfirmLiveMemoryProbeRequiresExplicitNonInteractiveConsent(t *testing.T) {
	var output bytes.Buffer
	req := &launchRequest{}
	if err := confirmLiveMemoryProbe(req, "full load required", strings.NewReader("yes\n"), &output, true); err != nil {
		t.Fatal(err)
	}
	if !req.AllowLiveMemoryProbe || !strings.Contains(output.String(), "contained live memory probe") {
		t.Fatalf("interactive consent was not retained: req=%#v output=%q", req, output.String())
	}
	if err := confirmLiveMemoryProbe(&launchRequest{}, "full load required", strings.NewReader(""), &output, false); err == nil || !strings.Contains(err.Error(), "--allow-live-memory-probe") {
		t.Fatalf("non-interactive probe must require an explicit flag, got %v", err)
	}
}

func TestRememberLiveMemoryProbeConsentPersistsPreference(t *testing.T) {
	isolateConfig(t)
	cfg := config.Defaults()
	var output bytes.Buffer
	rememberLiveMemoryProbeConsent(cfg, &output)
	if !cfg.AllowLiveMemoryProbe {
		t.Fatal("in-memory config did not retain live probe approval")
	}
	loaded, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	if !loaded.AllowLiveMemoryProbe {
		t.Fatal("saved config did not retain live probe approval")
	}
	if !strings.Contains(output.String(), "future launches will not ask again") {
		t.Fatalf("saved approval was not reported: %q", output.String())
	}
}

func TestParseLaunchArgsLoadsRememberedLiveMemoryProbeConsent(t *testing.T) {
	isolateConfig(t)
	t.Setenv("LLM_ALLOW_LIVE_MEMORY_PROBE", "true")
	req, err := parseLaunchArgs([]string{"model.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	if !req.AllowLiveMemoryProbe {
		t.Fatal("remembered live probe approval did not reach the launch request")
	}
}

func TestIsLlamaServerExecutableIgnoresArguments(t *testing.T) {
	if !isLlamaServerExecutable("/opt/bin/llama-server-cuda") || !isLlamaServerExecutable("/opt/bin/ik_llama-server") {
		t.Fatal("known server executable was not recognized")
	}
	if isLlamaServerExecutable("/usr/bin/rtk") || isLlamaServerExecutable("/tmp/ggrun") {
		t.Fatal("wrapper containing a --server-bin argument must not be recognized")
	}
}

func TestBackendMemoryMaxUsesDetectedBudgetAndHeadroom(t *testing.T) {
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 120000}}
	if runtime.GOOS != "linux" {
		if got := backendMemoryMaxMB(&launchRequest{}, caps); got != 0 {
			t.Fatalf("non-Linux launch cap = %d, want disabled", got)
		}
		return
	}
	if got := backendMemoryMaxMB(&launchRequest{}, caps); got != 120000 {
		t.Fatalf("detected free RAM cap = %d, want 120000", got)
	}
	if got := backendMemoryMaxMB(&launchRequest{RamBudgetMB: 96000}, caps); got != 96000 {
		t.Fatalf("RAM budget cap = %d, want 96000", got)
	}
	percentCaps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 128727, FreeMB: 125239}}
	if got := backendMemoryMaxMB(&launchRequest{RAMLimitPercent: 90}, percentCaps); got != 112366 {
		t.Fatalf("90%% whole-host cap = %d, want 112366", got)
	}
	if got := backendMemoryMaxMB(&launchRequest{RAMHeadroomMB: 8192}, caps); got != 111808 {
		t.Fatalf("headroom-adjusted cap = %d, want 111808", got)
	}
	if got := backendMemoryMaxMB(&launchRequest{RamBudgetMB: 64000, RAMHeadroomMB: 4096}, caps); got != 59904 {
		t.Fatalf("budget/headroom cap = %d, want 59904", got)
	}
	if got := backendMemoryMaxMB(&launchRequest{RAMHeadroomMB: 200000}, caps); got != 0 {
		t.Fatalf("non-positive cap = %d, want disabled", got)
	}
}

func TestBackendStartOptionsArePlacementStrategyIndependent(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("backend memory scopes are Linux-only")
	}
	req := &launchRequest{RAMHeadroomMB: 8192}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{FreeMB: 120000}}
	single := &placement.Strategy{Type: placement.SingleGPU}
	multi := &placement.Strategy{Type: placement.MultiGPUDense}

	singleOpts := backendStartOptions(req, caps, nil, nil, nil)
	multiOpts := backendStartOptions(req, caps, nil, nil, nil)

	if single.Type != placement.SingleGPU || multi.Type != placement.MultiGPUDense {
		t.Fatalf("test setup broken: single=%s multi=%s", single.Type, multi.Type)
	}
	if singleOpts.MemoryMaxMB != 111808 || multiOpts.MemoryMaxMB != 111808 {
		t.Fatalf("dense strategy memory scopes differ or use wrong cap: single=%d multi=%d", singleOpts.MemoryMaxMB, multiOpts.MemoryMaxMB)
	}
}

func TestHostMemoryPlacementRequiresExplicitContainmentBudget(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("host-memory containment is Linux-only")
	}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 120000}}
	hostStrategies := []*placement.Strategy{
		{Type: placement.CPUOnly},
		{Type: placement.DenseCPUOffload},
		{Type: placement.MoEOffload},
	}
	for _, strategy := range hostStrategies {
		if err := validateHostMemoryContainment(&launchRequest{}, caps, strategy); err == nil {
			t.Fatalf("%s placement accepted without an explicit RAM safety limit", strategy.Type)
		}
		if err := validateHostMemoryContainment(&launchRequest{RamBudgetMB: 96000}, caps, strategy); err != nil {
			t.Fatalf("%s placement rejected explicit RAM budget: %v", strategy.Type, err)
		}
		if err := validateHostMemoryContainment(&launchRequest{RAMHeadroomMB: 8192}, caps, strategy); err != nil {
			t.Fatalf("%s placement rejected explicit RAM headroom: %v", strategy.Type, err)
		}
		if err := validateHostMemoryContainment(&launchRequest{RAMLimitPercent: 90}, caps, strategy); err != nil {
			t.Fatalf("%s placement rejected RAM limit percent: %v", strategy.Type, err)
		}
	}
	for _, strategy := range []*placement.Strategy{{Type: placement.SingleGPU}, {Type: placement.MultiGPUDense}} {
		if err := validateHostMemoryContainment(&launchRequest{}, caps, strategy); err != nil {
			t.Fatalf("fully GPU-resident %s placement unexpectedly rejected: %v", strategy.Type, err)
		}
	}
}

func TestClaudeCodeParallelIsFeaturePolicyForDeepseek4(t *testing.T) {
	req := &launchRequest{ClaudeCode: true}
	model := &placement.ModelProfile{ModelArch: "deepseek4", CTXTrain: 1048576}
	be := &backendInfo{Tag: "llama"}
	opts := placementOptionsFromRequest(req, model, be, t.TempDir())
	if !opts.RequireMeasuredBuffers {
		t.Fatal("production placement must require measured buffer evidence")
	}
	if opts.Parallel != 4 {
		t.Fatalf("claude-code should request four slots over the shared mainline placement, got %d", opts.Parallel)
	}
	if opts.ContextSize != 0 || opts.AutoContextMax != 1048576 {
		t.Fatalf("claude-code fit should remain unresolved under a 1M cap, got ctx=%d cap=%d",
			opts.ContextSize, opts.AutoContextMax)
	}
	if !opts.AutoParallel || opts.ParallelSlotTarget != claudeSlotTarget {
		t.Fatalf("automatic Claude slots were not delegated to fit: auto=%t target=%d",
			opts.AutoParallel, opts.ParallelSlotTarget)
	}
	explicit := &launchRequest{ClaudeCode: true, CtxFlag: "262144"}
	if got := placementOptionsFromRequest(explicit, model, be, t.TempDir()).ContextSize; got != 262144 {
		t.Fatalf("explicit Claude Code context must win, got %d", got)
	}
	be = &backendInfo{Tag: "ik_llama"}
	// Unknown metadata gets a portable 131072 ceiling. The full fit search—not
	// option construction—selects the two 65k slots it can support, so every
	// candidate is accounted under its actual parallel key.
	opts = placementOptionsFromRequest(req, &placement.ModelProfile{ModelArch: "qwen3moe"}, be, t.TempDir())
	if opts.Parallel != 4 || !opts.AutoParallel {
		t.Fatalf("claude-code should pass a four-slot automatic ceiling into fit, got parallel=%d auto=%t",
			opts.Parallel, opts.AutoParallel)
	}
	if opts.ContextSize != 0 || opts.AutoContextMax != 131072 {
		t.Fatalf("unknown model fit should remain unresolved under the portable 131072 cap, got ctx=%d cap=%d",
			opts.ContextSize, opts.AutoContextMax)
	}
}

func TestClaudeCodeProfilesSelectExpectedAutomaticParallelism(t *testing.T) {
	model := &placement.ModelProfile{ModelArch: "deepseek4", CTXTrain: 1048576}
	be := &backendInfo{Tag: "llama"}
	for _, tc := range []struct {
		name string
		req  *launchRequest
		want int
	}{
		{"default_preserves_parallel_workflow", &launchRequest{ClaudeCode: true, Parallel: 1}, 4},
		{"default_preserves_higher_configured_parallel", &launchRequest{ClaudeCode: true, Parallel: 8}, 8},
		{"parallel_profile_preserves_parallel_workflow", &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileParallel}, 4},
		{"parallel_profile_overrides_stale_configured_parallel", &launchRequest{ClaudeCode: true, Parallel: 8, ClaudeProfile: claudeProfileParallel}, 4},
		{"interactive_keeps_single_foreground_slot", &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileInteractive}, 1},
		{"interactive_overrides_stale_configured_parallel", &launchRequest{ClaudeCode: true, Parallel: 8, ClaudeProfile: claudeProfileInteractive}, 1},
		{"interactive_keeps_explicit_parallel", &launchRequest{ClaudeCode: true, Parallel: 2, ParallelSet: true, ClaudeProfile: claudeProfileInteractive}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := placementOptionsFromRequest(tc.req, model, be, t.TempDir()).Parallel; got != tc.want {
				t.Fatalf("parallel=%d, want %d", got, tc.want)
			}
		})
	}
}

func TestClaudeWorkloadProfileScopesCacheEvidence(t *testing.T) {
	model := &placement.ModelProfile{ModelArch: "deepseek4", CTXTrain: 1048576}
	be := &backendInfo{Tag: "llama"}
	parallelDefault := &launchRequest{ClaudeCode: true, Parallel: 1}
	parallelExplicit := &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileParallel}
	interactive := &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileInteractive}

	parallelScope := requestWorkloadProfile(parallelDefault, model)
	if parallelScope == "" {
		t.Fatal("Claude Code default must have a non-empty workload cache scope")
	}
	if !strings.Contains(parallelScope, "agent-parallel-v4") {
		t.Fatalf("parallel workload did not invalidate pre-batch-tuner evidence: %q", parallelScope)
	}
	if got := requestWorkloadProfile(parallelExplicit, model); got != parallelScope {
		t.Fatalf("default and explicit agent-parallel should share behavior scope: got %q, want %q", got, parallelScope)
	}
	interactiveScope := requestWorkloadProfile(interactive, model)
	if interactiveScope == parallelScope {
		t.Fatalf("interactive and parallel profiles shared workload scope %q", interactiveScope)
	}
	if got := scopedProbeBackendTag(interactive, model, be); got == be.Tag {
		t.Fatalf("interactive profile reused the unscoped backend tag %q", got)
	}
	if got := placementOptionsFromRequest(interactive, model, be, t.TempDir()).WorkloadProfile; got != interactiveScope {
		t.Fatalf("placement workload scope=%q, want %q", got, interactiveScope)
	}
}

func TestWorkloadConcurrencyFollowsAgentDemandNotOnlyServerSlots(t *testing.T) {
	cases := []struct {
		name string
		req  *launchRequest
		want int
	}{
		{"interactive_default", &launchRequest{ClaudeCode: true, ClaudeProfile: claudeProfileInteractive, Parallel: 4}, 1},
		{"interactive_explicit", &launchRequest{ClaudeCode: true, ClaudeProfile: claudeProfileInteractive, Parallel: 4, ParallelSet: true}, 4},
		{"parallel_default", &launchRequest{ClaudeCode: true, ClaudeProfile: claudeProfileParallel}, 2},
		{"parallel_inherited_serial_default", &launchRequest{ClaudeCode: true, ClaudeProfile: claudeProfileParallel, Parallel: 1}, 2},
		{"parallel_configured_demand", &launchRequest{ClaudeCode: true, ClaudeProfile: claudeProfileParallel, Parallel: 4}, 4},
		{"parallel_declared_demand", &launchRequest{ClaudeCode: true, ClaudeProfile: claudeProfileParallel, Parallel: 2, ClaudeMaxActive: 4}, 4},
		{"bounded", &launchRequest{ClaudeCode: true, ClaudeProfile: claudeProfileParallel, ClaudeMaxActive: 32}, 8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := requestWorkloadConcurrency(tc.req); got != tc.want {
				t.Fatalf("workload concurrency=%d, want %d", got, tc.want)
			}
		})
	}
}

func TestPlacementEvidenceUsesExactBackendBuildIdentity(t *testing.T) {
	model := &placement.ModelProfile{ModelArch: "deepseek4", CTXTrain: 1048576}
	req := &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileInteractive}
	buildA := &backendInfo{Tag: "llama", Identity: "llama-server-cuda-build-a"}
	buildB := &backendInfo{Tag: "llama", Identity: "llama-server-cuda-build-b"}

	tagA := evidenceBackendCacheTag(buildA)
	tagB := evidenceBackendCacheTag(buildB)
	if tagA == tagB || tagA == buildA.Tag || tagB == buildB.Tag {
		t.Fatalf("backend evidence tags must isolate builds: A=%q B=%q", tagA, tagB)
	}
	if got := scopedProbeBackendTag(req, model, buildA); got == scopedProbeBackendTag(req, model, buildB) {
		t.Fatalf("probe scope reused evidence across backend builds: %q", got)
	}
	if got := placementOptionsFromRequest(req, model, buildA, t.TempDir()).BackendCacheTag; got != tagA {
		t.Fatalf("placement cache tag=%q, want build-scoped %q", got, tagA)
	}
}

func TestClaudeServerLogScopeTracksProfileBuildAndFinalArgs(t *testing.T) {
	cfg := config.Defaults()
	cfg.LogDir = t.TempDir()
	model := &placement.ModelProfile{Path: "model.gguf", ModelArch: "deepseek4", CTXTrain: 65536}
	interactive := &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileInteractive, Port: 8081}
	parallel := &launchRequest{ClaudeCode: true, Parallel: 1, ClaudeProfile: claudeProfileParallel, Port: 8081}
	buildA := &backendInfo{Tag: "llama", Identity: "build-a"}
	buildB := &backendInfo{Tag: "llama", Identity: "build-b"}
	argsA := []string{"/tmp/llama-server", "-m", "model.gguf", "-b", "512", "--port", "8081"}
	argsB := []string{"/tmp/llama-server", "-m", "model.gguf", "-b", "128", "--port", "8081"}

	scopeA := claudeLaunchLogScope(interactive, model, buildA, argsA)
	pathA := claudeServerLogPath(cfg, interactive.Port, scopeA)
	if pathA == claudeServerLogPath(cfg, parallel.Port, claudeLaunchLogScope(parallel, model, buildA, argsA)) {
		t.Fatal("interactive and parallel profiles shared a recoverable Claude log")
	}
	if pathA == claudeServerLogPath(cfg, interactive.Port, claudeLaunchLogScope(interactive, model, buildB, argsA)) {
		t.Fatal("different backend builds shared a recoverable Claude log")
	}
	if pathA == claudeServerLogPath(cfg, interactive.Port, claudeLaunchLogScope(interactive, model, buildA, argsB)) {
		t.Fatal("different final launch args shared a recoverable Claude log")
	}

	strategy := &placement.Strategy{Parallel: 1, ContextSize: 65536}
	log := "[ggrun] launch-scope: " + scopeA + "\n" +
		"health check OK model.gguf\n" +
		"n_slots = 1, n_ctx_slot = 65536\n"
	if !previousClaudeLogMatches(log, model, strategy, scopeA) {
		t.Fatal("scoped current log should be recoverable")
	}
	if previousClaudeLogMatches(log, model, strategy, "other-scope") {
		t.Fatal("log from another final launch scope was accepted for recovery")
	}
}

func TestClaudeCodeInteractiveProfileKeepsSSMPrefillBatch(t *testing.T) {
	req := &launchRequest{
		ClaudeCode:    true,
		Parallel:      1,
		ClaudeProfile: claudeProfileInteractive,
	}
	opts := placementOptionsFromRequest(req, &placement.ModelProfile{ModelArch: "deepseek4", CTXTrain: 1048576}, &backendInfo{Tag: "llama"}, t.TempDir())
	s := &placement.Strategy{ContextSize: opts.ContextSize, Parallel: opts.Parallel, BatchSize: 2048, HasSSM: true}
	claudeCodeSlotAdjust(s, &placement.ModelProfile{ModelArch: "deepseek4"}, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	if s.Parallel != 1 || s.BatchSize != 2048 {
		t.Fatalf("interactive Claude profile changed foreground prefill setup: parallel=%d batch=%d", s.Parallel, s.BatchSize)
	}
}

func TestParseClaudeProfile(t *testing.T) {
	isolateConfig(t)
	for _, tc := range []struct {
		args []string
		want string
	}{
		{[]string{"model.gguf", "--claude-code", "--claude-profile", "agent-interactive"}, claudeProfileInteractive},
		{[]string{"model.gguf", "--claude-code", "--claude-profile=AGENT-PARALLEL"}, claudeProfileParallel},
	} {
		req, err := parseLaunchArgs(tc.args)
		if err != nil {
			t.Fatalf("parse %v: %v", tc.args, err)
		}
		if req.ClaudeProfile != tc.want {
			t.Fatalf("profile=%q, want %q for %v", req.ClaudeProfile, tc.want, tc.args)
		}
	}
	if _, err := parseLaunchArgs([]string{"model.gguf", "--claude-profile", "fastest"}); err == nil {
		t.Fatal("invalid Claude profile was accepted")
	}
	if _, err := parseLaunchArgs([]string{"model.gguf", "--claude-profile", claudeProfileInteractive}); err == nil {
		t.Fatal("Claude profile without --claude-code was accepted")
	}
}

func TestParseClaudeReviewer(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--claude-code", "--claude-reviewer", "nanbeige"})
	if err != nil {
		t.Fatalf("parse --claude-reviewer nanbeige: %v", err)
	}
	if req.ClaudeReviewerOverride != claudeReviewerNanbeige {
		t.Fatalf("reviewer override = %q, want %q", req.ClaudeReviewerOverride, claudeReviewerNanbeige)
	}
	req, err = parseLaunchArgs([]string{"model.gguf", "--claude-code", "--claude-reviewer=QWEN"})
	if err != nil {
		t.Fatalf("parse --claude-reviewer=QWEN: %v", err)
	}
	if req.ClaudeReviewerOverride != claudeReviewerQwen {
		t.Fatalf("reviewer override = %q, want %q", req.ClaudeReviewerOverride, claudeReviewerQwen)
	}
	// auto is a valid explicit value and keeps the default selection behavior.
	req, err = parseLaunchArgs([]string{"model.gguf", "--claude-code", "--claude-reviewer", "auto"})
	if err != nil {
		t.Fatalf("parse --claude-reviewer auto: %v", err)
	}
	if req.ClaudeReviewerOverride != claudeReviewerAuto {
		t.Fatalf("reviewer override = %q, want %q", req.ClaudeReviewerOverride, claudeReviewerAuto)
	}
	// qwen2b is a valid explicit value that forces the small/light review-only
	// Qwen3.5-2B profile.
	req, err = parseLaunchArgs([]string{"model.gguf", "--claude-code", "--claude-reviewer", "qwen2b"})
	if err != nil {
		t.Fatalf("parse --claude-reviewer qwen2b: %v", err)
	}
	if req.ClaudeReviewerOverride != claudeReviewerQwen2B {
		t.Fatalf("reviewer override = %q, want %q", req.ClaudeReviewerOverride, claudeReviewerQwen2B)
	}
	// Invalid values are rejected.
	if _, err := parseLaunchArgs([]string{"model.gguf", "--claude-code", "--claude-reviewer", "mistral"}); err == nil {
		t.Fatal("invalid --claude-reviewer value was accepted")
	}
	// A non-auto override requires --claude-code.
	if _, err := parseLaunchArgs([]string{"model.gguf", "--claude-reviewer", "qwen"}); err == nil {
		t.Fatal("--claude-reviewer without --claude-code was accepted")
	}
}

func TestParseEmitServerArgvJSON(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--emit-server-argv-json"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.EmitServerArgvJSON {
		t.Fatal("--emit-server-argv-json was not retained for dry-run planning")
	}
}

func TestLaunchPlanEnvironmentMatchesServerChildCUDAContract(t *testing.T) {
	t.Setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
	oldQueue, hadQueue := os.LookupEnv("CUDA_SCALE_LAUNCH_QUEUES")
	if err := os.Unsetenv("CUDA_SCALE_LAUNCH_QUEUES"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if hadQueue {
			_ = os.Setenv("CUDA_SCALE_LAUNCH_QUEUES", oldQueue)
		} else {
			_ = os.Unsetenv("CUDA_SCALE_LAUNCH_QUEUES")
		}
	})
	env := launchPlanEnvironment(
		[]string{"llama-server", "--tensor-split", "1,0,0"},
		"CUDA_VISIBLE_DEVICES=2,0",
	)
	if got := env["CUDA_DEVICE_ORDER"]; got != "PCI_BUS_ID" {
		t.Fatalf("CUDA_DEVICE_ORDER=%q, want PCI_BUS_ID", got)
	}
	if got := env["CUDA_SCALE_LAUNCH_QUEUES"]; got != "4x" {
		t.Fatalf("CUDA_SCALE_LAUNCH_QUEUES=%q, want 4x", got)
	}
	if got := env["CUDA_VISIBLE_DEVICES"]; got != "2,0" {
		t.Fatalf("CUDA_VISIBLE_DEVICES=%q, want 2,0", got)
	}
}

func TestLaunchPlanEnvironmentIncludesStableBackendLibraries(t *testing.T) {
	t.Setenv("LD_LIBRARY_PATH", "")
	t.Setenv("LLM_SERVER_LIB_HUB", "")
	binDir := filepath.Join(t.TempDir(), "build-cuda", "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	backend := filepath.Join(binDir, "llama-server")
	if err := os.WriteFile(backend, []byte("backend"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(binDir, "libllama.so"), []byte("library"), 0o644); err != nil {
		t.Fatal(err)
	}

	env := launchPlanEnvironment([]string{backend}, "", backend)
	if got := env["LD_LIBRARY_PATH"]; got != binDir {
		t.Fatalf("LD_LIBRARY_PATH=%q, want stable backend directory %q", got, binDir)
	}
}

func TestTUILaunchArgsPassSelectedBackend(t *testing.T) {
	cfg := config.Defaults()
	cfg.Backend = "ik_llama"
	args := tuiLaunchArgs(&tui.LaunchRequest{
		ModelPath:   "model.gguf",
		Port:        8081,
		KVPlacement: "auto",
		KVQuality:   "mid",
		Backend:     "ik_llama",
		ClaudeCode:  true,
	}, cfg)
	if !hasAdjacentArg(args, "--backend", "ik_llama") {
		t.Fatalf("selected backend should be explicit so route-arch cannot override it, got %v", args)
	}
	if !hasAdjacentArg(args, "--ctx-size", "fit") {
		t.Fatalf("TUI fit context should map to CLI fit args, got %v", args)
	}
	if !hasArg(args, "--claude-code") {
		t.Fatalf("Claude Code toggle not preserved: %v", args)
	}
}

func TestParseLaunchArgsPlansDirectKVTypeOnce(t *testing.T) {
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "config"))
	req, err := parseLaunchArgs([]string{
		"model.gguf", "--cache-type-k", "q5_1", "--cache-type-v=q5_1",
	})
	if err != nil {
		t.Fatalf("parse direct KV cache type: %v", err)
	}
	if req.KVQuality != "q5_1" || len(req.ExtraArgs) != 0 {
		t.Fatalf("direct KV flags must become the planned type, got quality=%q extra=%v", req.KVQuality, req.ExtraArgs)
	}

	strategy, err := placement.Compute(&detect.Capabilities{
		CPU: detect.CPUInfo{Cores: 4}, RAM: detect.RAMInfo{TotalMB: 16384, FreeMB: 16384},
	}, &placement.ModelProfile{
		SizeBytes: 1, NumLayers: 32, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
	}, placement.Options{CPUMode: true, ContextSize: 32768, KVQuality: req.KVQuality})
	if err != nil {
		t.Fatalf("plan direct KV cache type: %v", err)
	}
	if strategy.KVType != "q5_1" {
		t.Fatalf("strategy KV type = %q, want q5_1", strategy.KVType)
	}
	args := strategy.Args("model.gguf", 8081)
	if !hasAdjacentArg(args, "--cache-type-k", "q5_1") || !hasAdjacentArg(args, "--cache-type-v", "q5_1") {
		t.Fatalf("strategy did not emit q5_1 K/V flags: %v", args)
	}
	if got := strings.Count(strings.Join(args, " "), "--cache-type-k"); got != 1 {
		t.Fatalf("cache type K flag emitted %d times, want once: %v", got, args)
	}
}

func TestParseLaunchArgsRejectsMixedKVTypes(t *testing.T) {
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "config"))
	_, err := parseLaunchArgs([]string{
		"model.gguf", "--cache-type-k", "q8_0", "--cache-type-v", "q5_1",
	})
	if err == nil || !strings.Contains(err.Error(), "mixed") {
		t.Fatalf("mixed cache types must fail before an unsafe placement, got %v", err)
	}
}

func TestTUILaunchArgsPreserveMaxContext(t *testing.T) {
	cfg := config.Defaults()
	args := tuiLaunchArgs(&tui.LaunchRequest{ModelPath: "model.gguf", CtxFlag: "max"}, cfg)
	if !hasAdjacentArg(args, "--ctx-size", "max") {
		t.Fatalf("TUI max context should pass CLI max, got %v", args)
	}
}

func TestTUILaunchArgsPassNonDefaultBackend(t *testing.T) {
	cfg := config.Defaults()
	cfg.Backend = "ik_llama"
	args := tuiLaunchArgs(&tui.LaunchRequest{ModelPath: "model.gguf", Backend: "custom"}, cfg)
	if !hasAdjacentArg(args, "--backend", "custom") {
		t.Fatalf("non-default backend selection should be explicit, got %v", args)
	}
}

func hasAdjacentArg(args []string, key, val string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == key && args[i+1] == val {
			return true
		}
	}
	return false
}

func TestParseModelMissingFileReportsModelPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.gguf")
	_, err := parseModel(path)
	if err == nil {
		t.Fatal("expected missing model to fail")
	}
	if !strings.Contains(err.Error(), "model file") || !strings.Contains(err.Error(), "missing.gguf") {
		t.Fatalf("expected model-file path error, got %v", err)
	}
	if strings.Contains(err.Error(), "parse_gguf.py failed") {
		t.Fatalf("missing model should not be reported as parser failure: %v", err)
	}
}

func TestParseModelRejectsIncompleteShardSetBeforePlacement(t *testing.T) {
	dir := t.TempDir()
	first := filepath.Join(dir, "partial-00001-of-00003.gguf")
	for _, path := range []string{first, filepath.Join(dir, "partial-00002-of-00003.gguf")} {
		if err := os.WriteFile(path, []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	_, err := parseModel(first)
	if err == nil || !strings.Contains(err.Error(), "incomplete sharded GGUF") || !strings.Contains(err.Error(), "00003") {
		t.Fatalf("incomplete model diagnostic = %v", err)
	}
}

func TestShouldPromoteMoEPlacement(t *testing.T) {
	cur := &placement.Strategy{Type: placement.MoEOffload, NCPUMoE: 37}
	next := &placement.Strategy{Type: placement.MoEOffload, NCPUMoE: 35}
	if !shouldPromoteMoEPlacement(cur, next) {
		t.Fatalf("expected fewer CPU MoE layers to promote")
	}
	if shouldPromoteMoEPlacement(cur, &placement.Strategy{Type: placement.MoEOffload, NCPUMoE: 37}) {
		t.Fatalf("equal CPU MoE layers must not promote")
	}
	if shouldPromoteMoEPlacement(&placement.Strategy{Type: placement.SingleGPU}, next) {
		t.Fatalf("non-MoE-offload current placement must not promote")
	}
}

func TestMeasuredPromotionBypassesPlacementCache(t *testing.T) {
	opts := measuredPromotionOptions(
		&launchRequest{CtxFlag: "32768"},
		&placement.ModelProfile{ModelArch: "qwen3moe", CTXTrain: 32768},
		&backendInfo{Tag: "llama"},
		t.TempDir(),
	)
	if !opts.SkipPlacementCache {
		t.Fatal("measured promotion must recompute instead of reloading the sparse placement it is meant to improve")
	}
}

func TestStartupLogCUDAOOM(t *testing.T) {
	log := "loading\n" +
		"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2206.07 MiB on device 0: cudaMalloc failed: out of memory\n" +
		"segmentation fault"
	device, allocMB, ok := startupLogCUDAOOM(log)
	if !ok || device != 0 || allocMB != 2207 {
		t.Fatalf("cuda oom parse = device %d alloc %d ok %v", device, allocMB, ok)
	}
}

func TestClassifyRuntimeCgroupOOM(t *testing.T) {
	if peakMB, ok := classifyRuntimeCgroupOOM(0, 73568*1024*1024); ok || peakMB != 0 {
		t.Fatalf("ordinary exit classified as cgroup OOM: peak=%d ok=%v", peakMB, ok)
	}
	if peakMB, ok := classifyRuntimeCgroupOOM(1, 73568*1024*1024); !ok || peakMB != 73568 {
		t.Fatalf("cgroup OOM classification = peak %d ok=%v, want 73568/true", peakMB, ok)
	}
	if peakMB, ok := classifyRuntimeCgroupOOM(2, 1024*1024+1); !ok || peakMB != 2 {
		t.Fatalf("cgroup OOM peak must round up: peak=%d ok=%v", peakMB, ok)
	}
}

func TestRuntimeLogCUDAOOMRecognizesVMMFormat(t *testing.T) {
	log := strings.Join([]string{
		"[launch] health check OK after 5m1s",
		"CUDA error: out of memory",
		"  current device: 0, in function alloc at ggml-cuda.cu:529",
		"  cuMemCreate(&handle, reserve_size, &prop, 0)",
	}, "\n")
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}}
	device, reserveMB, estimated, ok := runtimeLogCUDAOOM(log, caps, nil, nil)
	if !ok || !estimated || device != 0 || reserveMB != 2457 {
		t.Fatalf("VMM OOM = device %d reserve %d estimated=%v ok=%v", device, reserveMB, estimated, ok)
	}
	_, repeatedReserve, _, ok := runtimeLogCUDAOOM(log, caps, nil, map[int]int{0: reserveMB})
	if !ok || repeatedReserve != 4914 {
		t.Fatalf("repeated VMM OOM reserve = %d ok=%v, want 4914", repeatedReserve, ok)
	}
}

func TestRuntimeLogCUDAOOMPrefersExactAllocation(t *testing.T) {
	// The marker opens the runtime-growth window; this case is about parsing the
	// exact allocation size, not about when the failure happened.
	log := "srv  llama_server: model loaded\n" +
		"allocating 1679.00 MiB on device 2: cudaMalloc failed: out of memory"
	device, reserveMB, estimated, ok := runtimeLogCUDAOOM(log, nil, nil, nil)
	if !ok || estimated || device != 2 || reserveMB != 1679 {
		t.Fatalf("exact OOM = device %d reserve %d estimated=%v ok=%v", device, reserveMB, estimated, ok)
	}
}

func TestPreviousClaudeLogMatchesRuntimeShape(t *testing.T) {
	model := &placement.ModelProfile{Path: "/models/DeepSeek-V4-00001-of-00004.gguf"}
	strategy := &placement.Strategy{ContextSize: 1048576, Parallel: 4}
	const scope = "exact-final-launch-scope"
	log := "[ggrun] launch-scope: " + scope + "\n" +
		"loading model '/models/DeepSeek-V4-00001-of-00004.gguf'\n" +
		"initializing, n_slots = 4, n_ctx_slot = 262144, kv_unified = 'false'\n" +
		"[launch] health check OK after 5m1s\n"
	if !previousClaudeLogMatches(log, model, strategy, scope) {
		t.Fatal("matching previous Claude runtime log was rejected")
	}
	strategy.Parallel = 8
	if previousClaudeLogMatches(log, model, strategy, scope) {
		t.Fatal("log from a different parallel/context shape must not be recovered")
	}
}

func TestStartupComputeMeasurementMustMatchFailedGPU(t *testing.T) {
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	model := &placement.ModelProfile{Path: "/models/model.gguf", NumLayers: 43, NumExperts: 256}
	strategy := &placement.Strategy{
		ContextSize: 1048576,
		UBatchSize:  64,
		KVQuality:   "high",
		KVPlacement: "gpu",
		KVType:      "f16",
		Parallel:    1,
	}
	be := &backendInfo{Tag: "llama"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0}, {Index: 1}}}
	log := "CUDA1 compute buffer size = 100.00 MiB\n" +
		"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 8000.00 MiB on device 0: cudaMalloc failed: out of memory\n" +
		"ggml_gallocr_reserve_n: graph_reserve failed\n"

	measured := recordMeasuredLaunchProbes(nil, cfg, model, strategy, be, caps, log, nil, 0)
	device, _, isCompute, ok := startupLogCUDAOOMDetailed(log)
	if !ok || !isCompute || device != 0 {
		t.Fatalf("failed allocation parse = device %d compute=%v ok=%v", device, isCompute, ok)
	}
	if measured[device] != 0 {
		t.Fatalf("another GPU's probe must not suppress the failed GPU penalty: %v", measured)
	}
}

func TestRouteArchBackendKeepsRegisteredTag(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake-backend probe uses a shell script")
	}
	t.Setenv("LLM_APP_HOME", t.TempDir())
	backendPath := writeFakeBackend(t, "custom-server", "echo llama server help\n")
	if err := backends.Save([]backends.Backend{{Tag: "custom", Path: backendPath, RouteArch: "custom_moe"}}); err != nil {
		t.Fatalf("save backends: %v", err)
	}
	be := routeArchBackend(&backendInfo{Path: "/main/llama-server", Tag: "llama"}, &placement.ModelProfile{ModelArch: "custom_moe"}, &launchRequest{})
	if be == nil || be.Path != backendPath || be.Tag != "custom" {
		t.Fatalf("expected routed custom backend tag, got %#v", be)
	}
}

func TestRouteArchBackendPreservesIKDialectBehindRecipeTag(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake-backend probe uses a shell script")
	}
	t.Setenv("LLM_APP_HOME", t.TempDir())
	backendPath := writeFakeBackend(t, "hy3-server", "echo 'ikawrakow split-mode-graph'\n")
	if err := backends.Save([]backends.Backend{{Tag: "hy3", Path: backendPath, RouteArch: "hy_v3"}}); err != nil {
		t.Fatalf("save backends: %v", err)
	}
	be := routeArchBackend(&backendInfo{Path: "/main/llama-server", Tag: "llama"}, &placement.ModelProfile{ModelArch: "hy_v3"}, &launchRequest{})
	if be == nil || be.Tag != "hy3" || backendDialect(be) != "ik_llama" || !be.IsIK {
		t.Fatalf("expected HY3 identity with IK dialect, got %#v", be)
	}
	opts := placementOptionsFromRequest(&launchRequest{}, &placement.ModelProfile{}, be, t.TempDir())
	if opts.BackendTag != "ik_llama" {
		t.Fatalf("placement got recipe tag instead of IK dialect: %#v", opts)
	}
	if want := evidenceBackendCacheTag(be); opts.BackendCacheTag != want {
		t.Fatalf("placement probes are not isolated to the exact HY3 fork build: got=%q want=%q opts=%#v", opts.BackendCacheTag, want, opts)
	}
}

func TestHY3CompatibilityArgsUseOnlyDerivedMetadata(t *testing.T) {
	model := &placement.ModelProfile{
		ModelArch:                 "hy_v3",
		ExpertSharedCount:         1,
		ExpertSharedCountInferred: true,
		LeadingDense:              1,
		LeadingDenseInferred:      true,
	}
	got := hy3CompatibilityArgs(nil, model, &backendInfo{Tag: "hy3"})
	want := []string{
		"--override-kv", "hy_v3.expert_shared_count=int:1",
		"--override-kv", "hy_v3.leading_dense_block_count=int:1",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("HY3 derived args = %#v, want %#v", got, want)
	}

	got = hy3CompatibilityArgs([]string{"--override-kv", "hy_v3.expert_shared_count=int:2"}, model, &backendInfo{Tag: "hy3"})
	if !reflect.DeepEqual(got, []string{"--override-kv", "hy_v3.leading_dense_block_count=int:1"}) {
		t.Fatalf("explicit expert override must win, got %#v", got)
	}
	if got := hy3CompatibilityArgs(nil, model, &backendInfo{Tag: "llama"}); got != nil {
		t.Fatalf("non-HY3 backend must not receive compatibility args: %#v", got)
	}
}

func TestHY3TemplateArgsUseBundledTemplateWithoutOverridingUser(t *testing.T) {
	root := t.TempDir()
	template := filepath.Join(root, "models", "templates", "Hy3.jinja")
	if err := os.MkdirAll(filepath.Dir(template), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(template, []byte("template"), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(root, "build-cuda", "bin", "llama-server")
	if err := os.MkdirAll(filepath.Dir(bin), 0o755); err != nil {
		t.Fatal(err)
	}
	got := hy3TemplateArgs(nil, &backendInfo{Tag: "hy3", Path: bin})
	want := []string{"--chat-template-file", template}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("HY3 template args = %#v, want %#v", got, want)
	}
	if got := hy3TemplateArgs([]string{"--chat-template", "chatml"}, &backendInfo{Tag: "hy3", Path: bin}); got != nil {
		t.Fatalf("explicit user chat template must win: %#v", got)
	}
}

// writeGGUFWithTemplate writes a minimal GGUF whose metadata carries
// general.architecture and tokenizer.chat_template, so catalogTemplateArgs can
// detect a raise_exception guard in the model's own embedded template.
func writeGGUFWithTemplate(t *testing.T, path, arch, template string) {
	t.Helper()
	buf := new(bytes.Buffer)
	buf.WriteString("GGUF")
	_ = binary.Write(buf, binary.LittleEndian, uint32(3))
	_ = binary.Write(buf, binary.LittleEndian, uint64(0)) // tensor count
	_ = binary.Write(buf, binary.LittleEndian, uint64(2)) // kv count
	writeStr := func(s string) {
		_ = binary.Write(buf, binary.LittleEndian, uint64(len(s)))
		buf.WriteString(s)
	}
	writeStr("general.architecture")
	_ = binary.Write(buf, binary.LittleEndian, uint32(8))
	writeStr(arch)
	writeStr("tokenizer.chat_template")
	_ = binary.Write(buf, binary.LittleEndian, uint32(8))
	writeStr(template)
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
}

func TestCatalogTemplateArgsAppliesForQwen38AndNanbeige(t *testing.T) {
	cacheDir := t.TempDir()
	for _, tc := range []struct {
		name  string
		arch  string
		model string
	}{
		{"qwen35", "qwen35", "Qwen3.8-27B-UD-Q8_K_XL.gguf"},
		{"nanbeige", "nanbeige", "Nanbeige4.2-3B-Q4_K_M.gguf"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			modelPath := filepath.Join(t.TempDir(), tc.model)
			writeGGUFWithTemplate(t, modelPath, tc.arch, "<broken raise_exception('System message must be at the beginning.')>")
			model := &placement.ModelProfile{Path: modelPath, Basename: tc.model, ModelArch: tc.arch}
			req := &launchRequest{ModelPath: modelPath}
			cfg := &config.Config{CacheDir: cacheDir}
			got := catalogTemplateArgs(req, cfg, model)
			if len(got) != 2 || got[0] != "--chat-template-file" {
				t.Fatalf("catalog template args = %#v, want --chat-template-file", got)
			}
			if !strings.HasSuffix(got[1], ".jinja") {
				t.Fatalf("materialized template path %q must end in .jinja", got[1])
			}
			data, err := os.ReadFile(got[1])
			if err != nil {
				t.Fatal(err)
			}
			if len(data) == 0 {
				t.Fatal("materialized template is empty")
			}
		})
	}
}

func TestCatalogTemplateArgsUserOverrideWins(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "Qwen3.8-27B.gguf")
	writeGGUFWithTemplate(t, modelPath, "qwen35", "<broken raise_exception>")
	model := &placement.ModelProfile{Path: modelPath, Basename: "Qwen3.8-27B.gguf", ModelArch: "qwen35"}
	req := &launchRequest{ModelPath: modelPath, ExtraArgs: []string{"--chat-template", "chatml"}}
	if got := catalogTemplateArgs(req, &config.Config{CacheDir: cacheDir}, model); got != nil {
		t.Fatalf("explicit user chat template must win: %#v", got)
	}
}

func TestCatalogTemplateArgsUnknownModelUntouched(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "DeepSeek-V4.gguf")
	writeGGUFWithTemplate(t, modelPath, "deepseek4", "<broken raise_exception>")
	model := &placement.ModelProfile{Path: modelPath, Basename: "DeepSeek-V4.gguf", ModelArch: "deepseek4"}
	req := &launchRequest{ModelPath: modelPath}
	if got := catalogTemplateArgs(req, &config.Config{CacheDir: cacheDir}, model); got != nil {
		t.Fatalf("model with no catalog entry must be untouched: %#v", got)
	}
}

func TestCatalogTemplateArgsHealthyTemplateUntouched(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "Qwen3.8-27B.gguf")
	// Arch matches the catalog but the embedded template is healthy (no
	// raise_exception): the override must NOT be applied.
	writeGGUFWithTemplate(t, modelPath, "qwen35", "<healthy template>")
	model := &placement.ModelProfile{Path: modelPath, Basename: "Qwen3.8-27B.gguf", ModelArch: "qwen35"}
	req := &launchRequest{ModelPath: modelPath}
	if got := catalogTemplateArgs(req, &config.Config{CacheDir: cacheDir}, model); got != nil {
		t.Fatalf("healthy template must not be overridden: %#v", got)
	}
}

func TestCatalogTemplateArgsOverrideByName(t *testing.T) {
	cacheDir := t.TempDir()
	// A model with no catalog match (deepseek4) but an explicit --chat-template
	// override naming a catalog entry must force that entry's template.
	modelPath := filepath.Join(t.TempDir(), "DeepSeek-V4.gguf")
	writeGGUFWithTemplate(t, modelPath, "deepseek4", "<broken raise_exception>")
	model := &placement.ModelProfile{Path: modelPath, Basename: "DeepSeek-V4.gguf", ModelArch: "deepseek4"}
	req := &launchRequest{ModelPath: modelPath, ChatTemplateOverride: "qwen3.8-27b"}
	got := catalogTemplateArgs(req, &config.Config{CacheDir: cacheDir}, model)
	if len(got) != 2 || got[0] != "--chat-template-file" {
		t.Fatalf("catalog override must emit --chat-template-file, got %#v", got)
	}
	data, err := os.ReadFile(got[1])
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		t.Fatal("overridden template is empty")
	}
	// Unknown override name must be a no-op (validated at parse time).
	req.ChatTemplateOverride = "no-such-entry"
	if got := catalogTemplateArgs(req, &config.Config{CacheDir: cacheDir}, model); got != nil {
		t.Fatalf("unknown override must be a no-op: %#v", got)
	}
}

func TestParseLaunchArgsChatTemplateOverride(t *testing.T) {
	req, err := parseLaunchArgs([]string{"model.gguf", "--chat-template", "qwen3.8-27b"})
	if err != nil {
		t.Fatal(err)
	}
	if req.ChatTemplateOverride != "qwen3.8-27b" {
		t.Fatalf("ChatTemplateOverride = %q, want qwen3.8-27b", req.ChatTemplateOverride)
	}
	if hasChatTemplateOverride(req.ExtraArgs) {
		t.Fatalf("catalog override must not leak into ExtraArgs: %#v", req.ExtraArgs)
	}
	// A value that is NOT a catalog entry stays a passthrough backend flag.
	req2, err := parseLaunchArgs([]string{"model.gguf", "--chat-template", "chatml"})
	if err != nil {
		t.Fatal(err)
	}
	if req2.ChatTemplateOverride != "" {
		t.Fatalf("chatml is not a catalog entry, must not set override: %q", req2.ChatTemplateOverride)
	}
	if !hasChatTemplateOverride(req2.ExtraArgs) {
		t.Fatalf("--chat-template chatml must stay a passthrough flag: %#v", req2.ExtraArgs)
	}
}

// TestLaunchServerArgsEmitsCatalogTemplateForMatchedModel verifies the full
// launch arg assembly passes --chat-template-file for a catalog-matched model
// whose embedded template carries a raise_exception guard.
func TestLaunchServerArgsEmitsCatalogTemplateForMatchedModel(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "Qwen3.8-27B-UD-Q8_K_XL.gguf")
	writeGGUFWithTemplate(t, modelPath, "qwen35", "<broken raise_exception('System message must be at the beginning.')>")
	req := &launchRequest{ModelPath: modelPath}
	cfg := &config.Config{CacheDir: cacheDir}
	be := &backendInfo{Path: "llama-server", Tag: "llama", Help: ""}
	model := &placement.ModelProfile{Path: modelPath, Basename: filepath.Base(modelPath), ModelArch: "qwen35"}
	strategy := &placement.Strategy{
		ContextSize: 65536, KVType: "q8_0", KVQuality: "high",
		FlashAttention: true, Threads: 16, ThreadsBatch: 16, Parallel: 1,
	}
	args := buildLaunchServerArgs(req, cfg, be, nil, model, strategy)
	tpl := valueAfter(args, "--chat-template-file")
	if tpl == "" || !strings.HasSuffix(tpl, ".jinja") {
		t.Fatalf("launch must emit --chat-template-file with a .jinja path for a catalog-matched model, got %v", args)
	}
	data, err := os.ReadFile(tpl)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		t.Fatal("emitted template file is empty")
	}
}

func TestBackendBuildJobsCapsHeavyCompilers(t *testing.T) {
	if got := backendBuildJobs("cuda", 256); got != 8 {
		t.Fatalf("CUDA build jobs = %d, want 8", got)
	}
	if got := backendBuildJobs("cpu", 256); got != 16 {
		t.Fatalf("CPU build jobs = %d, want 16", got)
	}
	if got := backendBuildJobs("cuda", 4); got != 4 {
		t.Fatalf("small host CUDA build jobs = %d, want 4", got)
	}
	if got := backendBuildJobs("cuda", 0); got != 1 {
		t.Fatalf("invalid CPU count build jobs = %d, want 1", got)
	}
}

func TestRouteArchBackendKeepsExplicitBackend(t *testing.T) {
	be := routeArchBackend(&backendInfo{Path: "/main/llama-server", Tag: "llama"}, &placement.ModelProfile{ModelArch: "deepseek4"}, &launchRequest{Backend: "llama", BackendExplicit: true})
	if be == nil || be.Path != "/main/llama-server" || be.Tag != "llama" {
		t.Fatalf("explicit backend must not be route-arch overridden, got %#v", be)
	}
}

func TestConfiguredBackendExplicit(t *testing.T) {
	if !configuredBackendExplicit("llama") || !configuredBackendExplicit("custom") {
		t.Fatal("named configured backends must be explicit")
	}
	if configuredBackendExplicit("") || configuredBackendExplicit("auto") || configuredBackendExplicit("skip") {
		t.Fatal("empty/auto/legacy-skip backend must stay implicit")
	}
}

func TestUpdateRegisteredBackendListContinuesAfterForkFailure(t *testing.T) {
	forks := []backends.Backend{{Tag: "hy3"}, {Tag: "laguna"}, {Tag: "nanbeige42"}}
	var seen []string
	updater := func(tag string) error {
		seen = append(seen, tag)
		if tag == "laguna" {
			return errors.New("build failed")
		}
		return nil
	}
	errs := updateRegisteredBackendList(forks, updater)
	if !reflect.DeepEqual(seen, []string{"hy3", "laguna", "nanbeige42"}) {
		t.Fatalf("fork update stopped early: %#v", seen)
	}
	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "laguna") {
		t.Fatalf("fork errors = %#v, want isolated Laguna failure", errs)
	}
}

func TestBackendGPUCapableProbe(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake-backend probe uses a shell script")
	}
	cpuBin := writeFakeBackend(t, "cpu-server", "echo 'Available devices:'\n")
	gpuBin := writeFakeBackend(t, "gpu-server",
		"echo 'Available devices:'\necho '  CUDA0: NVIDIA GeForce RTX 4070 (11873 MiB, 11710 MiB free)'\n")

	if capable, probed := backendGPUCapable(cpuBin); !probed || capable {
		t.Fatalf("cpu-only build: want probed=true capable=false, got probed=%v capable=%v", probed, capable)
	}
	if capable, probed := backendGPUCapable(gpuBin); !probed || !capable {
		t.Fatalf("gpu build: want probed=true capable=true, got probed=%v capable=%v", probed, capable)
	}
	if _, probed := backendGPUCapable(filepath.Join(t.TempDir(), "nope")); probed {
		t.Fatal("missing binary must report probed=false so caps stays unchanged")
	}
}

func TestGateBackendGPUStripsGPUsForCPUBuild(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake-backend probe uses a shell script")
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Name: "RTX 4070", VRAMTotalMB: 12288}}}

	cpuBe := &backendInfo{Path: writeFakeBackend(t, "cpu-server", "echo 'Available devices:'\n")}
	if got := gateBackendGPU(cpuBe, caps); len(got.GPUs) != 0 {
		t.Fatalf("CPU-only backend: GPUs should be stripped, got %d", len(got.GPUs))
	}
	if len(caps.GPUs) != 1 {
		t.Fatal("gateBackendGPU must not mutate the caller's caps")
	}

	gpuBe := &backendInfo{Path: writeFakeBackend(t, "gpu-server",
		"echo 'Available devices:'\necho '  CUDA0: NVIDIA GeForce RTX 4070'\n")}
	if got := gateBackendGPU(gpuBe, caps); len(got.GPUs) != 1 {
		t.Fatalf("GPU-capable backend: GPUs must be kept, got %d", len(got.GPUs))
	}
}

func isolateConfig(t *testing.T) {
	t.Helper()
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "missing-config"))
	for _, k := range []string{
		"LLM_PORT", "LLM_CTX_SIZE", "LLM_KV_PLACEMENT", "LLM_KV_QUALITY",
		"LLM_SWA_FULL",
		"LLM_BACKEND", "LLAMA_SERVER", "LLM_HOST", "LLM_SPEC", "LLM_VISION",
		"LLM_APP_HOME",
	} {
		t.Setenv(k, "")
	}
}

func TestParseLaunchArgsLegacyModelFirst(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{
		"/models/test.gguf", "--dry-run", "--ctx-size", "fit",
		"--kv-placement", "gpu", "--kv-quality", "high", "--spec", "ngram",
		"--mmproj", "/models/mmproj.gguf", "--ram-budget", "48GB",
		"--", "--no-mmap",
	})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if req.ModelPath != "/models/test.gguf" {
		t.Fatalf("model mismatch: %s", req.ModelPath)
	}
	if req.CtxFlag != "fit" || req.KVPlacement != "gpu" || req.KVQuality != "high" {
		t.Fatalf("placement flags mismatch: %#v", req)
	}
	if req.Host != "127.0.0.1" {
		t.Fatalf("expected safe loopback host, got %q", req.Host)
	}
	if req.SpecMode != "ngram" || req.MMProjPath != "/models/mmproj.gguf" || req.RamBudgetMB != 48*1024 {
		t.Fatalf("advanced flags mismatch: %#v", req)
	}
	if !req.NoMMap {
		t.Fatalf("--no-mmap must feed placement, got %#v", req)
	}
	if len(req.ExtraArgs) != 0 {
		t.Fatalf("extra args mismatch: %v", req.ExtraArgs)
	}
}

func TestParseLaunchArgsDefaultKVQualityIsAuto(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"/models/test.gguf"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if req.KVQuality != "auto" {
		t.Fatalf("default KV quality must remain model-aware auto, got %q", req.KVQuality)
	}
}

func TestParseLaunchArgsSWAFullConfigAndCLIOverrides(t *testing.T) {
	isolateConfig(t)
	t.Setenv("LLM_SWA_FULL", "true")

	req, err := parseLaunchArgs([]string{"model.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	if !hasArg(req.ExtraArgs, "--swa-full") {
		t.Fatalf("configured full SWA cache was not retained: %v", req.ExtraArgs)
	}
	opts := placementOptionsFromRequest(req, &placement.ModelProfile{CTXTrain: 32768}, &backendInfo{Tag: "llama"}, t.TempDir())
	if !opts.SWAFull {
		t.Fatal("configured full SWA cache did not reach placement")
	}

	for _, args := range [][]string{
		{"model.gguf", "--no-swa-full"},
		{"model.gguf", "--swa-full=false"},
		{"model.gguf", "--", "--no-swa-full"},
	} {
		req, err = parseLaunchArgs(args)
		if err != nil {
			t.Fatalf("parse %v: %v", args, err)
		}
		if hasArg(req.ExtraArgs, "--swa-full") || hasArg(req.ExtraArgs, "--no-swa-full") {
			t.Fatalf("disable override leaked a backend flag for %v: %v", args, req.ExtraArgs)
		}
	}

	req, err = parseLaunchArgs([]string{"model.gguf", "--no-swa-full=false"})
	if err != nil || !hasArg(req.ExtraArgs, "--swa-full") {
		t.Fatalf("negative false override should enable full SWA: req=%#v err=%v", req, err)
	}
}

func TestParseLaunchArgsNoMMapFeedsPlacement(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--no-mmap", "-kv", "gpu"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.NoMMap {
		t.Fatalf("expected --no-mmap to set launch request")
	}
	if len(req.ExtraArgs) != 0 {
		t.Fatalf("--no-mmap must not remain a raw backend arg: %v", req.ExtraArgs)
	}
	opts := placementOptionsFromRequest(req, &placement.ModelProfile{CTXTrain: 32768}, &backendInfo{Tag: "llama"}, t.TempDir())
	if !opts.NoMMap {
		t.Fatalf("expected placement options to receive NoMMap")
	}
}

func TestParseLaunchArgsNoMMapAfterDelimiterStillFeedsPlacement(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--", "--no-mmap", "--draft-max", "8"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.NoMMap {
		t.Fatalf("expected passthrough --no-mmap to be promoted into placement")
	}
	want := []string{"--draft-max", "8"}
	if len(req.ExtraArgs) != len(want) || req.ExtraArgs[0] != want[0] || req.ExtraArgs[1] != want[1] {
		t.Fatalf("extra args mismatch: got %v want %v", req.ExtraArgs, want)
	}
}

func TestParseLaunchArgsEqualsForms(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{
		"--port=9090", "--ctx-size=max", "--backend=ik_llama",
		"--gpus=1,3", "--host=127.0.0.1", "--spec=draft", "--parallel=4",
		"--ram-limit-percent=88", "--allow-live-memory-probe=true", "model.gguf",
	})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if req.Port != 9090 || req.CtxFlag != "max" || req.Backend != "ik_llama" {
		t.Fatalf("equals flags mismatch: %#v", req)
	}
	if req.GPUsFlag != "1,3" || req.Host != "127.0.0.1" || req.SpecMode != "draft" || req.Parallel != 4 {
		t.Fatalf("equals placement mismatch: %#v", req)
	}
	if req.RAMLimitPercent != 88 {
		t.Fatalf("RAM limit percent = %d, want 88", req.RAMLimitPercent)
	}
	if !req.AllowLiveMemoryProbe {
		t.Fatal("equals-form live memory probe consent was not retained")
	}
}

func TestParseLaunchArgsSingleDashEqualsFormsAreNotOpaquePassthrough(t *testing.T) {
	isolateConfig(t)
	// -gpus and -kv-quality had a bare double-dash case in the "=value" switch
	// (--gpus=/--kv-quality=) but no single-dash alias, unlike every other
	// value-taking flag with one (-ctx=, -kv=, -t=, -cram=, ...). The
	// single-dash "=value" form matched neither switch and fell through to
	// ExtraArgs verbatim instead of setting GPUsFlag/KVQuality.
	req, err := parseLaunchArgs([]string{
		"-gpus=0,1", "-kv-quality=q5_1", "model.gguf",
	})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if req.GPUsFlag != "0,1" {
		t.Fatalf("-gpus=0,1 was not parsed into GPUsFlag: %#v", req)
	}
	if req.KVQuality != "q5_1" {
		t.Fatalf("-kv-quality=q5_1 was not parsed into KVQuality: %#v", req)
	}
	for _, extra := range req.ExtraArgs {
		if strings.HasPrefix(extra, "-gpus=") || strings.HasPrefix(extra, "-kv-quality=") {
			t.Fatalf("single-dash equals-form flag leaked into ExtraArgs instead of being parsed: %v", req.ExtraArgs)
		}
	}
}

func TestExplicitBatchFlagsFeedPlacementInsteadOfExtraArgs(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{
		"model.gguf", "--batch-size=512", "-ub", "256",
	})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.BatchSizeSet || req.BatchSize != 512 || !req.UBatchSizeSet || req.UBatchSize != 256 {
		t.Fatalf("explicit batch flags were not retained: %#v", req)
	}
	if len(req.ExtraArgs) != 0 {
		t.Fatalf("explicit placement flags must not remain late extra args: %v", req.ExtraArgs)
	}
	opts := placementOptionsFromRequest(req, &placement.ModelProfile{}, &backendInfo{Tag: "llama"}, t.TempDir())
	if !opts.BatchSizeExplicit || !opts.UBatchSizeExplicit || opts.BatchSize != 512 || opts.UBatchSize != 256 {
		t.Fatalf("explicit ubatch intent did not reach calibration scope: %#v", opts)
	}
	if _, err := parseLaunchArgs([]string{"model.gguf", "--batch-size", "128", "--ubatch-size", "256"}); err == nil {
		t.Fatal("batch smaller than microbatch was accepted")
	}
}

func TestBenchmarkCompatArgs(t *testing.T) {
	args := benchmarkCompatArgs([]string{"/models/test.gguf", "--benchmark", "--port", "9090"})
	if len(args) != 4 || args[0] != "--model" || args[1] != "test.gguf" || args[2] != "--port" || args[3] != "9090" {
		t.Fatalf("unexpected benchmark args: %v", args)
	}

	args = benchmarkCompatArgs([]string{"--model=/models/test.gguf", "--benchmark", "--port=9091"})
	if len(args) != 4 || args[0] != "--model" || args[1] != "test.gguf" || args[2] != "--port" || args[3] != "9091" {
		t.Fatalf("unexpected equals benchmark args: %v", args)
	}

	args = benchmarkCompatArgs([]string{"--model", "/models/test.gguf", "--benchmark"})
	if len(args) != 2 || args[0] != "--model" || args[1] != "test.gguf" {
		t.Fatalf("unexpected explicit model benchmark args: %v", args)
	}
}

func TestAutoStartupTimeoutDoublesHugeMoE(t *testing.T) {
	model := &placement.ModelProfile{
		SizeBytes: 146 * 1024 * 1024 * 1024,
		IsMoE:     true,
	}
	if got := autoStartupTimeout(model); got != 30*time.Minute {
		t.Fatalf("huge MoE timeout mismatch: got %v", got)
	}
}

func TestAutoStartupTimeoutDoublesBaseTimeout(t *testing.T) {
	model := &placement.ModelProfile{SizeBytes: 1024 * 1024}
	if got := autoStartupTimeout(model); got != 8*time.Minute {
		t.Fatalf("base timeout mismatch: got %v", got)
	}
}

func TestLongModelLoadWarningPrecedesHugeModelBoundary(t *testing.T) {
	model := &placement.ModelProfile{
		SizeBytes: 146 * 1024 * 1024 * 1024,
		IsMoE:     true,
	}
	warning := longModelLoadWarning(model, 30*time.Minute)
	for _, want := range []string{"Long first load ahead", "MoE model", "146.0 GiB", "more than once", "30m0s"} {
		if !strings.Contains(warning, want) {
			t.Fatalf("warning %q does not contain %q", warning, want)
		}
	}
	if got := longModelLoadWarning(&placement.ModelProfile{SizeBytes: 4 * 1024 * 1024 * 1024}, 8*time.Minute); got != "" {
		t.Fatalf("ordinary model received long-load warning: %q", got)
	}
}

func TestTunedBatchSurvivesBackendMeasuredRecompute(t *testing.T) {
	constraint := tunedBatchConstraintFor(&placement.Strategy{
		BatchSize: 256, UBatchSize: 128, BatchTuned: true, PerformanceTuned: true,
	})
	opts := placement.Options{BatchSize: 4096, UBatchSize: 512, RuntimePolicy: func(*placement.Strategy) {}}
	constraint.apply(&opts)
	if opts.BatchSize != 256 || opts.UBatchSize != 128 || !opts.BatchSizeExplicit || !opts.UBatchSizeExplicit || opts.RuntimePolicy != nil {
		t.Fatalf("tuned batch was not preserved as a recompute constraint: %+v", opts)
	}
	matching := constraint.retain(&placement.Strategy{BatchSize: 256, UBatchSize: 128})
	if !matching.BatchTuned || !matching.PerformanceTuned {
		t.Fatalf("matching recompute lost tuned identity: %+v", matching)
	}
	mismatch := constraint.retain(&placement.Strategy{BatchSize: 128, UBatchSize: 128})
	if mismatch.BatchTuned || mismatch.PerformanceTuned {
		t.Fatalf("altered recovery was mislabeled as the tuned candidate: %+v", mismatch)
	}
}

func TestExactAllocationDoesNotAuthorizeLateralMoESplit(t *testing.T) {
	current := &placement.Strategy{
		Type: placement.MoEOffload, NCPUMoE: 34,
		OTString:    "blk.(0|1).ffn_exps=CUDA0,exps=CPU",
		TensorSplit: []float64{0.27, 0.65, 0.09},
	}
	lateral := *current
	lateral.TensorSplit = []float64{0.26, 0.64, 0.10}
	if backendMeasuredRecomputeWorthVerifying(memoryEvidenceAllocated, current, &lateral) {
		t.Fatal("exact evidence for one argv must not trigger an unmeasured lateral split")
	}
	denser := lateral
	denser.NCPUMoE = 33
	if !backendMeasuredRecomputeWorthVerifying(memoryEvidenceAllocated, current, &denser) {
		t.Fatal("a denser measured re-plan should receive its own contained verification")
	}
	if !backendMeasuredRecomputeWorthVerifying(memoryEvidenceOraclePlanned, current, &lateral) {
		t.Fatal("planned evidence must continue through the recompute verification loop")
	}
}

func TestKnownCommandAcceptsUpdateAlias(t *testing.T) {
	if !knownCommand("update") {
		t.Fatal("expected update command to be known")
	}
	if !knownCommand("--update") {
		t.Fatal("expected legacy --update alias to be known")
	}
	if !knownCommand("freetoken") {
		t.Fatal("expected explicit FreeToken adapter command to be known")
	}
}

func TestFirstWordCouldBeAModelRejectsTyposAndPreservesLegacyLaunches(t *testing.T) {
	modelDir := t.TempDir()
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "missing-config"))
	t.Setenv("LLM_MODEL_DIR", modelDir)
	modelInDir := filepath.Join(modelDir, "named-model")
	if err := os.WriteFile(modelInDir, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	relativeModel := filepath.Join(t.TempDir(), "relative-model")
	if err := os.WriteFile(relativeModel, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, word := range []string{"missing.gguf", relativeModel, "named-model"} {
		if !firstWordCouldBeAModel(word) {
			t.Errorf("legacy model launch %q was rejected", word)
		}
	}
	if firstWordCouldBeAModel("flurb") {
		t.Fatal("misspelled command was accepted as a model launch")
	}
}

func TestParseBenchmarkArgsKeepsPositionalModelAndTrailingFlags(t *testing.T) {
	modelDir := t.TempDir()
	modelPath := filepath.Join(modelDir, "bench.gguf")
	writeGGUFWithTemplate(t, modelPath, "llama", "template")
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "missing-config"))
	t.Setenv("LLM_MODEL_DIR", modelDir)

	for _, args := range [][]string{
		{"bench.gguf", "--port", "9090"},
		{"--port", "9090", "bench.gguf"},
	} {
		port, model, err := parseBenchmarkArgs(args)
		if err != nil {
			t.Fatalf("parse %v: %v", args, err)
		}
		if port != 9090 || model != "bench.gguf" {
			t.Fatalf("parse %v = port %d model %q", args, port, model)
		}
	}
	if _, _, err := parseBenchmarkArgs([]string{"missing.gguf", "--port", "9090"}); err == nil || !strings.Contains(err.Error(), "missing.gguf") {
		t.Fatalf("missing positional model error = %v", err)
	}
	if _, _, err := parseBenchmarkArgs([]string{"bench.gguf", "other.gguf"}); err == nil || !strings.Contains(err.Error(), "at most one") {
		t.Fatalf("multiple positional models error = %v", err)
	}
}

func TestRequireBenchmarkServerFailsFastAndClearly(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	if err := requireBenchmarkServer(port, time.Second); err != nil {
		t.Fatalf("listening benchmark server rejected: %v", err)
	}
	if err := ln.Close(); err != nil {
		t.Fatal(err)
	}
	if err := requireBenchmarkServer(port, 100*time.Millisecond); err == nil || !strings.Contains(err.Error(), "no llama-server on port") || !strings.Contains(err.Error(), "--port") {
		t.Fatalf("closed benchmark port error = %v", err)
	}
}

func TestResolveCtxFlag(t *testing.T) {
	if got := resolveCtxFlag("fit", 131072); got != 0 {
		t.Fatalf("fit should resolve to auto 0, got %d", got)
	}
	if got := resolveCtxFlag("max", 131072); got != 131072 {
		t.Fatalf("max should resolve to native ctx, got %d", got)
	}
	if got := resolveCtxFlag("8192", 131072); got != 8192 {
		t.Fatalf("manual ctx mismatch: %d", got)
	}
}

func TestParseLaunchArgsFlagFirstLaunch(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"--cpu", "--ctx-size", "2048", "--parallel", "2", "model.gguf"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.CPUMode || req.CtxFlag != "2048" || req.Parallel != 2 || req.ModelPath != "model.gguf" {
		t.Fatalf("flag-first parse mismatch: %#v", req)
	}
}

func TestParseLaunchArgsBenchmark(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--benchmark", "--port", "9090"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.Benchmark || req.ModelPath != "model.gguf" || req.Port != 9090 {
		t.Fatalf("benchmark parse mismatch: %#v", req)
	}
}

func TestSelectBackendBackendFlagOverridesConfiguredServerBin(t *testing.T) {
	dir := t.TempDir()
	ikPath := filepath.Join(dir, "ik-llama-server")
	vulkanPath := filepath.Join(dir, "vulkan-llama-server")
	if err := os.WriteFile(ikPath, []byte("#!/bin/sh\necho ikawrakow split-mode-graph\n"), 0755); err != nil {
		t.Fatalf("write ik backend: %v", err)
	}
	if err := os.WriteFile(vulkanPath, []byte("#!/bin/sh\necho vulkan backend\n"), 0755); err != nil {
		t.Fatalf("write vulkan backend: %v", err)
	}

	caps := &detect.Capabilities{Backends: []detect.Backend{
		{Name: "llama-server", Path: vulkanPath},
	}}
	req := &launchRequest{
		ServerBin:       ikPath,
		AppHome:         t.TempDir(),
		Backend:         "vulkan",
		BackendExplicit: true,
	}
	be := selectBackend(caps, req)
	if be == nil || be.Path != vulkanPath || be.Tag != "vulkan" {
		t.Fatalf("expected explicit backend to override configured server bin, got %#v", be)
	}
}

func TestSelectBackendExplicitServerBinWins(t *testing.T) {
	dir := t.TempDir()
	ikPath := filepath.Join(dir, "ik-llama-server")
	vulkanPath := filepath.Join(dir, "vulkan-llama-server")
	if err := os.WriteFile(ikPath, []byte("#!/bin/sh\necho ikawrakow split-mode-graph\n"), 0755); err != nil {
		t.Fatalf("write ik backend: %v", err)
	}
	if err := os.WriteFile(vulkanPath, []byte("#!/bin/sh\necho vulkan backend\n"), 0755); err != nil {
		t.Fatalf("write vulkan backend: %v", err)
	}

	caps := &detect.Capabilities{Backends: []detect.Backend{
		{Name: "llama-server", Path: vulkanPath},
	}}
	req := &launchRequest{
		ServerBin:         ikPath,
		ServerBinExplicit: true,
		Backend:           "vulkan",
		BackendExplicit:   true,
	}
	be := selectBackend(caps, req)
	if be == nil || be.Path != ikPath || be.Tag != "ik_llama" {
		t.Fatalf("expected explicit server bin to win, got %#v", be)
	}
}

func TestChooseAutoBackendPrefersCanonicalWhenBothSupportArchitecture(t *testing.T) {
	canonical := &backendInfo{Path: "/app/.bin/llama-server-cuda", Tag: "llama"}
	global := &backendInfo{Path: "/home/me/.local/bin/llama-server", Tag: "ik_llama", IsIK: true}
	probe := func(path, arch string) (bool, bool) {
		if arch != "deepseek4" {
			t.Fatalf("probe arch = %q, want deepseek4", arch)
		}
		return true, true
	}
	got, _ := chooseAutoBackend([]autoBackendCandidate{
		{info: global},
		{info: canonical, canonical: true},
	}, "deepseek4", probe, nil)
	if got != canonical {
		t.Fatalf("auto backend = %#v, want canonical %#v", got, canonical)
	}
}

func TestChooseAutoBackendArchitectureSupportBeatsCanonicalPath(t *testing.T) {
	canonical := &backendInfo{Path: "/app/.bin/llama-server-cuda", Tag: "llama"}
	fork := &backendInfo{Path: "/fork/build/bin/llama-server", Tag: "fork"}
	probe := func(path, _ string) (bool, bool) {
		return path == fork.Path, true
	}
	got, _ := chooseAutoBackend([]autoBackendCandidate{
		{info: canonical, canonical: true},
		{info: fork},
	}, "future-arch", probe, nil)
	if got != fork {
		t.Fatalf("auto backend = %#v, want supporting fork %#v", got, fork)
	}
}

func TestChooseAutoBackendHonoursRequiredIKFamily(t *testing.T) {
	mainline := &backendInfo{Path: "/app/.bin/llama-server-cuda", Tag: "llama"}
	ik := &backendInfo{Path: "/app/.bin/ik_llama-server-cuda", Tag: "ik_llama", IsIK: true}
	unknownProbe := func(string, string) (bool, bool) { return false, false }
	got, _ := chooseAutoBackend([]autoBackendCandidate{
		{info: mainline, canonical: true},
		{info: ik, canonical: true},
	}, "minimax-m3", unknownProbe, nil)
	if got != ik {
		t.Fatalf("auto backend = %#v, want required ik %#v", got, ik)
	}
}

func TestChooseAutoBackendRejectsProfileIncompatibleCandidate(t *testing.T) {
	mainline := &backendInfo{Path: "/app/.bin/llama-server-cuda", Tag: "llama"}
	ik := &backendInfo{Path: "/app/.bin/ik_llama-server-cuda", Tag: "ik_llama", IsIK: true}
	got, _ := chooseAutoBackend([]autoBackendCandidate{
		{info: mainline, canonical: true, incompatible: true},
		{info: ik, canonical: true},
	}, "deepseek4", func(string, string) (bool, bool) { return true, true }, nil)
	if got != ik {
		t.Fatalf("q4-compatible V4 backend = %#v, want ik %#v", got, ik)
	}
}

// TestChooseAutoBackendPrefersFileBackedForLargeMoE guards FIX 2: for a
// large-CPU-expert MoE (like the ~94GB-CPU-expert DeepSeek-V4 that was
// OOM-killed under ik_llama's anonymous CUDA-host experts), auto selection must
// prefer the file-backed mainline backend when both it and ik_llama are in the
// same support class. This is exactly the case that previously lost: mainline
// was marked incompatible for bf16 by resolveKVQuality, so ik_llama won by
// default and the cgroup killed the launch.
func TestChooseAutoBackendPrefersFileBackedForLargeMoE(t *testing.T) {
	mainline := &backendInfo{Path: "/app/.bin/llama-server-cuda", Tag: "llama", Dialect: "llama"}
	ik := &backendInfo{Path: "/app/.bin/ik_llama-server-cuda", Tag: "ik_llama", Dialect: "ik_llama", IsIK: true}
	probe := func(string, string) (bool, bool) { return true, true }

	// Large MoE (94 GiB) with both backends viable: file-backed mainline wins
	// over anonymous ik_llama even though ik_llama is listed first.
	largeMoE := &placement.ModelProfile{ModelArch: "deepseek4", IsMoE: true, TotalSizeMB: 94 * 1024}
	got, _ := chooseAutoBackend([]autoBackendCandidate{
		{info: ik},
		{info: mainline},
	}, "deepseek4", probe, largeMoE)
	if got != mainline {
		t.Fatalf("auto backend for large-CPU-expert MoE = %#v, want file-backed mainline %#v", got, mainline)
	}

	// A small MoE (below the threshold) keeps the pre-existing behavior:
	// discovery order wins when nothing else separates the candidates, so ik_llama
	// listed first is chosen.
	smallMoE := &placement.ModelProfile{ModelArch: "deepseek4", IsMoE: true, TotalSizeMB: 4 * 1024}
	got, _ = chooseAutoBackend([]autoBackendCandidate{
		{info: ik},
		{info: mainline},
	}, "deepseek4", probe, smallMoE)
	if got != ik {
		t.Fatalf("auto backend for small MoE = %#v, want discovery-order ik %#v", got, ik)
	}

	// A large non-MoE model does not engage the tie-break either.
	largeDense := &placement.ModelProfile{ModelArch: "deepseek4", IsMoE: false, TotalSizeMB: 94 * 1024}
	got, _ = chooseAutoBackend([]autoBackendCandidate{
		{info: ik},
		{info: mainline},
	}, "deepseek4", probe, largeDense)
	if got != ik {
		t.Fatalf("auto backend for large dense = %#v, want discovery-order ik %#v", got, ik)
	}
}

// TestChooseAutoBackendFileBackedBeatsCanonicalForLargeMoE verifies that the
// memory tie-break outranks canonical locality for a large-CPU-expert MoE:
// survival under the cgroup is more important than which path is under
// APP_HOME, so a file-backed non-canonical backend beats a canonical anonymous
// one. The tie-break is scoped to the same support class, and to the
// large-CPU-expert-MoE case only.
func TestChooseAutoBackendFileBackedBeatsCanonicalForLargeMoE(t *testing.T) {
	mainline := &backendInfo{Path: "/app/.bin/llama-server-cuda", Tag: "llama", Dialect: "llama"}
	ikCanonical := &backendInfo{Path: "/app/.bin/ik_llama-server-cuda", Tag: "ik_llama", Dialect: "ik_llama", IsIK: true}
	probe := func(string, string) (bool, bool) { return true, true }
	largeMoE := &placement.ModelProfile{ModelArch: "deepseek4", IsMoE: true, TotalSizeMB: 94 * 1024}

	// File-backed non-canonical mainline beats canonical anonymous ik for a
	// large-CPU-expert MoE.
	got, _ := chooseAutoBackend([]autoBackendCandidate{
		{info: ikCanonical, canonical: true},
		{info: mainline},
	}, "deepseek4", probe, largeMoE)
	if got != mainline {
		t.Fatalf("file-backed mainline must beat canonical anonymous ik for large MoE, got %#v", got)
	}

	// Without the tie-break (small MoE), canonical locality wins as before.
	smallMoE := &placement.ModelProfile{ModelArch: "deepseek4", IsMoE: true, TotalSizeMB: 4 * 1024}
	got, _ = chooseAutoBackend([]autoBackendCandidate{
		{info: ikCanonical, canonical: true},
		{info: mainline},
	}, "deepseek4", probe, smallMoE)
	if got != ikCanonical {
		t.Fatalf("canonical ik must win over non-canonical mainline when tie-break not engaged, got %#v", got)
	}
}

// TestChooseAutoBackendProvenUnsupportedNotSelected reproduces the muse-glimmer
// defect: the canonical backend probes (false, true) — its loader was read and
// does not contain the architecture literal — and must NOT be auto-selected.
// Before the fix an all-support=0 set fell through to the canonical tie-break
// and launched a backend that died with "unknown model architecture" at load.
func TestChooseAutoBackendProvenUnsupportedNotSelected(t *testing.T) {
	mainline := &backendInfo{Path: "/app/.bin/llama-server-cuda", Tag: "llama"}
	fork := &backendInfo{Path: "/fork/build/bin/llama-server", Tag: "fork"}
	probe := func(path, _ string) (bool, bool) {
		return false, true // loader read, literal absent
	}
	got, unsupported := chooseAutoBackend([]autoBackendCandidate{
		{info: mainline, canonical: true},
		{info: fork},
	}, "muse-glimmer", probe, nil)
	if got != nil {
		t.Fatalf("proven-unsupported backend was auto-selected: %#v", got)
	}
	if unsupported != mainline {
		t.Fatalf("all-proven-unsupported set must name the best candidate to refuse, got %#v", unsupported)
	}
}

// TestChooseAutoBackendAllCandidatesUnsupportedRefuses covers the FAIL-CLOSED
// launch path: when every viable candidate proves unsupported, chooseAutoBackend
// returns nil and the caller refuses instead of launching.
func TestChooseAutoBackendAllCandidatesUnsupportedRefuses(t *testing.T) {
	mainline := &backendInfo{Path: "/app/.bin/llama-server-cuda", Tag: "llama"}
	vulkan := &backendInfo{Path: "/app/.bin/llama-server-vulkan", Tag: "vulkan"}
	probe := func(string, string) (bool, bool) { return false, true }
	got, unsupported := chooseAutoBackend([]autoBackendCandidate{
		{info: mainline, canonical: true},
		{info: vulkan, canonical: true},
	}, "muse-glimmer", probe, nil)
	if got != nil {
		t.Fatalf("all-proven-unsupported set selected %#v instead of refusing", got)
	}
	if unsupported == nil {
		t.Fatal("all-proven-unsupported set must report the candidate to name in the refusal message")
	}
}

// TestChooseAutoBackendMixedUnsupportedAndUnprobeableKeepsUnprobeable guards the
// fail-open side: an unprobeable backend (probe returned false, false) remains
// launchable and must be chosen over a set where every OTHER candidate is
// proven-unsupported. Refusing on a failed probe would be worse than the loader
// error it replaces.
func TestChooseAutoBackendMixedUnsupportedAndUnprobeableKeepsUnprobeable(t *testing.T) {
	mainline := &backendInfo{Path: "/app/.bin/llama-server-cuda", Tag: "llama"}
	unprobeable := &backendInfo{Path: "/home/me/.local/bin/llama-server", Tag: "llama"}
	probe := func(path, _ string) (bool, bool) {
		if path == unprobeable.Path {
			return false, false // could not run the probe: must stay launchable
		}
		return false, true
	}
	got, unsupported := chooseAutoBackend([]autoBackendCandidate{
		{info: mainline, canonical: true},
		{info: unprobeable},
	}, "muse-glimmer", probe, nil)
	if got != unprobeable {
		t.Fatalf("unprobeable backend must stay launchable and be chosen, got %#v", got)
	}
	if unsupported != nil {
		t.Fatalf("launchable unprobeable candidate must not be reported as unsupported, got %#v", unsupported)
	}
}

// TestChooseAutoBackendMandatedIKStillResolves guards the interplay with the
// mandated-ik branch: an all-support=0 set that came from "arch requires ik but
// the candidate is not ik" must NOT be refused by the FAIL-CLOSED gate — the
// caller routes that to preflightIKOnlyArch, which names the exact fix. Only
// probe-proven unsupported (false, true) is grounds for refusal.
func TestChooseAutoBackendMandatedIKStillResolves(t *testing.T) {
	mainline := &backendInfo{Path: "/app/.bin/llama-server-cuda", Tag: "llama"}
	ik := &backendInfo{Path: "/app/.bin/ik_llama-server-cuda", Tag: "ik_llama", IsIK: true}
	probe := func(string, string) (bool, bool) { return false, false }
	got, unsupported := chooseAutoBackend([]autoBackendCandidate{
		{info: mainline, canonical: true},
		{info: ik, canonical: true},
	}, "minimax-m3", probe, nil)
	if got != ik {
		t.Fatalf("required-ik family must still resolve to the ik candidate, got %#v", got)
	}
	if unsupported != nil {
		t.Fatalf("mandated-ik set must not be reported as probe-unsupported, got %#v", unsupported)
	}
}

// The auto baseline is f16, NOT the q8_0 that ik_llama's loader accepts. V4's
// attention weights are already FP8, so a q8_0 K-cache compounds the precision
// loss into repetition loops or '='-spam that no backend reports as an error.
func TestNormalizeArchKVRequestPinsDeepSeek4ToF16(t *testing.T) {
	for _, quality := range []string{"q4_0", "q8_0"} {
		req := &launchRequest{KVQuality: quality, KVTypeK: quality, KVTypeV: quality}
		normalizeArchKVRequest(req, &placement.ModelProfile{ModelArch: "deepseek4"})
		if req.KVQuality != "f16" || req.KVTypeK != "" || req.KVTypeV != "" {
			t.Fatalf("auto KV normalization from %s = quality %q, K %q, V %q", quality, req.KVQuality, req.KVTypeK, req.KVTypeV)
		}
	}

	// bf16 carries no requantization loss, so it is left alone.
	keep := &launchRequest{KVQuality: "bf16"}
	normalizeArchKVRequest(keep, &placement.ModelProfile{ModelArch: "deepseek4"})
	if keep.KVQuality != "bf16" {
		t.Fatalf("bf16 K-cache was rewritten to %q", keep.KVQuality)
	}

	// Other architectures keep whatever the user asked for.
	other := &launchRequest{KVQuality: "q4_0"}
	normalizeArchKVRequest(other, &placement.ModelProfile{ModelArch: "qwen3moe"})
	if other.KVQuality != "q4_0" {
		t.Fatalf("non-deepseek4 KV quality was rewritten to %q", other.KVQuality)
	}
}

func TestBackendFeatureCompatibilityDropsUnsupportedFullSWA(t *testing.T) {
	req := &launchRequest{ExtraArgs: []string{"--swa-full", "--metrics"}}
	be := &backendInfo{Path: "/ik/llama-server", Help: "--ctx-checkpoints-interval N"}
	applyBackendFeatureCompatibility(req, &placement.ModelProfile{ModelArch: "deepseek4"}, be)
	if hasArg(req.ExtraArgs, "--swa-full") || !hasArg(req.ExtraArgs, "--metrics") {
		t.Fatalf("backend feature normalization = %#v", req.ExtraArgs)
	}

	supported := &launchRequest{ExtraArgs: []string{"--swa-full"}}
	applyBackendFeatureCompatibility(supported, &placement.ModelProfile{ModelArch: "laguna", SlidingWindow: 512},
		&backendInfo{Path: "/laguna/llama-server", Help: "--swa-full"})
	if !hasArg(supported.ExtraArgs, "--swa-full") {
		t.Fatal("supported Full SWA flag was removed")
	}
}

// A model with no sliding-window layer cannot use a full SWA cache even when the
// backend advertises --swa-full in its --help (mainline llama-server does this
// and silently disables swa_full at load for n_swa==0 models). The deterministic
// GGUF gate must remove the flag before placement and the probe-key computation.
func TestBackendFeatureCompatibilityDropsFullSWAWhenModelHasNoSWAWindow(t *testing.T) {
	// Mainline advertises --swa-full in help, so the old help-surface gate never
	// fired; only the new SlidingWindow<=0 gate catches this.
	req := &launchRequest{ExtraArgs: []string{"--swa-full", "--metrics"}}
	be := &backendInfo{Path: "/llama/llama-server", Help: "--swa-full --ctx-checkpoints-interval N"}
	applyBackendFeatureCompatibility(req, &placement.ModelProfile{ModelArch: "qwen3moe", SlidingWindow: 0}, be)
	if hasArg(req.ExtraArgs, "--swa-full") || !hasArg(req.ExtraArgs, "--metrics") {
		t.Fatalf("model with SlidingWindow<=0 kept --swa-full after backend feature normalization: %#v", req.ExtraArgs)
	}
}

// A model WITH a sliding window must keep the flag even when the backend help is
// empty (unknown help surface is not evidence of unsupported).
func TestBackendFeatureCompatibilityKeepsFullSWAWhenModelHasSWAWindow(t *testing.T) {
	req := &launchRequest{ExtraArgs: []string{"--swa-full"}}
	applyBackendFeatureCompatibility(req, &placement.ModelProfile{ModelArch: "qwen3moe", SlidingWindow: 512}, &backendInfo{Path: "/llama/llama-server"})
	if !hasArg(req.ExtraArgs, "--swa-full") {
		t.Fatal("model with a sliding-window layer lost --swa-full")
	}
}

// An explicit user-supplied --swa-full on a no-SWA model must be preserved and
// fail closed, matching the -khad explicit-passthrough behavior. Explicit intent
// is carried in req.OriginalArgs (see userExplicitBackendFlag).
func TestBackendFeatureCompatibilityPreservesExplicitFullSWAonNoSWAModel(t *testing.T) {
	req := &launchRequest{ExtraArgs: []string{"--swa-full"}, OriginalArgs: []string{"--swa-full"}}
	applyBackendFeatureCompatibility(req, &placement.ModelProfile{ModelArch: "qwen3moe", SlidingWindow: 0}, &backendInfo{Path: "/ik/llama-server", Help: "--swa-full"})
	if !hasArg(req.ExtraArgs, "--swa-full") {
		t.Fatal("explicit user-supplied --swa-full was silently disabled on a no-SWA model")
	}
}

func TestBackendFeatureCompatibilityDisablesGeneratedDeepSeek4KHadamard(t *testing.T) {
	req := &launchRequest{}
	be := &backendInfo{Path: "/ik/llama-server", IsIK: true}
	applyBackendFeatureCompatibility(req, &placement.ModelProfile{ModelArch: "deepseek4"}, be)
	if reason := req.DisabledBackendFlags["-khad"]; reason == "" {
		t.Fatal("known unsupported generated -khad was not disabled")
	}
	got := applyDisabledBackendFlags([]string{"llama-server", "-khad", "-muge"}, req.DisabledBackendFlags)
	want := []string{"llama-server", "-muge"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("compatibility-filtered args = %#v, want %#v", got, want)
	}
}

func TestBackendFeatureCompatibilityPreservesExplicitKHadamard(t *testing.T) {
	req := &launchRequest{ExtraArgs: []string{"-khad"}}
	be := &backendInfo{Path: "/ik/llama-server", IsIK: true}
	applyBackendFeatureCompatibility(req, &placement.ModelProfile{ModelArch: "deepseek4"}, be)
	if _, disabled := req.DisabledBackendFlags["-khad"]; disabled {
		t.Fatal("explicit user-supplied -khad was silently disabled")
	}
}

// Both backend families are held to f16. ik_llama's loader accepting a q8_0
// K-cache for deepseek4 is not permission to use one: the old rule read that
// acceptance as support and promoted ik launches to q8_0, which is the silent
// corruption path. Mainline rejects it outright, so only ik ever got there.
func TestBackendFeatureCompatibilityPinsDeepSeek4KVToF16(t *testing.T) {
	for _, be := range []*backendInfo{
		{Path: "/ik/llama-server", IsIK: true},
		{Path: "/llama/llama-server"},
	} {
		for _, quality := range []string{"q4_0", "q8_0"} {
			req := &launchRequest{KVQuality: quality, KVTypeK: quality, KVTypeV: quality}
			applyBackendFeatureCompatibility(req, &placement.ModelProfile{ModelArch: "deepseek4"}, be)
			if req.KVQuality != "f16" || req.KVTypeK != "" || req.KVTypeV != "" {
				t.Fatalf("DeepSeek4 KV on %s from %s = quality %q, K %q, V %q",
					be.Path, quality, req.KVQuality, req.KVTypeK, req.KVTypeV)
			}
		}
	}
}

func TestRouteArchBackendKeepsExplicitServerBinary(t *testing.T) {
	explicit := &backendInfo{Path: "/chosen/llama-server", Tag: "llama"}
	got := routeArchBackend(explicit, &placement.ModelProfile{ModelArch: "laguna"}, &launchRequest{ServerBinExplicit: true})
	if got != explicit {
		t.Fatalf("explicit server binary was architecture-routed: %#v", got)
	}
}

func TestSelectBackendSkipsLoaderFailedCUDA(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake backend probe uses a shell script")
	}
	appHome := t.TempDir()
	binDir := filepath.Join(appHome, ".bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	cuda := filepath.Join(binDir, "ik_llama-server-cuda")
	vulkan := filepath.Join(binDir, "llama-server-vulkan")
	if err := os.WriteFile(cuda, []byte("#!/bin/sh\necho 'error while loading shared libraries: libnccl.so.2: cannot open shared object file' >&2\nexit 127\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(vulkan, []byte("#!/bin/sh\necho 'usage: llama-server [--help] [--version]'\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	req := &launchRequest{
		ServerBin: cuda,
		AppHome:   appHome,
		Backend:   "auto",
	}
	be := selectBackend(&detect.Capabilities{}, req)
	if be == nil || be.Path != vulkan {
		t.Fatalf("loader-failed CUDA should lose to working Vulkan, got %#v", be)
	}
}

func TestSelectBackendLegacyConfiguredSkipUsesConfiguredServerBin(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake backend probe uses a shell script")
	}
	mainline := writeFakeBackend(t, "llama-server", "echo 'mainline llama.cpp'\n")
	req := &launchRequest{
		ServerBin:       mainline,
		Backend:         "skip",
		BackendExplicit: false,
	}
	be := selectBackend(&detect.Capabilities{}, req)
	if be == nil || be.Path != mainline {
		t.Fatalf("legacy launcher-only backend setting did not use configured server: %#v", be)
	}
}

func TestSelectBackendUsesConfiguredAppHomeForExplicitIK(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake backend probe uses a shell script")
	}
	appHome := t.TempDir()
	ikPath := filepath.Join(appHome, ".src", "ik_llama.cpp", "build", "bin", "llama-server")
	if err := os.MkdirAll(filepath.Dir(ikPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(ikPath, []byte("#!/bin/sh\necho 'ikawrakow split-mode-graph'\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	mainline := writeFakeBackend(t, "llama-server-cuda", "echo 'mainline llama.cpp'\n")
	globalIK := writeFakeBackend(t, "global-ik-llama-server", "echo 'ikawrakow split-mode-graph'\n")
	req := &launchRequest{
		ServerBin:       mainline,
		AppHome:         appHome,
		Backend:         "ik_llama",
		BackendExplicit: true,
	}

	be := selectBackend(&detect.Capabilities{Backends: []detect.Backend{{Name: "ik_llama", Path: globalIK}}}, req)
	if be == nil || be.Path != ikPath || !be.IsIK || be.Tag != "ik_llama" {
		t.Fatalf("configured APP_HOME IK backend was not selected: %#v", be)
	}
}

func TestSelectBackendMissingExplicitNameNeverFallsBack(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake backend probe uses a shell script")
	}
	mainline := writeFakeBackend(t, "llama-server-cuda", "echo 'mainline llama.cpp'\n")
	req := &launchRequest{
		ServerBin:       mainline,
		AppHome:         t.TempDir(),
		Backend:         "missing-explicit-backend",
		BackendExplicit: true,
	}
	if be := selectBackend(&detect.Capabilities{}, req); be != nil {
		t.Fatalf("missing explicit backend silently fell back to %#v", be)
	}
	message := backendUnavailableMessage(req)
	if !strings.Contains(message, req.Backend) || !strings.Contains(message, req.AppHome) {
		t.Fatalf("explicit backend diagnostic is not actionable: %q", message)
	}
}

// TestBackendUnavailableMessageSurfacesProvenUnsupportedReason guards the
// FAIL-CLOSED message path: when auto selection refuses because every candidate
// was probe-proven to lack the architecture, backendUnavailableMessage must
// surface that specific reason instead of the generic "no llama-server binary
// found" fallback. This is the message the user sees instead of the cryptic
// "unknown model architecture" loader error.
func TestBackendUnavailableMessageSurfacesProvenUnsupportedReason(t *testing.T) {
	req := &launchRequest{BackendUnavailableReason: backendUnavailableReason("muse-glimmer", "/x/llama-server")}
	message := backendUnavailableMessage(req)
	if !strings.Contains(message, "muse-glimmer") {
		t.Fatalf("FAIL-CLOSED reason was not surfaced verbatim: %q", message)
	}
	if strings.Contains(message, "no llama-server binary found") {
		t.Fatalf("FAIL-CLOSED reason fell through to the generic fallback: %q", message)
	}
	// An empty reason keeps the pre-existing generic fallback for explicit
	// backend requests.
	req2 := &launchRequest{Backend: "some-backend", BackendExplicit: true, AppHome: "/tmp/app"}
	msg2 := backendUnavailableMessage(req2)
	if !strings.Contains(msg2, "some-backend") || strings.Contains(msg2, "muse-glimmer") {
		t.Fatalf("generic fallback changed behaviour: %q", msg2)
	}
}

// TestBackendUnavailableReasonNovelArchNamesMainline guards the actionable hint
// for a NOVEL architecture (one with no reviewed recipe): the message must point
// at advancing the mainline llama.cpp backend or installing a fork, not at a
// nonexistent recipe install line.
func TestBackendUnavailableReasonNovelArchNamesMainline(t *testing.T) {
	if len(backends.RecipesForArch("muse-glimmer")) > 0 {
		t.Skip("muse-glimmer unexpectedly has a reviewed recipe; pick a novel arch for this fixture")
	}
	message := backendUnavailableReason("muse-glimmer", "/x/llama-server")
	if !strings.Contains(message, "muse-glimmer") {
		t.Fatalf("message lacks the architecture: %q", message)
	}
	if !strings.Contains(message, "newer llama.cpp mainline") {
		t.Fatalf("novel-arch message does not name advancing the mainline: %q", message)
	}
	if !strings.Contains(message, "search open llama.cpp PRs") || !strings.Contains(message, "fork") {
		t.Fatalf("novel-arch message does not offer a fork search: %q", message)
	}
}

// TestBackendUnavailableReasonRecipeArchKeepsRecipeHint guards that an arch with
// a reviewed recipe keeps the install-<recipe> hint rather than the mainline
// wording.
func TestBackendUnavailableReasonRecipeArchKeepsRecipeHint(t *testing.T) {
	// Use a non-helper reviewed recipe: RecipesForArch excludes helper-only
	// entries, so a helper-only arch would (correctly) take the novel-arch
	// wording and this fixture would test the wrong branch.
	arch := ""
	for _, recipe := range backends.Recipes() {
		if !recipe.HelperOnly {
			arch = recipe.RouteArch
			break
		}
	}
	if arch == "" {
		t.Skip("no non-helper reviewed recipe in this build")
	}
	message := backendUnavailableReason(arch, "/x/llama-server")
	if !strings.Contains(message, "ggrun backend install") {
		t.Fatalf("recipe-backed arch lost the install hint: %q", message)
	}
	if strings.Contains(message, "newer llama.cpp mainline") {
		t.Fatalf("recipe-backed arch wrongly named the mainline path: %q", message)
	}
}

// TestOfferMainlineBackendUpdate guards the interactive offer seam: an
// unsupported NOVEL architecture with a FAIL-CLOSED reason prompts, and an
// accepted answer runs the update; declined, non-terminal, recipe-backed, and
// non-FAIL-CLOSED inputs never prompt or run.
func TestOfferMainlineBackendUpdate(t *testing.T) {
	if len(backends.RecipesForArch("muse-glimmer")) > 0 {
		t.Skip("muse-glimmer unexpectedly has a reviewed recipe; pick a novel arch for this fixture")
	}
	novelReq := func() *launchRequest {
		return &launchRequest{BackendUnavailableReason: backendUnavailableReason("muse-glimmer", "/x/llama-server")}
	}
	novelModel := &placement.ModelProfile{ModelArch: "muse-glimmer"}

	ran := 0
	noop := func() error { ran++; return nil }

	// Accept on a terminal runs the update.
	var out bytes.Buffer
	ok := offerMainlineBackendUpdateWith(novelReq(), novelModel, false, strings.NewReader("y\n"), &out, true, noop)
	if !ok || ran != 1 {
		t.Fatalf("accepted prompt: ok=%v ran=%d (want true,1)", ok, ran)
	}
	if !strings.Contains(out.String(), "muse-glimmer") {
		t.Fatalf("prompt did not name the architecture: %q", out.String())
	}

	// Decline on a terminal does not run.
	ran = 0
	out.Reset()
	if ok := offerMainlineBackendUpdateWith(novelReq(), novelModel, false, strings.NewReader("n\n"), &out, true, noop); ok || ran != 0 {
		t.Fatalf("declined prompt: ok=%v ran=%d (want false,0)", ok, ran)
	}

	// Non-terminal never prompts nor runs.
	ran = 0
	if ok := offerMainlineBackendUpdateWith(novelReq(), novelModel, false, strings.NewReader("y\n"), io.Discard, false, noop); ok || ran != 0 {
		t.Fatalf("non-terminal: ok=%v ran=%d (want false,0)", ok, ran)
	}

	// assumeYes runs without prompting.
	ran = 0
	if ok := offerMainlineBackendUpdateWith(novelReq(), novelModel, true, strings.NewReader(""), io.Discard, false, noop); !ok || ran != 1 {
		t.Fatalf("assume-yes: ok=%v ran=%d (want true,1)", ok, ran)
	}

	// A recipe-backed architecture is not eligible (reviewedRecipeRequiredForMain
	// handles it before this offer). hy_v3 is a non-helper reviewed recipe.
	if len(backends.RecipesForArch("hy_v3")) == 0 {
		t.Skip("hy_v3 reviewed recipe missing from catalog")
	}
	ran = 0
	if ok := offerMainlineBackendUpdateWith(novelReq(), &placement.ModelProfile{ModelArch: "hy_v3"}, false, strings.NewReader("y\n"), io.Discard, true, noop); ok || ran != 0 {
		t.Fatalf("recipe-backed arch should not offer a mainline update: ok=%v ran=%d", ok, ran)
	}

	// No FAIL-CLOSED reason (e.g. explicit-backend fallback) is not eligible.
	ran = 0
	if ok := offerMainlineBackendUpdateWith(&launchRequest{}, novelModel, false, strings.NewReader("y\n"), io.Discard, true, noop); ok || ran != 0 {
		t.Fatalf("no FAIL-CLOSED reason should not offer: ok=%v ran=%d", ok, ran)
	}

	// An update failure returns false (caller keeps the dead-end error path).
	ran = 0
	fail := func() error { ran++; return errors.New("build failed") }
	if ok := offerMainlineBackendUpdateWith(novelReq(), novelModel, true, strings.NewReader(""), io.Discard, false, fail); ok || ran != 1 {
		t.Fatalf("failed update: ok=%v ran=%d (want false,1)", ok, ran)
	}
}

func TestOfferDiscoveredArchForkInstallsOpenPR(t *testing.T) {
	if len(backends.RecipesForArch("muse-glimmer")) > 0 {
		t.Skip("muse-glimmer unexpectedly has a reviewed recipe")
	}
	req := &launchRequest{BackendUnavailableReason: backendUnavailableReason("muse-glimmer", "/x/llama-server")}
	model := &placement.ModelProfile{
		ModelArch: "muse-glimmer",
		Path:      "/models/Muse-Glimmer-30B-UD-Q8_K_XL.gguf",
		Basename:  "Muse-Glimmer-30B-UD-Q8_K_XL.gguf",
	}
	pr := backends.ArchForkPR{
		Arch: "muse-glimmer", Number: 99, Title: "add muse-glimmer",
		URL:    "https://github.com/ggml-org/llama.cpp/pull/99",
		GitURL: "https://github.com/example/llama.cpp.git",
		Branch: "muse", Commit: "cccccccccccccccccccccccccccccccccccccccc",
	}
	search := func(context.Context, *placement.ModelProfile) ([]backends.ArchForkPR, error) {
		return []backends.ArchForkPR{pr}, nil
	}
	var installed backends.Recipe
	install := func(recipe backends.Recipe) error { installed = recipe; return nil }

	var out bytes.Buffer
	ok := offerDiscoveredArchForkWith(req, model, false, strings.NewReader("y\n"), &out, true, search, install)
	if !ok || installed.GitURL != pr.GitURL || installed.RouteArch != "muse-glimmer" || installed.Commit != pr.Commit {
		t.Fatalf("accepted fork: ok=%v recipe=%+v out=%q", ok, installed, out.String())
	}
	if installed.Tag != "muse-glimmer-30b" {
		t.Fatalf("fork manager tag=%q, want model name muse-glimmer-30b", installed.Tag)
	}
	if !strings.Contains(out.String(), "PR #99") || !strings.Contains(out.String(), "separate fork") ||
		!strings.Contains(out.String(), "muse-glimmer-30b") {
		t.Fatalf("prompt did not name an isolated model-named fork: %q", out.String())
	}

	installed = backends.Recipe{}
	out.Reset()
	if ok := offerDiscoveredArchForkWith(req, model, false, strings.NewReader("n\n"), &out, true, search, install); ok || installed.GitURL != "" {
		t.Fatalf("declined fork still installed: ok=%v recipe=%+v", ok, installed)
	}

	installed = backends.Recipe{}
	out.Reset()
	if ok := offerDiscoveredArchForkWith(req, model, false, strings.NewReader("y\n"), &out, false, search, install); ok || installed.GitURL != "" {
		t.Fatalf("non-terminal installed a fork: ok=%v recipe=%+v", ok, installed)
	}
	if !strings.Contains(out.String(), "ggrun backend add") {
		t.Fatalf("non-terminal did not print the add command: %q", out.String())
	}

	installed = backends.Recipe{}
	if ok := offerDiscoveredArchForkWith(req, model, true, strings.NewReader(""), io.Discard, false, search, install); !ok || installed.GitURL != pr.GitURL {
		t.Fatalf("assume-yes fork: ok=%v recipe=%+v", ok, installed)
	}

	if ok := offerDiscoveredArchForkWith(req, &placement.ModelProfile{ModelArch: "hy_v3"}, true, strings.NewReader(""), io.Discard, false, search, install); ok {
		t.Fatal("recipe-backed arch searched GitHub instead of using the reviewed recipe")
	}
	if ok := offerDiscoveredArchForkWith(&launchRequest{}, model, true, strings.NewReader(""), io.Discard, false, search, install); ok {
		t.Fatal("no FAIL-CLOSED reason still searched for a fork")
	}
	empty := func(context.Context, *placement.ModelProfile) ([]backends.ArchForkPR, error) { return nil, nil }
	if ok := offerDiscoveredArchForkWith(req, model, true, strings.NewReader(""), io.Discard, false, empty, install); ok {
		t.Fatal("empty search installed a fork")
	}

	cited := pr
	cited.Cited = true
	searchCited := func(context.Context, *placement.ModelProfile) ([]backends.ArchForkPR, error) {
		return []backends.ArchForkPR{cited}, nil
	}
	out.Reset()
	installed = backends.Recipe{}
	if ok := offerDiscoveredArchForkWith(req, model, false, strings.NewReader("n\n"), &out, true, searchCited, install); ok {
		t.Fatal("declined cited fork still installed")
	}
	if !strings.Contains(out.String(), "Hugging Face model card cites") {
		t.Fatalf("cited PR prompt omitted Hugging Face: %q", out.String())
	}
}

func TestParseLaunchArgsCarriesConfiguredAppHome(t *testing.T) {
	isolateConfig(t)
	appHome := t.TempDir()
	t.Setenv("LLM_APP_HOME", appHome)
	req, err := parseLaunchArgs([]string{"model.gguf", "--backend", "ik_llama"})
	if err != nil {
		t.Fatal(err)
	}
	if req.AppHome != appHome {
		t.Fatalf("request APP_HOME = %q, want %q", req.AppHome, appHome)
	}
}

func TestDetectBackendCUDAHelpMentionVulkanStaysLlama(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake-backend probe uses a shell script")
	}
	bin := writeFakeBackend(t, "llama-server-cuda", "echo 'Vulkan appears in generic help text'\n")
	info := detectBackend(bin)
	if info.Tag != "llama" {
		t.Fatalf("CUDA/mainline path should stay llama even when help mentions Vulkan, got %#v", info)
	}
}

func TestBackendMatchesVulkanAliases(t *testing.T) {
	info := &backendInfo{Path: "/home/me/llama.cpp/build-vulkan/bin/llama-server", Tag: "vulkan"}
	if !backendMatches(info, "llama-server", "vulkan") {
		t.Fatalf("expected vulkan backend match")
	}
	if !backendMatches(info, "llama-server", "llama-vk") {
		t.Fatalf("expected llama-vk backend alias match")
	}
}

// A metal- or vulkan-tagged backend is still mainline llama.cpp. The tag says
// which device dialect placement must emit, not which fork the binary came
// from. setup-home.sh records LLM_BACKEND="llama" for a metal install, and
// detectBackend tags every darwin build "metal", so an exact-tag rule left
// every macOS install unable to resolve the backend it had just installed:
// "selected backend \"llama\" was not found under APP_HOME".
func TestBackendMatchesTreatsMetalAndVulkanAsLlama(t *testing.T) {
	for _, tag := range []string{"llama", "vulkan", "metal"} {
		info := &backendInfo{Path: "/app/.bin/llama-server", Tag: tag}
		if !backendMatches(info, "llama-server", "llama") {
			t.Fatalf("a %s-tagged backend must satisfy a llama request", tag)
		}
	}
	// A different fork must not be swept in by the same rule.
	ik := &backendInfo{Path: "/app/.bin/ik_llama-server", Tag: "ik_llama"}
	if backendMatches(ik, "ik_llama-server", "llama") {
		t.Fatal("ik_llama is a separate fork and must not satisfy a llama request")
	}
	// An explicit device request stays narrow.
	if backendMatches(&backendInfo{Tag: "llama"}, "llama-server", "metal") {
		t.Fatal("an explicit metal request must not accept a plain llama build")
	}
}

func TestResolveModelPathUsesConfiguredModelDir(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(model, []byte("gguf"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	got := resolveModelPath("model.gguf", dir)
	if got != model {
		t.Fatalf("expected configured model dir path, got %s", got)
	}
}

func TestApplyTuneCacheAutoSelectsBest(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	cachePath := filepath.Join(cacheDir, "tune_model.gguf_4_hw12345678_vulkan.json")
	doc := `{
		"model": "model.gguf",
		"baseline_gen_tps": 100.0,
		"baseline_wins": false,
		"best_config": {
			"name": "threads12",
			"flags": {"--threads": "12"},
			"gen_tps": 120.0,
			"pp_tps": 300.0
		},
		"rounds": 1,
		"tuned_at": "2026-05-28T00:00:00Z"
	}`
	if err := os.WriteFile(cachePath, []byte(doc), 0644); err != nil {
		t.Fatalf("write tune cache: %v", err)
	}

	args := applyTuneCache(&launchRequest{ModelPath: modelPath}, []string{"llama-server", "--threads", "8"}, cacheDir, "vulkan", false, nil)
	if !hasArgValue(args, "--threads", "12") {
		t.Fatalf("expected cached --threads override, got %v", args)
	}
}

func TestApplyTuneCacheSkipsMemoryExpandingOverrideWhenVRAMHeadroomIsLow(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	cachePath := filepath.Join(cacheDir, "tune_model.gguf_4_hw12345678_vulkan.json")
	doc := `{
		"model": "model.gguf",
		"baseline_gen_tps": 100.0,
		"baseline_wins": false,
		"best_config": {
			"name": "larger-ubatch",
			"flags": {"-ub": "2048"},
			"gen_tps": 120.0,
			"pp_tps": 300.0
		},
		"rounds": 1,
		"tuned_at": "2026-06-02T00:00:00Z"
	}`
	if err := os.WriteFile(cachePath, []byte(doc), 0644); err != nil {
		t.Fatalf("write tune cache: %v", err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, VRAMUsedMB: 21000}}}
	base := []string{"llama-server", "--device", "Vulkan0", "-ub", "1024"}

	args := applyTuneCache(&launchRequest{ModelPath: modelPath, TuneCache: cachePath}, base, cacheDir, "vulkan", false, caps)
	if !hasArgValue(args, "-ub", "1024") {
		t.Fatalf("expected low-headroom guard to keep base -ub, got %v", args)
	}
}

func TestApplyTuneCacheAllowsNonVRAMOverrideWhenVRAMHeadroomIsLow(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	cachePath := filepath.Join(cacheDir, "tune_model.gguf_4_hw12345678_vulkan.json")
	doc := `{
		"model": "model.gguf",
		"baseline_gen_tps": 100.0,
		"baseline_wins": false,
		"best_config": {
			"name": "threads12",
			"flags": {"--threads": "12"},
			"gen_tps": 120.0,
			"pp_tps": 300.0
		},
		"rounds": 1,
		"tuned_at": "2026-06-02T00:00:00Z"
	}`
	if err := os.WriteFile(cachePath, []byte(doc), 0644); err != nil {
		t.Fatalf("write tune cache: %v", err)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "RTX 3090 Ti", VRAMTotalMB: 24564, VRAMUsedMB: 24000}}}
	base := []string{"llama-server", "--device", "Vulkan0", "--threads", "8"}

	args := applyTuneCache(&launchRequest{ModelPath: modelPath, TuneCache: cachePath}, base, cacheDir, "vulkan", false, caps)
	if !hasArgValue(args, "--threads", "12") {
		t.Fatalf("expected non-VRAM override to apply, got %v", args)
	}
}

func TestApplyTuneCacheDoesNotCrossBackend(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	cachePath := filepath.Join(cacheDir, "tune_model.gguf_4_hw12345678_vulkan.json")
	doc := `{
		"model": "model.gguf",
		"baseline_gen_tps": 100.0,
		"baseline_wins": false,
		"best_config": {
			"name": "threads12",
			"flags": {"--threads": "12"},
			"gen_tps": 120.0,
			"pp_tps": 300.0
		},
		"rounds": 1,
		"tuned_at": "2026-05-28T00:00:00Z"
	}`
	if err := os.WriteFile(cachePath, []byte(doc), 0644); err != nil {
		t.Fatalf("write tune cache: %v", err)
	}

	args := applyTuneCache(&launchRequest{ModelPath: modelPath}, []string{"llama-server", "--threads", "8"}, cacheDir, "llama", false, nil)
	if !hasArgValue(args, "--threads", "8") {
		t.Fatalf("expected backend-scoped cache to be ignored, got %v", args)
	}
}

func TestBestTuneCachePathFiltersHardwareHash(t *testing.T) {
	cacheDir := t.TempDir()
	cachePath := filepath.Join(cacheDir, "tune_model.gguf_4_hwdeadbeef_vulkan.json")
	doc := `{
		"model": "model.gguf",
		"baseline_gen_tps": 100.0,
		"baseline_wins": false,
		"best_config": {
			"name": "threads12",
			"flags": {"--threads": "12"},
			"gen_tps": 120.0,
			"pp_tps": 300.0
		},
		"rounds": 1,
		"tuned_at": "2026-05-28T00:00:00Z"
	}`
	if err := os.WriteFile(cachePath, []byte(doc), 0644); err != nil {
		t.Fatalf("write tune cache: %v", err)
	}
	agentPath := filepath.Join(cacheDir, "tune_model.gguf_4_hwdeadbeef_vulkan_agent-parallel-v1-p2.json")
	agentDoc := `{
		"model": "model.gguf",
		"workload": "agent-parallel-v1-p2",
		"baseline_gen_tps": 100.0,
		"baseline_wins": false,
		"best_config": {"name": "agent-batch", "flags": {"-b": "256", "-ub": "256"}, "gen_tps": 999.0, "pp_tps": 300.0},
		"rounds": 4,
		"tuned_at": "2026-05-29T00:00:00Z"
	}`
	if err := os.WriteFile(agentPath, []byte(agentDoc), 0644); err != nil {
		t.Fatalf("write agent tune cache: %v", err)
	}
	if got := bestTuneCachePath(cacheDir, "model.gguf", "vulkan", false, "", "badc0ffe"); got != "" {
		t.Fatalf("expected wrong-hardware cache to be ignored, got %s", got)
	}
	if got := bestTuneCachePath(cacheDir, "model.gguf", "vulkan", false, "", "deadbeef"); got != cachePath {
		t.Fatalf("expected matching hardware cache, got %s", got)
	}
	if got := bestTuneCachePath(cacheDir, "model.gguf", "vulkan", false, "agent-parallel-v1-p2", "deadbeef"); got != agentPath {
		t.Fatalf("expected matching agent workload cache, got %s", got)
	}
}

func TestApplyTuneCacheSkipsAutomaticGenericTuneForClaudeCode(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	cachePath := filepath.Join(cacheDir, "tune_model.gguf_4_hwdeadbeef_vulkan.json")
	doc := `{
		"model": "model.gguf",
		"baseline_gen_tps": 100.0,
		"baseline_wins": false,
		"best_config": {"name": "threads12", "flags": {"--threads": "12"}, "gen_tps": 120.0, "pp_tps": 300.0},
		"rounds": 1,
		"tuned_at": "2026-05-28T00:00:00Z"
	}`
	if err := os.WriteFile(cachePath, []byte(doc), 0644); err != nil {
		t.Fatalf("write generic tune cache: %v", err)
	}
	base := []string{"llama-server", "--threads", "8"}
	got := applyTuneCache(&launchRequest{ModelPath: modelPath, ClaudeCode: true}, base, cacheDir, "vulkan", false, nil)
	if !hasArgValue(got, "--threads", "8") {
		t.Fatalf("automatic generic tune changed Claude Code args: %v", got)
	}
	got = applyTuneCache(&launchRequest{ModelPath: modelPath, ClaudeCode: true, TuneCache: cachePath}, base, cacheDir, "vulkan", false, nil)
	if !hasArgValue(got, "--threads", "12") {
		t.Fatalf("explicit Claude Code tune was not honored: %v", got)
	}
}

func hasArgValue(args []string, flag, value string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag && args[i+1] == value {
			return true
		}
	}
	return false
}

// valueAfter returns the argument immediately following flag, or "" when flag
// is the last argument or absent.
func valueAfter(args []string, flag string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag {
			return args[i+1]
		}
	}
	return ""
}

func TestBackendSearchPathsIncludeAppHomeBackend(t *testing.T) {
	appHome := filepath.Join(t.TempDir(), "ggrun")
	t.Setenv("LLM_APP_HOME", appHome)
	paths := backendSearchPaths()
	want := filepath.Join(appHome, ".bin", "llama-server")
	for _, path := range paths {
		if path == want {
			return
		}
	}
	t.Fatalf("missing app-home backend path %s in %#v", want, paths)
}

func TestFirstPositionalSkipsParallelValue(t *testing.T) {
	// --parallel takes a value; "2" must not be mistaken for the model arg.
	got := firstPositional([]string{"--parallel", "2", "unsloth/Qwen-GGUF", "--download"})
	if got != "unsloth/Qwen-GGUF" {
		t.Fatalf("expected repo positional, got %q", got)
	}
	got = firstPositional([]string{"-c", "32768", "model.gguf"})
	if got != "model.gguf" {
		t.Fatalf("expected model.gguf, got %q", got)
	}
	got = firstPositional([]string{"--ram-headroom", "2G", "org/model-GGUF", "--download"})
	if got != "org/model-GGUF" {
		t.Fatalf("--ram-headroom value was treated as positional: got %q", got)
	}
	got = firstPositional([]string{"--ram-limit-percent", "90", "org/model-GGUF", "--download"})
	if got != "org/model-GGUF" {
		t.Fatalf("--ram-limit-percent value was treated as positional: got %q", got)
	}
	got = firstPositional([]string{"--claude-profile", "agent-interactive", "org/model-GGUF", "--download"})
	if got != "org/model-GGUF" {
		t.Fatalf("--claude-profile value was treated as positional: got %q", got)
	}
}

func TestParseLaunchArgsRejectsInvalidSafetyFlags(t *testing.T) {
	cases := []struct {
		name string
		args []string
	}{
		{"port text", []string{"model.gguf", "--port", "abc"}},
		{"port zero", []string{"model.gguf", "--port=0"}},
		{"parallel text", []string{"model.gguf", "--parallel", "many"}},
		{"parallel zero", []string{"model.gguf", "--parallel=0"}},
		{"vram headroom text", []string{"model.gguf", "--vram-headroom", "two-gig"}},
		{"ram headroom negative", []string{"model.gguf", "--ram-headroom=-2G"}},
		{"ram percent zero", []string{"model.gguf", "--ram-limit-percent=0"}},
		{"ram percent high", []string{"model.gguf", "--ram-limit-percent", "101"}},
		{"gpu token", []string{"model.gguf", "--gpus", "0,fast"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			isolateConfig(t)
			if _, err := parseLaunchArgs(tc.args); err == nil {
				t.Fatalf("parseLaunchArgs(%v) accepted invalid input", tc.args)
			}
		})
	}
}

func TestPlacementOptionsNeverMapsInvalidGPUToZero(t *testing.T) {
	opts := placementOptionsFromRequest(
		&launchRequest{GPUsFlag: "not-a-gpu"},
		&placement.ModelProfile{}, &backendInfo{Tag: "llama"}, t.TempDir(),
	)
	if len(opts.GPUs) != 0 {
		t.Fatalf("invalid GPU input became placement GPUs %v", opts.GPUs)
	}
}

func TestApplyGPUVisibilitySetsEnv(t *testing.T) {
	t.Setenv("CUDA_VISIBLE_DEVICES", "")
	req := &launchRequest{GPUsFlag: "2,0"}
	env := applyGPUVisibility(req, "ik_llama")
	if env != "CUDA_VISIBLE_DEVICES=0,2" {
		t.Fatalf("unexpected env assignment: %q", env)
	}
	if os.Getenv("CUDA_VISIBLE_DEVICES") != "0,2" {
		t.Fatalf("CUDA_VISIBLE_DEVICES not set: %q", os.Getenv("CUDA_VISIBLE_DEVICES"))
	}

	t.Setenv("GGML_VK_VISIBLE_DEVICES", "")
	env = applyGPUVisibility(&launchRequest{GPUsFlag: "1"}, "vulkan")
	if env != "GGML_VK_VISIBLE_DEVICES=1" {
		t.Fatalf("unexpected vulkan env assignment: %q", env)
	}
}

func TestApplyGPUVisibilityNoFlagNoEnv(t *testing.T) {
	if env := applyGPUVisibility(&launchRequest{}, "ik_llama"); env != "" {
		t.Fatalf("expected no env assignment without --gpus, got %q", env)
	}
	if env := applyGPUVisibility(&launchRequest{GPUsFlag: "abc"}, "ik_llama"); env != "" {
		t.Fatalf("expected no env assignment for invalid --gpus, got %q", env)
	}
}

func TestRuntimeGPUCapabilitiesMatchesVisibilityRenumbering(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, Name: "large", VRAMTotalMB: 24576},
		{Index: 1, Name: "slow", VRAMTotalMB: 12288},
		{Index: 2, Name: "fast", VRAMTotalMB: 12282},
	}}
	runtime, mapping := runtimeGPUCapabilities(caps, &launchRequest{GPUsFlag: "2,1"})
	if runtime == nil || len(runtime.GPUs) != 2 {
		t.Fatalf("runtime GPU filter mismatch: %#v", runtime)
	}
	if runtime.GPUs[0].Name != "slow" || runtime.GPUs[0].Index != 0 || runtime.GPUs[1].Name != "fast" || runtime.GPUs[1].Index != 1 {
		t.Fatalf("visible GPU order/renumber mismatch: %#v", runtime.GPUs)
	}
	if mapping[0] != 1 || mapping[1] != 2 || physicalGPUIndex(1, mapping) != 2 {
		t.Fatalf("visible-to-physical mapping mismatch: %#v", mapping)
	}
}

func TestRuntimeGPUCapabilitiesForLaunchRejectsTheLiveV4WorkerOvercommit(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 23840},
		{Index: 1, VRAMTotalMB: 11909},
		{Index: 2, VRAMTotalMB: 11710},
	}}
	req := &launchRequest{ReviewerReservation: &placement.CompanionReservation{
		Name: claudeNanoCompanionName, VRAMMB: 8000,
	}}
	strategy := &placement.Strategy{CompanionPlacements: []placement.CompanionPlacement{{
		Name: claudeNanoCompanionName, GPU: 1,
	}}}
	restore := runtimeVRAMUsedMB
	defer func() { runtimeVRAMUsedMB = restore }()
	runtimeVRAMUsedMB = func(index int) int {
		if index == 1 {
			return 8766
		}
		return 0
	}

	runtime, _ := runtimeGPUCapabilitiesForLaunch(caps, req, strategy)
	if got := runtime.GPUs[1].VRAMFreeMB(); got != 3143 {
		t.Fatalf("live CUDA1 free VRAM = %d MiB, want 3143", got)
	}
	if got := runtime.GPUs[2].VRAMFreeMB(); got != 11710 {
		t.Fatalf("idle CUDA2 free VRAM = %d MiB, want 11710", got)
	}
	devs := []preflightDevice{
		{Name: "CUDA1", ModelMB: 2925, ComputeMB: 599},
		{Name: "CUDA2", ModelMB: 414, ComputeMB: 599},
	}
	device, deficit, summary := preflightWorstDeficit(devs, runtime.GPUs, nil, nil)
	if device != 1 || deficit != 381 {
		t.Fatalf("live V4 admission = CUDA%d deficit %d, want CUDA1 deficit 381; %s", device, deficit, summary)
	}
	if !strings.Contains(summary, "CUDA2 1013/11710 MiB") {
		t.Fatalf("idle destination GPU missing from admission evidence: %s", summary)
	}
}

func TestRuntimeGPUCapabilitiesForLaunchKeepsReservationAsFloor(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 1, VRAMTotalMB: 11909}}}
	req := &launchRequest{ReviewerReservation: &placement.CompanionReservation{
		Name: claudeNanoCompanionName, VRAMMB: 9216,
	}}
	strategy := &placement.Strategy{CompanionPlacements: []placement.CompanionPlacement{{
		Name: claudeNanoCompanionName, GPU: 1,
	}}}
	restore := runtimeVRAMUsedMB
	defer func() { runtimeVRAMUsedMB = restore }()
	runtimeVRAMUsedMB = func(int) int { return 7000 }

	runtime, _ := runtimeGPUCapabilitiesForLaunch(caps, req, strategy)
	if got := runtime.GPUs[0].VRAMFreeMB(); got != 2693 {
		t.Fatalf("planned reservation floor left %d MiB, want 2693", got)
	}
}

func TestRejectedMemoryArgvIdentityBlocksOnlyTheExactFailedPlan(t *testing.T) {
	failed := []string{"llama-server", "-m", "model.gguf", "--n-cpu-moe", "36", "--swa-full"}
	safe := []string{"llama-server", "-m", "model.gguf", "--n-cpu-moe", "37", "--swa-full"}
	withoutSWA := []string{"llama-server", "-m", "model.gguf", "--n-cpu-moe", "36"}
	recovery := newLaunchMemoryRecovery()
	recovery.reject(failed)
	if !recovery.isRejected(failed) || recovery.isRejected(safe) {
		t.Fatalf("rejected membership failed: failed=%v safe=%v", recovery.isRejected(failed), recovery.isRejected(safe))
	}
	if !recovery.hasRejections() || recovery.rejectionCount() != 1 {
		t.Fatalf("rejection state = has %v count %d, want true/1", recovery.hasRejections(), recovery.rejectionCount())
	}

	if changed, rejected := recovery.recomputeDecision(safe, failed); !changed || !rejected {
		t.Fatalf("exact failed argv decision = changed %v rejected %v", changed, rejected)
	}
	if changed, rejected := recovery.recomputeDecision(failed, safe); !changed || rejected {
		t.Fatalf("safe layer recovery decision = changed %v rejected %v", changed, rejected)
	}
	if changed, rejected := recovery.recomputeDecision(failed, withoutSWA); !changed || rejected {
		t.Fatalf("SWA-disabled recovery decision = changed %v rejected %v", changed, rejected)
	}
	if changed, rejected := recovery.recomputeDecision(failed, failed); changed || rejected {
		t.Fatalf("identical argv decision = changed %v rejected %v", changed, rejected)
	}
}

func TestRecoveredLaunchRetainsFirstSafeAllocationBaselineUnlessForced(t *testing.T) {
	recovery := newLaunchMemoryRecovery()
	recovery.reject([]string{"llama-server", "--n-cpu-moe", "35"})

	if !retainRecoveredBaselineForAllocationPromotion(&launchRequest{Calibrate: calibrateAuto}, recovery) {
		t.Fatal("automatic allocation promotion was allowed after recovery")
	}
	if !retainRecoveredBaselineForAllocationPromotion(&launchRequest{Calibrate: calibrateOff}, recovery) {
		t.Fatal("disabled optimization was allowed to promote a recovered allocation")
	}
	if retainRecoveredBaselineForAllocationPromotion(&launchRequest{Calibrate: calibrateOn}, recovery) {
		t.Fatal("explicit forced calibration was unexpectedly suppressed")
	}
	if retainRecoveredBaselineForAllocationPromotion(&launchRequest{Calibrate: calibrateAuto}, newLaunchMemoryRecovery()) {
		t.Fatal("clean launch was treated as memory-recovered")
	}
}

func TestSharedLaunchRecoveryRefusesRejectedEntryArgv(t *testing.T) {
	args := []string{"llama-server", "--n-cpu-moe", "35", "--swa-full"}
	recovery := newLaunchMemoryRecovery()
	recovery.reject(args)

	_, _, _, err := startLaunchWithCUDAOOMRecoveryState(
		nil, nil, nil, nil, nil, nil, args, time.Second, recovery,
	)
	if err == nil || !strings.Contains(err.Error(), "rejected earlier in this launch lifecycle") {
		t.Fatalf("rejected lifecycle entry returned %v", err)
	}
}

func TestValidateExactAdmissionArgvRejectsAnyRewrite(t *testing.T) {
	original := []string{
		"llama-server", "-m", "model.gguf", "-b", "128", "-ub", "64",
		"--tensor-split", "0,1",
	}
	expected := formatCommand(original)
	if err := validateExactAdmissionArgv(true, expected, original); err != nil {
		t.Fatalf("identical exact candidate was rejected: %v", err)
	}

	mutated := append([]string(nil), original...)
	mutated[len(mutated)-1] = "1,0"
	if err := validateExactAdmissionArgv(true, expected, mutated); err == nil ||
		!strings.Contains(err.Error(), "argv changed during admission") {
		t.Fatalf("changed exact candidate was accepted: %v", err)
	}
	if err := validateExactAdmissionArgv(false, expected, mutated); err != nil {
		t.Fatalf("ordinary fit recovery was incorrectly constrained: %v", err)
	}
}

func TestExactAdmissionErrorCoversEveryRewriteClass(t *testing.T) {
	cause := errors.New("probe")
	cases := []struct {
		class   exactAdmissionClass
		detail  string
		cause   error
		want    string
		wantErr error
	}{
		{exactAdmissionSpec, "", nil, "speculation re-plan", nil},
		{exactAdmissionCompat, "remove --swa-full", nil, "compatibility adjustment", nil},
		{exactAdmissionCompanion, "", nil, "speculative companion was rejected", nil},
		{exactAdmissionMemory, " on CUDA2 (617 MiB deficit)", nil, "memory admission on CUDA2", nil},
		{exactAdmissionMMap, "", cause, "mmap pageability", cause},
		{exactAdmissionCUDAOOM, " on device 0 allocating 9000 MiB", cause, "CUDA OOM on device 0", cause},
	}
	if len(cases) != 6 {
		t.Fatalf("rewrite classes = %d, want speculation/compat/companion/memory/mmap/cuda-oom", len(cases))
	}
	for _, tc := range cases {
		err := exactAdmissionError(tc.class, tc.detail, tc.cause)
		if err == nil || !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), "refusing") {
			t.Fatalf("%s error = %v, want substring %q", tc.class, err, tc.want)
		}
		if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
			t.Fatalf("%s did not wrap cause: %v", tc.class, err)
		}
		if !isStableExactAdmissionFailure(err) {
			t.Fatalf("%s was not classified as deterministic admission evidence: %v", tc.class, err)
		}
	}
	if isStableExactAdmissionFailure(errors.New("health timeout")) {
		t.Fatal("a transient start error became persistent negative admission evidence")
	}
}

func TestExactAdmissionRefusesRejectedLifecycleArgv(t *testing.T) {
	args := []string{"llama-server", "--n-cpu-moe", "35", "-ub", "8192"}
	recovery := newLaunchMemoryRecovery()
	recovery.reject(args)
	_, _, gotArgs, err := startLaunchExactAdmission(
		&launchRequest{SpecMode: "off"}, nil, nil, nil, nil, nil, args, time.Second, recovery,
	)
	if err == nil || !strings.Contains(err.Error(), "rejected earlier in this launch lifecycle") {
		t.Fatalf("rejected challenger returned %v", err)
	}
	if formatCommand(gotArgs) != formatCommand(args) {
		t.Fatalf("rejected challenger mutated argv: %v", gotArgs)
	}
}

func TestExactAdmissionRefusesSpeculationReplan(t *testing.T) {
	args := []string{"llama-server", "-m", "model.gguf", "--draft", "draft.gguf"}
	strategy := &placement.Strategy{
		Draft: &placement.DraftConfig{Type: placement.DraftModel, VerifiedLaunchIdentity: "not-this-argv"},
	}
	_, _, gotArgs, err := startLaunchExactAdmission(
		&launchRequest{SpecMode: "auto"}, &config.Config{}, &placement.ModelProfile{},
		strategy, &backendInfo{}, &detect.Capabilities{}, args, time.Second, newLaunchMemoryRecovery(),
	)
	if err == nil || !strings.Contains(err.Error(), "speculation re-plan") {
		t.Fatalf("speculation rewrite was allowed: %v", err)
	}
	if formatCommand(gotArgs) != formatCommand(args) {
		t.Fatalf("speculation refusal mutated argv: %v", gotArgs)
	}
}

func TestRestoreBypassesRejectedMemoryArgv(t *testing.T) {
	args := []string{"llama-server", "--n-cpu-moe", "35", "--swa-full"}
	recovery := newLaunchMemoryRecovery()
	recovery.reject(args)
	// The restore boundary is exempt from the isRejected gate: it is the outer
	// launcher's unconditional fallback (failed-promotion restore, calibration
	// winner restart). Re-gating it would convert a working fallback into a
	// dead box. SpecMode "off" skips the speculation block and a nil strategy
	// skips the preflight, so the body proceeds to the real start and fails on
	// nil be/caps with a non-gate error — proving the gate was bypassed.
	req := &launchRequest{SpecMode: "off"}
	_, _, _, err := restoreLaunchWithCUDAOOMRecoveryState(
		req, nil, nil, nil, nil, nil, args, time.Second, recovery,
	)
	if err == nil || strings.Contains(err.Error(), "rejected earlier in this launch lifecycle") {
		t.Fatalf("restore boundary re-gated a rejected argv: %v", err)
	}
}

func TestOOMOvershootEnforcesDocumentedFloor(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 2, VRAMTotalMB: 12288}}}
	if got := oomOvershoot(caps, 2, 75); got != 512 {
		t.Fatalf("75 MiB allocation penalty = %d MiB, want 512 MiB floor", got)
	}
}

func TestOOMOvershootKeepsLargerMeasuredDeficit(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 2, VRAMTotalMB: 12288, VRAMUsedMB: 12000}}}
	if got := oomOvershoot(caps, 2, 1000); got != 712 {
		t.Fatalf("1000 MiB allocation against 288 MiB free = %d MiB, want 712 MiB", got)
	}
}

func TestClaudeCodeAutocompactWindow(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want int
	}{
		{"parallel4_65k_slot", []string{"--ctx-size", "262144", "--parallel", "4"}, 65536},
		{"parallel8_32k_slot", []string{"--ctx-size", "262144", "--parallel", "8"}, 32768},
		{"no_parallel_defaults_to_1", []string{"--ctx-size", "65536"}, 65536},
		{"tiny_slot_floors_at_2k", []string{"--ctx-size", "8192", "--parallel", "8"}, 2048},
		{"missing_ctx_keeps_legacy_window", []string{"--parallel", "4"}, 200000},
		{"short_ctx_alias", []string{"-c", "131072", "-np", "4"}, 32768},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := claudeCodeAutocompactWindow(tc.args, 0); got != tc.want {
				t.Fatalf("claudeCodeAutocompactWindow(%v) = %d, want %d", tc.args, got, tc.want)
			}
		})
	}
}

func TestClaudeCodeAutocompactWindowCapsAtActualCtx(t *testing.T) {
	// A requested ctx that the model's native context_length caps must not size
	// the autocompact window: sizing it off the raw --ctx-size let a Muse request
	// overflow the real 131k slot before compaction fired (250k requested, 131k
	// actual). The actual strategy context is the true ceiling.
	if got := claudeCodeAutocompactWindow([]string{"--ctx-size", "250000"}, 131072); got != 131072 {
		t.Fatalf("window must cap at the actual context: got %d, want 131072", got)
	}
	if got := claudeCodeAutocompactWindow([]string{"--ctx-size", "250000", "--parallel", "2"}, 131072); got != 65536 {
		t.Fatalf("window must divide the capped actual ctx by parallel: got %d, want 65536", got)
	}
	// When no actual ctx is known (diagnostic path), the raw --ctx-size is used
	// (rounded to a 256-multiple, the existing behavior).
	if got := claudeCodeAutocompactWindow([]string{"--ctx-size", "250000"}, 0); got != 249856 {
		t.Fatalf("no actual ctx must keep the raw window: got %d, want 249856", got)
	}
}

func TestClaudeCodeAutocompactPct(t *testing.T) {
	if got := claudeCodeAutocompactPct([]string{"--ctx-size", "262144", "--parallel", "4"}, 0); got != 75 {
		t.Fatalf("known-window autocompact pct = %d, want 75", got)
	}
	if got := claudeCodeAutocompactPct([]string{"--parallel", "4"}, 0); got != 25 {
		t.Fatalf("unknown-window autocompact pct = %d, want 25", got)
	}
}

func TestArgIntValue(t *testing.T) {
	args := []string{"-m", "model.gguf", "--ctx-size", "4096", "--parallel", "4", "--flag"}
	if got := argIntValue(args, "--ctx-size", "-c"); got != 4096 {
		t.Fatalf("--ctx-size = %d, want 4096", got)
	}
	if got := argIntValue(args, "--parallel", "-np"); got != 4 {
		t.Fatalf("--parallel = %d, want 4", got)
	}
	if got := argIntValue(args, "--missing"); got != -1 {
		t.Fatalf("--missing = %d, want -1", got)
	}
	// trailing flag with no value must not panic or misparse
	if got := argIntValue(args, "--flag"); got != -1 {
		t.Fatalf("--flag (no value) = %d, want -1", got)
	}
	// last-wins: a user value appended after the strategy's must override it, to
	// mirror llama.cpp/ik_llama (which honor the final repeated flag).
	dup := []string{"--ctx-size", "262144", "--parallel", "4", "--ctx-size", "16384"}
	if got := argIntValue(dup, "--ctx-size", "-c"); got != 16384 {
		t.Fatalf("last-wins --ctx-size = %d, want 16384", got)
	}
	// an unparseable later value is skipped, falling back to the last parseable one
	if got := argIntValue([]string{"--ctx-size", "8192", "--ctx-size", "max"}, "--ctx-size"); got != 8192 {
		t.Fatalf("last parseable --ctx-size = %d, want 8192", got)
	}
}

func TestClaudeCodeAutocompactWindowLastWinsOnUserOverride(t *testing.T) {
	// The backend honors the final context value, so the explicit window must too.
	args := []string{"--ctx-size", "262144", "--parallel", "4", "--ctx-size", "16384"}
	if got := claudeCodeAutocompactWindow(args, 0); got != 4096 {
		t.Fatalf("autocompact window with user override = %d, want 4096", got)
	}
}

func TestClaudeCodeSearchMCPArgsRespectsUserConfig(t *testing.T) {
	if got := claudeCodeSearchMCPArgs([]string{"--mcp-config", "mine.json"}); got != nil {
		t.Fatalf("expected nil when user passed --mcp-config, got %v", got)
	}
}

func TestClaudeCodeSearchMCPArgsEnablesResearchTools(t *testing.T) {
	binDir := t.TempDir()
	uvx := filepath.Join(binDir, "uvx")
	if err := os.WriteFile(uvx, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", binDir)

	got := claudeCodeSearchMCPArgs(nil)
	joined := strings.Join(got, " ")
	for _, want := range []string{"--mcp-config", "duckduckgo-mcp-server", "--allowedTools", "mcp__ddg-search__search", "mcp__ddg-search__fetch_content"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("missing %q in research MCP args: %v", want, got)
		}
	}

	got = claudeCodeSearchMCPArgs([]string{"--allowed-tools", "mine"})
	if hasArg(got, "--allowedTools") || hasArg(got, "--allowed-tools") {
		t.Fatalf("user allowed-tools must not be overridden, got %v", got)
	}
}

func TestClaudeCodePermissionArgsDefaultsToLocalAuto(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "")
	got := claudeCodePermissionArgs(nil)
	if len(got) != 2 || got[0] != "--permission-mode" || got[1] != "auto" {
		t.Fatalf("local Claude launch must use the routed Auto reviewer, got %v", got)
	}
}

func TestClaudeCodePermissionArgsRespectsOverrides(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "auto")
	if got := claudeCodePermissionArgs(nil); len(got) != 2 || got[1] != "auto" {
		t.Fatalf("environment override not respected: %v", got)
	}
	if got := claudeCodePermissionArgs([]string{"--permission-mode", "plan"}); got != nil {
		t.Fatalf("explicit CLI mode must win, got %v", got)
	}
	if got := claudeCodePermissionArgs([]string{"--permission-mode=manual"}); got != nil {
		t.Fatalf("explicit equals-form CLI mode must win, got %v", got)
	}
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "inherit")
	if got := claudeCodePermissionArgs(nil); got != nil {
		t.Fatalf("inherit must preserve settings.json mode, got %v", got)
	}
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "not-a-mode")
	if got := claudeCodePermissionArgs(nil); len(got) != 2 || got[1] != "auto" {
		t.Fatalf("invalid override must fail safe to routed Auto, got %v", got)
	}
}

func TestClaudeCodeAliasArgs(t *testing.T) {
	base := []string{"-m", "model.gguf", "--port", "8081"}
	// claude-code mode appends --alias local so /v1/models advertises "local"
	got := claudeCodeAliasArgs(base, true)
	if argIndexOf(got, "--alias") < 0 || got[argIndexOf(got, "--alias")+1] != "local" {
		t.Fatalf("expected --alias local appended, got %v", got)
	}
	// non-claude-code mode is a no-op
	if got := claudeCodeAliasArgs(base, false); len(got) != len(base) {
		t.Fatalf("expected no change outside claude-code mode, got %v", got)
	}
	// a user-set alias is respected (not doubled)
	user := []string{"-m", "model.gguf", "--alias", "mymodel"}
	if got := claudeCodeAliasArgs(user, true); len(got) != len(user) {
		t.Fatalf("expected user --alias preserved without doubling, got %v", got)
	}
	if got := claudeCodeAliasArgs([]string{"-a", "x"}, true); len(got) != 2 {
		t.Fatalf("expected short -a alias respected, got %v", got)
	}
}

func TestClaudeCodeCacheArgs(t *testing.T) {
	base := []string{"llama-server", "-m", "model.gguf"}
	got := claudeCodeCacheArgs(base, true, "--cache-prompt --cache-reuse N", true)
	if !hasArgValue(got, "--cache-reuse", "256") {
		t.Fatalf("expected Claude cache reuse default, got %v", got)
	}
	if got := claudeCodeCacheArgs(base, false, "--cache-reuse N", true); len(got) != len(base) {
		t.Fatalf("expected no cache change outside Claude mode, got %v", got)
	}
	if got := claudeCodeCacheArgs(base, true, "--cache-prompt", true); len(got) != len(base) {
		t.Fatalf("expected unsupported backend to remain unchanged, got %v", got)
	}
	if got := claudeCodeCacheArgs(base, true, "--cache-reuse N", false); len(got) != len(base) {
		t.Fatalf("expected recurrent context to skip unsupported cache shifting, got %v", got)
	}
	for _, user := range [][]string{
		{"llama-server", "--cache-reuse", "0"},
		{"llama-server", "--cache-reuse=0"},
		{"llama-server", "--no-cache-prompt"},
	} {
		got := claudeCodeCacheArgs(user, true, "--cache-reuse N", true)
		if len(got) != len(user) {
			t.Fatalf("expected user cache override preserved, input %v got %v", user, got)
		}
	}
}

func TestClaudeCodeShiftableContext(t *testing.T) {
	if claudeCodeShiftableContext(&placement.ModelProfile{ModelArch: "laguna"}, &placement.Strategy{}) {
		t.Fatal("Laguna multi-position RoPE must not enable cache shifting")
	}
	if claudeCodeShiftableContext(&placement.ModelProfile{ModelArch: "qwen35"}, &placement.Strategy{HasSSM: true}) {
		t.Fatal("recurrent context must not enable cache shifting")
	}
	if claudeCodeShiftableContext(&placement.ModelProfile{ModelArch: "qwen35"}, &placement.Strategy{MMProjPath: "/models/mmproj.gguf"}) {
		t.Fatal("multimodal context must not enable cache shifting")
	}
	if !claudeCodeShiftableContext(&placement.ModelProfile{ModelArch: "qwen35"}, &placement.Strategy{}) {
		t.Fatal("ordinary transformer context should enable cache shifting")
	}
}

func argIndexOf(args []string, want string) int {
	for i, a := range args {
		if a == want {
			return i
		}
	}
	return -1
}

func TestClaudeCodeEnvDisablesIdleTimeoutForLocalBackend(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "real-key")
	t.Setenv("API_TIMEOUT_MS", "")
	t.Setenv("API_FORCE_IDLE_TIMEOUT", "")
	t.Setenv("CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS", "")
	t.Setenv("CLAUDE_ENABLE_BYTE_WATCHDOG", "")
	t.Setenv("CLAUDE_ENABLE_STREAM_WATCHDOG", "")
	t.Setenv("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "")
	t.Setenv("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "")
	t.Setenv("CLAUDE_CODE_EFFORT_LEVEL", "")
	env := claudeCodeEnv("0.0.0.0", 8081, []string{"llama-server", "--ctx-size", "1048576", "--parallel", "4"}, 1048576)

	if envHasPrefix(env, "ANTHROPIC_API_KEY=") {
		t.Fatalf("claude-code env must drop real ANTHROPIC_API_KEY: %v", env)
	}
	for _, want := range []string{
		"ANTHROPIC_BASE_URL=http://127.0.0.1:8081",
		"API_TIMEOUT_MS=2147483647",
		"API_FORCE_IDLE_TIMEOUT=0",
		"CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS=2147483647",
		"CLAUDE_ENABLE_BYTE_WATCHDOG=0",
		"CLAUDE_ENABLE_STREAM_WATCHDOG=0",
		"CLAUDE_CODE_EFFORT_LEVEL=xhigh",
		"CLAUDE_CODE_AUTO_COMPACT_WINDOW=262144",
		"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=75",
	} {
		if !envContains(env, want) {
			t.Fatalf("missing %s in claude-code env: %v", want, env)
		}
	}

	t.Setenv("CLAUDE_CODE_EFFORT_LEVEL", "max")
	overridden := claudeCodeEnv("127.0.0.1", 8081, nil, 0)
	if !envContains(overridden, "CLAUDE_CODE_EFFORT_LEVEL=max") {
		t.Fatalf("explicit Claude effort override was not preserved: %v", overridden)
	}
}

func envContains(env []string, want string) bool {
	for _, kv := range env {
		if kv == want {
			return true
		}
	}
	return false
}

func envHasPrefix(env []string, prefix string) bool {
	for _, kv := range env {
		if strings.HasPrefix(kv, prefix) {
			return true
		}
	}
	return false
}

func TestClaudeCodeSlotAdjust(t *testing.T) {
	cases := []struct {
		name         string
		ctx, par     int
		claudeCode   bool
		explicit     bool
		wantParallel int
	}{
		{"large_ctx_keeps_4", 262144, 4, true, false, 4},
		{"fit_32k_drops_to_1", 32768, 4, true, false, 1}, // the MiniMax-M3 regression: 8k slots
		{"128k_drops_to_2", 131072, 4, true, false, 2},   // 65k slots
		{"tiny_ctx_floors_at_1", 8192, 4, true, false, 1},
		{"not_claude_mode_untouched", 32768, 4, false, false, 4},
		{"parallel_1_untouched", 32768, 1, true, false, 1},
		{"explicit_parallel_kept", 65536, 2, true, true, 2}, // user tuning a big MoE: 2x32k slots
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := &placement.Strategy{ContextSize: tc.ctx, Parallel: tc.par}
			claudeCodeSlotAdjust(s, nil, tc.claudeCode, tc.explicit, false, false)
			if s.Parallel != tc.wantParallel {
				t.Fatalf("ctx=%d par=%d cc=%v: got parallel %d, want %d", tc.ctx, tc.par, tc.claudeCode, s.Parallel, tc.wantParallel)
			}
		})
	}
}

func TestClaudeCodeHybridUsesFairPromptBatch(t *testing.T) {
	s := &placement.Strategy{ContextSize: 1048576, Parallel: 2, BatchSize: 2048, UBatchSize: 512, HasSSM: true}
	claudeCodeSlotAdjust(s, &placement.ModelProfile{HasSSM: 1}, true, false, false, false)
	if s.BatchSize != claudeHybridBatch || s.UBatchSize != claudeHybridBatch {
		t.Fatalf("hybrid Claude batch/ubatch=%d/%d, want %d/%d", s.BatchSize, s.UBatchSize, claudeHybridBatch, claudeHybridBatch)
	}

	nonClaude := &placement.Strategy{ContextSize: 1048576, Parallel: 2, BatchSize: 2048, HasSSM: true}
	claudeCodeSlotAdjust(nonClaude, &placement.ModelProfile{HasSSM: 1}, false, false, false, false)
	if nonClaude.BatchSize != 2048 {
		t.Fatalf("non-Claude batch was changed: %d", nonClaude.BatchSize)
	}
}

func TestClaudeCodeHybridRepairsMissingVerifiedConfigSemantics(t *testing.T) {
	s := &placement.Strategy{ContextSize: 262144, Parallel: 2, BatchSize: 2048, UBatchSize: 128}
	model := &placement.ModelProfile{ModelArch: "qwen35", HasSSM: 1}
	claudeCodeSlotAdjust(s, model, true, true, false, false)
	if !s.HasSSM {
		t.Fatal("model-derived hybrid semantics were not restored")
	}
	if s.BatchSize != claudeHybridBatch || s.UBatchSize != claudeHybridBatch {
		t.Fatalf("restored hybrid batch/ubatch=%d/%d, want %d/%d", s.BatchSize, s.UBatchSize, claudeHybridBatch, claudeHybridBatch)
	}
}

func TestClaudeCodeHybridExplicitBatchOverridesFairnessCap(t *testing.T) {
	s := &placement.Strategy{ContextSize: 65536, Parallel: 4, BatchSize: 512, UBatchSize: 512, HasSSM: true}
	claudeCodeSlotAdjust(s, &placement.ModelProfile{HasSSM: 1}, true, true, true, false)
	if s.BatchSize != 512 {
		t.Fatalf("explicit hybrid Claude batch=%d, want 512", s.BatchSize)
	}
}

func TestClaudeCodeHybridExplicitUBatchRaisesAutomaticFairBatch(t *testing.T) {
	s := &placement.Strategy{ContextSize: 65536, Parallel: 2, BatchSize: 2048, UBatchSize: 512, HasSSM: true}
	claudeCodeSlotAdjust(s, &placement.ModelProfile{HasSSM: 1}, true, true, false, true)
	if s.BatchSize != 512 || s.UBatchSize != 512 {
		t.Fatalf("explicit ubatch was silently clamped: batch/ubatch=%d/%d", s.BatchSize, s.UBatchSize)
	}
}

func TestClaudeCodeHybridSingleSlotKeepsPlacementBatch(t *testing.T) {
	for _, tc := range []struct {
		name          string
		ctx, parallel int
		explicit      bool
	}{
		{"single_slot", 1048576, 1, false},
		// The automatic 4-slot default becomes one slot at this context. Verify
		// batch fairness is evaluated after that normalization.
		{"auto_reduced_to_single_slot", 32768, 4, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &placement.Strategy{ContextSize: tc.ctx, Parallel: tc.parallel, BatchSize: 2048, HasSSM: true}
			claudeCodeSlotAdjust(s, &placement.ModelProfile{HasSSM: 1}, true, tc.explicit, false, false)
			if s.Parallel != 1 {
				t.Fatalf("parallel=%d, want final single slot", s.Parallel)
			}
			if s.BatchSize != 2048 {
				t.Fatalf("single-slot hybrid batch=%d, want placement-selected 2048", s.BatchSize)
			}
		})
	}
}

func TestClaudeCodeSamplingArgs(t *testing.T) {
	base := []string{"-m", "model.gguf"}
	got := claudeCodeSamplingArgs(base, true, nil)
	for _, want := range []string{"--presence-penalty", "--repeat-penalty", "--repeat-last-n", "--top-k", "--top-p", "--min-p"} {
		if !hasArg(got, want) {
			t.Fatalf("expected %s in claude-code sampling defaults, got %v", want, got)
		}
	}
	// non-claude-code: untouched
	if got := claudeCodeSamplingArgs(base, false, nil); len(got) != len(base) {
		t.Fatalf("expected no sampling flags outside claude-code mode, got %v", got)
	}
	// user-set flag wins: not doubled, others still added
	user := []string{"-m", "model.gguf", "--presence-penalty", "1.5"}
	got = claudeCodeSamplingArgs(user, true, nil)
	n := 0
	for _, a := range got {
		if a == "--presence-penalty" {
			n++
		}
	}
	if n != 1 || !hasArg(got, "--top-k") {
		t.Fatalf("expected user presence-penalty kept once + other defaults added, got %v", got)
	}
}

func TestClaudeCodeSamplingArgsDeepSeek4(t *testing.T) {
	base := []string{"-m", "model.gguf"}
	model := &placement.ModelProfile{ModelArch: "deepseek4"}
	got := claudeCodeSamplingArgs(base, true, model)
	for _, want := range []string{"--temp", "--top-k", "--top-p", "--min-p", "--reasoning-budget"} {
		if !hasArg(got, want) {
			t.Fatalf("expected %s in deepseek4 claude-code defaults, got %v", want, got)
		}
	}
	if got[argIndexOf(got, "--top-k")+1] != "40" || got[argIndexOf(got, "--min-p")+1] != "0.05" || got[argIndexOf(got, "--reasoning-budget")+1] != "0" {
		t.Fatalf("unexpected deepseek4 defaults: %v", got)
	}

	user := []string{"-m", "model.gguf", "--reasoning-budget", "-1", "--top-k", "10"}
	got = claudeCodeSamplingArgs(user, true, model)
	if got[argIndexOf(got, "--reasoning-budget")+1] != "-1" || got[argIndexOf(got, "--top-k")+1] != "10" {
		t.Fatalf("user deepseek4 sampling overrides should win, got %v", got)
	}
}

// A models dir full of symlinks (e.g. shards linked from another disk) must be
// sized via the link targets. Summing entry.Info() (lstat) once shrank a 146GB
// sharded model to 365 bytes; the parseModel drift-rescale then crushed
// ExpertBytes with it and placement pinned all expert layers onto one GPU.
func TestTotalModelSizeFollowsSymlinkedShards(t *testing.T) {
	realDir := t.TempDir()
	linkDir := t.TempDir()
	var want int64
	for i := 1; i <= 3; i++ {
		name := fmt.Sprintf("big-%05d-of-00003.gguf", i)
		data := bytes.Repeat([]byte{0xAB}, 1000*i)
		if err := os.WriteFile(filepath.Join(realDir, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(filepath.Join(realDir, name), filepath.Join(linkDir, name)); err != nil {
			t.Fatal(err)
		}
		want += int64(len(data))
	}

	if got := totalModelSize(filepath.Join(linkDir, "big-00001-of-00003.gguf")); got != want {
		t.Fatalf("symlinked shards: totalModelSize = %d, want %d", got, want)
	}
	if got := totalModelSize(filepath.Join(realDir, "big-00001-of-00003.gguf")); got != want {
		t.Fatalf("real shards: totalModelSize = %d, want %d", got, want)
	}
}

func TestShouldPromoteMoEPlacementIncludesSubpinSqueeze(t *testing.T) {
	current := &placement.Strategy{
		Type:     placement.MoEOffload,
		NCPUMoE:  32,
		OTString: `blk\.(0|1)\.ffn_((gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA0,exps=CPU`,
	}
	fewerCPULayers := &placement.Strategy{
		Type:     placement.MoEOffload,
		NCPUMoE:  31,
		OTString: current.OTString,
	}
	if !shouldPromoteMoEPlacement(current, fewerCPULayers) {
		t.Fatal("expected fewer CPU MoE layers to promote")
	}

	subpinSqueeze := &placement.Strategy{
		Type:     placement.MoEOffload,
		NCPUMoE:  32,
		OTString: current.OTString + `,blk\.(2)\.ffn_(gate_up|up_gate|gate|up)_exps.*=CUDA0`,
	}
	if !shouldPromoteMoEPlacement(current, subpinSqueeze) {
		t.Fatal("expected same-NCPUMoE subpin squeeze to promote")
	}

	same := *current
	if shouldPromoteMoEPlacement(current, &same) {
		t.Fatal("unchanged placement must not promote")
	}
}

// TestWaitForShutdownOrCrashDetectsProcessDeath guards the exact bug this
// function fixes: cmdLaunch's "Press Ctrl+C to stop" wait used to block only
// on the shutdown signal, so a backend that crashed on its own (a real CUDA
// OOM well after health check, reproduced 2026-07-08/09 on a long request)
// left the wrapper silently hung forever with no idea its child had died.
func TestWaitForShutdownOrCrashDetectsProcessDeath(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX short-lived process")
	}
	cmd := exec.Command("sh", "-c", "exit 0")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start fake process: %v", err)
	}
	// Production always reaps via a background cmd.Wait() (server.go's
	// StartWithTimeoutTo) — that's what populates Cmd.ProcessState, which
	// IsRunning() checks. Without it the child would sit as a zombie and
	// never look "not running", which would silently mask this test.
	go func() { _ = cmd.Wait() }()
	p := &server.Process{Cmd: cmd}
	sigCh := make(chan os.Signal, 1)

	done := make(chan bool, 1)
	go func() { done <- waitForShutdownOrCrash(p, sigCh) }()

	select {
	case crashed := <-done:
		if !crashed {
			t.Fatal("expected crashed=true when the process exits on its own")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("waitForShutdownOrCrash did not detect process death in time")
	}
}

func TestWaitForShutdownOrCrashRespondsToSignal(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX long-lived process")
	}
	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start fake process: %v", err)
	}
	defer func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}()
	p := &server.Process{Cmd: cmd}
	sigCh := make(chan os.Signal, 1)

	done := make(chan bool, 1)
	go func() { done <- waitForShutdownOrCrash(p, sigCh) }()

	sigCh <- os.Interrupt
	select {
	case crashed := <-done:
		if crashed {
			t.Fatal("expected crashed=false when a shutdown signal arrives while the process is still running")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("waitForShutdownOrCrash did not respond to the signal in time")
	}
}

func TestRuntimeOOMInvalidatesActiveProfileAndPlacementCache(t *testing.T) {
	cacheDir := t.TempDir()
	cfg := &config.Config{CacheDir: cacheDir}
	model := &placement.ModelProfile{Path: "/models/test.gguf", Name: "test", ModelArch: "test", SizeBytes: 1234}
	backend := &backendInfo{Tag: "llama", Identity: "backend-build"}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 128000}, CPU: detect.CPUInfo{Model: "cpu", Cores: 8}}
	req := &launchRequest{CtxFlag: "4096", KVQuality: "q8_0", KVPlacement: "gpu", Parallel: 1}
	req.ProfilePolicyIdentity = requestedLaunchPolicyIdentity(req, model)
	args := []string{"llama-server", "-m", model.Path, "--ctx-size", "4096"}
	scope := launchProfileScope(req, model, backend, caps)
	store := controller.Store{CacheDir: cacheDir}
	profile, err := store.Begin(controller.Profile{Scope: scope, ArgsHash: controller.HashArgs(args)})
	if err != nil {
		t.Fatal(err)
	}
	for _, state := range []controller.State{
		controller.StateAllocationVerified, controller.StateLoadHealthy, controller.StateFunctionalVerified,
		controller.StateCacheVerified, controller.StatePerformanceVerified, controller.StateActive,
	} {
		if _, err := store.Transition(scope, profile.ID, state, "", "test"); err != nil {
			t.Fatal(err)
		}
	}
	placementPath := filepath.Join(cacheDir, "placements", "test.place")
	strategy := &placement.Strategy{Type: placement.MoEOffload, PlacementCachePath: placementPath, NCPUMoE: 2}
	if err := placement.SavePlacementCache(placementPath, placement.StrategyToCacheEntry(strategy)); err != nil {
		t.Fatal(err)
	}
	if err := invalidateRuntimeOOMLaunch(req, cfg, model, backend, caps, strategy, args, "CUDA OOM"); err != nil {
		t.Fatal(err)
	}
	record, err := store.Load(scope)
	if err != nil {
		t.Fatal(err)
	}
	if record.Active != nil {
		t.Fatalf("runtime-failed argv remained active: %#v", record.Active)
	}
	if _, err := os.Stat(placementPath); !os.IsNotExist(err) {
		t.Fatalf("stale placement cache survived runtime OOM: %v", err)
	}
}

func TestLaunchProfileScopeGroupsAdjustedArgvUnderRequestedPolicy(t *testing.T) {
	model := &placement.ModelProfile{Path: "/models/test.gguf", SizeBytes: 1234, ModelArch: "test"}
	backend := &backendInfo{Tag: "llama", Identity: "backend-build-a"}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 128000}, CPU: detect.CPUInfo{Model: "cpu", Cores: 8}}
	req := &launchRequest{CtxFlag: "131072", KVQuality: "q8_0", KVPlacement: "gpu", Parallel: 1, UBatchSize: 512}
	req.ProfilePolicyIdentity = requestedLaunchPolicyIdentity(req, model)

	before := launchProfileScope(req, model, backend, caps)
	// Simulate a measured/controller adjustment. Exact argv changes, but it is a
	// candidate for the same requested serving policy and must share Active/LKG.
	req.UBatchSize, req.UBatchSizeSet = 128, true
	req.ExtraArgs = append(req.ExtraArgs, "--metrics")
	after := launchProfileScope(req, model, backend, caps)
	if before != after {
		t.Fatalf("controller-adjusted argv changed profile family: %s != %s", before, after)
	}
	updatedBackend := &backendInfo{Tag: "llama", Identity: "backend-build-b"}
	if got := launchProfileScope(req, model, updatedBackend, caps); got != before {
		t.Fatalf("backend rebuild escaped its LKG family: %s != %s", got, before)
	}
	if got := launchProfileScope(req, model, &backendInfo{Tag: "ik_llama", Identity: "backend-build-c"}, caps); got == before {
		t.Fatal("different backend family reused the same lifecycle scope")
	}

	other := &launchRequest{CtxFlag: "65536", KVQuality: "q8_0", KVPlacement: "gpu", Parallel: 1, UBatchSize: 512}
	if launchProfileScope(other, model, backend, caps) == before {
		t.Fatal("different requested context policy reused the same profile family")
	}
}

func TestLaunchProfileScopeSeparatesClaudeCompanionIdentity(t *testing.T) {
	model := &placement.ModelProfile{Path: "/models/test.gguf", SizeBytes: 1234, ModelArch: "test"}
	backend := &backendInfo{Tag: "llama"}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 128000}}
	qwen := &launchRequest{ClaudeCode: true, ReviewerProfile: &claudeCompanionProfile{
		Name: claudeReviewerCompanionName, ModelPath: "/models/qwen.gguf", BackendPath: "/bin/mainline",
	}}
	nano := &launchRequest{ClaudeCode: true, ReviewerProfile: &claudeCompanionProfile{
		Name: claudeNanoCompanionName, ModelPath: "/models/nano.gguf", BackendPath: "/bin/nanbeige",
	}}
	if launchProfileScope(qwen, model, backend, caps) == launchProfileScope(nano, model, backend, caps) {
		t.Fatal("Qwen and Nano companions reused one lifecycle scope")
	}
	nanoQ8 := *nano
	nanoQ8.ReviewerProfile = &claudeCompanionProfile{
		Name: claudeNanoCompanionName, ModelPath: "/models/nano.gguf", BackendPath: "/bin/nanbeige", KVType: "q8_0",
	}
	nano.ReviewerProfile.KVType = "q4_0"
	if launchProfileScope(nano, model, backend, caps) == launchProfileScope(&nanoQ8, model, backend, caps) {
		t.Fatal("Q4 and Q8 companion profiles reused one lifecycle scope")
	}
	nanoLargerContext := *nano
	nanoLargerContext.ReviewerProfile = &claudeCompanionProfile{
		Name: claudeNanoCompanionName, ModelPath: "/models/nano.gguf", BackendPath: "/bin/nanbeige",
		KVType: "q4_0", ContextTokens: 262144,
	}
	nano.ReviewerProfile.ContextTokens = 131072
	if launchProfileScope(nano, model, backend, caps) == launchProfileScope(&nanoLargerContext, model, backend, caps) {
		t.Fatal("different companion context capacities reused one lifecycle scope")
	}
	nano.ClaudeReviewerDisabled = true
	if launchProfileScope(qwen, model, backend, caps) == launchProfileScope(nano, model, backend, caps) {
		t.Fatal("separate companion and main-model fallback reused one lifecycle scope")
	}
}

func TestReleaseReadingsAtBaseline(t *testing.T) {
	baseline := &detect.Capabilities{
		RAM:  detect.RAMInfo{FreeMB: 100000},
		GPUs: []detect.GPU{{Index: 0, VRAMUsedMB: 500}, {Index: 2, VRAMUsedMB: 900}},
	}
	if !releaseReadingsAtBaseline(baseline, 99500, map[int]int{0: 520, 2: 940}) {
		t.Fatal("small post-stop accounting drift should be accepted")
	}
	if releaseReadingsAtBaseline(baseline, 98000, map[int]int{0: 500, 2: 900}) {
		t.Fatal("unreleased main-model RAM was accepted")
	}
	if releaseReadingsAtBaseline(baseline, 100000, map[int]int{0: 700, 2: 900}) {
		t.Fatal("unreleased main-model VRAM was accepted")
	}
}

// ik_llama's CUDA host buffer prefaults and page-locks the whole allocation, so
// an 84 GiB CPU-expert plan stalls the launch before a single weight is read.
// GGML_CUDA_NO_PINNED routes those tensors to an ordinary CPU buffer instead.
func TestHostExpertPinningEnvDisablesPinnedIKHostBuffer(t *testing.T) {
	ik := &backendInfo{Path: "/ik/llama-server", IsIK: true}

	for _, args := range [][]string{
		{"-m", "model.gguf", "--n-cpu-moe", "32"},
		{"-m", "model.gguf", "-ot", `blk\.(0|1)\.ffn.*=CUDA0,exps=CPU`},
	} {
		got := hostExpertPinningEnv(ik, args)
		if len(got) != 1 || got[0] != "GGML_CUDA_NO_PINNED=1" {
			t.Fatalf("CPU-expert ik launch env = %v, want GGML_CUDA_NO_PINNED=1 (args %v)", got, args)
		}
	}

	// A fully GPU-resident plan keeps pinning: there is no large host buffer to
	// pay for, and pinned memory is genuinely faster for prompt processing.
	if got := hostExpertPinningEnv(ik, []string{"-m", "model.gguf", "-ngl", "999", "--n-cpu-moe", "0"}); got != nil {
		t.Fatalf("GPU-resident ik launch disabled pinning: %v", got)
	}

	// Mainline allocates CPU experts differently and is not subject to this.
	mainline := &backendInfo{Path: "/llama/llama-server"}
	if got := hostExpertPinningEnv(mainline, []string{"--n-cpu-moe", "32"}); got != nil {
		t.Fatalf("mainline launch disabled pinning: %v", got)
	}
	if got := hostExpertPinningEnv(nil, []string{"--n-cpu-moe", "32"}); got != nil {
		t.Fatalf("nil backend produced env %v", got)
	}
}

// The recorder files whatever allocation size failed as runtime graph growth,
// so the log window decides whether that number means anything. A launch that
// dies while allocating its KV buffer must teach the planner nothing: KV is
// budgeted separately, and recording it as growth reserves it twice. This is
// the real 2026-08-03 poisoning — CUDA0=5504 was that plan's KV buffer.
func TestRuntimeLogCUDAOOMIgnoresFailuresBeforeModelLoaded(t *testing.T) {
	model := &placement.ModelProfile{NumLayers: 43, IsMoE: true}

	loadTime := strings.Join([]string{
		"llama_kv_cache:      CUDA0 KV buffer size =  5504.00 MiB",
		"ggml-cuda.cu:104: CUDA error",
		"CUDA error: out of memory",
		"  current device: 0, in function alloc at ggml-cuda.cu:529",
	}, "\n")
	if _, reserve, _, ok := runtimeLogCUDAOOM(loadTime, nil, model, map[int]int{}); ok {
		t.Fatalf("a load-time OOM was recorded as runtime growth (%d MiB)", reserve)
	}

	// The same failure, but after the model is serving, IS runtime growth.
	runtime := strings.Join([]string{
		"llama_kv_cache:      CUDA0 KV buffer size =  5504.00 MiB",
		"srv  llama_server: model loaded",
		"slot create_check: created context checkpoint 3 of 16",
		"ggml-cuda.cu:104: CUDA error",
		"CUDA error: out of memory",
		"  current device: 0, in function alloc at ggml-cuda.cu:529",
	}, "\n")
	if _, _, _, ok := runtimeLogCUDAOOM(runtime, nil, model, map[int]int{}); !ok {
		t.Fatal("a post-load OOM was not recognised as runtime growth")
	}
}

// A log that never reached "model loaded" yields no growth at all, whatever it
// contains. Guessing here corrupts the cache for every later launch on the key.
func TestRuntimeGrowthWindowRequiresALoadCompleteMarker(t *testing.T) {
	if _, ok := runtimeGrowthWindowStart([]string{"loading model", "still loading"}); ok {
		t.Fatal("growth window opened without a load-complete marker")
	}
	at, ok := runtimeGrowthWindowStart([]string{"loading", "srv  llama_server: model loaded", "serving"})
	if !ok || at != 1 {
		t.Fatalf("growth window start = (%d, %v), want (1, true)", at, ok)
	}
}

// minimax-m3 could not be launched at all: RequiredBackendForArch mandates the
// ik family, ggrun resolved an ik backend whose libllama.so carries the
// "minimax-m3" architecture literal, and then refused it for being ik --
// "no proven main-model backend for architecture". The guard exists so a
// reviewed fork normally beats an ik build that merely claims an architecture,
// but it must not reject the family the architecture itself requires.
func TestReviewedRecipeAcceptsIKWhenTheArchRequiresIK(t *testing.T) {
	arch := "minimax-m3"
	if len(backends.RecipesForArch(arch)) == 0 {
		t.Skipf("no reviewed recipe registered for %s", arch)
	}
	if got := backends.RequiredBackendForArch(arch); !strings.EqualFold(got, "ik_llama") {
		t.Fatalf("fixture assumes %s requires ik_llama, got %q", arch, got)
	}

	// A binary that genuinely carries the architecture: the literal is
	// NUL-delimited exactly as BackendSupportsArch requires.
	dir := t.TempDir()
	supporting := filepath.Join(dir, "llama-server")
	body := append([]byte("\x00"+arch+"\x00"), []byte("\x00llama\x00")...)
	if err := os.WriteFile(supporting, body, 0o755); err != nil {
		t.Fatal(err)
	}
	if ok, probed := backends.BackendSupportsArch(supporting, arch); !ok || !probed {
		t.Fatalf("fixture binary should probe as supporting: ok=%v probed=%v", ok, probed)
	}

	if got := reviewedRecipeRequiredForMain(arch, &backendInfo{Path: supporting, IsIK: true}); got != nil {
		t.Errorf("ik backend carrying %s was still sent to the reviewed recipe %q", arch, got.Name)
	}
	// Mainline carrying it was already accepted and must stay accepted.
	if got := reviewedRecipeRequiredForMain(arch, &backendInfo{Path: supporting, IsIK: false}); got != nil {
		t.Errorf("mainline carrying %s was sent to the reviewed recipe %q", arch, got.Name)
	}

	// A backend that does not carry the architecture still needs the recipe,
	// which is the case the guard exists for.
	lacking := filepath.Join(dir, "llama-server-bare")
	if err := os.WriteFile(lacking, []byte("\x00llama\x00qwen3moe\x00"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := reviewedRecipeRequiredForMain(arch, &backendInfo{Path: lacking, IsIK: true}); got == nil {
		t.Error("an ik backend without the architecture must still require the reviewed recipe")
	}
}

// MiniMax-M3 could not run in any mode. Under mmap its CPU-side experts are
// file-backed, so the plan correctly reports a tiny host footprint -- but page
// cache is still charged to the cgroup, and with MemoryHigh == MemoryMax the
// reclaim threshold and the kill threshold arrive together. ~107 GiB of
// reclaimable experts under a 114 GiB cap walked straight into the OOM killer
// (measured: cgroup peak 114558 MiB, oom_kill=1) when the kernel could have
// dropped clean pages instead.
func TestMMapBackedPlanGetsAReclaimBandBeforeTheKillThreshold(t *testing.T) {
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 114844},
		CPU: detect.CPUInfo{Cores: 8},
	}
	req := &launchRequest{RAMLimitPercent: 95}
	mainline := &backendInfo{CPUExpertMMapCapability: placement.CPUExpertMMapFileBacked}
	anonymous := &backendInfo{CPUExpertMMapCapability: placement.CPUExpertMMapAnonymous}

	mmapArgs := []string{"-m", "m3.gguf", "--n-cpu-moe", "44", "-ot", "exps=CPU"}
	residentArgs := append(append([]string{}, mmapArgs...), "--no-mmap")

	if !argsMMapBackedExperts(mainline, mmapArgs) {
		t.Fatal("CPU experts with no --no-mmap must count as mmap-backed")
	}
	if argsMMapBackedExperts(mainline, residentArgs) {
		t.Fatal("--no-mmap means the experts are resident, not file-backed")
	}
	if argsMMapBackedExperts(mainline, []string{"-m", "dense.gguf"}) {
		t.Fatal("a plan with no CPU experts has no file-backed expert bytes")
	}
	if argsMMapBackedExperts(anonymous, mmapArgs) {
		t.Fatal("an anonymous CPU-expert loader must never receive a reclaim band")
	}

	// Resident: the bytes are anonymous and cannot be reclaimed, so a hard cap
	// at the plan's budget is exactly right and must not be loosened.
	res := backendStartOptions(req, caps, mainline, nil, residentArgs)
	if res.MemoryHighMB != res.MemoryMaxMB {
		t.Errorf("resident plan should keep a hard cap, got high=%d max=%d", res.MemoryHighMB, res.MemoryMaxMB)
	}
	if res.MemoryMaxMB != backendMemoryMaxMB(req, caps) {
		t.Errorf("resident ceiling moved off the plan budget: %d", res.MemoryMaxMB)
	}

	// mmap: reclaim at the budget, kill only at the whole-host limit.
	mm := backendStartOptions(req, caps, mainline, nil, mmapArgs)
	if mm.MemoryHighMB != backendMemoryMaxMB(req, caps) {
		t.Errorf("reclaim threshold should stay at the plan budget, got %d", mm.MemoryHighMB)
	}
	if mm.MemoryMaxMB <= mm.MemoryHighMB {
		t.Errorf("mmap plan has no reclaim band: high=%d max=%d", mm.MemoryHighMB, mm.MemoryMaxMB)
	}
	// The percent target becomes reclaim pressure, so the hard ceiling is the RAM
	// that was actually free -- the physical bound the plan could reach in any
	// case. It must never exceed that, and must leave a real band below it.
	if mm.MemoryMaxMB > caps.RAM.FreeMB {
		t.Errorf("hard ceiling %d exceeds free RAM %d", mm.MemoryMaxMB, caps.RAM.FreeMB)
	}
	if mm.MemoryHighMB >= mm.MemoryMaxMB {
		t.Errorf("no reclaim band: high=%d max=%d", mm.MemoryHighMB, mm.MemoryMaxMB)
	}

	// An explicitly named --ram-budget is a ceiling the user asked for by name
	// and must stay hard, even under mmap.
	named := backendStartOptions(&launchRequest{RamBudgetMB: 90000, RAMLimitPercent: 95}, caps, mainline, nil, mmapArgs)
	if named.MemoryHighMB != named.MemoryMaxMB {
		t.Errorf("an explicit --ram-budget must stay a hard cap, got high=%d max=%d", named.MemoryHighMB, named.MemoryMaxMB)
	}

	// With no percent target there is nothing to reinterpret as pressure, so
	// containment must be exactly as it was.
	bare := backendStartOptions(&launchRequest{RamBudgetMB: 100000}, caps, mainline, nil, mmapArgs)
	if bare.MemoryHighMB != bare.MemoryMaxMB {
		t.Errorf("no ram-limit-percent must leave the cap unchanged, got high=%d max=%d", bare.MemoryHighMB, bare.MemoryMaxMB)
	}
}

func TestMMapPageabilityContradictionUsesExactExpertScale(t *testing.T) {
	strategy := &placement.Strategy{
		Type:                     placement.MoEOffload,
		MMap:                     true,
		CPUExpertMMapCapability:  placement.CPUExpertMMapFileBacked,
		ReclaimableHostWeightsMB: 80000,
		PlannedHostFootprintMB:   4000,
		CRAM:                     8000,
	}
	if contradicted, limit := mmapPageabilityContradicted(strategy, 18000); contradicted || limit != 22000 {
		t.Fatalf("ordinary working set was treated as anonymous experts: contradicted=%v limit=%d", contradicted, limit)
	}
	if contradicted, _ := mmapPageabilityContradicted(strategy, 70000); !contradicted {
		t.Fatal("large anonymous CPU-expert allocation did not contradict file-backed capability")
	}
	anonymous := *strategy
	anonymous.CPUExpertMMapCapability = placement.CPUExpertMMapAnonymous
	if contradicted, _ := mmapPageabilityContradicted(&anonymous, 70000); contradicted {
		t.Fatal("known-anonymous plan was incorrectly treated as a new pageability contradiction")
	}
}

func TestProbedMMapCapabilityCannotBeForgedByBackendTag(t *testing.T) {
	ik := &backendInfo{Tag: "llama", Dialect: "llama", IsIK: true, Help: "ikawrakow --model", Identity: "ik-build"}
	capability, _ := probedCPUExpertMMapCapability(ik)
	if capability != placement.CPUExpertMMapAnonymous {
		t.Fatalf("misleading friendly tag authorized ik anonymous experts: %q", capability)
	}
	mainline := &backendInfo{Tag: "custom-name", Dialect: "llama", Help: "--model FNAME --ctx-size N", Identity: "main-build"}
	capability, _ = probedCPUExpertMMapCapability(mainline)
	if capability != placement.CPUExpertMMapFileBacked {
		t.Fatalf("probed mapped loader was not recognized: %q", capability)
	}
	unknown := &backendInfo{Tag: "llama", Dialect: "llama", Identity: "opaque-build"}
	capability, _ = probedCPUExpertMMapCapability(unknown)
	if capability != placement.CPUExpertMMapUnknown {
		t.Fatalf("unprobed loader did not fail closed: %q", capability)
	}
}

func TestDaemonMemoryScopeMatchesProductionCapabilityRules(t *testing.T) {
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 114844}}
	req := &launchRequest{RAMLimitPercent: 95}
	args := []string{"llama-server", "--n-cpu-moe", "44", "-ot", "exps=CPU"}
	mainline := &backendInfo{CPUExpertMMapCapability: placement.CPUExpertMMapFileBacked}
	high, max, err := daemonMemoryScope(req, caps, mainline, args, 0)
	if err != nil || high != backendMemoryMaxMB(req, caps) || max <= high {
		t.Fatalf("daemon did not inherit production mmap band: high=%d max=%d err=%v", high, max, err)
	}
	high, max, err = daemonMemoryScope(req, caps, mainline, args, 114000)
	if err != nil || high != backendMemoryMaxMB(req, caps) || max != 114000 {
		t.Fatalf("daemon hard override lost the production soft threshold: high=%d max=%d err=%v", high, max, err)
	}
	anonymous := &backendInfo{CPUExpertMMapCapability: placement.CPUExpertMMapAnonymous}
	high, max, err = daemonMemoryScope(req, caps, anonymous, args, 0)
	if err != nil || high != max {
		t.Fatalf("anonymous loader received a daemon reclaim band: high=%d max=%d err=%v", high, max, err)
	}
}

// TestMeasuredFootprintCgroupMaxMB guards Fix B's sizing rule: the cgroup limit
// is set to the measured post-launch non-reclaimable footprint + headroom,
// clamped to the whole-host ceiling. This is what keeps a correct plan from
// dying (headroom absorbs jitter and checkpoint growth) while still containing a
// runaway to its own scope (the limit never exceeds the user's ceiling).
func TestMeasuredFootprintCgroupMaxMB(t *testing.T) {
	// A measured footprint comfortably under the ceiling gets measured+headroom.
	if got := measuredFootprintCgroupMaxMB(90000, 4096, 0, 111500); got != 94096 {
		t.Fatalf("measured+headroom sizing: got %d, want 94096", got)
	}
	// measured+headroom over the ceiling is clamped to the ceiling (fail-closed).
	if got := measuredFootprintCgroupMaxMB(110000, 4096, 0, 111500); got != 111500 {
		t.Fatalf("over-ceiling must clamp to ceiling: got %d, want 111500", got)
	}
	// Zero headroom disables the auto re-size entirely.
	if got := measuredFootprintCgroupMaxMB(90000, 0, 0, 111500); got != 0 {
		t.Fatalf("zero headroom must disable auto re-size, got %d", got)
	}
	// A zero/invalid measured footprint disables the re-size.
	if got := measuredFootprintCgroupMaxMB(0, 4096, 0, 111500); got != 0 {
		t.Fatalf("zero measured footprint must disable auto re-size, got %d", got)
	}
	// The canary has not filled the prompt cache yet, so a larger planned
	// host-footprint+CRAM requirement remains available for later requests.
	if got := measuredFootprintCgroupMaxMB(70000, 4096, 82000, 100000); got != 82000 {
		t.Fatalf("planned cache-capacity floor: got %d, want 82000", got)
	}
	// The whole-host ceiling remains absolute even when the plan asks for more.
	if got := measuredFootprintCgroupMaxMB(70000, 4096, 82000, 80000); got != 80000 {
		t.Fatalf("planned floor must clamp to ceiling: got %d, want 80000", got)
	}
}

// TestMeasuredFootprintPlannedFloor guards Fix-B's planned-floor for strategies
// that derive no PlannedHostFootprintMB. The Qwen3.8 27B crash was exactly this:
// a MultiGPUDense placement (PlannedHostFootprintMB == 0) made the floor collapse
// to bare CRAM, so the measured-footprint resize wrote ~11 GB = the prompt-cache
// budget as the scope ceiling and the server OOM'd against its own scope on a
// long prompt. The floor must never shrink the scope below the pre-launch plan
// ceiling when the plan priced no host cost.
func TestMeasuredFootprintPlannedFloor(t *testing.T) {
	ceiling := 122000
	measured := 10000

	// A plan with a real host footprint plus CRAM preserves that floor.
	dense := &placement.Strategy{Type: placement.MultiGPUDense, PlannedHostFootprintMB: 6000, CRAM: 11264}
	if got := measuredFootprintPlannedFloor(dense, measured, ceiling); got != 17264 {
		t.Fatalf("planned host footprint + CRAM floor: got %d, want 17264", got)
	}
	// The same fully-resident plan with NO derived footprint must NOT collapse to
	// CRAM: the floor becomes the whole-host ceiling, so the resize can only
	// tighten to measured+headroom and never clamp the scope below the plan
	// ceiling (the Qwen3.8 crash).
	noFootprint := &placement.Strategy{Type: placement.MultiGPUDense, PlannedHostFootprintMB: 0, CRAM: 11264}
	if got := measuredFootprintPlannedFloor(noFootprint, measured, ceiling); got != ceiling {
		t.Fatalf("no-host-footprint floor must be the whole-host ceiling, got %d", got)
	}
	// A plan whose host footprint + CRAM under-counts the measured backend keeps
	// measured as the honest floor (CRAM not yet filled, checkpoint reserve below
	// reality), never the stale plan estimate.
	under := &placement.Strategy{Type: placement.MoEOffload, PlannedHostFootprintMB: 2000, CRAM: 1024}
	if got := measuredFootprintPlannedFloor(under, measured, ceiling); got != measured {
		t.Fatalf("under-counting plan floor must be the measured footprint, got %d", got)
	}
	// A nil strategy yields no floor at all (pure measured+headroom sizing).
	if got := measuredFootprintPlannedFloor(nil, measured, ceiling); got != 0 {
		t.Fatalf("nil strategy floor must be 0, got %d", got)
	}
}

// TestDenseScopeNotClampedToCRAM guards the end-to-end guarantee: after a
// successful health check the cgroup must be re-sized to measured footprint +
// headroom (not the low plan budget), and a fully-resident placement must never
// be clamped back to its CRAM value. This is the policy promise the Qwen3.8
// crash broke.
func TestDenseScopeNotClampedToCRAM(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("host memory containment is Linux-only")
	}
	req := &launchRequest{RAMLimitPercent: 95, CgroupHeadroomMB: 4096}
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 114844},
		CPU: detect.CPUInfo{Cores: 8},
	}
	ceiling := backendMemoryMaxMB(req, caps)
	if ceiling <= 0 {
		t.Fatalf("expected a positive whole-host ceiling, got %d", ceiling)
	}
	// A dense placement that now derives its real host footprint (Fix-B part 2).
	// The floor preserves host footprint + CRAM, but the measured backend exceeds
	// it, so the resize returns measured+headroom -- never the low plan budget.
	dense := &placement.Strategy{Type: placement.MultiGPUDense, PlannedHostFootprintMB: 3000, CRAM: 11264}
	measured := 27000 // server + CUDA host buffers + KV + checkpoints for Qwen3.8-27B
	floor := measuredFootprintPlannedFloor(dense, measured, ceiling)
	if floor != measured {
		t.Fatalf("dense plan under-counting the measured backend must floor at measured, got %d", floor)
	}
	newMax := measuredFootprintCgroupMaxMB(measured, req.CgroupHeadroomMB, floor, ceiling)
	if newMax != measured+req.CgroupHeadroomMB {
		t.Fatalf("dense scope must be re-sized to measured+headroom (%d), got %d (clamped to CRAM?)",
			measured+req.CgroupHeadroomMB, newMax)
	}
	if newMax <= dense.CRAM {
		t.Fatalf("dense scope re-sized to %d MiB, at or below the CRAM budget %d MiB (the Qwen3.8 OOM)", newMax, dense.CRAM)
	}
	// A fully-resident placement with NO derived footprint (DenseCPUOffload /
	// CPUOnly still do not derive one) must not collapse to bare CRAM either:
	// the floor is the pre-launch whole-host ceiling, so Fix-B only ever keeps
	// or raises the plan ceiling for such a plan -- never clamps it below the
	// backend's real need.
	noFootprint := &placement.Strategy{Type: placement.DenseCPUOffload, PlannedHostFootprintMB: 0, CRAM: 11264}
	if floor := measuredFootprintPlannedFloor(noFootprint, measured, ceiling); floor != ceiling {
		t.Fatalf("no-footprint dense plan must floor at the whole-host ceiling, got %d", floor)
	}
	if newMax := measuredFootprintCgroupMaxMB(measured, req.CgroupHeadroomMB, ceiling, ceiling); newMax != ceiling {
		t.Fatalf("no-footprint plan must stay at the pre-launch ceiling, got %d", newMax)
	}
}

// TestMeasuredFootprintCgroupResizeMath guards the end-to-end sizing path through
// the launch request: headroom wiring (config default 4096) feeds the pure sizing
// rule, and an explicit --ram-budget disables it.
func TestMeasuredFootprintCgroupResizeMath(t *testing.T) {
	req := &launchRequest{RAMLimitPercent: 95, CgroupHeadroomMB: 4096}
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 114844},
		CPU: detect.CPUInfo{Cores: 8},
	}
	ceiling := backendMemoryMaxMB(req, caps)
	if ceiling <= 0 {
		t.Fatalf("expected a positive whole-host ceiling, got %d", ceiling)
	}
	// A footprint just under the ceiling gets measured+headroom; the ceiling is
	// only ever the clamp, never loosened.
	measured := ceiling - 8192
	got := measuredFootprintCgroupMaxMB(measured, req.CgroupHeadroomMB, 0, ceiling)
	if got != measured+req.CgroupHeadroomMB {
		t.Fatalf("measured+headroom under the ceiling: got %d, want %d", got, measured+req.CgroupHeadroomMB)
	}
	// A footprint whose +headroom overshoots the ceiling is clamped to it.
	if got := measuredFootprintCgroupMaxMB(ceiling-1024, req.CgroupHeadroomMB, 0, ceiling); got != ceiling {
		t.Fatalf("overshoot must clamp to ceiling: got %d, want %d", got, ceiling)
	}
}

// TestValidateHostMemoryContainmentFailClosed guards Fix B.4: a --no-mmap plan
// whose planned footprint + cgroup headroom exceeds the whole-host ceiling is
// refused BEFORE launch, converting the 41-layer V4 crash (planned ~116 GB vs a
// ~111 GB ceiling) into a pre-launch error instead of an OOM-killed server.
func TestValidateHostMemoryContainmentFailClosed(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("host memory containment is Linux-only")
	}
	caps := &detect.Capabilities{
		RAM: detect.RAMInfo{TotalMB: 128512, FreeMB: 114844},
		CPU: detect.CPUInfo{Cores: 8},
	}
	req := &launchRequest{RAMLimitPercent: 95, CgroupHeadroomMB: 4096}

	// A --no-mmap plan whose footprint is inside the ceiling passes.
	ok := &placement.Strategy{Type: placement.MoEOffload, MMap: false, PlannedHostFootprintMB: 100000}
	if err := validateHostMemoryContainment(req, caps, ok); err != nil {
		t.Fatalf("in-ceiling plan must pass: %v", err)
	}
	// A --no-mmap plan whose footprint + headroom blows the ceiling is refused.
	over := &placement.Strategy{Type: placement.MoEOffload, MMap: false, PlannedHostFootprintMB: 111000}
	if err := validateHostMemoryContainment(req, caps, over); err == nil {
		t.Fatal("over-ceiling --no-mmap plan must be refused")
	}
	// A large future prompt cache must also fit. The old headroom-only gate
	// accepted this plan, then the post-canary resize silently clamped away part
	// of CRAM and exposed the backend to another late cache-save OOM.
	cacheOver := &placement.Strategy{Type: placement.MoEOffload, MMap: false, PlannedHostFootprintMB: 108000, CRAM: 8192}
	if err := validateHostMemoryContainment(req, caps, cacheOver); err == nil {
		t.Fatal("host footprint + CRAM above the ceiling must be refused")
	}
	// An mmap-backed plan is NOT refused: page cache can be reclaimed, so the
	// mmap reclaim band absorbs overshoot (backendStartOptions moves the hard
	// ceiling up to the host reclaim ceiling).
	mmap := &placement.Strategy{
		Type: placement.MoEOffload, MMap: true, PlannedHostFootprintMB: 111000,
		ReclaimableHostWeightsMB: 90000, CPUExpertMMapCapability: placement.CPUExpertMMapFileBacked,
	}
	if err := validateHostMemoryContainment(req, caps, mmap); err != nil {
		t.Fatalf("mmap plan must not be fail-closed by footprint: %v", err)
	}
	anonymousMMap := *mmap
	anonymousMMap.CPUExpertMMapCapability = placement.CPUExpertMMapAnonymous
	if err := validateHostMemoryContainment(req, caps, &anonymousMMap); err == nil {
		t.Fatal("anonymous CPU experts must not bypass resident containment just because mmap is enabled")
	}
	// Zero headroom disables the gate (auto re-size off).
	noHeadroom := &launchRequest{RAMLimitPercent: 95, CgroupHeadroomMB: 0}
	if err := validateHostMemoryContainment(noHeadroom, caps, over); err != nil {
		t.Fatalf("zero headroom must disable the fail-closed gate: %v", err)
	}
	// The planned-footprint field drives the gate; without it the gate is inert.
	noFootprint := &placement.Strategy{Type: placement.MoEOffload, MMap: false}
	if err := validateHostMemoryContainment(req, caps, noFootprint); err != nil {
		t.Fatalf("no planned footprint must not trigger the gate: %v", err)
	}
}

// MODEL_DIR is a single path, so every download landed on whichever filesystem
// it sits on -- here the 456 GB root volume with 67 GB free, while the 1.9 TB
// volume holding the large quants had 935 GB. The only way to put a model there
// was to download it elsewhere and symlink it in by hand.
func TestDownloadAcceptsAnExplicitDestination(t *testing.T) {
	const fallback = "/home/mik/ggrun-project/ggrun/models"
	alt := "/home/mik/2tb-disk/AI_Models"

	for _, form := range [][]string{
		{"unsloth/Inkling-Small-GGUF", "--dir", alt},
		{"unsloth/Inkling-Small-GGUF", "--dir=" + alt},
		{"--dir", alt, "unsloth/Inkling-Small-GGUF"},
		{"unsloth/Inkling-Small-GGUF", "-dir", alt},
		{"unsloth/Inkling-Small-GGUF", "--model-dir", alt},
	} {
		repo, dir, _, err := downloadOptionsFromArgs(form, fallback)
		if err != nil {
			t.Fatalf("%v: %v", form, err)
		}
		if repo != "unsloth/Inkling-Small-GGUF" {
			t.Errorf("%v: repo = %q", form, repo)
		}
		if dir != alt {
			t.Errorf("%v: dir = %q, want %q", form, dir, alt)
		}
	}

	// No flag means the configured MODEL_DIR, unchanged.
	repo, dir, _, err := downloadOptionsFromArgs([]string{"unsloth/Inkling-Small-GGUF"}, fallback)
	if err != nil || repo != "unsloth/Inkling-Small-GGUF" || dir != fallback {
		t.Errorf("bare download changed behaviour: repo=%q dir=%q err=%v", repo, dir, err)
	}

	if _, _, _, err := downloadOptionsFromArgs([]string{"--dir"}, fallback); err == nil {
		t.Error("a --dir with no value must be rejected, not silently ignored")
	}
	if _, _, _, err := downloadOptionsFromArgs([]string{"--dir", alt}, fallback); err == nil {
		t.Error("a destination with no repository must be rejected")
	}
	if _, _, _, err := downloadOptionsFromArgs(nil, fallback); err == nil {
		t.Error("no arguments must be rejected")
	}
}

// A destination typed by hand -- which is how the TUI collects it -- routinely
// starts with ~. Without expansion that creates a directory literally named "~"
// in the working directory and the model lands on the wrong disk, which is the
// exact failure this flag exists to avoid.
func TestExpandPathResolvesTildeAndEnv(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		t.Skip("no home directory available")
	}
	if got := expandPath("~/2tb-disk/AI_Models"); got != filepath.Join(home, "2tb-disk/AI_Models") {
		t.Fatalf("tilde not expanded: %q", got)
	}
	if got := expandPath("~"); got != home {
		t.Fatalf("bare tilde not expanded: %q", got)
	}
	t.Setenv("GGRUN_TEST_DEST", "/mnt/2tb")
	if got := expandPath("$GGRUN_TEST_DEST/models"); got != "/mnt/2tb/models" {
		t.Fatalf("env not expanded: %q", got)
	}
	if got := expandPath("  /mnt/2tb/models  "); got != "/mnt/2tb/models" {
		t.Fatalf("surrounding space not trimmed: %q", got)
	}
	// A path that is neither ~ nor env-bearing must survive untouched apart from
	// being made absolute.
	if got := expandPath("/mnt/2tb/AI_Models"); got != "/mnt/2tb/AI_Models" {
		t.Fatalf("absolute path altered: %q", got)
	}
}

// Without an explicit quant the downloader drops into its interactive numbered
// picker, so a download cannot be started from anything without a terminal --
// which rules out scripted and unattended use entirely.
func TestDownloadAcceptsAnExplicitQuant(t *testing.T) {
	const fallback = "/models"
	for _, form := range [][]string{
		{"unsloth/Inkling-Small-GGUF", "--quant", "UD-Q3_K_XL"},
		{"unsloth/Inkling-Small-GGUF", "--quant=UD-Q3_K_XL"},
		{"--quant", "UD-Q3_K_XL", "unsloth/Inkling-Small-GGUF"},
		{"unsloth/Inkling-Small-GGUF", "-quant", "UD-Q3_K_XL"},
	} {
		repo, _, quant, err := downloadOptionsFromArgs(form, fallback)
		if err != nil {
			t.Fatalf("%v: %v", form, err)
		}
		if repo != "unsloth/Inkling-Small-GGUF" || quant != "UD-Q3_K_XL" {
			t.Errorf("%v: repo=%q quant=%q", form, repo, quant)
		}
	}

	// Both flags together is the case that actually motivated this: a 111 GiB
	// quant that has to land on the other disk, started without a terminal.
	repo, dir, quant, err := downloadOptionsFromArgs(
		[]string{"unsloth/Inkling-Small-GGUF", "--dir", "/home/mik/2tb-disk/AI_Models", "--quant", "UD-Q3_K_XL"},
		fallback)
	if err != nil {
		t.Fatalf("combined flags: %v", err)
	}
	if repo != "unsloth/Inkling-Small-GGUF" || dir != "/home/mik/2tb-disk/AI_Models" || quant != "UD-Q3_K_XL" {
		t.Errorf("combined flags: repo=%q dir=%q quant=%q", repo, dir, quant)
	}

	// Omitting it must keep the previous behaviour: empty means "ask".
	if _, _, quant, err := downloadOptionsFromArgs([]string{"org/repo"}, fallback); err != nil || quant != "" {
		t.Errorf("bare download must not preselect a quant: quant=%q err=%v", quant, err)
	}
	if _, _, _, err := downloadOptionsFromArgs([]string{"org/repo", "--quant"}, fallback); err == nil {
		t.Error("a --quant with no value must be rejected, not silently ignored")
	}
}

func TestMeasuredLaunchProbesNetsNanoBeigeUnderMeasurementKey(t *testing.T) {
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	profile := &claudeCompanionProfile{
		Name: claudeNanoCompanionName, MeasurementKey: claudeNanoCompanionName + "-ctx65536-kv-q4_0",
	}
	// Write the measured VRAM exactly as recordReviewerVRAM does, under the
	// KV-qualified MeasurementKey — the real storage location.
	if err := placement.RecordCompanionVRAM(cfg.CacheDir, profile.companionMeasurementKey(), 2916); err != nil {
		t.Fatalf("seed companion VRAM: %v", err)
	}
	req := &launchRequest{ReviewerProfile: profile}
	model := &placement.ModelProfile{IsMoE: true}
	strategy := &placement.Strategy{CompanionPlacements: []placement.CompanionPlacement{{Name: claudeNanoCompanionName, GPU: 1}}}
	be := &backendInfo{Tag: "llama"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 1, VRAMTotalMB: 12288}}}
	// recordMeasuredLaunchProbes requires a non-empty server log and is MoE-gated.
	recordMeasuredLaunchProbes(req, cfg, model, strategy, be, caps, "fake-server-log\n", nil, 0)
	// The NanoBeige worker's VRAM must now be nettable under its MeasurementKey
	// (the fix resolves the key via the frozen profile). Without the fix the
	// lookup under cp.Name misses and the worker stays booked as CUDA overhead.
	if got := placement.MeasuredCompanionVRAMMB(cfg.CacheDir, profile.companionMeasurementKey()); got != 2916 {
		t.Fatalf("companion measurement not found under MeasurementKey: got %d", got)
	}
}

func TestRuntimeOOMReplanRefusesIdenticalFailedArgv(t *testing.T) {
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	req := &launchRequest{CtxFlag: "32768", Parallel: 1}
	model := &placement.ModelProfile{SizeBytes: 1, NumLayers: 32, HeadCountKV: 8, KeyLength: 128, ValueLength: 128}
	be := &backendInfo{Tag: "llama", Identity: "build"}
	caps := &detect.Capabilities{CPU: detect.CPUInfo{Cores: 4}, RAM: detect.RAMInfo{TotalMB: 16384, FreeMB: 16384}}

	// Derive a first argv, then ask for a re-plan after a runtime OOM of that
	// same argv. The crashed argv is rejected on the shared lifecycle recovery,
	// and an identical re-plan is refused outright.
	firstStrategy, firstArgs, err := replanAfterRuntimeOOM(req, cfg, model, be, caps, nil, newLaunchMemoryRecovery())
	if err != nil {
		t.Fatalf("first runtime OOM replan failed: %v", err)
	}
	if firstStrategy == nil {
		t.Fatal("first replan returned no strategy")
	}
	// Re-planning after the exact argv that just crashed must reject that argv
	// and refuse an identical relaunch: a fresh derivation that reproduces the
	// failed placement must not be handed back to be re-run identically.
	_, _, err = replanAfterRuntimeOOM(req, cfg, model, be, caps, firstArgs, newLaunchMemoryRecovery())
	if err == nil || !strings.Contains(err.Error(), "refusing an identical relaunch") {
		t.Fatalf("re-plan after the crashed argv should refuse an identical relaunch, got %v", err)
	}
}

func TestComputeServerArgsAppliesCachedCalibrationWinner(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(39 * 1024)
	cfg.CacheDir = t.TempDir()
	// Compute the base estimate first, then seed a kv-alternate decision under
	// the exact scope that strategy produces — so a fresh recompute of the same
	// shape must consume the winner rather than the raw estimate.
	base, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil || base == nil {
		t.Fatalf("base compute: %v", err)
	}
	scopeKey := calibrationScopeKey(req, model, be, caps, base)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: "kv-alternate", DefaultTPS: 20, WinnerTPS: 24,
		ValidationLevel: placement.CalibrationValidationWorkflow,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}
	// Recompute the same shape (as the daemon's computeServerArgs would) and
	// confirm the cached winner is consumed, not the raw estimate.
	recomputed, err := placement.Compute(caps, model, placementOptionsFromRequest(req, model, be, cfg.CacheDir))
	if err != nil {
		t.Fatalf("recompute: %v", err)
	}
	applied := applyCalibrationDecision(req, cfg, model, be, caps, recomputed)
	if applied == recomputed {
		t.Fatal("daemon compute path did not consume the cached calibration winner")
	}
}

// --no-cached-config must reach placement as SkipCachedConfig so the launch
// derives fresh (placement cache, probe caches, and measured KV all ignored).
func TestParseLaunchArgsNoCachedConfigFeedsPlacement(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--no-cached-config"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.NoCachedConfig {
		t.Fatalf("expected --no-cached-config to set the launch request")
	}
	opts := placementOptionsFromRequest(req, &placement.ModelProfile{CTXTrain: 32768}, &backendInfo{Tag: "llama"}, t.TempDir())
	if !opts.SkipCachedConfig {
		t.Fatalf("expected placement options to receive SkipCachedConfig")
	}
}

// The help text is the only place a user learns the full --claude-reviewer
// value set; "off" was selectable but absent from the enumeration, and the
// bare list said nothing about what each value actually seats.
func TestClaudeReviewerHelpEnumeratesOffAndDescribesEachValue(t *testing.T) {
	help := usageText()
	// Assert inside the --claude-reviewer block only: bare words like "off"
	// appear under other flags too, and a global Contains would pass even if
	// the enumeration lost the value again.
	start := strings.Index(help, "--claude-reviewer")
	if start < 0 {
		t.Fatal("help text has no --claude-reviewer entry")
	}
	block := help[start:]
	if end := strings.Index(block, "\n  --"); end >= 0 {
		block = block[:end]
	}
	for _, want := range []string{"off", "auto", "qwen2b", "nanbeige"} {
		if !strings.Contains(block, want) {
			t.Errorf("--claude-reviewer help block does not enumerate %q: %q", want, block)
		}
	}
	// Each description must match engine semantics (resolveClaudeCompanionProfile):
	// auto/qwen share the 4B worker+reviewer, qwen2b is review-only, nanbeige is
	// the big worker, off seats nothing.
	if !strings.Contains(block, "Qwen3.5-4B") {
		t.Errorf("help block does not say which model auto/qwen resolve to: %q", block)
	}
	if !strings.Contains(block, "review-only") {
		t.Errorf("help block does not describe qwen2b as review-only: %q", block)
	}
	if !strings.Contains(block, "Nanbeige4.2") {
		t.Errorf("help block does not name the nanbeige worker: %q", block)
	}
	if !strings.Contains(block, "self-review") {
		t.Errorf("help block does not say what off means for reviews: %q", block)
	}
}

func TestHelpAdvertisesMeasuredBandwidthDetection(t *testing.T) {
	if help := usageText(); !strings.Contains(help, "detect --bandwidth") {
		t.Fatalf("help does not advertise the hardware bandwidth profiler")
	}
}

func TestHelpAdvertisesLaunchOptimizerStatus(t *testing.T) {
	help := usageText()
	if !strings.Contains(help, "status [model.gguf]") {
		t.Fatalf("help does not advertise ggrun status: %s", help)
	}
}

// With --claude-reviewer off no reviewer process starts, so a dry-run that
// promises one contradicts the real launch. The notice must switch to what
// actually happens.
func TestDryRunClaudeNoticeOmittedWhenReviewerOff(t *testing.T) {
	capture := captureStdout(t)
	printDryRunClaudeNotice(true, true)
	out := capture()
	if strings.Contains(out, "starts the local Auto reviewer") {
		t.Errorf("reviewer-off dry-run still promised a reviewer: %q", out)
	}
	if !strings.Contains(out, "no separate reviewer") {
		t.Errorf("reviewer-off dry-run did not explain self-review: %q", out)
	}
}

// The default notice must keep promising the companion, and a plain launch
// must print nothing at all.
func TestDryRunClaudeNoticeKeptWhenReviewerSeated(t *testing.T) {
	capture := captureStdout(t)
	printDryRunClaudeNotice(true, false)
	out := capture()
	if !strings.Contains(out, "starts the local Auto reviewer/router") {
		t.Errorf("seated-reviewer dry-run lost its notice: %q", out)
	}

	capture = captureStdout(t)
	printDryRunClaudeNotice(false, false)
	if out := capture(); out != "" {
		t.Errorf("non-Claude dry-run printed %q, want silence", out)
	}
}

// Every explicit --claude-reviewer value now hits the same requires-claude-code
// gate. "auto" used to bypass it, silently accepting a flag that could never
// take effect outside Claude Code mode.
func TestClaudeReviewerAutoRequiresClaudeCodeLikeOtherValues(t *testing.T) {
	isolateConfig(t)
	for _, value := range []string{"auto", "qwen", "qwen2b", "nanbeige", "off"} {
		_, err := parseLaunchArgs([]string{"model.gguf", "--claude-reviewer", value})
		if err == nil {
			t.Errorf("--claude-reviewer %s without --claude-code was accepted", value)
			continue
		}
		if !strings.Contains(err.Error(), "--claude-code") {
			t.Errorf("--claude-reviewer %s failed with %v, want the --claude-code gate", value, err)
		}
	}
}

// --inventory lets a dry-run plan for a machine other than the one running
// ggrun, or for this one while another process holds its VRAM. The flag is
// planning-only: it must be parsed but must never be usable to launch, because
// invariant 4 makes exact-argv admission against present hardware the authority.
func TestParseLaunchArgsRetainsInventoryForPlanning(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--inventory", "/tmp/box.json"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if req.InventoryPath != "/tmp/box.json" {
		t.Fatalf("--inventory was not retained: %q", req.InventoryPath)
	}
}

// A missing value must be an error, not a silently empty path that would fall
// back to planning against the live machine.
func TestParseLaunchArgsInventoryNeedsAValue(t *testing.T) {
	isolateConfig(t)
	if _, err := parseLaunchArgs([]string{"model.gguf", "--inventory"}); err == nil {
		t.Fatal("--inventory with no value was accepted")
	}
}

// The flag must not be swallowed by the "skip the next token" walk that handles
// value-taking flags, or the path would be re-parsed as a positional argument.
func TestInventoryValueIsNotTreatedAsAPositionalArg(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--inventory", "/tmp/box.json"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if req.ModelPath != "model.gguf" {
		t.Fatalf("the inventory path displaced the model path: %q", req.ModelPath)
	}
}

// The `--flag=value` spelling never reaches the bare-token switch, and an
// unhandled token falls through to ExtraArgs. Before this was handled, an
// `--inventory=/path` was silently swallowed: the plan ran against the LIVE
// machine while the user believed they had modelled another, and the literal
// token was handed to the backend.
func TestInventoryEqualsFormIsParsedNotSwallowed(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--inventory=/tmp/box.json", "--swa-full"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if req.InventoryPath != "/tmp/box.json" {
		t.Fatalf("--inventory=path was not parsed: %q (ExtraArgs=%v)", req.InventoryPath, req.ExtraArgs)
	}
	for _, a := range req.ExtraArgs {
		if a == "--inventory=/tmp/box.json" {
			t.Fatal("--inventory=path leaked into ExtraArgs and would be sent to the backend")
		}
	}
}

// An empty value must fail closed. Accepting it would leave InventoryPath empty
// and quietly plan against the live machine, which is the opposite of the
// command's intent.
func TestInventoryEmptyEqualsFormIsAnError(t *testing.T) {
	isolateConfig(t)
	if _, err := parseLaunchArgs([]string{"model.gguf", "--inventory="}); err == nil {
		t.Fatal("--inventory= with no path was accepted; it would plan against the live machine")
	}
}

// --ctx-checkpoints must produce exactly ONE value in the argv.
//
// It previously had no parse case, so the token fell through to ExtraArgs and
// the emitted argv carried the coordinate twice: ggrun's derived value and the
// user's. llama.cpp keeps the last, so the override appeared to work — but two
// values for one coordinate is the partial overlay invariant 2 forbids, and it
// makes "diff the final full argv" comparisons ambiguous.
func TestCtxCheckpointsOverrideReplacesTheDerivedValue(t *testing.T) {
	isolateConfig(t)

	// Both spellings must parse into the request rather than ExtraArgs.
	for _, form := range [][]string{
		{"model.gguf", "--ctx-checkpoints", "32"},
		{"model.gguf", "--ctx-checkpoints=32"},
		{"model.gguf", "-ctxcp", "32"},
	} {
		req, err := parseLaunchArgs(form)
		if err != nil {
			t.Fatalf("%v: parse: %v", form, err)
		}
		if !req.MaxCheckpointsSet || req.MaxCheckpoints != 32 {
			t.Errorf("%v did not set the override: set=%v value=%d",
				form, req.MaxCheckpointsSet, req.MaxCheckpoints)
		}
		for _, a := range req.ExtraArgs {
			if strings.HasPrefix(a, "--ctx-checkpoints") || strings.HasPrefix(a, "-ctxcp") {
				t.Errorf("%v leaked the flag into ExtraArgs: %q — it would be "+
					"appended to the derived value and emitted twice", form, a)
			}
		}
	}
}

// An explicit 0 is a real decision (disable checkpoints), not "unset", so the
// setter needs its own bool. Without it a 0 override would be indistinguishable
// from the zero value and silently ignored.
func TestCtxCheckpointsZeroIsAnExplicitOverride(t *testing.T) {
	isolateConfig(t)
	req, err := parseLaunchArgs([]string{"model.gguf", "--ctx-checkpoints", "0"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.MaxCheckpointsSet {
		t.Error("--ctx-checkpoints 0 was treated as unset; the user asked to " +
			"disable checkpoints and would silently get the derived cap instead")
	}
	if req.MaxCheckpoints != 0 {
		t.Errorf("value = %d, want 0", req.MaxCheckpoints)
	}
}
