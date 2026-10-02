package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// fakeUBatchOracle answers the no-alloc oracle per -ub value.
func fakeUBatchOracle(t *testing.T, rows map[string]string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell oracle fixture")
	}
	script := "#!/bin/sh\nub=\nctx=\nprev=\nfor a in \"$@\"; do [ \"$prev\" = -ub ] && ub=$a; { [ \"$prev\" = -c ] || [ \"$prev\" = --ctx-size ]; } && ctx=$a; prev=$a; done\ncase $ub:$ctx in\n"
	for key, out := range rows {
		if !strings.Contains(key, ":") {
			key += ":*" // any context
		}
		script += key + ") printf '" + out + "' ;;\n"
	}
	script += "*) echo 'no such rung' >&2; exit 3 ;;\nesac\n"
	fit := filepath.Join(t.TempDir(), "llama-fit-params")
	if err := os.WriteFile(fit, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return fit
}

func ubatchRaiseFixture(t *testing.T) (*detect.Capabilities, *placement.ModelProfile, *placement.Strategy, []string, []preflightDevice, *configForPreflight) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 12000}, {Index: 1, VRAMTotalMB: 24000},
	}}
	model := &placement.ModelProfile{Path: filepath.Join(t.TempDir(), "m.gguf"), NumLayers: 48, IsMoE: true}
	s := &placement.Strategy{Type: placement.MoEOffload, ContextSize: 65536, BatchSize: 2048, UBatchSize: 64,
		Parallel: 1, KVQuality: "q8_0", KVPlacement: "gpu", NCPUMoE: 47}
	args := []string{"llama-server", "-m", model.Path, "-c", "65536", "-b", "2048", "-ub", "64"}
	current := []preflightDevice{{Name: "CUDA0", ModelMB: 4000, ContextMB: 2000, ComputeMB: 2100},
		{Name: "CUDA1", ModelMB: 10000, ContextMB: 5000, ComputeMB: 1800}}
	return caps, model, s, args, current, &configForPreflight{CacheDir: t.TempDir()}
}

func TestStagedPrefillUBatchRaisePicksLargestAdmittedRung(t *testing.T) {
	caps, model, s, args, current, cfg := ubatchRaiseFixture(t)
	fit := fakeUBatchOracle(t, map[string]string{
		// Does not fit: 4000+2000+7000 > 12000.
		"512": `CUDA0 4000 2000 7000\nCUDA1 10000 5100 2400\n`,
		"256": `CUDA0 4000 2000 2200\nCUDA1 10000 5020 2100\n`,
		"128": `CUDA0 4000 2000 2150\nCUDA1 10000 5000 1900\n`,
	})
	got, ctx := stagedPrefillUBatchRaise(fit, args, current, cfg, caps, model, s, "llama", []int{512, 256, 128}, false)
	if got != 256 || ctx != s.ContextSize {
		t.Fatalf("raised to %d, want the largest admitted rung 256", got)
	}
	// The measured rung becomes compute evidence for later planning; the
	// rejected one stays unknown rather than borrowing it.
	opts := placement.Options{CacheDir: cfg.CacheDir, BackendCacheTag: "llama", RequireMeasuredBuffers: true}
	s.TensorSplit = []float64{.3, .7}
	for ub, recorded := range map[int]bool{256: true, 512: false} {
		ledger := placement.BuildResourceLedger(caps, model, placement.WithUBatch(s, model, ub), opts)
		if (len(ledger.MissingComputeGPUs) == 0) != recorded {
			t.Fatalf("ubatch %d compute recorded=%v, want %v: %+v", ub, !recorded, recorded, ledger.MissingComputeGPUs)
		}
	}
}

func TestStagedPrefillUBatchRaiseRejectsMovedWeightsAndFailures(t *testing.T) {
	caps, model, s, args, current, cfg := ubatchRaiseFixture(t)
	fit := fakeUBatchOracle(t, map[string]string{
		// Fits, but the backend placed weights differently: not the same plan.
		"512": `CUDA0 5000 2000 2000\nCUDA1 9000 5100 2400\n`,
		// 256 is absent: the oracle fails for it.
		"128": `CUDA0 4000 2000 2150\nCUDA1 10000 5000 1900\n`,
	})
	if got, _ := stagedPrefillUBatchRaise(fit, args, current, cfg, caps, model, s, "llama", []int{512, 256, 128}, true); got != 128 {
		t.Fatalf("raised to %d, want 128 (512 moved weights, 256 failed)", got)
	}
	none := fakeUBatchOracle(t, map[string]string{"512": `CUDA0 4000 2000 9000\n`})
	if got, _ := stagedPrefillUBatchRaise(none, args, current, cfg, caps, model, s, "llama", []int{512}, false); got != 0 {
		t.Fatalf("a deficit rung was admitted: %d", got)
	}
	if got, _ := stagedPrefillUBatchRaise("", args, current, cfg, caps, model, s, "llama", []int{512}, true); got != 0 {
		t.Fatal("raised without a backend oracle")
	}
}

// Claude Code mode is the main agent path. Its recorded launch arguments must
// reach the same staged-prefill raise as plain serving, and its explicit
// choices (ubatch, extra slots) must still block it.
func TestClaudeCodeLaunchGetsStagedPrefillRaise(t *testing.T) {
	model := &placement.ModelProfile{Path: filepath.Join(t.TempDir(), "moe.gguf"), NumLayers: 48, IsMoE: true}
	be := &backendInfo{Tag: "llama", Dialect: "llama", Identity: "llama-server-test",
		Help: "--op-offload, --no-op-offload  whether to offload host tensor operations to device (default: true)"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 12000, BandwidthMBps: 15760}, {Index: 1, VRAMTotalMB: 24000, BandwidthMBps: 15760}}}
	plan := &placement.Strategy{Type: placement.MoEOffload, ContextSize: 723968, BatchSize: 2048, UBatchSize: 64,
		Parallel: 1, NCPUMoE: 50, KVPlacement: "gpu", KVQuality: "q8_0"}
	claudeArgs := []string{model.Path, "--port", "8081", "--ctx-size", "fit", "--kv-placement", "auto",
		"--parallel", "1", "--support-expert", "auto", "--claude-code"}
	cases := []struct {
		name  string
		extra []string
		want  bool
	}{
		{"recorded claude-code launch", nil, true},
		{"explicit ubatch", []string{"-ub", "64"}, false},
		{"two slots", []string{"--parallel", "2"}, false},
	}
	for _, c := range cases {
		req, err := parseLaunchArgs(append(append([]string(nil), claudeArgs...), c.extra...))
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if !req.ClaudeCode {
			t.Fatalf("%s: --claude-code not parsed", c.name)
		}
		s := placement.WithUBatch(plan, model, plan.UBatchSize)
		s.Parallel = max(1, req.Parallel)
		claudeCodeSlotAdjust(s, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
		got := placement.StagedPrefillUBatchRaiseCandidates(caps, placementOptionsFromRequest(req, model, be, t.TempDir()), s)
		if (len(got) > 0) != c.want {
			t.Fatalf("%s: raise candidates %v, want raise=%v", c.name, got, c.want)
		}
	}
}

// With an automatic context the microbatch is reserved first and context is
// fitted to what remains, never below minRaiseContext. The fixture is
// GLM-5.3-Flash-shaped: compute grows with ubatch x context, so its context
// rows alone (200 MiB at 262k) cannot pay the deficit at any context, yet the
// oracle admits ubatch 512 below 262k. An explicit context never shrinks.
func TestStagedPrefillUBatchRaiseReservesMicrobatchBeforeAutoContext(t *testing.T) {
	caps, model, s, _, current, cfg := ubatchRaiseFixture(t)
	s.ContextSize = 262144
	// Unmeasured cards reserve the default CUDA overhead; keep 12000 MiB usable.
	for i := range caps.GPUs {
		caps.GPUs[i].VRAMTotalMB += placement.UnmeasuredCUDAOverheadMB
	}
	args := []string{"llama-server", "-m", model.Path, "-c", "262144", "-b", "2048", "-ub", "64"}
	fit := fakeUBatchOracle(t, map[string]string{
		// Short on CUDA0 at the planned context.
		"512:262144": `CUDA0 4000 200 9000\nCUDA1 10000 5100 2400\n`,
		// Fits at the floor. The line through both answers crosses 10,976 MiB
		// (12,000 less the unmeasured-growth reserve) at 198,773 tokens,
		// 198,656 on the granule.
		"512:131072": `CUDA0 4000 100 4500\nCUDA1 10000 2550 1200\n`,
		"512:198656": `CUDA0 4000 152 6820\nCUDA1 10000 3865 1819\n`,
		"256":        `CUDA0 4000 2000 2200\nCUDA1 10000 5020 2100\n`,
	})
	if ub, ctx := stagedPrefillUBatchRaise(fit, args, current, cfg, caps, model, s, "llama", []int{512, 256}, true); ub != 512 || ctx != 198656 {
		t.Fatalf("auto context: ubatch %d at %d, want 512 at 198656", ub, ctx)
	}
	if ub, ctx := stagedPrefillUBatchRaise(fit, args, current, cfg, caps, model, s, "llama", []int{512, 256}, false); ub != 256 || ctx != 262144 {
		t.Fatalf("explicit context shrank: ubatch %d at %d", ub, ctx)
	}
	// Short even at minRaiseContext: fall back to a smaller rung.
	floor := fakeUBatchOracle(t, map[string]string{
		"512:262144": `CUDA0 4000 200 9000\nCUDA1 10000 5100 2400\n`,
		"512:131072": `CUDA0 4000 100 7000\nCUDA1 10000 2550 1200\n`,
		"256":        `CUDA0 4000 2000 2200\nCUDA1 10000 5020 2100\n`,
	})
	if ub, _ := stagedPrefillUBatchRaise(floor, args, current, cfg, caps, model, s, "llama", []int{512, 256}, true); ub != 256 {
		t.Fatalf("ubatch %d taken below the minimum automatic context", ub)
	}
}

// A raised rung has never run, so the oracle's exact boundary leaves nothing
// for its runtime growth. GLM-5.3-Flash at ubatch 512 fitted there aborted in
// warmup. Unmeasured growth is reserved; a measured value replaces it.
func TestStagedPrefillUBatchRaiseReservesUnmeasuredRuntimeGrowth(t *testing.T) {
	caps, model, s, _, current, cfg := ubatchRaiseFixture(t)
	for i := range caps.GPUs {
		caps.GPUs[i].VRAMTotalMB += placement.UnmeasuredCUDAOverheadMB
	}
	s.ContextSize = 262144
	args := []string{"llama-server", "-m", model.Path, "-c", "262144", "-b", "2048", "-ub", "64"}
	// 300 MiB free on CUDA0 at the planned context: inside the reserve.
	fit := fakeUBatchOracle(t, map[string]string{
		"512:262144": `CUDA0 4000 2000 5700\nCUDA1 10000 5100 2400\n`,
		"512:131072": `CUDA0 4000 1000 5700\nCUDA1 10000 2550 2400\n`,
	})
	if ub, ctx := stagedPrefillUBatchRaise(fit, args, current, cfg, caps, model, s, "llama", []int{512}, false); ub != 0 {
		t.Fatalf("ubatch 512 admitted at %d with 300 MiB for unmeasured growth", ctx)
	}
	if err := placement.RecordRuntimeGraphGrowth(cfg.CacheDir, model, 262144, 512, s.KVQuality, s.KVPlacement, "llama",
		caps.GPUs, s.Parallel, map[int]int{0: 200, 1: 200}); err != nil {
		t.Fatal(err)
	}
	if ub, ctx := stagedPrefillUBatchRaise(fit, args, current, cfg, caps, model, s, "llama", []int{512}, false); ub != 512 || ctx != 262144 {
		t.Fatalf("measured 200 MiB growth: ubatch %d at %d, want 512 at 262144", ub, ctx)
	}
}

// A relaunch replays the raised plan. A larger rung was rejected because it is
// short even at minRaiseContext, which does not depend on the context a launch
// starts from, so the relaunch keeps the same argv.
func TestStagedPrefillUBatchRaiseIsStableAcrossRelaunches(t *testing.T) {
	caps, model, s, _, current, cfg := ubatchRaiseFixture(t)
	for i := range caps.GPUs {
		caps.GPUs[i].VRAMTotalMB += placement.UnmeasuredCUDAOverheadMB
	}
	fit := fakeUBatchOracle(t, map[string]string{
		"1024:262144": `CUDA0 4000 200 15000\nCUDA1 10000 5100 2400\n`,
		"1024:198656": `CUDA0 4000 152 13640\nCUDA1 10000 3865 2400\n`,
		"1024:131072": `CUDA0 4000 100 8100\nCUDA1 10000 2550 1200\n`,
		"512:262144":  `CUDA0 4000 200 9000\nCUDA1 10000 5100 2400\n`,
		"512:131072":  `CUDA0 4000 100 4500\nCUDA1 10000 2550 1200\n`,
		"512:198656":  `CUDA0 4000 152 6820\nCUDA1 10000 3865 1819\n`,
	})
	s.ContextSize = 262144
	args := []string{"llama-server", "-m", model.Path, "-c", "262144", "-b", "2048", "-ub", "64"}
	ub, ctx := stagedPrefillUBatchRaise(fit, args, current, cfg, caps, model, s, "llama", []int{1024, 512}, true)
	if ub != 512 || ctx != 198656 {
		t.Fatalf("first launch: ubatch %d at %d, want 512 at 198656", ub, ctx)
	}
	s.ContextSize, s.UBatchSize = ctx, ub
	args = []string{"llama-server", "-m", model.Path, "-c", "198656", "-b", "2048", "-ub", "512"}
	if again, _ := stagedPrefillUBatchRaise(fit, args, current, cfg, caps, model, s, "llama", []int{1024}, true); again != 0 {
		t.Fatalf("relaunch raised again to %d", again)
	}
}

// Placement and microbatch compete for VRAM. Among placements admitted in one
// launch the cheaper staged-prefill plus decode plan wins, even when it keeps
// more experts on the CPU (GLM-5.3-Flash: 43 CPU layers at ubatch 128 beat 41
// at ubatch 64). Unknown costs leave the current plan alone.
func TestChooseStagedPrefillPlanAcrossAdmittedPlacements(t *testing.T) {
	caps, _, _, _, current, cfg := ubatchRaiseFixture(t)
	for i := range caps.GPUs {
		caps.GPUs[i].VRAMTotalMB += placement.UnmeasuredCUDAOverheadMB
		caps.GPUs[i].BandwidthMBps = 15760
	}
	caps.HostMemoryBandwidthMBps = 80000
	model := &placement.ModelProfile{Path: filepath.Join(t.TempDir(), "glm.gguf"), NumLayers: 46, LeadingDense: 3,
		IsMoE: true, NumExperts: 288, ExpertUsedCount: 8, ExpertBytes: 130000 << 20}
	plan := func(ncpu int, cuda1Model string) admittedPlan {
		s := &placement.Strategy{Type: placement.MoEOffload, ContextSize: 65536, BatchSize: 2048, UBatchSize: 64,
			Parallel: 1, NCPUMoE: ncpu, KVQuality: "q8_0", KVPlacement: "gpu"}
		args := []string{"llama-server", "-m", model.Path, "-c", "65536", "-ub", "64", "--n-cpu-moe", strconv.Itoa(ncpu)}
		devs := []preflightDevice{{Name: "CUDA0", ModelMB: 4000, ContextMB: 2000, ComputeMB: 2100},
			{Name: "CUDA1", ModelMB: map[string]int{"a": 10000, "b": 16000}[cuda1Model], ContextMB: 5000, ComputeMB: 1800}}
		return admittedPlan{s, args, devs}
	}
	fit := fakeUBatchOracle(t, map[string]string{
		// Plan with fewer GPU experts has room for ubatch 128 ...
		"128": `CUDA0 4000 2000 3000\nCUDA1 10000 5000 3000\n`,
	})
	plans := []admittedPlan{plan(43, "a"), plan(41, "b")}
	opts := placement.Options{BackendTag: "llama", BackendHelp: "--op-offload, --no-op-offload"}
	tag := func(*placement.Strategy) string { return "llama" }
	got, ub, _ := chooseStagedPrefillPlan(fit, plans, cfg, caps, model, opts, tag, false)
	if got != 0 || ub != 128 {
		t.Fatalf("chose plan %d at ubatch %d; want the 43-CPU-layer plan raised to 128", got, ub)
	}
	unknown := opts
	unknown.BackendTag = "ik_llama"
	if got, ub, _ := chooseStagedPrefillPlan(fit, plans, cfg, caps, model, unknown, tag, false); got != 1 || ub != 0 {
		t.Fatalf("unknown policy switched plans: plan %d ubatch %d", got, ub)
	}
	_ = current
}
