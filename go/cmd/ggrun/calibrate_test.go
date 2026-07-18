package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func calibrateTestSetup(sizeMB int) (*launchRequest, *config.Config, *placement.ModelProfile, *backendInfo, *detect.Capabilities) {
	cfg := config.Defaults()
	cfg.CacheDir = ""
	model := &placement.ModelProfile{
		Path: "model.gguf", Basename: "model", IsMoE: true,
		TotalSizeMB: sizeMB, NumLayers: 60, NumExperts: 128,
	}
	be := &backendInfo{Tag: "ik_llama", Identity: "ik-build-4641", Help: "--reasoning ARG"}
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 32000},
			{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 8000},
		},
		RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 131072},
		CPU: detect.CPUInfo{Cores: 16},
	}
	req := &launchRequest{Port: 8081, Calibrate: calibrateAuto}
	return req, cfg, model, be, caps
}

func TestCalibrationPlanSkipsLargeModelInAuto(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(60 * 1024) // 60 GB MoE
	strategy := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	if got := calibrationPlan(req, cfg, model, be, caps, strategy); got != nil {
		t.Fatalf("auto calibration must skip a 60GB MoE (restart too costly), got %d candidates", len(got))
	}
}

func TestCalibrationPlanForcedOnIgnoresSizeGate(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(60 * 1024)
	req.Calibrate = calibrateOn
	strategy := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	if got := calibrationPlan(req, cfg, model, be, caps, strategy); len(got) < 2 {
		t.Fatalf("forced calibration must run on a big MoE, got %d candidates", len(got))
	}
}

func TestCalibrationPlanSkipsSingleGPU(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(8 * 1024)
	caps.GPUs = caps.GPUs[:1] // one GPU only
	strategy := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	if got := calibrationPlan(req, cfg, model, be, caps, strategy); got != nil {
		t.Fatalf("single GPU offers no placement alternatives, got %d", len(got))
	}
}

func TestCalibrationPlanSkipsWhenDecisionCached(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(8 * 1024)
	cfg.CacheDir = t.TempDir()
	strategy := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	// Seed a prior decision for this exact scope.
	scopeKey := calibrationScopeKey(req, model, be, caps)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: "kv-alternate", DefaultTPS: 20, WinnerTPS: 24,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}
	if got := calibrationPlan(req, cfg, model, be, caps, strategy); got != nil {
		t.Fatalf("a cached decision must suppress re-calibration, got %d candidates", len(got))
	}
}

func TestCalibrationPlanSmallMoEOffered(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(8 * 1024)
	strategy := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	got := calibrationPlan(req, cfg, model, be, caps, strategy)
	if len(got) < 2 {
		t.Fatalf("a small multi-GPU MoE with no cached decision should calibrate, got %d", len(got))
	}
	if got[0].Name != "default" {
		t.Fatalf("candidate 0 must be the default, got %q", got[0].Name)
	}
}

func TestApplyCalibrationDecisionRestoresWinner(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(8 * 1024)
	cfg.CacheDir = t.TempDir()
	base := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	scopeKey := calibrationScopeKey(req, model, be, caps)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: "kv-alternate", DefaultTPS: 20, WinnerTPS: 24.5,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}
	got := applyCalibrationDecision(req, cfg, model, be, caps, base)
	if got == base {
		t.Fatal("cached kv-alternate winner was not applied")
	}
	if got.KVPlacement != "gpu" {
		t.Fatalf("kv-alternate should restore KV=gpu, got %q", got.KVPlacement)
	}
	// The base estimate must be untouched for the next caller.
	if base.KVPlacement != "cpu" {
		t.Fatalf("base strategy mutated to %q", base.KVPlacement)
	}
}

func TestApplyCalibrationDecisionIgnoresDefaultWinner(t *testing.T) {
	req, cfg, model, be, caps := calibrateTestSetup(8 * 1024)
	cfg.CacheDir = t.TempDir()
	base := &placement.Strategy{Type: placement.MoEOffload, KVPlacement: "cpu", NCPUMoE: 40}
	scopeKey := calibrationScopeKey(req, model, be, caps)
	if _, err := placement.SaveCalibrationDecision(cfg.CacheDir, placement.CalibrationDecision{
		ScopeKey: scopeKey, Winner: "default", DefaultTPS: 22, WinnerTPS: 22,
	}); err != nil {
		t.Fatalf("seed decision: %v", err)
	}
	got := applyCalibrationDecision(req, cfg, model, be, caps, base)
	if got != base {
		t.Fatal("a default winner must leave the estimated strategy in place")
	}
}

func TestParseCalibrateMode(t *testing.T) {
	for _, ok := range []string{"auto", "on", "off", "AUTO", " On "} {
		if _, err := parseCalibrateMode(ok); err != nil {
			t.Fatalf("parseCalibrateMode(%q): %v", ok, err)
		}
	}
	if _, err := parseCalibrateMode("yes"); err == nil {
		t.Fatal("parseCalibrateMode must reject unknown modes")
	}
}
