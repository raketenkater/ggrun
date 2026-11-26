package main

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestExpertPackAdmissionMissKeepsRaisedBaselineServing(t *testing.T) {
	h := newBudgetHarness(t, time.Minute, time.Minute)
	h.strategy.Type = placement.MoEOffload
	h.strategy.ContextSize = 143360
	h.strategy.BatchSize, h.strategy.UBatchSize = 2048, 1024
	h.strategy.NCPUMoE = 27
	h.strategy.OTString = "fixture-experts=CUDA0"
	h.serverArgs = buildLaunchServerArgs(h.req, h.cfg, h.be, h.caps, h.model, h.strategy)
	h.recovery.observeProductionLoad(h.serverArgs, time.Minute)
	pack := *h.strategy
	pack.NCPUMoE--
	h.candidates = []placement.CalibrationCandidate{{Name: "default", Strategy: h.strategy}, {Name: "expert-pack", Strategy: &pack}}
	h.fx.prescreen = func(*launchRequest, *config.Config, *placement.ModelProfile, *backendInfo, *detect.Capabilities, *detect.Capabilities, *placement.Strategy, []string) (bool, string, string) {
		h.events = append(h.events, "prescreen")
		return true, "memory", "candidate exceeds guard by 66 MiB"
	}
	p, strategy, args, decision := h.run()
	if p != h.baseline || strategy != h.strategy || !reflect.DeepEqual(args, h.serverArgs) {
		t.Fatal("rejected optional packing displaced the healthy serving baseline")
	}
	if len(h.starts) != 0 || len(h.restarts) != 0 || strings.Contains(strings.Join(h.events, ";"), "stop ") {
		t.Fatalf("admission miss stopped or reloaded the server: %v", h.events)
	}
	if strategy.UBatchSize != 1024 || strategy.ContextSize != 143360 || strategy.NCPUMoE != 27 {
		t.Fatal("recovery changed the admitted baseline")
	}
	if decision == nil || decision.FinalistOutcome != "unavailable" || decision.Winner != "default" {
		t.Fatalf("failed proposal claimed a measured win: %+v", decision)
	}
}
