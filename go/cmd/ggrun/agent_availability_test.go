package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestNoCompanionPermissionDefaultPreservesExplicitChoices(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "off")
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "")
	if got := claudeCodePermissionArgs(nil); len(got) != 2 || got[1] != "manual" {
		t.Fatalf("no reviewer must use normal prompts rather than timing out self-classification: %v", got)
	}
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "auto")
	if got := claudeCodePermissionArgs(nil); len(got) != 2 || got[1] != "auto" {
		t.Fatalf("explicit environment choice was lost: %v", got)
	}
	if got := claudeCodePermissionArgs([]string{"--permission-mode=auto"}); got != nil {
		t.Fatalf("explicit CLI mode was overridden: %v", got)
	}
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "inherit")
	if got := claudeCodePermissionArgs(nil); got != nil {
		t.Fatalf("inherit was overridden: %v", got)
	}
}

func TestPermissionDefaultUsesSeatedReviewerNotOnlyEnvironment(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "")
	// The request flag and launch-time companion rejection can leave the env
	// untouched. The actual local-client boundary must still select prompts.
	if got := claudeCodePermissionArgs(nil, false); len(got) != 2 || got[1] != "manual" {
		t.Fatalf("absent seated reviewer selected self-classification: %v", got)
	}
	if got := claudeCodePermissionArgs(nil, true); len(got) != 2 || got[1] != "auto" {
		t.Fatalf("seated reviewer lost Auto mode: %v", got)
	}
	t.Setenv("GGRUN_CLAUDE_PERMISSION_MODE", "auto")
	if got := claudeCodePermissionArgs(nil, false); len(got) != 2 || got[1] != "auto" {
		t.Fatalf("runtime availability overrode the operator's Auto choice: %v", got)
	}
	if got := claudeCodePermissionArgs([]string{"--permission-mode=auto"}, false); got != nil {
		t.Fatalf("runtime availability overrode explicit CLI permission mode: %v", got)
	}
}

func TestMeasuredScopeKeepsFutureCacheAndGrowthAllowance(t *testing.T) {
	// Portable sizing invariant: a canary that exceeds a static base must not
	// consume the cache/growth allowance of later agent turns. Applies to CPU,
	// dense offload and MoE, independent of GPU count or a particular model.
	for _, kind := range []placement.StrategyType{placement.CPUOnly, placement.DenseCPUOffload, placement.MoEOffload} {
		s := &placement.Strategy{Type: kind, PlannedHostFootprintMB: 6000, CRAM: 8000}
		const measured, growth, ceiling = 15000, 4096, 64000
		floor := measuredFootprintPlannedFloor(s, measured, ceiling)
		got := measuredFootprintCgroupMaxMB(measured, growth, floor, ceiling)
		if got < measured+s.CRAM+growth {
			t.Fatalf("%v: cache consumed future growth allowance: got %d", kind, got)
		}
		if limited := measuredFootprintCgroupMaxMB(measured, growth, floor, 20000); limited != 20000 {
			t.Fatalf("%v: crossed the resource ceiling: %d", kind, limited)
		}
	}
}
