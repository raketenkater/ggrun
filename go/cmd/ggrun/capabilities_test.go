package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestCapabilityReportNamesTheCostOfEveryGap is the point of the report. A
// checklist of missing binaries is not actionable; the reason ggrun ran for
// months without a placement oracle is that nothing said what its absence cost.
func TestCapabilityReportNamesTheCostOfEveryGap(t *testing.T) {
	caps := []capability{
		{Name: "placement oracle [glm]", OK: false, Detail: "not built",
			Cost: "launch preflight is skipped; every candidate costs a real model load"},
		{Name: "systemd-run", OK: true, Detail: "/usr/bin/systemd-run"},
	}
	var buf bytes.Buffer
	printCapabilities(&buf, caps)
	out := buf.String()

	if !strings.Contains(out, "MISSING") {
		t.Fatalf("a degraded capability must be marked, got:\n%s", out)
	}
	if !strings.Contains(out, "every candidate costs a real model load") {
		t.Fatalf("the cost of a gap must be stated, got:\n%s", out)
	}
	if !strings.Contains(out, "1 capability") {
		t.Fatalf("the report must count degraded capabilities, got:\n%s", out)
	}
}

// TestCapabilityReportStaysQuietWhenHealthy keeps the report from crying wolf:
// a fully-provisioned rig must produce no warnings to scroll past.
func TestCapabilityReportStaysQuietWhenHealthy(t *testing.T) {
	caps := []capability{
		{Name: "systemd-run", OK: true, Detail: "/usr/bin/systemd-run"},
		{Name: "python3", OK: true, Detail: "/usr/bin/python3"},
	}
	var buf bytes.Buffer
	printCapabilities(&buf, caps)
	out := buf.String()
	if strings.Contains(out, "MISSING") || strings.Contains(out, "degraded:") {
		t.Fatalf("a healthy rig must report nothing degraded, got:\n%s", out)
	}
	if !strings.Contains(out, "Nothing is degraded") {
		t.Fatalf("expected an explicit all-clear, got:\n%s", out)
	}
}

// TestCapabilitySummaryLineIsEmptyWhenHealthy guards the launch banner: it must
// add a line only when there is something to say.
func TestCapabilitySummaryLineIsEmptyWhenHealthy(t *testing.T) {
	healthy := []capability{{Name: "python3", OK: true}}
	if got := capabilitySummaryLine(healthy); got != "" {
		t.Fatalf("healthy rig must add no banner line, got %q", got)
	}
	degraded := []capability{{Name: "python3", OK: true}, {Name: "nvcc", OK: false}}
	got := capabilitySummaryLine(degraded)
	if !strings.Contains(got, "nvcc") {
		t.Fatalf("banner must name what is missing, got %q", got)
	}
	if strings.Contains(got, "python3") {
		t.Fatalf("banner must not list healthy capabilities, got %q", got)
	}
}
