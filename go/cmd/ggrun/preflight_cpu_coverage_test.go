package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/memprobe"
)

// A CPU-only launch hides every device, so its contained probe sees no device
// allocations. That is complete evidence for a CPU plan; with a device visible
// the same summary still is not.
func TestCPUOnlyProbeCoverageNeedsNoDeviceEvents(t *testing.T) {
	hostOnly := memprobe.Summary{Loaded: true}
	if !guardedProbeCoverageComplete(hostOnly, true, true, false) {
		t.Fatal("a CPU-only probe without device events was treated as incomplete")
	}
	if guardedProbeCoverageComplete(hostOnly, true, true, true) {
		t.Fatal("a GPU probe without device events was treated as complete")
	}
	for _, tc := range []struct {
		name                   string
		summary                memprobe.Summary
		guard, cgroup, visible bool
	}{
		{"guard not loaded", memprobe.Summary{DeviceEvents: true}, true, true, true},
		{"no guard library", memprobe.Summary{Loaded: true}, false, true, false},
		{"no cgroup stats", memprobe.Summary{Loaded: true}, true, false, false},
	} {
		if guardedProbeCoverageComplete(tc.summary, tc.guard, tc.cgroup, tc.visible) {
			t.Fatalf("%s: incomplete coverage accepted", tc.name)
		}
	}
	if !guardedProbeCoverageComplete(memprobe.Summary{Loaded: true, DeviceEvents: true}, true, true, true) {
		t.Fatal("complete GPU coverage rejected")
	}
}

// Host-only evidence from a complete CPU-only probe is reused on relaunch;
// evidence that measured nothing at all is not.
func TestHostOnlyMemoryEvidenceIsReused(t *testing.T) {
	dir := t.TempDir()
	coverage := memprobe.Coverage{GuardLoaded: true, CgroupV2: true, Complete: true}
	host := memprobe.HostMemory{ModelBytes: 3 << 30, ContextBytes: 4 << 30, ComputeBytes: 256 << 20}
	if _, err := memprobe.Save(dir, memprobe.Plan{Key: "cpu", Evidence: memprobe.EvidenceGuardedAllocated,
		BackendIdentity: "ik", Outcome: "fit", Coverage: coverage, Host: host}); err != nil {
		t.Fatal(err)
	}
	got, ok := loadMemoryEvidence(dir, "cpu")
	if !ok || len(got.Devices) != 1 || got.Devices[0].Name != "Host" || got.Devices[0].ModelMB != 3072 {
		t.Fatalf("host-only evidence not reused: ok=%v %#v", ok, got)
	}
	if _, err := memprobe.Save(dir, memprobe.Plan{Key: "empty", Evidence: memprobe.EvidenceGuardedAllocated,
		BackendIdentity: "ik", Outcome: "fit", Coverage: coverage}); err != nil {
		t.Fatal(err)
	}
	if _, ok := loadMemoryEvidence(dir, "empty"); ok {
		t.Fatal("evidence that measured nothing was reused")
	}
}
