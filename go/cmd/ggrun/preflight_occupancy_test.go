package main

import "testing"

// The 2026-09-08 GLM 5.3 Flash control serve, as the backend reported it.
// Capacities are the rig's real cards: 4070 12282, 3090 Ti 24564, 3060 12288.
//
// Committed: 41357 MiB of 49134 (84%). Free: 1758 / 2889 / 3130 MiB, none of
// which can hold one ~4.5 GB routed-expert layer.
func glmServeDevices() []preflightDevice {
	return []preflightDevice{
		{Name: "CUDA0", ModelMB: 5089, ContextMB: 580, ComputeMB: 4855},
		{Name: "CUDA1", ModelMB: 14242, ContextMB: 2318, ComputeMB: 5114},
		{Name: "CUDA2", ModelMB: 4013, ContextMB: 290, ComputeMB: 4855},
	}
}

// TestAllocationLedgerCountsComputeBuffers pins the wiring defect itself.
//
// preflightDevice keeps ComputeMB and UnaccountedMB disjoint, so a ledger that
// copies only UnaccountedMB silently loses the compute buffer. On this serve
// that was 14824 MiB -- 30% of the rig -- and it made occupancy report 54%
// where the devices had actually committed 84%.
func TestAllocationLedgerCountsComputeBuffers(t *testing.T) {
	alloc, computeByGPU := allocationLedgerFromDevices(glmServeDevices())

	for _, want := range []struct {
		idx     int
		compute int
	}{{0, 4855}, {1, 5114}, {2, 4855}} {
		if got := computeByGPU[want.idx]; got != want.compute {
			t.Errorf("CUDA%d compute buffer: got %d MiB, want %d", want.idx, got, want.compute)
		}
		// The ledger term placement reads must carry the compute buffer. A
		// device with no separate allocator residue must still report it.
		if got := alloc.UnaccountedByGPU[want.idx]; got != want.compute {
			t.Errorf("CUDA%d ledger residue: got %d MiB, want %d (compute buffer dropped)",
				want.idx, got, want.compute)
		}
	}

	committed := 0
	for idx := range alloc.ModelByGPU {
		committed += alloc.ModelByGPU[idx] + alloc.ContextByGPU[idx] + alloc.UnaccountedByGPU[idx]
	}
	if want := 41356; committed != want {
		t.Fatalf("total committed VRAM: got %d MiB, want %d -- the ledger must account for every "+
			"byte the backend allocated, or the packer offers seats that do not exist", committed, want)
	}
}

// TestAllocationLedgerKeepsResidueDisjoint guards the other direction: a device
// that reports both a compute buffer and an unexplained allocator peak must
// carry the sum, not one or the other.
func TestAllocationLedgerKeepsResidueDisjoint(t *testing.T) {
	alloc, computeByGPU := allocationLedgerFromDevices([]preflightDevice{
		{Name: "CUDA0", ModelMB: 1000, ContextMB: 100, ComputeMB: 400, UnaccountedMB: 250},
		{Name: "Host", ModelMB: 90000, ContextMB: 0, ComputeMB: 12, UnaccountedMB: 34},
	})
	if got, want := alloc.UnaccountedByGPU[0], 650; got != want {
		t.Errorf("CUDA0 residue: got %d MiB, want %d (compute+unaccounted)", got, want)
	}
	// computeByGPU feeds RecordMeasuredComputeBuffers and must stay the compute
	// buffer alone, never the combined residue.
	if got, want := computeByGPU[0], 400; got != want {
		t.Errorf("CUDA0 recorded compute buffer: got %d MiB, want %d", got, want)
	}
	if got, want := alloc.UnaccountedHostMB, 46; got != want {
		t.Errorf("host residue: got %d MiB, want %d (compute+unaccounted)", got, want)
	}
	if got, want := alloc.ModelHostMB, 90000; got != want {
		t.Errorf("host model: got %d MiB, want %d", got, want)
	}
}
