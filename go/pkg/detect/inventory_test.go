package detect

import (
	"strings"
	"testing"
)

// An inventory is a planning input that lets a plan be computed for a machine
// other than this one, or for this one while another process holds its VRAM.
// Because a wrong inventory produces a plausible-looking plan for hardware that
// does not exist, the decoder is strict: a half-populated capture must fail
// loudly rather than plan against zero VRAM.

func TestCapabilitiesRoundTripsThroughJSON(t *testing.T) {
	original := &Capabilities{
		OS:   "linux",
		Arch: "amd64",
		GPUs: []GPU{
			{Index: 0, Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12282, Driver: "580",
				PCIGen: 3, PCILanes: 16, BandwidthMBps: 12192, MemBandwidthMBps: 504048},
			{Index: 1, Name: "NVIDIA GeForce RTX 3060", VRAMTotalMB: 12288, Driver: "580",
				PCIGen: 3, PCILanes: 8, BandwidthMBps: 6269, MemBandwidthMBps: 360048},
		},
		RAM: RAMInfo{TotalMB: 217096, FreeMB: 200000},
		CPU: CPUInfo{Cores: 14, Threads: 28},
	}

	data, err := original.JSON()
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := LoadCapabilities(data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got.GPUs) != 2 {
		t.Fatalf("GPUs = %d, want 2", len(got.GPUs))
	}
	if got.GPUs[0].VRAMTotalMB != 12282 || got.GPUs[0].PCILanes != 16 {
		t.Errorf("GPU identity did not survive the round trip: %+v", got.GPUs[0])
	}
	// Bandwidth is part of the inventory even though placement identity no longer
	// keys on it: the planner still orders devices by it.
	if got.GPUs[1].BandwidthMBps != 6269 {
		t.Errorf("bandwidth did not survive the round trip: %d", got.GPUs[1].BandwidthMBps)
	}
	if got.CPU.Cores != 14 || got.RAM.TotalMB != 217096 {
		t.Errorf("host facts did not survive the round trip: %+v %+v", got.CPU, got.RAM)
	}
}

// A GPU with no declared capacity cannot be budgeted. Planning it as zero would
// silently fit nothing and look like a legitimate "does not fit" answer.
func TestInventoryRejectsAGPUWithNoVRAM(t *testing.T) {
	_, err := LoadCapabilities([]byte(`{"gpus":[{"index":0,"name":"GHOST"}]}`))
	if err == nil {
		t.Fatal("a GPU with no vram_total_mb was accepted")
	}
	if !strings.Contains(err.Error(), "vram_total_mb") {
		t.Errorf("error does not name the missing field: %v", err)
	}
}

// A mistyped field must fail rather than be silently dropped, or a hand-written
// inventory would plan against whatever the zero value happens to mean.
func TestInventoryRejectsUnknownFields(t *testing.T) {
	_, err := LoadCapabilities([]byte(`{"gpus":[{"index":0,"name":"X","VRAMTotalMb":12282}]}`))
	if err == nil {
		t.Fatal("a misspelled vram field was silently ignored")
	}
}

func TestInventoryRejectsMalformedInput(t *testing.T) {
	for _, in := range []string{"", "not json", "[1,2,3]", `{"gpus":"two"}`} {
		if _, err := LoadCapabilities([]byte(in)); err == nil {
			t.Errorf("malformed inventory %q was accepted", in)
		}
	}
}

// Missing optional host fields are unknown, not invalid: a GPU-only capture is
// still a usable planning input.
func TestInventoryAcceptsMissingOptionalHostFields(t *testing.T) {
	got, err := LoadCapabilities([]byte(`{"gpus":[{"index":0,"name":"X","vram_total_mb":8192}]}`))
	if err != nil {
		t.Fatalf("a GPU-only inventory was rejected: %v", err)
	}
	if len(got.GPUs) != 1 || got.GPUs[0].VRAMTotalMB != 8192 {
		t.Fatalf("unexpected decode: %+v", got)
	}
}

// A CPU-only inventory is meaningful (dense models can be planned for RAM), so
// it must not be rejected merely for having no devices.
func TestInventoryAcceptsNoGPUs(t *testing.T) {
	if _, err := LoadCapabilities([]byte(`{"ram":{"total_mb":64000,"free_mb":60000},"cpu":{"cores":8}}`)); err != nil {
		t.Fatalf("a CPU-only inventory was rejected: %v", err)
	}
}
