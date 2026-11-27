package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Invariant: an inventory plan only proceeds when every card's index is the one
// ggrun's own detection would give that set of cards (0..N-1 in list order and,
// when every card has a bus id, in ascending bus-id order). Saved host records
// are keyed by an index-free identity but store per-card values by index, so any
// other numbering would apply them to the wrong cards.
func TestValidateInventoryNumbering(t *testing.T) {
	gpu := func(index int, bus string, vram int) detect.GPU {
		return detect.GPU{Index: index, Name: "card", VRAMTotalMB: vram, PCIBusID: bus}
	}
	const b1, b2, b3 = "00000000:01:00.0", "00000000:02:00.0", "00000000:03:00.0"
	cases := []struct {
		name    string
		gpus    []detect.GPU
		wantErr string // substring; "" means valid
	}{
		{name: "no gpus", gpus: nil},
		{name: "one card", gpus: []detect.GPU{gpu(0, b1, 12288)}},
		{name: "bus order", gpus: []detect.GPU{gpu(0, b1, 12288), gpu(1, b2, 24576), gpu(2, b3, 12288)}},
		{name: "no bus ids", gpus: []detect.GPU{gpu(0, "", 24576), gpu(1, "", 12288)}},
		{name: "some bus ids", gpus: []detect.GPU{gpu(0, b2, 24576), gpu(1, "", 12288)}},
		{name: "duplicate index", gpus: []detect.GPU{gpu(0, b1, 12288), gpu(0, b2, 12288)}, wantErr: "index 0 is given to two GPUs"},
		{name: "negative index", gpus: []detect.GPU{gpu(-1, b1, 12288), gpu(0, b2, 12288)}, wantErr: "negative index -1"},
		{name: "gap", gpus: []detect.GPU{gpu(0, b1, 12288), gpu(2, b3, 12288)}, wantErr: "index 1 is missing"},
		{name: "duplicate bus id", gpus: []detect.GPU{gpu(0, b1, 12288), gpu(1, b1, 24576)}, wantErr: "pci_bus_id " + b1 + " is given to two GPUs"},
		{name: "duplicate bus id without full set", gpus: []detect.GPU{gpu(0, b1, 12288), gpu(1, b1, 24576), gpu(2, "", 8192)}, wantErr: "is given to two GPUs"},
		{name: "indexes swapped against bus order", gpus: []detect.GPU{gpu(0, b2, 12288), gpu(1, b1, 24576)},
			wantErr: "index 0: " + b1},
		{name: "listed out of bus order with matching indexes", gpus: []detect.GPU{gpu(1, b2, 12288), gpu(0, b1, 24576)},
			wantErr: "not numbered the way ggrun numbers"},
		{name: "no bus ids listed out of index order", gpus: []detect.GPU{gpu(1, "", 12288), gpu(0, "", 24576)},
			wantErr: "listed at position 0"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateInventoryNumbering(tc.gpus)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("valid numbering rejected: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

// Every inventory the plan and console goldens use is valid under the rule, so
// the rule cannot change a recorded plan.
func TestGoldenInventoriesHaveDetectNumbering(t *testing.T) {
	inventories := map[string]string{"console/gpu2": goldenPlanInventory(goldenPlanGPU(0, 2048, 16, 224000))}
	for name, body := range goldenPlanInventories {
		inventories[name] = body
	}
	for name, body := range inventories {
		caps, err := detect.LoadCapabilities([]byte(body))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if err := validateInventoryNumbering(caps.GPUs); err != nil {
			t.Errorf("%s: golden inventory rejected: %v", name, err)
		}
	}
}
