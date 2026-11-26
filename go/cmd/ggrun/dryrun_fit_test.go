package main

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

func TestDryRunRefusesDenseOffloadKVThatDoesNotFitItsGPU(t *testing.T) {
	ledger := &placement.ResourceLedger{Devices: []placement.DeviceResourceLedger{
		{GPU: 0, FreeMB: 1587, ContextMB: 4000, SlackMB: -2758},
		{GPU: 1, FreeMB: 3040, ContextMB: 0, SlackMB: -40},
		{GPU: 2, FreeMB: 934, ContextMB: 300, SlackMB: 12},
	}}
	got := dryRunFitRefusal(&placement.Strategy{Type: placement.DenseCPUOffload, ResourceLedger: ledger})
	if !strings.Contains(got, "plan does not fit as estimated") || !strings.Contains(got, "CUDA0") ||
		!strings.Contains(got, "-2758") {
		t.Fatalf("refusal = %q", got)
	}
	if strings.Contains(got, "CUDA1") || strings.Contains(got, "CUDA2") {
		t.Fatalf("refusal names a card whose KV fits or that holds none: %q", got)
	}
}

func TestDryRunKeepsPlansWhoseLedgerFits(t *testing.T) {
	fits := &placement.ResourceLedger{Devices: []placement.DeviceResourceLedger{
		{GPU: 0, ContextMB: 4000, SlackMB: 0},
	}}
	short := &placement.ResourceLedger{Devices: []placement.DeviceResourceLedger{
		{GPU: 0, ContextMB: 4000, SlackMB: -1},
	}}
	for name, strategy := range map[string]*placement.Strategy{
		"nil":             nil,
		"no ledger":       {Type: placement.DenseCPUOffload},
		"fits":            {Type: placement.DenseCPUOffload, ResourceLedger: fits},
		"other plan type": {Type: placement.MoEOffload, ResourceLedger: short},
	} {
		if got := dryRunFitRefusal(strategy); got != "" {
			t.Fatalf("%s: unexpected refusal %q", name, got)
		}
	}
}
