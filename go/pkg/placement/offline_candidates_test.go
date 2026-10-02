package placement

import (
	"fmt"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// TestOfflinePrefillCandidateComparison recomputes complete candidates derived
// from the recorded 2026-09-29 serving baseline (723,968 context, q8 KV on GPU,
// parallel 1, batch 2048, split .22/.72/.06, every routed expert in host
// memory) and pins what the offline model may and may not claim about them.
// Only ubatch 64 has measured compute rows; everything else needs admission.
// Run with -v to print the comparison used in the handoff.
func TestOfflinePrefillCandidateComparison(t *testing.T) {
	f, model := loadMiMoFixture(t)
	model.IsMoE, model.NumExperts, model.ExpertUsedCount = true, 256, 8
	// Header facts (tools/gguf/parse_gguf.py on the real shards): one leading
	// dense block and 114,192 MiB of routed experts, all kept in host memory.
	model.ExpertBytes, model.LeadingDense = 114192*1048576, 1
	model.TotalSizeMB, model.SizeBytes = 134861, 134861*1048576
	caps := &detect.Capabilities{HostMemoryBandwidthMBps: 80000, GPUs: []detect.GPU{
		{Index: 0, Name: "RTX 4070", MemBandwidthMBps: 504000, BandwidthMBps: 15760},
		{Index: 1, Name: "RTX 3090 Ti", MemBandwidthMBps: 1008000, BandwidthMBps: 15760},
		{Index: 2, Name: "RTX 3060", MemBandwidthMBps: 360000, BandwidthMBps: 7880},
	}}
	mainline := Options{BackendTag: "llama", BackendHelp: mainlineHelp, WorkloadConcurrency: 1}
	noOffload := mainline
	noOffload.HostWeightOffload = "off"
	base := &Strategy{Type: MoEOffload, ContextSize: f.Assumptions.Context, KVType: "q8_0", KVQuality: "q8_0",
		KVPlacement: "gpu", SWAFull: false, Parallel: 1, BatchSize: 2048, UBatchSize: 64, SplitMode: "layer",
		TensorSplit: f.Assumptions.TensorSplit, NCPUMoE: 50, FlashAttention: true}
	// Recorded model rows plus measured ubatch-64 compute; later rows unknown.
	ledgerFor := func(s *Strategy) ResourceLedger {
		kv := estimatedModelContextShares(model, s, caps.GPUs, computeKVTotalMBForStrategy(model, s))
		l := ResourceLedger{Fits: s.UBatchSize == 64, NeedsAdmission: s.UBatchSize != 64}
		for i, m := range []int{4282, 10609, 1192} {
			l.Devices = append(l.Devices, DeviceResourceLedger{GPU: i, Active: true, ModelMB: m,
				ContextMB: kv[i], RequiredMB: m + kv[i], BandwidthMBps: caps.GPUs[i].MemBandwidthMBps})
		}
		return l
	}
	type row struct {
		name string
		s    *Strategy
		opts Options
	}
	rows := []row{{"baseline ub64", base, mainline}}
	for _, ub := range []int{128, 256, 512, 2048} {
		s := cloneStrategy(base)
		s.UBatchSize = ub
		rows = append(rows, row{fmt.Sprintf("ubatch %d", ub), s, mainline})
	}
	rows = append(rows, row{"no-op-offload ub64", base, noOffload})

	baseKV := computeKVTotalMBForStrategy(model, base)
	prevTransfer := 0.0
	t.Logf("%-20s %8s %6s %18s %9s %10s %12s %s", "candidate", "KV MiB", "dKV", "KV/GPU MiB", "touch", "GiB/token", "link-bound", "memory")
	for i, r := range rows {
		s := r.s
		// Complete configurations: only the tested coordinate may differ.
		if s.ContextSize != base.ContextSize || s.KVType != base.KVType || s.Parallel != 1 ||
			s.KVPlacement != base.KVPlacement || splitCompactKey(s.TensorSplit) != splitCompactKey(base.TensorSplit) {
			t.Fatalf("%s is a partial overlay", r.name)
		}
		kv := computeKVTotalMBForStrategy(model, s)
		shares := estimatedModelContextShares(model, s, caps.GPUs, kv)
		ledger := ledgerFor(s)
		est := EstimateStrategyCost(caps, model, s, r.opts, ledger)
		h := est.HostExpert
		memory := "measured compute (serving baseline)"
		if ledger.NeedsAdmission {
			memory = "compute unmeasured: exact admission required"
		}
		transfer, bound := "-", "compute unpriced"
		if h.PrefillStaged {
			gib := h.TransferSecPerToken * float64(h.CopyRateMBps) / 1024
			transfer, bound = fmt.Sprintf("%.3f", gib), fmt.Sprintf("<=%.1f tok/s", 1/h.TransferSecPerToken)
			if i > 0 && h.TransferSecPerToken >= prevTransfer {
				t.Fatalf("%s: larger microbatch did not reduce per-token transfer", r.name)
			}
			prevTransfer = h.TransferSecPerToken
		}
		t.Logf("%-20s %8d %+6d %18v %9.4f %10s %12s %s", r.name, kv, kv-baseKV, shares, h.TouchFraction, transfer, bound, memory)

		if h.StoredMB != 114192 {
			t.Fatalf("%s: %v MiB of host experts priced, want all 114192", r.name, h.StoredMB)
		}
		if shares[2] != 0 {
			t.Fatalf("%s: KV charged to the MTP/output-only GPU: %v", r.name, shares)
		}
		switch s.UBatchSize {
		case 64, 128:
			if kv != 8477 {
				t.Fatalf("%s: KV %d, want 8477 (256 SWA cells)", r.name, kv)
			}
		case 256:
			if kv != 8503 { // +25.8984375 MiB of window
				t.Fatalf("%s: KV %d, want 8503", r.name, kv)
			}
		}
		if ledger.NeedsAdmission && (est.Feasible || !est.NeedsAdmission || est.Confidence != "unknown-memory") {
			t.Fatalf("%s: unmeasured memory reported as fit: %+v", r.name, est)
		}
		if r.opts.HostWeightOffload == "off" && (h.PrefillStaged || h.ComputePriced) {
			t.Fatalf("CPU execution alternative claims a priced prefill: %+v", h)
		}
	}
}
