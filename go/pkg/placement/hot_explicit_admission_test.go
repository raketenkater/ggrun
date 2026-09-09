package placement

import "testing"

func TestExplicitHotSlotsCreditDemotedBytesAndChargeHost(t *testing.T) {
	caps, m, base, opts := hotExpertFixture()
	m.NumExperts = 16
	m.RoutedExpertLayerBytes = []int64{64 * hotExpertTestMiB, 16 * hotExpertTestMiB, 16 * hotExpertTestMiB, 16 * hotExpertTestMiB}
	ledger := ResourceLedger{Exact: true, Fits: true, Evidence: "live-allocated", Devices: []DeviceResourceLedger{
		{GPU: 3, FreeMB: 100, ModelMB: 64, RequiredMB: 96, SlackMB: 4},
		{GPU: 7, FreeMB: 120, ModelMB: 16, RequiredMB: 20, SlackMB: 100},
	}, Host: HostResourceLedger{FreeMB: 2000, ModelMB: 100, RequiredMB: 100, SlackMB: 1900}}
	candidate := cloneStrategy(base)
	shape, got, drops, err := hotExpertFreeLayersForSlots(caps, m, candidate, opts, ledger, 4)
	if err != nil {
		t.Fatalf("explicit cache could fit after demotion: %v", err)
	}
	if drops != 1 || shape == nil {
		t.Fatalf("wrong demotion: drops=%d shape=%+v", drops, shape)
	}
	if got.Exact || !got.Fits || got.Devices[0].SlackMB != 68 || got.Host.RequiredMB != 164 || got.Host.SlackMB != 1836 {
		t.Fatalf("demotion accounting inconsistent: %+v", got)
	}
	if base.NCPUMoE != 3 || base.OTString == candidate.OTString || ledger.Devices[0].SlackMB != 4 {
		t.Fatal("original packed plan or ledger mutated")
	}
	if err := assignHotExpertCache(candidate, shape, 4); err != nil {
		t.Fatal(err)
	}
	applyHotExpertCacheLedger(&got, candidate, false)
	if !got.Fits || got.Devices[0].RequiredMB+got.Devices[0].SlackMB != 100 {
		t.Fatalf("explicit cache not fully charged: %+v", got)
	}
}

// Auto must be able to form its first displacement challenger without a manual
// cache-on launch. A candidate remains derived and preserves the packed plan.
func TestAutoHotDisplacementCandidateDoesNotRequirePriorWin(t *testing.T) {
	for _, priorLoss := range []bool{false, true} {
		caps, m, base, opts := hotExpertFixture()
		opts.HotExperts = "auto"
		opts.CacheDir = t.TempDir()
		m.NumExperts = 16
		m.ExpertUsedCount = 4
		m.RoutedExpertLayerBytes = []int64{64 * hotExpertTestMiB, 16 * hotExpertTestMiB, 16 * hotExpertTestMiB, 16 * hotExpertTestMiB}
		if priorLoss {
			if err := RecordHotExpertDisplacementProof(opts.CacheDir, m.Path, measuredLoss(AllocationPlacementIdentity(base, m))); err != nil {
				t.Fatal(err)
			}
		}
		ledger := ResourceLedger{Exact: true, Fits: true, Devices: []DeviceResourceLedger{
			{GPU: 3, FreeMB: 100, ModelMB: 64, RequiredMB: 96, SlackMB: 4},
			{GPU: 7, FreeMB: 120, ModelMB: 16, RequiredMB: 20, SlackMB: 100},
		}, Host: HostResourceLedger{FreeMB: 2000, ModelMB: 100, RequiredMB: 100, SlackMB: 1900}}
		got, err := hotExpertPriorityCandidate(caps, m, base, opts, ledger)
		if err != nil {
			t.Fatalf("priorLoss=%v: automatic challenger blocked: %v", priorLoss, err)
		}
		if got.HotExpertCacheSlots <= 0 || got.NCPUMoE != base.NCPUMoE+1 {
			t.Fatalf("missing cache/demotion: %+v", got)
		}
		if got.ResourceLedger == nil || got.ResourceLedger.Exact || !got.ResourceLedger.Fits {
			t.Fatal("candidate must fit the derived ledger without claiming exact admission")
		}
		if got.HotExpertCacheFreeBaseline == nil || got.HotExpertCacheFreeBaseline.NCPUMoE != base.NCPUMoE || base.HotExpertCacheSlots != 0 || ledger.Devices[0].SlackMB != 4 {
			t.Fatal("packed fallback or source ledger mutated")
		}
	}
}
