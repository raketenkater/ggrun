package placement

import "testing"

func TestResolvedHotCacheGrowthAcrossPolicies(t *testing.T) {
	for _, policy := range []string{"auto", "on", "8"} {
		t.Run(policy, func(t *testing.T) {
			caps, model, strategy, opts := hotExpertFixture()
			opts.CacheDir = t.TempDir()
			opts.BackendTag = "merged-backend"
			opts.HotExperts = policy
			strategy.ContextSize, strategy.UBatchSize = 4096, 256
			strategy.KVQuality, strategy.KVType, strategy.KVPlacement = "q8_0", "q8_0", "gpu"
			strategy.HotExpertCacheSlots = 8
			model.TotalSizeMB, model.SizeBytes = 32, 32*hotExpertTestMiB
			if err := RecordRuntimeGraphGrowth(opts.CacheDir, model, 4096, 256, "q8_0", "gpu", backendCacheTag(opts), caps.GPUs, 1, map[int]int{3: 274, 7: 506}); err != nil {
				t.Fatal(err)
			}
			resolved := opts
			resolved.HotExpertCacheSlots = 8
			if err := RecordRuntimeGraphGrowth(opts.CacheDir, model, 4096, 256, "q8_0", "gpu", backendCacheTag(resolved), caps.GPUs, 1, map[int]int{3: 700}); err != nil {
				t.Fatal(err)
			}
			if policy == "8" {
				opts.HotExpertCacheSlots = 8
			}
			ledger := BuildResourceLedger(caps, model, strategy, opts)
			want := map[int]int{3: 700, 7: 506}
			for _, device := range ledger.Devices {
				if !device.RuntimeMeasured || device.RuntimeMB != want[device.GPU] {
					t.Fatalf("policy %s GPU %d: growth=%d measured=%v, want %d", policy, device.GPU, device.RuntimeMB, device.RuntimeMeasured, want[device.GPU])
				}
			}
		})
	}
}

// Candidate enumeration may call finalization repeatedly before any server is
// started. That must not consume the pin needed by the eventual launch.
func TestHotCandidatePlanningRetainsBootstrapPin(t *testing.T) {
	caps, model, base, opts := hotExpertFixture()
	opts.CacheDir = t.TempDir()
	opts.HotExperts = "on"
	opts.BackendCacheTag = "merged-test"
	model.SizeBytes, model.TotalSizeMB = 32*hotExpertTestMiB, 32
	model.ExpertBytes = model.SizeBytes
	base.ContextSize, base.BatchSize, base.UBatchSize = 4096, 512, 128
	base.KVPlacement, base.KVQuality, base.KVType = "cpu", "high", "q8_0"
	base.PlanFreeVRAM = map[int]int{3: 24576, 7: 24576}
	if err := RecordHotExpertBootstrapPin(opts.CacheDir, model.Path, base); err != nil {
		t.Fatal(err)
	}
	if err := RecordMeasuredAllocation(opts.CacheDir, model, base.ContextSize, base.UBatchSize, base.KVQuality, base.KVPlacement, backendCacheTag(opts), caps.GPUs, base.Parallel, MeasuredAllocation{
		Evidence: "allocation-verified", PlacementIdentity: AllocationPlacementIdentity(base, model),
		ContextTotalMB: 100, ContextHostMB: 100, ModelByGPU: map[int]int{3: 1000, 7: 1000},
		ModelHostMB: 100, UnaccountedByGPU: map[int]int{3: 100, 7: 100},
	}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		got, err := finalizeHotExpertCache(caps, model, opts, cloneStrategy(base))
		if err != nil || got == nil || got.HotExpertCacheSlots == 0 {
			t.Fatalf("candidate %d: got=%+v err=%v", i, got, err)
		}
		if !MeasuredHotExpertBootstrapPin(opts.CacheDir, model.Path).Valid() {
			t.Fatal("planning consumed the bootstrap pin before activation")
		}
	}
}
