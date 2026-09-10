package placement

import "testing"

func TestOverlayParentReserveReachesExactAllocationAndSlotSizing(t *testing.T) {
	caps, model, s, opts := hotExpertFixture()
	opts.CacheDir = t.TempDir()
	opts.BackendCacheTag = "overlay@build"
	opts.RuntimeGrowthBaseBackendTag = "base@build"
	opts.WorkloadProfile = "agent-v1"
	opts.HotExperts = "auto"
	s.ContextSize, s.UBatchSize = 4096, 256
	s.KVQuality, s.KVType, s.KVPlacement = "q8_0", "q8_0", "cpu"
	model.NumExperts = 1000
	parent := opts
	parent.BackendCacheTag = opts.RuntimeGrowthBaseBackendTag
	if err := RecordRuntimeGraphGrowth(opts.CacheDir, model, 4096, 256, "q8_0", "cpu", backendCacheTag(parent), caps.GPUs, 1, map[int]int{3: 700, 7: 506}); err != nil {
		t.Fatal(err)
	}
	// Parent evidence must never make the overlay's allocation exact.
	if got := BuildResourceLedger(caps, model, s, opts); got.Exact {
		t.Fatal("parent growth granted overlay allocation authority")
	}
	allocation := MeasuredAllocation{Evidence: "allocation-verified", PlacementIdentity: AllocationPlacementIdentity(s, model), ContextTotalMB: 100, ContextHostMB: 100, ModelByGPU: map[int]int{3: 1000, 7: 1000}, UnaccountedByGPU: map[int]int{3: 100, 7: 100}, ModelHostMB: 100}
	if err := RecordMeasuredAllocation(opts.CacheDir, model, 4096, 256, "q8_0", "cpu", backendCacheTag(opts), caps.GPUs, 1, allocation); err != nil {
		t.Fatal(err)
	}
	with := BuildResourceLedger(caps, model, s, opts)
	withoutOpts := opts
	withoutOpts.RuntimeGrowthBaseBackendTag = ""
	without := BuildResourceLedger(caps, model, s, withoutOpts)
	if !with.Exact || !with.Fits {
		t.Fatalf("matching overlay allocation lost: %+v", with)
	}
	for i, d := range with.Devices {
		want := map[int]int{3: 700, 7: 506}[d.GPU]
		if !d.RuntimeMeasured || d.RuntimeMB != want || without.Devices[i].SlackMB-d.SlackMB != want {
			t.Fatalf("floor missing from exact device: %+v", d)
		}
	}
	shape := &hotExpertCacheShape{perSlotBytesByGPU: map[int]int64{3: 100 * hotExpertTestMiB, 7: 100 * hotExpertTestMiB}, fixedBytesByGPU: map[int]int64{3: hotExpertTestMiB, 7: hotExpertTestMiB}}
	if hotExpertCacheMaxSlots(model, shape, with) >= hotExpertCacheMaxSlots(model, shape, without) {
		t.Fatal("runtime reserve did not constrain discretionary cache sizing")
	}
}

func TestOverlayParentFloorIsScopedAndNeverLowersLocalGrowth(t *testing.T) {
	for _, scenario := range []string{"serve", "wrong workload", "wrong backend", "wrong parallel", "oom only", "measured zero"} {
		t.Run(scenario, func(t *testing.T) {
			caps, model, s, opts := hotExpertFixture()
			opts.CacheDir = t.TempDir()
			opts.BackendCacheTag = "overlay@build"
			opts.RuntimeGrowthBaseBackendTag = "base@build"
			opts.WorkloadProfile = "agent-v1"
			opts.HotExpertCacheSlots = 8
			parent := opts
			parent.BackendCacheTag = opts.RuntimeGrowthBaseBackendTag
			parent.HotExpertCacheSlots = 0
			parallel := 1
			values := map[int]int{3: 700, 7: 506}
			switch scenario {
			case "wrong workload":
				parent.WorkloadProfile = "other"
			case "wrong backend":
				parent.BackendCacheTag = "base@other"
			case "wrong parallel":
				parallel = 2
			case "measured zero":
				values = map[int]int{3: 0, 7: 0}
			}
			if scenario == "oom only" {
				if err := RecordRuntimeGraphGrowthFromOOM(opts.CacheDir, model, 4096, 256, "q8_0", "cpu", backendCacheTag(parent), caps.GPUs, 1, 7, 506, false); err != nil {
					t.Fatal(err)
				}
			} else if err := RecordRuntimeGraphGrowth(opts.CacheDir, model, 4096, 256, "q8_0", "cpu", backendCacheTag(parent), caps.GPUs, parallel, values); err != nil {
				t.Fatal(err)
			}
			if err := RecordRuntimeGraphGrowth(opts.CacheDir, model, 4096, 256, "q8_0", "cpu", backendCacheTag(opts), caps.GPUs, 1, map[int]int{3: 900}); err != nil {
				t.Fatal(err)
			}
			got := runtimeGrowthFloor(model, s, opts, caps.GPUs)
			if got[3] != 900 {
				t.Fatalf("local reserve lowered: %v", got)
			}
			value, known := got[7]
			wantKnown := scenario == "serve" || scenario == "measured zero"
			if known != wantKnown || (scenario == "serve" && value != 506) || (scenario == "measured zero" && value != 0) {
				t.Fatalf("unexpected parent carry: %v", got)
			}
		})
	}
}

func TestObservedRuntimeFloorAvoidsDoubleCounting(t *testing.T) {
	for _, tc := range []struct {
		name          string
		graph, growth int
		known         bool
		want          int
	}{
		{"already included", 800, 300, true, 800},
		{"partly included", 600, 300, true, 800},
		{"larger observed peak", 900, 300, true, 900},
		{"unknown breakdown", 800, 300, false, 1100},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := DeviceResourceLedger{GraphMB: tc.graph, RequiredMB: 1000 + tc.graph, FreeMB: 5000, SlackMB: 4000 - tc.graph}
			applyObservedRuntimeFloor(&d, tc.growth, 400, 100, tc.known)
			if d.RequiredMB != 1000+tc.want || d.RequiredMB+d.SlackMB != d.FreeMB || d.GraphMB+d.RuntimeMB != tc.want {
				t.Fatalf("bad conservation: %+v", d)
			}
		})
	}
}
