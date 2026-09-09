package placement

import "testing"

func TestHotExpertLowerSlotCandidateIsBoundedAndRecomputed(t *testing.T) {
	caps, model, base, opts := hotExpertFixture()
	model.NumExperts = 16
	model.ExpertUsedCount = 8
	shape, err := hotExpertCacheShapeFor(caps, model, base, opts)
	if err != nil {
		t.Fatal(err)
	}
	if err := assignHotExpertCache(base, shape, 15); err != nil {
		t.Fatal(err)
	}
	oldCharge := cloneIntMap(base.HotExpertCacheVRAMByGPU)
	base.HotExpertCacheSteps = 512
	base.HotExpertCacheHits = 100
	base.HotExpertCacheMisses = 200
	base.BackendSupportsHotExpertCache = true
	base.ResourceLedger = &ResourceLedger{Host: HostResourceLedger{FreeMB: 64000, ModelMB: 10000, RequiredMB: 12000, SlackMB: 52000}, Exact: false, Fits: true, Devices: []DeviceResourceLedger{
		{GPU: 3, FreeMB: 12000, RequiredMB: 8000 + oldCharge[3], SlackMB: 4000 - oldCharge[3], HotExpertCacheMB: oldCharge[3]},
		{GPU: 7, FreeMB: 12000, RequiredMB: 8000 + oldCharge[7], SlackMB: 4000 - oldCharge[7], HotExpertCacheMB: oldCharge[7]},
	}}
	base.HotExpertCacheFreeBaseline = &Strategy{Type: base.Type, NCPUMoE: base.NCPUMoE, OTString: base.OTString}
	opts.HotExperts = "auto"
	lower := hotExpertLowerSlotCandidate(caps, model, base, opts)
	if lower == nil || lower.HotExpertCacheSlots != 8 {
		t.Fatalf("lower candidate=%+v, want 8 slots", lower)
	}
	if lower.ResourceLedger.Host != base.ResourceLedger.Host || lower.NCPUMoE != base.NCPUMoE || lower.OTString != base.OTString {
		t.Fatal("smaller cache changed residency or host account")
	}
	if !hasAdjacentArgPlacement(lower.Args("model.gguf", 8081), "--moe-expert-cache", "8") {
		t.Fatal("smaller cache not emitted on argv")
	}
	if lower.HotExpertCacheVRAMByGPU[3] >= base.HotExpertCacheVRAMByGPU[3] || lower.ResourceLedger.Exact || !lower.ResourceLedger.Fits {
		t.Fatalf("lower cache geometry/ledger invalid: %+v", lower)
	}
	if lower.HotExpertCacheSteps != 0 || lower.HotExpertCacheHits != 0 || lower.HotExpertCacheMisses != 0 {
		t.Fatal("lower candidate inherited runtime telemetry")
	}
	if base.HotExpertCacheSlots != 15 || base.ResourceLedger.Devices[0].HotExpertCacheMB != oldCharge[3] {
		t.Fatal("original candidate was mutated")
	}
	numeric := opts
	numeric.HotExperts = "15"
	if hotExpertLowerSlotCandidate(caps, model, base, numeric) != nil {
		t.Fatal("numeric hot-expert request received automatic fallback")
	}
	for _, tc := range []struct {
		name   string
		change func(*Strategy, *Options)
	}{
		{"host-deficit", func(s *Strategy, o *Options) { s.ResourceLedger.Host.SlackMB = -1 }},
		{"missing-device", func(s *Strategy, o *Options) { s.ResourceLedger.Devices = s.ResourceLedger.Devices[:1] }},
		{"inconsistent-charge", func(s *Strategy, o *Options) { s.ResourceLedger.Devices[0].HotExpertCacheMB++ }},
		{"minimum-size", func(s *Strategy, o *Options) { s.HotExpertCacheSlots = 8 }},
		{"explicit-slots", func(s *Strategy, o *Options) { o.HotExpertCacheSlots = 15 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			candidate := cloneStrategy(base)
			ledger := *base.ResourceLedger
			ledger.Devices = append([]DeviceResourceLedger(nil), ledger.Devices...)
			candidate.ResourceLedger = &ledger
			o := opts
			tc.change(candidate, &o)
			if hotExpertLowerSlotCandidate(caps, model, candidate, o) != nil {
				t.Fatal("invalid fallback was offered")
			}
		})
	}

}
