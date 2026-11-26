package placement

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestExpertPackIsAChallengerOfTheResolvedBaseline(t *testing.T) {
	for _, gpuCount := range []int{1, 2} {
		for _, explicit := range []bool{false, true} {
			t.Run(fmt.Sprintf("gpus=%d/explicit=%v", gpuCount, explicit), func(t *testing.T) {
				caps := &detect.Capabilities{
					GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, BandwidthMBps: 32000}},
					RAM:  detect.RAMInfo{TotalMB: 131072, FreeMB: 131072}, CPU: detect.CPUInfo{Cores: 16},
				}
				if gpuCount == 2 {
					caps.GPUs = append(caps.GPUs, detect.GPU{Index: 1, VRAMTotalMB: 12288, BandwidthMBps: 8000})
				}
				model := &ModelProfile{
					Path: "fixture.gguf", IsMoE: true, TotalSizeMB: 64 * 1024, NumLayers: 60, NumExperts: 128,
					ExpertBytes: 56 << 30, NonExpertBytes: 8 << 30, ContextSize: 8192, CTXTrain: 262144,
					HiddenSize: 4096, HeadCountKV: 8, KeyLength: 128, ValueLength: 128,
				}
				opts := Options{ContextSize: 8192, KVPlacement: "cpu", KVQuality: "q8_0", Parallel: 1,
					BatchSize: 2048, UBatchSize: 1024, BatchSizeExplicit: true, UBatchSizeExplicit: true, CacheDir: t.TempDir()}
				base, err := Compute(caps, model, opts)
				if err != nil || base.Type != MoEOffload {
					t.Fatalf("fixture placement: %+v, %v", base, err)
				}
				// Model the conservative prior baseline before new allocation evidence
				// allows one more expert layer. The serving object must stay intact.
				base.NCPUMoE++
				before := *base
				if !explicit {
					opts.ContextSize, opts.AutoContextMax, opts.UBatchSize = 262144, 262144, 512
					opts.BatchSizeExplicit, opts.UBatchSizeExplicit = false, false
				}
				candidates := CalibrationCandidates(caps, model, base, opts)
				if len(candidates) < 2 || candidates[0].Strategy != base || candidates[0].Name != "default" {
					t.Fatal("the serving placement must remain candidate zero")
				}
				var pack *Strategy
				for _, c := range candidates {
					if c.Name == "expert-pack" {
						pack = c.Strategy
					}
				}
				if pack == nil {
					t.Fatal("allocation improvement missing from measured candidate frontier")
				}
				if pack == base || pack.ContextSize != base.ContextSize || pack.Parallel != base.Parallel ||
					pack.BatchSize != base.BatchSize || pack.UBatchSize != base.UBatchSize ||
					pack.KVType != base.KVType || pack.KVPlacement != base.KVPlacement || pack.NCPUMoE >= base.NCPUMoE {
					t.Fatalf("expert re-pack changed resolved constraints: base=%+v candidate=%+v", base, pack)
				}
				if !reflect.DeepEqual(*base, before) {
					t.Fatal("candidate generation mutated the live baseline")
				}
			})
		}
	}
}
