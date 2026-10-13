package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// Qwen3.8-27B-UD-Q5_K_S.gguf header fields (the same fixture as
// placement's qwen38Q5KSProfile), served by an ik_llama build whose walk
// assigns blocks by bytes.
func ikQwen38Profile() *placement.ModelProfile {
	return &placement.ModelProfile{
		Path: "Qwen3.8-27B-UD-Q5_K_S.gguf", Basename: "Qwen3.8-27B-UD-Q5_K_S", ModelArch: "qwen35",
		SizeBytes: 18665753504, TotalSizeMB: 17801, CTXTrain: 262144, ContextSize: 262144,
		NumLayers: 65, NextNPredictLayers: 1, HasSSM: 1, FullAttnInterval: 4,
		HeadCount: 24, HeadCountKV: 4, KeyLength: 256, ValueLength: 256, RopeDim: 64,
		EmbeddingLength: 5120, HiddenSize: 5120, FeedForwardLength: 17408, VocabSize: 248320,
		SSMConvKernel: 4, SSMStateSize: 128, SSMGroupCount: 16, SSMInnerSize: 6144, SSMTimeStepRank: 48,
		NonExpertBytes: 18654756864, OutputBytes: 1042944000, TokenEmbdBytes: 715161600,
		NonExpertLayerBytes: []int64{
			227605376, 232479616, 261766016, 260179968, 256891776, 249928576, 274996096, 236955648,
			243374976, 236002176, 227646336, 216844288, 218307456, 205323136, 210443136, 204064768,
			231824256, 222731136, 215522176, 215943168, 252713856, 274996096, 261479296, 275908608,
			268032896, 268032896, 268032896, 272427008, 268032896, 264100736, 263854976, 264071168,
			252713856, 252713856, 249928576, 251660288, 263854976, 268032896, 257301376, 261531648,
			263854976, 263854976, 263854976, 262801408, 259922816, 258939776, 263854976, 251660288,
			263854976, 258939776, 268032896, 262801408, 279870336, 268032896, 275692416, 274638848,
			291707776, 303545216, 303545216, 299583488, 303545216, 303545216, 291707776, 299583488,
			351008768,
		},
		ByteWeightedLayerSplit: true,
	}
}

// The --claude-code box: RTX 4070, RTX 3090 Ti, and an RTX 3060 holding the
// 5,040 MiB reviewer (A2-q38-27b-claude-code-converge).
func ikClaudeCodeBox() *detect.Capabilities {
	return &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12282, VRAMUsedMB: 690,
				BandwidthMBps: 15760, MemBandwidthMBps: 504048},
			{Index: 1, Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24564, VRAMUsedMB: 1113,
				BandwidthMBps: 15760, MemBandwidthMBps: 1008096},
			{Index: 2, Name: "NVIDIA GeForce RTX 3060", VRAMTotalMB: 12288, VRAMUsedMB: 586 + 5040,
				BandwidthMBps: 7880, MemBandwidthMBps: 360048},
		},
		RAM: detect.RAMInfo{TotalMB: 262144, FreeMB: 200000},
		CPU: detect.CPUInfo{Cores: 24},
	}
}

// ikDeviceMB is the weights plus KV the byte walk puts on each device under
// a strategy's split, as the resource ledger prices them.
func ikDeviceMB(t *testing.T, caps *detect.Capabilities, model *placement.ModelProfile, s *placement.Strategy) []int {
	t.Helper()
	ledger := placement.BuildResourceLedger(caps, model, s, placement.Options{CacheDir: t.TempDir(), RequireMeasuredBuffers: true})
	out := make([]int, len(ledger.Devices))
	for i, d := range ledger.Devices {
		out[i] = d.ModelMB + d.ContextMB
	}
	return out
}

// The first --claude-code preflight (A2-q38-27b-claude-code-kvfix/serve.log):
// --ctx-size 661504 --parallel 4 -b 128 -ub 128 --tensor-split 0.27,0.58,0.15
// failed on "CUDA0 allocation 7051 MiB (deficit 648 MiB)", and each derate
// after it (593,920, 518,144, 483,328) failed again, the deficit moving from
// CUDA0 to CUDA1, until the four-start admission budget ended the launch: each
// re-split was priced by the slot rule.
//
// This drives the launch's DoesNotFit branch against a simulated backend that
// places blocks by the byte walk (checked against the backend logs in
// placement's iklayersplit_test.go) under the allocation guard's limit: free
// VRAM less PlanningCUDAOverheadByGPU, 1,024 MiB per unmeasured card. (Before
// the planner reserved that overhead, this test stood it in as a 755 MiB
// "reserve" of used VRAM, the shortfall the logged failure implied.) Priced by
// the walk, the first derate fits every card: the deficit does not move.
// Priced by the slot rule, the derates overfill a card again, which is the
// logged shape.
func TestIKKVDerateKeepsEveryDeviceWithinCapacity(t *testing.T) {
	const starts = 4
	run := func(byteWalk bool) (fails []preflightOutcome, converged bool) {
		cfg := &config.Config{CacheDir: t.TempDir()}
		backend := ikQwen38Profile()
		model := ikQwen38Profile()
		caps := ikClaudeCodeBox()
		guard := placement.PlanningCUDAOverheadByGPU(cfg.CacheDir, caps.GPUs)
		be := &backendInfo{Tag: "ik_llama", Identity: "ik", IsIK: true, Help: "       --fit   automatically determine"}
		req := &launchRequest{CtxFlag: "fit", ClaudeCode: true, Parallel: 4, ParallelSet: true, KVQuality: "q8_0"}
		applyBackendFeatureCompatibility(req, model, be)
		if !model.ByteWeightedLayerSplit {
			t.Fatal("the ik backend fact was not set")
		}
		model.ByteWeightedLayerSplit = byteWalk
		strategy := &placement.Strategy{
			Type: placement.MultiGPUDense, ContextSize: 661504, ContextAuto: true, Parallel: 4,
			BatchSize: 128, UBatchSize: 128, KVType: "q8_0", KVQuality: "q8_0", KVPlacement: "gpu",
			TensorSplit: []float64{0.27, 0.58, 0.15}, SplitMode: "layer",
		}
		args := buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
		recovery := newLaunchMemoryRecovery()
		penalty := map[int]int{}
		for start := 0; start < starts; start++ {
			held := ikDeviceMB(t, caps, backend, strategy)
			outcome := preflightOutcome{Device: -1}
			for i, g := range caps.GPUs {
				if over := held[i] - (g.VRAMFreeMB() - guard[g.Index]); over > 0 {
					kv := placement.StrategyDeviceKVCacheMB(backend, strategy, "q8_0", false, i)
					outcome = preflightOutcome{DoesNotFit: true, Device: i, AllocMB: kv, AllocMBMeasured: true, DeficitMB: over, IsKVCache: true}
					break
				}
			}
			t.Logf("walk=%v start %d: ctx %d split %v holds %v; failure %+v", byteWalk, start, strategy.ContextSize, strategy.TensorSplit, held, outcome)
			if outcome.Device < 0 {
				return fails, true
			}
			fails = append(fails, outcome)
			recovery.reject(args)
			recovery.rejectContext(strategy, contextReclaimTokens(model, strategy, args, outcome.DeficitMB, outcome.Device))
			recovery.notePricedContextStep(strategy, majorityDevicePricedTokens(model, strategy, args, outcome.DeficitMB, outcome.Device))
			if contextDeficitOutstripsDevice(model, strategy, args, outcome.DeficitMB, outcome.Device) {
				recovery.rejectContextOutstripped(strategy)
			}
			next, nextArgs, _, err := recoverPreflightOOM(req, cfg, model, be, caps, caps, nil, strategy, args, penalty, outcome, recovery)
			if err != nil {
				t.Logf("walk=%v: recovery failed closed: %v", byteWalk, err)
				return fails, false
			}
			strategy, args = next, nextArgs
		}
		return fails, false
	}

	fails, converged := run(true)
	if len(fails) == 0 || fails[0].Device != 0 || fails[0].DeficitMB < 648 || fails[0].AllocMB < 7050 || fails[0].AllocMB > 7052 {
		t.Fatalf("the simulated first preflight does not reproduce the logged CUDA0 7051 MiB failure (deficit >= 648 MiB): %+v", fails)
	}
	if !converged || len(fails) != 1 {
		t.Fatalf("byte-walk recovery: converged=%v after failures %+v; want the first derate to fit every card", converged, fails)
	}
	if slotFails, _ := run(false); len(slotFails) < 2 {
		t.Fatalf("slot-rule recovery failed only %+v; the test no longer reproduces the logged failure", slotFails)
	}
}

// The KV share that sizes a context cut for a compute or oracle deficit
// (deviceKVShareMB) is read from the argv. Under a byte-weighted backend it is
// the KV of the blocks the walk puts on the device: CUDA0 of the failed
// preflight allocated 7051 MiB, where the split fraction gives 0.27 of the
// model-wide KV. A mainline backend keeps the fraction.
func TestIKDeviceKVShareFollowsTheWalk(t *testing.T) {
	model := ikQwen38Profile()
	args := []string{"ik_llama-server", "--ctx-size", "661504", "-b", "128", "-ub", "128",
		"--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--parallel", "4", "-ngl", "999",
		"--tensor-split", "0.27,0.58,0.15", "--split-mode", "layer"}
	s := &placement.Strategy{ContextSize: 661504, Parallel: 4, BatchSize: 128, UBatchSize: 128, KVType: "q8_0",
		KVPlacement: "gpu", TensorSplit: []float64{0.27, 0.58, 0.15}}
	if got := deviceKVShareMB(model, args, 0); got < 7050 || got > 7052 {
		t.Fatalf("CUDA0 share %d MiB, the backend allocated 7051", got)
	}
	for device := range s.TensorSplit {
		if got, want := deviceKVShareMB(model, args, device), placement.StrategyDeviceKVCacheMB(model, s, "q8_0", false, device); got != want {
			t.Errorf("CUDA%d argv share %d MiB, strategy share %d", device, got, want)
		}
	}
	// -b below -ub sets the backend's microbatch, which the walk's compute
	// term depends on.
	small := append(append([]string(nil), args...), "-b", "64", "-ub", "128")
	s.UBatchSize = 64
	if got, want := deviceKVShareMB(model, small, 0), placement.StrategyDeviceKVCacheMB(model, s, "q8_0", false, 0); got != want {
		t.Errorf("-b 64 -ub 128: share %d MiB, strategy at ubatch 64 %d", got, want)
	}
	model.ByteWeightedLayerSplit = false
	want := int(float64(placement.EstimateKVCacheMB(model, 661504, "q8_0", false)) * 0.27)
	if got := deviceKVShareMB(model, args, 0); got != want {
		t.Errorf("mainline CUDA0 share %d MiB, want the split fraction's %d", got, want)
	}
}
