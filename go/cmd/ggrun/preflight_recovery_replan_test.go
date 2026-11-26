package main

import (
	"errors"
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// deviceReplanFixture is a dense two-GPU plan whose failed device holds no
// movable expert layer, so a compute-bound deficit at ubatch 64 has no
// deterministic lever left. That is the 2026-11-26 fresh-install shape: the
// first 3-GPU preflight came up short, ubatch reached 64, and recovery then
// derated context until the planner left a GPU unused.
func deviceReplanFixture(t *testing.T, ctxFlag string, ubatch int) (*launchRequest, *config.Config, *placement.ModelProfile, *backendInfo, *detect.Capabilities, *placement.Strategy, []string) {
	t.Helper()
	model := fitTestModel(131072, 6000)
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "GPU A", VRAMTotalMB: 12000, VRAMUsedMB: 300},
			{Index: 1, Name: "GPU B", VRAMTotalMB: 12000, VRAMUsedMB: 300},
		},
		RAM: detect.RAMInfo{TotalMB: 262144, FreeMB: 200000},
		CPU: detect.CPUInfo{Cores: 24},
	}
	req := &launchRequest{CtxFlag: ctxFlag, Parallel: 1, ParallelSet: true, KVQuality: "mid", KVPlacement: "gpu", RAMLimitPercent: 95}
	be := fitTestBackend()
	cfg := &config.Config{CacheDir: t.TempDir()}
	current := &placement.Strategy{
		Type: placement.MultiGPUDense, ContextSize: 65536, ContextAuto: automaticContextRequest(req),
		BatchSize: 2048, UBatchSize: ubatch, Parallel: 1, KVType: "q8_0", KVPlacement: "gpu",
		TensorSplit: []float64{0.5, 0.5}, SplitMode: "layer", Threads: 8, ThreadsBatch: 8, MMap: true,
	}
	args := buildLaunchServerArgs(req, cfg, be, caps, model, current)
	return req, cfg, model, be, caps, current, args
}

// computeBoundOracleOutcome is an oracle total deficit that the failed
// device's own graph buffer covers, so oracleComputeBoundOutcome classes it as
// a compute-buffer failure.
func computeBoundOracleOutcome() preflightOutcome {
	return preflightOutcome{
		Device: 0, DoesNotFit: true, DeficitMB: 400, DeviceComputeMB: 900,
		Evidence: memoryPlanEvidence{Level: memoryEvidenceOraclePlanned},
	}
}

// disprovedRecovery mirrors the launch loop at the ubatch floor: an earlier
// planner re-plan was already refused, and the plan just refused is recorded.
func disprovedRecovery(current *placement.Strategy, args []string) *launchMemoryRecovery {
	r := newLaunchMemoryRecovery()
	r.plannerDisproved = true
	r.reject(args)
	r.rejectContext(current, 0)
	return r
}

type stubDeviceReplan struct {
	calls   int
	opts    placement.Options
	penalty map[int]int
	share0  float64
	err     error
}

// install replaces the device re-plan with one that moves load off device 0
// at the requested context, or fails with s.err.
func (s *stubDeviceReplan) install(t *testing.T, base *placement.Strategy) {
	t.Helper()
	original := replanAfterDeviceOOM
	t.Cleanup(func() { replanAfterDeviceOOM = original })
	if s.share0 == 0 {
		s.share0 = 0.3
	}
	replanAfterDeviceOOM = func(caps *detect.Capabilities, model *placement.ModelProfile, opts placement.Options, penalty map[int]int) (*placement.Strategy, error) {
		s.calls++
		s.opts = opts
		s.penalty = map[int]int{}
		for k, v := range penalty {
			s.penalty[k] = v
		}
		if s.err != nil {
			return nil, s.err
		}
		next := *base
		next.ContextSize = opts.ContextSize
		next.UBatchSize = opts.UBatchSize
		next.TensorSplit = []float64{s.share0, 1 - s.share0}
		s.share0 /= 2
		return &next, nil
	}
}

// A compute-bound deficit at the ubatch floor must try the device-penalty
// re-plan before any context derate, keep the context, and return a complete
// candidate that only the caller's next exact preflight can admit.
func TestComputeBoundDeficitAtUBatchFloorTriesDeviceReplanBeforeContext(t *testing.T) {
	req, cfg, model, be, caps, current, args := deviceReplanFixture(t, "fit", 64)
	stub := &stubDeviceReplan{}
	stub.install(t, current)
	recovery := disprovedRecovery(current, args)
	oomPenalty := map[int]int{}

	next, nextArgs, method, err := recoverPreflightOOM(req, cfg, model, be, caps, caps, nil,
		current, args, oomPenalty, computeBoundOracleOutcome(), recovery)
	if err != nil || method != "device-replanned" {
		t.Fatalf("recovery = %q, %v; want the device re-plan before any context derate", method, err)
	}
	if stub.calls != 1 {
		t.Fatalf("device re-plan called %d times, want 1", stub.calls)
	}
	if stub.penalty[0] != 400 || len(stub.penalty) != 1 {
		t.Fatalf("penalty %v, want the 400 MiB deficit charged to the failed device only", stub.penalty)
	}
	// The rejected context set a ceiling below 65,536; the lever must pin the
	// context instead of letting that ceiling derate it.
	if stub.opts.ContextSize != 65536 || stub.opts.AutoContextMax != 0 ||
		stub.opts.UBatchSize != 64 || !stub.opts.UBatchSizeExplicit || stub.opts.Parallel != 1 {
		t.Fatalf("re-plan options did not pin context/ubatch/slots: ctx=%d max=%d ub=%d explicit=%v np=%d",
			stub.opts.ContextSize, stub.opts.AutoContextMax, stub.opts.UBatchSize, stub.opts.UBatchSizeExplicit, stub.opts.Parallel)
	}
	if next == nil || next.ContextSize != current.ContextSize || !next.ContextAuto || next.UBatchSize != 64 {
		t.Fatalf("device re-plan changed the workload: %+v", next)
	}
	if oomPenalty[0] != 400 {
		t.Fatalf("returned candidate did not commit its penalty: %v", oomPenalty)
	}

	// Admission stays with exact preflight: the candidate is a complete argv
	// built from its Strategy, nothing about it is recorded as proven, and it
	// is not pre-rejected.
	rebuilt := buildLaunchServerArgs(req, cfg, be, caps, model, next)
	if launchArgsIdentity(nextArgs) != launchArgsIdentity(rebuilt) {
		t.Fatal("device re-plan returned a partial argv overlay")
	}
	if next.PerformanceTuned {
		t.Fatal("a recovery candidate must not carry a performance decision")
	}
	if recovery.acceptedContext != 0 || recovery.acceptedUBatch != 0 || recovery.isRejected(nextArgs) {
		t.Fatalf("recovery recorded proof for an unadmitted candidate: ctx=%d ub=%d rejected=%v",
			recovery.acceptedContext, recovery.acceptedUBatch, recovery.isRejected(nextArgs))
	}
	if !candidateRelievesFailedDevice(args, nextArgs, 0) {
		t.Fatal("candidate does not move load off the failed device")
	}
}

// A failing device re-plan leaves today's ladder: the same result the launch
// gets with the lever already spent, and no penalty left behind.
func TestFailedDeviceReplanFallsBackToTheContextLadder(t *testing.T) {
	type result struct {
		args    []string
		method  string
		err     error
		penalty map[int]int
		calls   int
	}
	run := func(spent bool) result {
		req, cfg, model, be, caps, current, args := deviceReplanFixture(t, "fit", 64)
		stub := &stubDeviceReplan{err: errors.New("no placement fits")}
		stub.install(t, current)
		recovery := disprovedRecovery(current, args)
		if spent {
			recovery.deviceReplans = maxComputeBoundDeviceReplans
		}
		oomPenalty := map[int]int{}
		_, nextArgs, method, err := recoverPreflightOOM(req, cfg, model, be, caps, caps, nil,
			current, args, oomPenalty, computeBoundOracleOutcome(), recovery)
		return result{nextArgs, method, err, oomPenalty, stub.calls}
	}
	base := run(true)
	if base.calls != 0 {
		t.Fatalf("a spent lever was called %d times", base.calls)
	}
	got := run(false)
	if got.calls != 1 {
		t.Fatalf("device re-plan called %d times, want 1", got.calls)
	}
	if got.method == "device-replanned" || got.method != base.method || (got.err == nil) != (base.err == nil) ||
		launchArgsIdentity(got.args) != launchArgsIdentity(base.args) {
		t.Fatalf("failed re-plan changed the fallback: got %q err=%v, want %q err=%v", got.method, got.err, base.method, base.err)
	}
	penalty := got.penalty
	if len(penalty) != 0 {
		t.Fatalf("a refused re-plan left its penalty behind: %v", penalty)
	}
}

// An explicit context is never derated: a fitting re-plan keeps it, and a
// failing re-plan fails closed.
func TestDeviceReplanKeepsAnExplicitContextAndFailsClosed(t *testing.T) {
	req, cfg, model, be, caps, current, args := deviceReplanFixture(t, "65536", 64)
	stub := &stubDeviceReplan{}
	stub.install(t, current)
	next, _, method, err := recoverPreflightOOM(req, cfg, model, be, caps, caps, nil,
		current, args, map[int]int{}, computeBoundOracleOutcome(), disprovedRecovery(current, args))
	if err != nil || method != "device-replanned" || next.ContextSize != 65536 || next.ContextAuto {
		t.Fatalf("explicit-context re-plan = %q, %v, %+v", method, err, next)
	}

	req, cfg, model, be, caps, current, args = deviceReplanFixture(t, "65536", 64)
	failing := &stubDeviceReplan{err: errors.New("no placement fits")}
	failing.install(t, current)
	next, _, method, err = recoverPreflightOOM(req, cfg, model, be, caps, caps, nil,
		current, args, map[int]int{}, computeBoundOracleOutcome(), disprovedRecovery(current, args))
	if err == nil {
		t.Fatalf("explicit context with a failing re-plan must fail closed, got %q ctx=%d", method, next.ContextSize)
	}
	if failing.calls != 1 {
		t.Fatalf("device re-plan called %d times, want 1", failing.calls)
	}
}

// The lever is bounded per launch: when its candidate is refused and the
// deficit returns, recovery does not re-plan again.
func TestDeviceReplanAttemptBoundHolds(t *testing.T) {
	req, cfg, model, be, caps, current, args := deviceReplanFixture(t, "fit", 64)
	stub := &stubDeviceReplan{}
	stub.install(t, current)
	recovery := disprovedRecovery(current, args)
	oomPenalty := map[int]int{}
	next, nextArgs, method, err := recoverPreflightOOM(req, cfg, model, be, caps, caps, nil,
		current, args, oomPenalty, computeBoundOracleOutcome(), recovery)
	if err != nil || method != "device-replanned" {
		t.Fatalf("first recovery = %q, %v", method, err)
	}
	for round := 0; round < 3; round++ {
		// Exact preflight refused the candidate with the same deficit.
		recovery.reject(nextArgs)
		recovery.rejectContext(next, 0)
		_, _, method, _ = recoverPreflightOOM(req, cfg, model, be, caps, caps, nil,
			next, nextArgs, oomPenalty, computeBoundOracleOutcome(), recovery)
		if method == "device-replanned" {
			t.Fatalf("round %d re-planned again past the bound", round+2)
		}
	}
	if stub.calls != maxComputeBoundDeviceReplans {
		t.Fatalf("device re-plan called %d times, bound is %d", stub.calls, maxComputeBoundDeviceReplans)
	}
	if !recovery.plannerWasDisproved() {
		t.Fatal("a refused device re-plan must disprove the planner estimate")
	}
}

// Deficits that are not compute-bound, or not yet at the ubatch floor, keep
// today's path and never reach the device re-plan lever.
func TestNonComputeBoundOrAboveFloorDeficitSkipsDeviceReplan(t *testing.T) {
	cases := []struct {
		name    string
		ubatch  int
		outcome preflightOutcome
		want    string
	}{
		{"compute above floor derates ubatch first", 512, computeBoundOracleOutcome(), "ubatch-derate"},
		{"kv allocation", 64, preflightOutcome{Device: 0, DoesNotFit: true, DeficitMB: 400, AllocMB: 900, AllocMBMeasured: true, IsKVCache: true,
			Evidence: memoryPlanEvidence{Level: memoryEvidenceAllocated}}, ""},
		{"oracle total larger than the graph", 64, preflightOutcome{Device: 0, DoesNotFit: true, DeficitMB: 1400, DeviceComputeMB: 900,
			Evidence: memoryPlanEvidence{Level: memoryEvidenceOraclePlanned}}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req, cfg, model, be, caps, current, args := deviceReplanFixture(t, "fit", tc.ubatch)
			stub := &stubDeviceReplan{}
			stub.install(t, current)
			_, _, method, _ := recoverPreflightOOM(req, cfg, model, be, caps, caps, nil,
				current, args, map[int]int{}, tc.outcome, disprovedRecovery(current, args))
			if stub.calls != 0 || method == "device-replanned" {
				t.Fatalf("device re-plan reached (calls=%d, method=%q)", stub.calls, method)
			}
			if tc.want != "" && method != tc.want {
				t.Fatalf("method %q, want %q", method, tc.want)
			}
		})
	}
}
