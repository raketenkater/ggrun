package main

import (
	"math/rand"
	"strconv"
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

const ikFitHelp = "--fit automatically determine which tensors to offload\n--fit-margin N safety margin in MiB"

func fitMarginOf(t *testing.T, args []string) int {
	t.Helper()
	for i := 0; i+1 < len(args); i++ {
		if args[i] == fitMarginFlag {
			n, err := strconv.Atoi(args[i+1])
			if err != nil {
				t.Fatalf("bad margin %q", args[i+1])
			}
			return n
		}
	}
	return -1
}

func gpusN(n int) []detect.GPU {
	out := make([]detect.GPU, n)
	for i := range out {
		out[i] = detect.GPU{Index: i}
	}
	return out
}

// The backend's fit budget (driver-free less margin) must stay within the
// guard's limit on every device, or admission rejects every re-plan.
func TestBackendFitBudgetStaysInsideTheGuardOnEveryDevice(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 2000; trial++ {
		n := 1 + rng.Intn(3)
		gpus := gpusN(n)
		limits := make([]int, n)
		free := map[int]int{}
		for i := 0; i < n; i++ {
			free[i] = 1000 + rng.Intn(24000)
			limits[i] = rng.Intn(free[i] + 2000)
		}
		args := guardedFitArgs([]string{"srv", "--fit"}, ikFitHelp, nil, limits, free, gpus)
		m := fitMarginOf(t, args)
		if m < 0 {
			m = 1024 // ik's own default when no margin is passed
		}
		for i := 0; i < n; i++ {
			if budget := free[i] - m; budget > limits[i] && budget > 0 {
				t.Fatalf("trial %d device %d: fit budget %d exceeds guard %d (free %d, margin %d)",
					trial, i, budget, limits[i], free[i], m)
			}
		}
	}
}

// Observed on a fresh install: the 4070 reported 7218 MiB free with the
// reviewer resident, while the guard allowed 5094 (6144 MiB reviewer
// reservation, 1024 unmeasured overhead).
func TestReviewerReservationIsPassedToTheBackendFit(t *testing.T) {
	args := guardedFitArgs([]string{"srv", "--fit"}, ikFitHelp, nil, []int{5094}, map[int]int{0: 7218}, gpusN(1))
	if m := fitMarginOf(t, args); m != 2304 {
		t.Fatalf("margin = %d, want 2304 (7218-5094 rounded up to 256)", m)
	}
}

func TestFitArgsUnchangedWhenTheBackendDoesNotFit(t *testing.T) {
	free := map[int]int{0: 7218}
	limits := []int{5094}
	cases := map[string]struct {
		args, user []string
		help       string
		free       map[int]int
	}{
		"explicit layout":    {args: []string{"srv", "-ngl", "999"}, help: ikFitHelp, free: free},
		"fit off":            {args: []string{"srv", "-ngl", "999", "--fit", "off"}, help: ikFitHelp, free: free},
		"no margin flag":     {args: []string{"srv", "--fit"}, help: "--fit automatically determine", free: free},
		"user margin":        {args: []string{"srv", "--fit", "--fit-margin", "300"}, user: []string{"--fit-margin", "300"}, help: ikFitHelp, free: free},
		"driver unreadable":  {args: []string{"srv", "--fit"}, help: ikFitHelp, free: nil},
		"device not sampled": {args: []string{"srv", "--fit"}, help: ikFitHelp, free: map[int]int{1: 7218}},
	}
	for name, c := range cases {
		got := guardedFitArgs(c.args, c.help, c.user, limits, c.free, gpusN(1))
		if len(got) != len(c.args) {
			t.Errorf("%s: args changed to %v", name, got)
			continue
		}
		for i := range got {
			if got[i] != c.args[i] {
				t.Errorf("%s: args changed to %v", name, got)
				break
			}
		}
	}
}

// The binding only tightens: where ik's default margin is already within the
// guard (a measured overhead below 1024, or a guard above driver free) the argv
// is unchanged.
func TestNoMarginWhenIKDefaultIsAlreadyInsideTheGuard(t *testing.T) {
	for _, limit := range []int{9000, 7700, 8000 - ikDefaultFitMarginMB} {
		args := guardedFitArgs([]string{"srv", "--fit", "on"}, ikFitHelp, nil, []int{limit}, map[int]int{0: 8000}, gpusN(1))
		if m := fitMarginOf(t, args); m != -1 {
			t.Fatalf("limit %d: margin %d added although ik's default is within the guard", limit, m)
		}
	}
}

// Memory evidence is keyed by argv; small drift in free VRAM must not change it.
func TestFitMarginIsStableUnderSmallDrift(t *testing.T) {
	a := guardedFitArgs([]string{"srv", "--fit"}, ikFitHelp, nil, []int{5094}, map[int]int{0: 7218}, gpusN(1))
	b := guardedFitArgs([]string{"srv", "--fit"}, ikFitHelp, nil, []int{5094}, map[int]int{0: 7300}, gpusN(1))
	if fitMarginOf(t, a) != fitMarginOf(t, b) {
		t.Fatalf("margins %d and %d differ for an 82 MiB drift", fitMarginOf(t, a), fitMarginOf(t, b))
	}
}

// A re-plan that re-binds args replaces ggrun's previous margin, not appends.
func TestRebindingReplacesThePreviousMargin(t *testing.T) {
	first := guardedFitArgs([]string{"srv", "--fit"}, ikFitHelp, nil, []int{5094}, map[int]int{0: 7218}, gpusN(1))
	second := guardedFitArgs(first, ikFitHelp, nil, []int{4300}, map[int]int{0: 7218}, gpusN(1))
	count := 0
	for _, a := range second {
		if a == fitMarginFlag {
			count++
		}
	}
	if count != 1 || fitMarginOf(t, second) != 3072 {
		t.Fatalf("rebind gave %v", second)
	}
}

func TestGuardLimitsAreFreeLessPlanningOverhead(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, VRAMTotalMB: 12282, VRAMUsedMB: 20, VRAMReservedMB: 6144},
		{Index: 1, VRAMTotalMB: 1000, VRAMUsedMB: 500},
	}}
	got := allocationGuardLimitsMB(t.TempDir(), caps)
	want0 := 12282 - 20 - 6144 - placement.UnmeasuredCUDAOverheadMB
	if len(got) != 2 || got[0] != want0 || got[1] != 0 {
		t.Fatalf("limits = %v, want [%d 0]", got, want0)
	}
}

// Every argv the launcher builds for one plan carries the same margin. When
// only the admission loop added it, each re-plan's rebuilt argv differed from
// the admitted one, and the launch spent its five measured re-plans "verifying"
// an unchanged placement (2026-10-14).
func TestBuiltLaunchArgsCarryOneStableFitMargin(t *testing.T) {
	savedUsed, savedFree := runtimeVRAMUsedMB, fitDriverFreeVRAM
	runtimeVRAMUsedMB = func(int) int { return 0 }
	fitDriverFreeVRAM = func(*detect.Capabilities) map[int]int { return map[int]int{0: 7218} }
	defer func() { runtimeVRAMUsedMB, fitDriverFreeVRAM = savedUsed, savedFree }()

	model := fitTestModel(262144, 9000)
	caps := fitTestCaps(12282)
	caps.GPUs[0].VRAMReservedMB = 6144 // the Auto reviewer's reservation
	be := &backendInfo{Path: "/x/ik_llama-server-cuda", Dialect: "ik_llama", Tag: "ik_llama", IsIK: true, Identity: "test", Help: ikFitHelp}
	req := &launchRequest{CtxFlag: "fit", Parallel: 1, ParallelSet: true, RAMLimitPercent: 95}
	cfg := &config.Config{CacheDir: t.TempDir()}
	s := &placement.Strategy{Type: placement.DenseCPUOffload, BackendSupportsFit: true, ContextSize: 32768, Parallel: 1, BatchSize: 2048, UBatchSize: 512}

	first := buildLaunchServerArgs(req, cfg, be, caps, model, s)
	second := buildLaunchServerArgs(req, cfg, be, caps, model, s)
	if formatCommand(first) != formatCommand(second) {
		t.Fatalf("rebuilt argv differs:\n%v\n%v", first, second)
	}
	m := fitMarginOf(t, first)
	limit := allocationGuardLimitsMB(cfg.CacheDir, caps)[0]
	if m < 0 || 7218-m > limit {
		t.Fatalf("margin %d leaves the fit budget %d above the guard's %d: %v", m, 7218-m, limit, first)
	}
}

func TestReviewerSeatShrinksToItsMeasurementOnly(t *testing.T) {
	req := &launchRequest{ReviewerReservation: &placement.CompanionReservation{Name: "r", VRAMMB: 6144}}
	seatReviewerAtMeasurement(req, 0)
	seatReviewerAtMeasurement(req, 7000)
	if req.ReviewerReservation.VRAMMB != 6144 {
		t.Fatalf("an absent or larger measurement changed the seat to %d", req.ReviewerReservation.VRAMMB)
	}
	seatReviewerAtMeasurement(req, 5045)
	if req.ReviewerReservation.VRAMMB != 5045 {
		t.Fatalf("seat %d, want the measured 5045", req.ReviewerReservation.VRAMMB)
	}
	seatReviewerAtMeasurement(nil, 1)
	seatReviewerAtMeasurement(&launchRequest{}, 1)
}

// With the reviewer resident and seated at its measurement, the admission's view
// of the card matches the driver, so ik's own margin is already the guard's
// overhead and the argv needs no extra margin (the fresh-install 4070 case).
func TestMeasuredReviewerLeavesTheGuardAtTheDriversView(t *testing.T) {
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 12282, VRAMUsedMB: 409, PCIBusID: "00000000:17:00.0"}}}
	strategy := &placement.Strategy{CompanionPlacements: []placement.CompanionPlacement{{Name: "r", GPU: 0}}}
	req := &launchRequest{ReviewerReservation: &placement.CompanionReservation{Name: "r", VRAMMB: 6144}}
	seatReviewerAtMeasurement(req, 5045)
	live := func(int) int { return 409 + 5064 } // driver-reported use with the reviewer resident
	runtime, _ := runtimeGPUCapabilitiesWithUsage(caps, req, strategy, live)
	driverFree := 12282 - live(0)
	limit := allocationGuardLimitsMB(t.TempDir(), runtime)[0]
	if got := driverFree - limit; got > ikDefaultFitMarginMB {
		t.Fatalf("guard withholds %d MiB beyond ik's own margin with the reviewer measured", got-ikDefaultFitMarginMB)
	}
	args := guardedFitArgs([]string{"srv", "--fit"}, ikFitHelp, nil, []int{limit}, map[int]int{0: driverFree}, runtime.GPUs)
	if m := fitMarginOf(t, args); m != -1 {
		t.Fatalf("margin %d added although the guard matches the driver", m)
	}
}

// An argv built before the reviewer starts must not count the reviewer's seat
// as free: that turned into a 7168 MiB margin and put 522 MiB of a 6.9 GB model
// on the GPU (2026-10-14).
func TestUnstartedCompanionSeatIsNotCountedAsFree(t *testing.T) {
	gpus := []detect.GPU{{Index: 0, VRAMTotalMB: 12282, VRAMUsedMB: 409 + 6144}} // planned seat in the runtime view
	limit := 12282 - (409 + 6144) - placement.UnmeasuredCUDAOverheadMB
	args := guardedFitArgs([]string{"srv", "--fit"}, ikFitHelp, nil, []int{limit}, map[int]int{0: 11873}, gpus)
	if m := fitMarginOf(t, args); m != -1 {
		t.Fatalf("margin %d counts the unstarted reviewer's seat as free", m)
	}
}
