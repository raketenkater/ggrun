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

// fitBindingCaps is a three-GPU host. The launch is restricted with --gpus 1,2,
// so the backend sees physical 1 as CUDA0 and physical 2 as CUDA1. Physical 0
// has the largest would-be gap: a margin taken over the physical inventory
// instead of the runtime devices would be 9216, not 4096.
func fitBindingCaps() *detect.Capabilities {
	return &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "GPU A", PCIBusID: "00000000:01:00.0", VRAMTotalMB: 12282, VRAMUsedMB: 300, VRAMReservedMB: 8000},
			{Index: 1, Name: "GPU B", PCIBusID: "00000000:02:00.0", VRAMTotalMB: 24564, VRAMUsedMB: 500, VRAMReservedMB: 1000},
			{Index: 2, Name: "GPU C", PCIBusID: "00000000:03:00.0", VRAMTotalMB: 12288, VRAMUsedMB: 400, VRAMReservedMB: 3000},
		},
		RAM: detect.RAMInfo{TotalMB: 262144, FreeMB: 200000},
		CPU: detect.CPUInfo{Cores: 24},
	}
}

// fitBindingDriverCSV is nvidia-smi's "pci.bus_id, memory.free" for the three
// physical GPUs, before the reviewer has started.
const fitBindingDriverCSV = "00000000:01:00.0, 11900\n00000000:02:00.0, 23000\n00000000:03:00.0, 8000\n"

const fitBindingReviewer = "claude-auto-reviewer"

func fitBindingIK() *backendInfo {
	return &backendInfo{Path: "/x/ik_llama-server-cuda", Dialect: "ik_llama", Tag: "ik_llama", IsIK: true, Identity: "test", Help: goldenPlanIKHelp}
}

// fitBindingPlan is a backend-fit plan under --gpus 1,2 with the reviewer's
// 6144 MiB seat planned on physical GPU seatPhysical (-1: on the CPU).
func fitBindingPlan(seatPhysical int) (*launchRequest, *placement.Strategy) {
	req := &launchRequest{CtxFlag: "fit", Parallel: 1, ParallelSet: true, RAMLimitPercent: 95, GPUsFlag: "1,2",
		ReviewerReservation: &placement.CompanionReservation{Name: fitBindingReviewer, VRAMMB: 6144}}
	s := &placement.Strategy{Type: placement.DenseCPUOffload, BackendSupportsFit: true, ContextSize: 32768, Parallel: 1, BatchSize: 2048, UBatchSize: 512,
		CompanionPlacements: []placement.CompanionPlacement{{Name: fitBindingReviewer, GPU: seatPhysical}}}
	return req, s
}

// fitBindingProbe records what the binding asked of the host.
type fitBindingProbe struct {
	runtime     []detect.Capabilities // the views handed to the driver reading
	usedQueried map[int]bool          // physical indexes whose live usage was read
}

// stubFitBindingHost replaces both live readings, so no nvidia-smi runs, and
// points the home directory at a temp dir: with no system probe in the cache
// dir, the overhead lookup would otherwise search ~/.cache/ggrun.
func stubFitBindingHost(t *testing.T, driver func(*detect.Capabilities) map[int]int) *fitBindingProbe {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	p := &fitBindingProbe{usedQueried: map[int]bool{}}
	savedUsed, savedFree := runtimeVRAMUsedMB, fitDriverFreeVRAM
	runtimeVRAMUsedMB = func(physical int) int {
		p.usedQueried[physical] = true
		return 0
	}
	fitDriverFreeVRAM = func(runtime *detect.Capabilities) map[int]int {
		if runtime != nil {
			view := *runtime
			view.GPUs = append([]detect.GPU(nil), runtime.GPUs...)
			p.runtime = append(p.runtime, view)
		}
		return driver(runtime)
	}
	t.Cleanup(func() { runtimeVRAMUsedMB, fitDriverFreeVRAM = savedUsed, savedFree })
	return p
}

func fitBindingDriver(csv string) func(*detect.Capabilities) map[int]int {
	return func(runtime *detect.Capabilities) map[int]int {
		if runtime == nil {
			return nil
		}
		return freeVRAMByRuntimeDevice(csv, runtime.GPUs)
	}
}

// fitDev is one runtime device as the binding's formula reads it; used
// includes a planned companion seat.
type fitDev struct{ total, used, reserved, driverFree int }

// documentedFitMargin is guardedFitArgs' documented margin written out
// independently: ceil256 of the largest min(driverFree, total-used) - limit,
// with limit = total - used - reserved - the unmeasured CUDA overhead, or -1
// when that gap is within ik's default.
func documentedFitMargin(devs []fitDev) int {
	gap := 0
	for _, d := range devs {
		free := min(d.driverFree, d.total-d.used)
		gap = max(gap, free-(d.total-d.used-d.reserved-placement.UnmeasuredCUDAOverheadMB))
	}
	if gap <= ikDefaultFitMarginMB {
		return -1
	}
	return (gap + fitMarginStepMB - 1) / fitMarginStepMB * fitMarginStepMB
}

func fitMarginCount(args []string) int {
	n := 0
	for _, a := range args {
		if a == fitMarginFlag {
			n++
		}
	}
	return n
}

func hasArgPair(args []string, name, value string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == name && args[i+1] == value {
			return true
		}
	}
	return false
}

// The ik --fit-margin binding, end to end through buildLaunchServerArgs on a
// renumbered --gpus view. The launch-plan golden never reaches the margin
// branch (its stubbed driver free equals the inventory's), so this pins the
// binding before the launch path is restructured.
//
// Runtime view with the reviewer seated on physical 2:
//
//	CUDA0 = physical 1: limit 24564-500-1000-1024 = 22040, seen free min(23000, 24064) = 23000, gap 960
//	CUDA1 = physical 2: limit 12288-6544-3000-1024 = 1720, seen free min(8000, 5744) = 5744, gap 4024
//
// margin = ceil256(4024) = 4096. With the seat charged to physical 1 instead it
// is 2048, with no seat no margin, and over all three physical GPUs 9216.
func TestFitMarginBindingThroughBuildLaunchServerArgs(t *testing.T) {
	seatOn2 := []fitDev{{24564, 500, 1000, 23000}, {12288, 400 + 6144, 3000, 8000}}
	if got := documentedFitMargin(seatOn2); got != 4096 {
		t.Fatalf("fixture: documented margin %d, want the hand-computed 4096", got)
	}
	model := fitTestModel(262144, 9000)
	build := func(cfg *config.Config, be *backendInfo, req *launchRequest, s *placement.Strategy) []string {
		return buildLaunchServerArgs(req, cfg, be, fitBindingCaps(), model, s)
	}
	forbidden := func(t *testing.T) func(*detect.Capabilities) map[int]int {
		return func(*detect.Capabilities) map[int]int {
			t.Errorf("driver free VRAM read although the binding cannot apply")
			return nil
		}
	}

	t.Run("margin is the formula over the runtime devices", func(t *testing.T) {
		p := stubFitBindingHost(t, fitBindingDriver(fitBindingDriverCSV))
		req, s := fitBindingPlan(2)
		args := build(&config.Config{CacheDir: t.TempDir()}, fitBindingIK(), req, s)
		if m := fitMarginOf(t, args); m != documentedFitMargin(seatOn2) {
			t.Fatalf("margin %d, want %d: %v", m, documentedFitMargin(seatOn2), args)
		}
		if len(p.runtime) == 0 {
			t.Fatal("the binding never read driver free VRAM")
		}
		view := p.runtime[len(p.runtime)-1]
		if len(view.GPUs) != 2 || view.GPUs[0].Index != 0 || view.GPUs[1].Index != 1 ||
			view.GPUs[0].PCIBusID != "00000000:02:00.0" || view.GPUs[1].PCIBusID != "00000000:03:00.0" {
			t.Fatalf("driver read for %+v, want CUDA0 = physical 1 and CUDA1 = physical 2", view.GPUs)
		}
		if p.usedQueried[0] || !p.usedQueried[1] || !p.usedQueried[2] {
			t.Fatalf("live usage read for physical %v, want 1 and 2 only", p.usedQueried)
		}
	})

	t.Run("three rebuilds give one identical margin", func(t *testing.T) {
		stubFitBindingHost(t, fitBindingDriver(fitBindingDriverCSV))
		req, s := fitBindingPlan(2)
		cfg := &config.Config{CacheDir: t.TempDir()}
		first := build(cfg, fitBindingIK(), req, s)
		for i := 0; i < 3; i++ {
			args := build(cfg, fitBindingIK(), req, s)
			if formatCommand(args) != formatCommand(first) {
				t.Fatalf("rebuild %d differs:\n%v\n%v", i+1, first, args)
			}
			if n := fitMarginCount(args); n != 1 {
				t.Fatalf("rebuild %d carries %d %s: %v", i+1, n, fitMarginFlag, args)
			}
		}
	})

	t.Run("a user margin is kept", func(t *testing.T) {
		stubFitBindingHost(t, fitBindingDriver(fitBindingDriverCSV))
		req, s := fitBindingPlan(2)
		req.ExtraArgs = []string{fitMarginFlag, "300"}
		args := build(&config.Config{CacheDir: t.TempDir()}, fitBindingIK(), req, s)
		if n, m := fitMarginCount(args), fitMarginOf(t, args); n != 1 || m != 300 {
			t.Fatalf("%d %s, value %d; want the user's single 300: %v", n, fitMarginFlag, m, args)
		}
	})

	t.Run("the seat is charged to its physical index", func(t *testing.T) {
		for _, c := range []struct {
			seat     int
			devs     []fitDev
			want     int
			wantUsed [2]int
		}{
			{seat: 2, devs: seatOn2, want: 4096, wantUsed: [2]int{500, 400 + 6144}},
			{seat: 1, devs: []fitDev{{24564, 500 + 6144, 1000, 23000}, {12288, 400, 3000, 8000}}, want: 2048, wantUsed: [2]int{500 + 6144, 400}},
			{seat: -1, devs: []fitDev{{24564, 500, 1000, 23000}, {12288, 400, 3000, 8000}}, want: -1, wantUsed: [2]int{500, 400}},
		} {
			if got := documentedFitMargin(c.devs); got != c.want {
				t.Fatalf("fixture seat %d: documented margin %d, want %d", c.seat, got, c.want)
			}
			p := stubFitBindingHost(t, fitBindingDriver(fitBindingDriverCSV))
			req, s := fitBindingPlan(c.seat)
			args := build(&config.Config{CacheDir: t.TempDir()}, fitBindingIK(), req, s)
			if m := fitMarginOf(t, args); m != c.want {
				t.Errorf("seat on physical %d: margin %d, want %d: %v", c.seat, m, c.want, args)
			}
			if len(p.runtime) == 0 {
				t.Fatalf("seat on physical %d: the binding never read driver free VRAM", c.seat)
			}
			view := p.runtime[len(p.runtime)-1]
			if len(view.GPUs) != 2 || view.GPUs[0].VRAMUsedMB != c.wantUsed[0] || view.GPUs[1].VRAMUsedMB != c.wantUsed[1] {
				t.Errorf("seat on physical %d: runtime view %+v, want used %v", c.seat, view.GPUs, c.wantUsed)
			}
		}
	})

	t.Run("no margin for mainline help", func(t *testing.T) {
		stubFitBindingHost(t, forbidden(t))
		req, s := fitBindingPlan(2)
		s.BackendFitTakesValue = true
		be := &backendInfo{Path: "/x/llama-server", Dialect: "llama", Tag: "llama", Identity: "test", Help: goldenPlanMainlineHelp + goldenPlanMainlineFitHelp}
		args := build(&config.Config{CacheDir: t.TempDir()}, be, req, s)
		if !hasArgPair(args, "--fit", "on") || fitMarginCount(args) != 0 {
			t.Fatalf("mainline argv %v, want --fit on and no %s", args, fitMarginFlag)
		}
	})

	t.Run("no margin for fit off", func(t *testing.T) {
		stubFitBindingHost(t, forbidden(t))
		req, s := fitBindingPlan(2)
		s.Type, s.TensorSplit, s.BackendFitTakesValue = placement.MultiGPUDense, []float64{0.5, 0.5}, true
		args := build(&config.Config{CacheDir: t.TempDir()}, fitBindingIK(), req, s)
		if !hasArgPair(args, "--fit", "off") || fitMarginCount(args) != 0 {
			t.Fatalf("explicit layout argv %v, want --fit off and no %s", args, fitMarginFlag)
		}
	})

	for name, driver := range map[string]func(*detect.Capabilities) map[int]int{
		"no margin when the driver is unreadable": func(*detect.Capabilities) map[int]int { return nil },
		"no margin when a device is not sampled":  fitBindingDriver("00000000:01:00.0, 11900\n00000000:02:00.0, 23000\n"),
	} {
		t.Run(name, func(t *testing.T) {
			p := stubFitBindingHost(t, driver)
			req, s := fitBindingPlan(2)
			args := build(&config.Config{CacheDir: t.TempDir()}, fitBindingIK(), req, s)
			if len(p.runtime) == 0 {
				t.Fatal("the binding never read driver free VRAM")
			}
			if !hasArg(args, "--fit") || fitMarginCount(args) != 0 {
				t.Fatalf("argv %v, want --fit and no %s", args, fitMarginFlag)
			}
		})
	}
}
