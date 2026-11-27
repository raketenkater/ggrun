package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/advisor"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// companionSeatHost is the golden matrix's mixed3 inventory: physical 0 = 12 GB,
// 1 = 24 GB, 2 = 12 GB on the slowest link, 300 MiB used on each.
func companionSeatHost(t *testing.T) *detect.Capabilities {
	t.Helper()
	inventory := filepath.Join(t.TempDir(), "inventory.json")
	if err := os.WriteFile(inventory, []byte(goldenPlanInventories["mixed3"]), 0o600); err != nil {
		t.Fatal(err)
	}
	caps, err := loadInventory(inventory)
	if err != nil {
		t.Fatal(err)
	}
	if len(caps.GPUs) != 3 {
		t.Fatalf("inventory has %d GPUs, want 3", len(caps.GPUs))
	}
	return caps
}

// companionSeatDriverCSV is nvidia-smi's "pci.bus_id, memory.free" for that
// host before the reviewer starts.
const companionSeatDriverCSV = "00000000:01:00.0, 11988\n00000000:02:00.0, 24276\n00000000:03:00.0, 11988\n"

// companionSeatRequest is a --claude-code launch with a typed context and the
// reviewer reservation built as claudeReviewerReservation builds it (fixed
// 6144 MiB, preference from the production candidate list).
func companionSeatRequest(caps *detect.Capabilities, gpus string, model *placement.ModelProfile) *launchRequest {
	defaults := config.Defaults()
	req := &launchRequest{
		ModelPath: model.Path, Port: 8081, CtxFlag: "32768",
		KVPlacement: defaults.KVPlacement, KVQuality: defaults.KVQuality,
		Parallel: 1, ParallelSet: true, RAMLimitPercent: 95,
		ClaudeCode: true, GPUsFlag: gpus,
	}
	req.ReviewerReservation = &placement.CompanionReservation{
		Name: fitBindingReviewer, VRAMMB: 6144,
		GPUPreference: claudeReviewerGPUCandidates(caps, req), AllowCPU: false,
	}
	return req
}

// The reviewer seat, the card the plan charges and the --fit-margin binding's
// runtime view agree on one physical card for every --gpus selection, through
// the real placement.Compute and buildLaunchServerArgs. Before the fix --gpus 1
// and 2 failed with "no GPU has room", and --gpus 1,2 charged physical 2 while
// the seat (read as physical by the launcher) named physical 1.
func TestCompanionSeatAgreesThroughLaunchArgsUnderGPURestriction(t *testing.T) {
	const reservation = 6144
	for _, c := range []struct {
		gpus     string
		wantSeat int
	}{
		{gpus: "", wantSeat: 2}, // unrestricted: slowest link first, the 24 GB main card last
		{gpus: "0", wantSeat: 0},
		{gpus: "1", wantSeat: 1},
		{gpus: "2", wantSeat: 2},
		{gpus: "1,2", wantSeat: 1},
		{gpus: "2,1", wantSeat: 2}, // the user's order is the preference; visibility stays sorted
		{gpus: "0,2", wantSeat: 0},
	} {
		t.Run("gpus="+c.gpus, func(t *testing.T) {
			p := stubFitBindingHost(t, fitBindingDriver(companionSeatDriverCSV))
			caps := companionSeatHost(t)
			// About 49 GB: larger than every selection, so the plan is the
			// backend-fit DenseCPUOffload the --fit-margin binding applies to.
			model := goldenPlanDense("synthetic/dense-xl.gguf", 80, 8192, 64, 28672, 600<<20, 131072)
			be := goldenPlanBackend(t, "ik")
			cacheDir := t.TempDir()
			req := companionSeatRequest(caps, c.gpus, model)

			strategy, err := placement.Compute(caps, model, placementOptionsFromRequestCaps(req, model, be, cacheDir, caps))
			if err != nil {
				t.Fatalf("placement: %v", err)
			}
			if strategy.Type != placement.DenseCPUOffload {
				t.Fatalf("strategy %s, want the backend-fit DenseCPUOffload this fixture is built for", strategy.Type)
			}

			selected := []int{0, 1, 2}
			if c.gpus != "" {
				var perr error
				if selected, perr = parseGPUIndices(c.gpus); perr != nil {
					t.Fatal(perr)
				}
			}
			sort.Ints(selected)

			seat := -2
			for _, cp := range strategy.CompanionPlacements {
				if cp.Name == fitBindingReviewer {
					seat = cp.GPU
				}
			}
			if seat != c.wantSeat {
				t.Fatalf("reviewer seat on physical %d, want %d (placements %+v)", seat, c.wantSeat, strategy.CompanionPlacements)
			}
			if visibleGPUForPhysical(req, seat) < 0 {
				t.Fatalf("seat physical %d is outside --gpus %q", seat, c.gpus)
			}

			// The plan's ledger is the backend-visible view of the selection;
			// the reservation sits on the seat's visible index only.
			if len(strategy.PlanFreeVRAM) != len(selected) {
				t.Fatalf("plan ledger %v, want %d visible cards", strategy.PlanFreeVRAM, len(selected))
			}
			for visible, physical := range selected {
				want := caps.GPUs[physical].VRAMFreeMB()
				if physical == seat {
					want -= reservation
				}
				if got := strategy.PlanFreeVRAM[visible]; got != want {
					t.Fatalf("plan charges visible %d (physical %d) free %d, want %d; ledger %v",
						visible, physical, got, want, strategy.PlanFreeVRAM)
				}
			}

			args := buildLaunchServerArgs(req, &config.Config{CacheDir: cacheDir}, be, caps, model, strategy)
			if !backendFitsItself(args) {
				t.Fatalf("argv does not hand layout to the backend's --fit, so the binding was not exercised: %v", args)
			}
			if len(p.runtime) == 0 {
				t.Fatal("the --fit-margin binding never read driver free VRAM")
			}
			view := p.runtime[len(p.runtime)-1]
			if len(view.GPUs) != len(selected) {
				t.Fatalf("binding view %+v, want %d visible cards", view.GPUs, len(selected))
			}
			for visible, physical := range selected {
				g := view.GPUs[visible]
				wantUsed := caps.GPUs[physical].VRAMUsedMB
				if physical == seat {
					wantUsed += reservation
				}
				if g.Index != visible || g.PCIBusID != caps.GPUs[physical].PCIBusID || g.VRAMUsedMB != wantUsed {
					t.Fatalf("binding view CUDA%d = %+v, want physical %d (%s) with used %d",
						visible, g, physical, caps.GPUs[physical].PCIBusID, wantUsed)
				}
			}
			for physical := range p.usedQueried {
				if visibleGPUForPhysical(req, physical) < 0 {
					t.Fatalf("live usage read for unselected physical %d", physical)
				}
			}
			if len(p.usedQueried) != len(selected) {
				t.Fatalf("live usage read for %v, want the selection %v", p.usedQueried, selected)
			}
		})
	}
}

// The post-launch probe reads cards by backend-visible index, so a physical
// seat is netted on its visible index, and a seat outside the selection on none.
func TestCompanionVRAMNettingUsesVisibleIndex(t *testing.T) {
	cacheDir := t.TempDir()
	if err := placement.RecordCompanionVRAM(cacheDir, fitBindingReviewer, 2916); err != nil {
		t.Fatal(err)
	}
	strategy := &placement.Strategy{CompanionPlacements: []placement.CompanionPlacement{
		{Name: fitBindingReviewer, GPU: 2}, {Name: "cpu-helper", GPU: -1},
	}}
	for _, c := range []struct {
		gpus string
		want map[int]int
	}{
		{gpus: "", want: map[int]int{2: 2916}},
		{gpus: "1,2", want: map[int]int{1: 2916}},
		{gpus: "2", want: map[int]int{0: 2916}},
		{gpus: "0,1", want: map[int]int{}},
	} {
		got := companionVRAMByVisibleGPU(&launchRequest{GPUsFlag: c.gpus}, cacheDir, strategy)
		if len(got) != len(c.want) {
			t.Fatalf("--gpus %q: netting %v, want %v", c.gpus, got, c.want)
		}
		for k, v := range c.want {
			if got[k] != v {
				t.Fatalf("--gpus %q: netting %v, want %v", c.gpus, got, c.want)
			}
		}
	}
}

func stubPlacementAdvisorConsentIO(t *testing.T, answer string, interactive bool) *bytes.Buffer {
	t.Helper()
	out := &bytes.Buffer{}
	saved := placementAdvisorConsentIO
	placementAdvisorConsentIO = func() (io.Reader, io.Writer, bool) {
		return strings.NewReader(answer), out, interactive
	}
	t.Cleanup(func() { placementAdvisorConsentIO = saved })
	return out
}

func stubSupportIncidentCount(t *testing.T) *int {
	t.Helper()
	called := 0
	saved := runSupportIncidentFn
	runSupportIncidentFn = func(_ context.Context, _ *config.Config, _ *detect.Capabilities, _ string, _ advisor.Incident, _ bool) (advisor.Decision, advisor.RunReport, error) {
		called++
		return advisor.Decision{}, advisor.RunReport{}, errors.New("helper must not run in this test")
	}
	t.Cleanup(func() { runSupportIncidentFn = saved })
	return &called
}

// A reviewer seat that cannot be found is classified, deterministic, and never
// reaches the support advisor, so the launch prints the placement error itself.
// The error comes from the real launch planner so text and classifier cannot
// drift apart.
func TestCompanionSeatFailureIsClassifiedAndNeverConsultsAdvisor(t *testing.T) {
	t.Setenv("GGRUN_ADVISOR_CONSENT", "on") // consent would approve; the gate must still block
	stubFitBindingHost(t, fitBindingDriver(companionSeatDriverCSV))
	caps := companionSeatHost(t)
	model := goldenPlanDense("synthetic/dense-small.gguf", 32, 4096, 32, 14336, 130<<20, 131072)
	be := goldenPlanBackend(t, "ik")
	req := companionSeatRequest(caps, "2", model)
	req.CtxFlag = "" // automatic context: the error arrives wrapped, as on the launch
	req.ReviewerReservation.VRAMMB = 30000

	_, seatErr := placement.Compute(caps, model, placementOptionsFromRequestCaps(req, model, be, t.TempDir(), caps))
	if seatErr == nil || !strings.Contains(seatErr.Error(), "companion "+fitBindingReviewer+" needs 30000 MB VRAM but no GPU has room") {
		t.Fatalf("placement error %v, want the reviewer seat failure", seatErr)
	}
	code := classifyAdvisorFailure(seatErr)
	if code != "companion_seat_unavailable" {
		t.Fatalf("seat failure classified %q, want companion_seat_unavailable", code)
	}
	if shouldEscalateToAdvisor(code, escalationSignal{retriesExhausted: true, sameCodeRecurred: true}) {
		t.Fatal("a companion seat failure escalated to the advisor")
	}

	out := stubPlacementAdvisorConsentIO(t, "y\n", true)
	called := stubSupportIncidentCount(t)
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	cfg.SupportExpert = "on"
	computeStrategy := func(*launchRequest) (*placement.Strategy, error) {
		t.Fatal("placement recomputed for a deterministic seat failure")
		return nil, nil
	}
	got, err := escalatePlacementFailure(req, cfg, model, be, caps, code, seatErr, computeStrategy)
	if !errors.Is(err, seatErr) || got != nil {
		t.Fatalf("escalation returned %+v, %v; want the original seat error", got, err)
	}
	if *called != 0 {
		t.Fatalf("support helper consulted %d times for a seat failure", *called)
	}
	if strings.Contains(out.String(), "Support advisor") {
		t.Fatalf("advisor prompt shown for a deterministic seat failure: %q", out.String())
	}

	// The final error names what the user can do, on both launch exits.
	for _, prefix := range []string{"Error computing placement", "Error computing resident placement"} {
		report := placementFailureReport(prefix, req, err)
		if !strings.HasPrefix(report, prefix+": "+seatErr.Error()) ||
			!strings.Contains(report, "\n[claude-code] ") || !strings.Contains(report, "--claude-reviewer off") ||
			!strings.Contains(report, "--gpus selection with room") {
			t.Fatalf("seat failure report lacks the cause or the reviewer hint:\n%s", report)
		}
	}
	// No hint without a reviewer reservation, or for any other placement error.
	noReviewer := *req
	noReviewer.ReviewerReservation = nil
	if report := placementFailureReport("Error computing placement", &noReviewer, seatErr); strings.Contains(report, "--claude-reviewer") {
		t.Fatalf("reviewer hint without a reviewer reservation: %s", report)
	}
	for _, other := range []error{
		errors.New("automatic context fit found no admissible plan between 4096 and 8192 tokens"),
		errors.New("memory preflight failed: too tight"),
	} {
		if report := placementFailureReport("Error computing placement", req, other); strings.Contains(report, "--claude-reviewer") {
			t.Fatalf("reviewer hint for a non-seat failure: %s", report)
		}
	}
}

// When a placement failure does reach the advisor gate, the user sees the
// placement error before the consent prompt (interactive) or before the rerun
// hint (non-interactive), and a decline returns the original error.
func TestPlacementErrorPrecedesAdvisorPrompt(t *testing.T) {
	t.Setenv("GGRUN_ADVISOR_CONSENT", "prompt")
	cfg := config.Defaults()
	cfg.CacheDir = t.TempDir()
	cfg.SupportExpert = "on"
	req := &launchRequest{CtxFlag: "131072", KVPlacement: "gpu", KVQuality: "q8_0", Parallel: 1}
	model := &placement.ModelProfile{ModelArch: "llama"}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, Name: "GPU", VRAMTotalMB: 24576}}, RAM: detect.RAMInfo{TotalMB: 128000, FreeMB: 64000}}
	be := &backendInfo{Tag: "llama", Identity: "build"}
	novel := errors.New("automatic context fit found no admissible plan: some novel planner refusal")
	if code := classifyAdvisorFailure(novel); code != "unclassified_launch_failure" {
		t.Fatalf("fixture error classified %q, want it unclassified so the prompt is shown", code)
	}
	computeStrategy := func(*launchRequest) (*placement.Strategy, error) {
		t.Fatal("placement recomputed after a declined consultation")
		return nil, nil
	}

	for _, interactive := range []bool{true, false} {
		out := stubPlacementAdvisorConsentIO(t, "n\n", interactive)
		called := stubSupportIncidentCount(t)
		got, err := escalatePlacementFailure(req, cfg, model, be, caps, "unclassified_launch_failure", novel, computeStrategy)
		if !errors.Is(err, novel) || got != nil {
			t.Fatalf("interactive=%v: escalation returned %+v, %v; want the original error", interactive, got, err)
		}
		if *called != 0 {
			t.Fatalf("interactive=%v: helper consulted %d times after a decline", interactive, *called)
		}
		text := out.String()
		cause := strings.Index(text, "[placement] "+novel.Error())
		ask := strings.Index(text, "Support advisor wants to")
		if !interactive {
			ask = strings.Index(text, "support advisor would")
		}
		if cause < 0 || ask < 0 || cause > ask {
			t.Fatalf("interactive=%v: cause at %d, advisor text at %d, want the cause first:\n%s", interactive, cause, ask, text)
		}
		// The launch's final line refers to the printed cause instead of
		// repeating it.
		report := placementFailureReport("Error computing placement", req, err)
		if strings.Contains(report, novel.Error()) || !strings.Contains(report, "see the [placement] line above") {
			t.Fatalf("interactive=%v: final error repeats or loses the cause: %s", interactive, report)
		}
	}
	// A cause the advisor never printed is still reported in full.
	if report := placementFailureReport("Error computing placement", req, novel); report != "Error computing placement: "+novel.Error() {
		t.Fatalf("unshown cause reported as %q", report)
	}
}
