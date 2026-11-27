package placement

import (
	"sort"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// companionRestrictionCaps is a three-GPU host whose cards differ, so a seat on
// the wrong physical card shows in every assertion: 0 = 12 GB fast link,
// 1 = 24 GB fast link, 2 = 12 GB slowest link.
func companionRestrictionCaps() *detect.Capabilities {
	return strategyFitCaps(
		detect.GPU{Index: 0, VRAMTotalMB: 12288, VRAMUsedMB: 300, BandwidthMBps: 15760},
		detect.GPU{Index: 1, VRAMTotalMB: 24576, VRAMUsedMB: 500, BandwidthMBps: 15760},
		detect.GPU{Index: 2, VRAMTotalMB: 12288, VRAMUsedMB: 700, BandwidthMBps: 7880},
	)
}

// visibleIndexOf is the runtime (CUDA_VISIBLE_DEVICES) index of a physical
// card under a restriction: its position in the sorted selection.
func visibleIndexOf(restriction []int, physical int) int {
	if len(restriction) == 0 {
		return physical
	}
	sorted := append([]int(nil), restriction...)
	sort.Ints(sorted)
	for visible, idx := range sorted {
		if idx == physical {
			return visible
		}
	}
	return -1
}

// A companion's GPUPreference and its reported seat are physical device
// indexes, and the plan charges the reservation to that same physical card,
// under every --gpus restriction. Before the fix the preference was matched
// against the renumbered runtime view: --gpus 1 and --gpus 2 found no card,
// --gpus 1,2 seated on physical 2 for a preference of [1,2], and the seat was
// reported as a runtime index the launcher then read as physical.
func TestCompanionSeatIsPhysicalUnderGPURestriction(t *testing.T) {
	const reservation = 6144
	cases := []struct {
		name        string
		restriction []int
		preference  []int
		allowCPU    bool
		want        int // physical seat; -1 = CPU
	}{
		{name: "unrestricted preference", preference: []int{1}, want: 1},
		{name: "unrestricted automatic", want: 2},
		{name: "only 0 preferred", restriction: []int{0}, preference: []int{0}, want: 0},
		{name: "only 0 automatic", restriction: []int{0}, want: 0},
		{name: "only 1 preferred", restriction: []int{1}, preference: []int{1}, want: 1},
		{name: "only 1 automatic", restriction: []int{1}, want: 1},
		{name: "only 2 preferred", restriction: []int{2}, preference: []int{2}, want: 2},
		{name: "only 2 automatic", restriction: []int{2}, want: 2},
		{name: "1,2 preferring 1", restriction: []int{1, 2}, preference: []int{1, 2}, want: 1},
		{name: "1,2 preferring 2", restriction: []int{1, 2}, preference: []int{2, 1}, want: 2},
		{name: "1,2 automatic picks the slow link", restriction: []int{1, 2}, want: 2},
		{name: "0,2 preferring 0", restriction: []int{0, 2}, preference: []int{0, 2}, want: 0},
		{name: "0,2 preferring 2", restriction: []int{0, 2}, preference: []int{2, 0}, want: 2},
		{name: "0,2 automatic picks the slow link", restriction: []int{0, 2}, want: 2},
		{name: "preferred card outside the selection is skipped", restriction: []int{1, 2}, preference: []int{0, 2}, want: 2},
		{name: "no preferred card selected, CPU allowed", restriction: []int{1}, preference: []int{0}, allowCPU: true, want: -1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			caps := companionRestrictionCaps()
			opts := strategyFitOptions()
			opts.ContextSize = 8192
			opts.CacheDir = t.TempDir()
			opts.GPUs = c.restriction
			preference := append([]int(nil), c.preference...)
			opts.Companions = []CompanionReservation{{Name: "reviewer", VRAMMB: reservation, GPUPreference: preference, AllowCPU: c.allowCPU}}
			s, err := Compute(caps, strategyFitDenseModel(2000, 32768), opts)
			if err != nil {
				t.Fatalf("compute: %v", err)
			}
			if len(s.CompanionPlacements) != 1 || s.CompanionPlacements[0].Name != "reviewer" {
				t.Fatalf("companion placements %+v, want one reviewer seat", s.CompanionPlacements)
			}
			if got := s.CompanionPlacements[0].GPU; got != c.want {
				t.Fatalf("seat on physical %d, want %d", got, c.want)
			}
			if len(opts.Companions[0].GPUPreference) != len(c.preference) {
				t.Fatalf("caller's preference was rewritten: %v", opts.Companions[0].GPUPreference)
			}
			for i := range c.preference {
				if opts.Companions[0].GPUPreference[i] != c.preference[i] {
					t.Fatalf("caller's preference was rewritten: %v", opts.Companions[0].GPUPreference)
				}
			}
			// The plan's ledger is the runtime view: every selected card, and
			// the reservation on exactly the seat's runtime index.
			selected := c.restriction
			if len(selected) == 0 {
				selected = []int{0, 1, 2}
			}
			if len(s.PlanFreeVRAM) != len(selected) {
				t.Fatalf("plan ledger %v, want %d selected cards", s.PlanFreeVRAM, len(selected))
			}
			seatVisible := -1
			if c.want >= 0 {
				seatVisible = visibleIndexOf(c.restriction, c.want)
			}
			for _, physical := range selected {
				visible := visibleIndexOf(c.restriction, physical)
				want := caps.GPUs[physical].VRAMFreeMB()
				if visible == seatVisible {
					want -= reservation
				}
				if got, ok := s.PlanFreeVRAM[visible]; !ok || got != want {
					t.Fatalf("physical %d (runtime %d): plan free %d (present %v), want %d; ledger %v",
						physical, visible, got, ok, want, s.PlanFreeVRAM)
				}
			}
		})
	}
}

// A seat that cannot be found is still an error that names the companion: a
// card too small, and a preference that names no selected card (never widened
// to the planner's own choice).
func TestCompanionSeatFailureUnderGPURestrictionNamesTheCompanion(t *testing.T) {
	cases := []struct {
		name        string
		restriction []int
		preference  []int
		vramMB      int
		wantText    string
	}{
		{name: "selected card too small", restriction: []int{2}, preference: []int{2}, vramMB: 20000, wantText: "companion reviewer needs 20000 MB VRAM but no GPU has room"},
		{name: "automatic, no selected card fits", restriction: []int{0, 2}, vramMB: 20000, wantText: "companion reviewer needs 20000 MB VRAM but no GPU has room"},
		{name: "preferred selected card full, other selected card not used", restriction: []int{1, 2}, preference: []int{2}, vramMB: 13000, wantText: "companion reviewer needs 13000 MB VRAM but no GPU has room"},
		{name: "no preferred card selected", restriction: []int{1}, preference: []int{0}, vramMB: 6144, wantText: "none of its preferred GPUs [0] is in the --gpus selection"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			opts := strategyFitOptions()
			opts.ContextSize = 8192
			opts.CacheDir = t.TempDir()
			opts.GPUs = c.restriction
			opts.Companions = []CompanionReservation{{Name: "reviewer", VRAMMB: c.vramMB, GPUPreference: c.preference}}
			s, err := Compute(companionRestrictionCaps(), strategyFitDenseModel(2000, 32768), opts)
			if err == nil {
				t.Fatalf("planned %+v, want a companion seat error", s.CompanionPlacements)
			}
			if !strings.Contains(err.Error(), c.wantText) || !strings.Contains(err.Error(), "but no GPU has room") {
				t.Fatalf("error %q, want it to contain %q", err, c.wantText)
			}
		})
	}
}

// The runtime/physical maps are identity-free: an unrestricted plan gets no
// map (so nothing is translated), and a restricted one maps each renumbered
// index back to its card.
func TestRestrictGPUsWithPhysicalMapsRuntimeToPhysical(t *testing.T) {
	caps := companionRestrictionCaps()
	if out, m, err := restrictGPUsWithPhysical(caps, nil); err != nil || out != caps || m != nil {
		t.Fatalf("unrestricted: caps changed=%v map=%v err=%v", out != caps, m, err)
	}
	out, m, err := restrictGPUsWithPhysical(caps, []int{2, 0})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.GPUs) != 2 || out.GPUs[0].Index != 0 || out.GPUs[1].Index != 1 ||
		out.GPUs[0].VRAMUsedMB != 300 || out.GPUs[1].VRAMUsedMB != 700 {
		t.Fatalf("restricted view %+v, want physical 0 then 2 renumbered 0,1", out.GPUs)
	}
	if len(m) != 2 || m[0] != 0 || m[1] != 2 {
		t.Fatalf("runtime->physical %v, want map[0:0 1:2]", m)
	}
	seats := companionSeatsOnPhysicalGPUs([]CompanionPlacement{{Name: "a", GPU: 1}, {Name: "b", GPU: -1}}, m)
	if seats[0].GPU != 2 || seats[1].GPU != -1 {
		t.Fatalf("seats %+v, want runtime 1 -> physical 2 and the CPU seat kept", seats)
	}
}
