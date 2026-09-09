package placement

import (
	"fmt"
	"sort"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Residency occupancy answers a question the fit stage never asked: not "is
// every GPU participating?" but "is every GPU actually carrying weight?".
//
// numGPUsExcluded counts a GPU as used when its tensor-split share maps to at
// least one dense layer. Measured 2026-09-02 on GLM 5.3 Flash (137 GB of
// weights, 48 GB of VRAM): CUDA2 held a 0.06 share -- 2.7 dense layers, which
// passes that test -- and finished the run at 496 MiB of 12288, a 0.61 MiB
// compute buffer, and 0% SM. All GPU-resident experts sat on a single owner,
// 18.1 GB of VRAM (37% of the rig) was idle, and the placement reported three
// of three GPUs in use. Participation is not occupancy.
//
// This file computes occupancy from the backend's own measured ledger and
// reports how many further whole routed-expert layers each device could seat.
// It deliberately produces no placement of its own: it is the evidence a
// re-pack needs, and the detector that says a plan has not reached the
// hardware's stable maximum.

// ExpertSeat is one device's occupancy against its own capacity.
type ExpertSeat struct {
	GPUIndex int
	// CapacityMB is the device total as the detector reported it.
	CapacityMB int
	// OccupiedMB is model + context + unaccounted from the measured ledger.
	OccupiedMB int
	// MarginMB is the measured runtime growth held back for this device.
	// It is never a fraction of anything: see marginByGPU.
	MarginMB int
	// EntryCostMB is what this device must start paying to own expert layers
	// when it does not already, charged at what a real owner was measured to
	// pay on this same launch.
	EntryCostMB int
	// Seats is how many further whole routed-expert layers fit after margin
	// and entry cost. Zero when the device's margin is unmeasured.
	Seats int
	// MarginMeasured is false when no exact runtime-growth evidence exists for
	// this device, in which case Seats is forced to zero (invariant 4).
	MarginMeasured bool
}

// FreeMB reports the unclaimed VRAM on this device before margin/entry cost.
func (s ExpertSeat) FreeMB() int {
	if free := s.CapacityMB - s.OccupiedMB; free > 0 {
		return free
	}
	return 0
}

// OccupancyPct reports how much of the device is carrying anything at all.
func (s ExpertSeat) OccupancyPct() int {
	if s.CapacityMB <= 0 {
		return 0
	}
	return int(float64(s.OccupiedMB) * 100 / float64(s.CapacityMB))
}

// ExpertSeatReport is the whole-rig occupancy picture for one launch.
type ExpertSeatReport struct {
	Seats []ExpertSeat
	// ExpertLayerMB is the largest routed-expert layer, the unit a seat is
	// counted in. A seat is whole-layer because the planner emits whole expert
	// layers only (see enableAutomaticSubLayerExpertPins).
	ExpertLayerMB int
	// ExpertsOnCPU is how many routed-expert layers the plan left on the host.
	ExpertsOnCPU int
	// IdleMB is total unclaimed VRAM across every device.
	IdleMB int
	// CapacityMB is the rig total.
	CapacityMB int
}

// TotalSeats is how many further expert layers the rig could seat.
func (r ExpertSeatReport) TotalSeats() int {
	n := 0
	for _, s := range r.Seats {
		n += s.Seats
	}
	return n
}

// UnderPacked reports whether the rig could seat at least one more expert layer
// while expert layers remain on the host. This is the "not at stable max"
// predicate: it is pure arithmetic over measured bytes and the model's own
// layer size, so it carries no rig-specific constant and behaves the same on
// any hardware (invariant 8).
func (r ExpertSeatReport) UnderPacked() bool {
	return r.ExpertsOnCPU > 0 && r.TotalSeats() > 0
}

// Summary renders the report for a launch banner or a recorded decision.
func (r ExpertSeatReport) Summary() string {
	if len(r.Seats) == 0 {
		return "no GPU occupancy evidence"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "occupancy %d/%d MiB (%d%%), %d MiB idle",
		r.CapacityMB-r.IdleMB, r.CapacityMB,
		pctOf(r.CapacityMB-r.IdleMB, r.CapacityMB), r.IdleMB)
	for _, s := range r.Seats {
		fmt.Fprintf(&b, "; CUDA%d %d%%", s.GPUIndex, s.OccupancyPct())
		switch {
		case !s.MarginMeasured:
			b.WriteString(" (margin unmeasured, no seats offered)")
		case s.Seats > 0:
			fmt.Fprintf(&b, " (+%d seat(s))", s.Seats)
		}
	}
	return b.String()
}

func pctOf(part, whole int) int {
	if whole <= 0 {
		return 0
	}
	return int(float64(part) * 100 / float64(whole))
}

// ComputeExpertSeats builds the occupancy report from a measured allocation.
//
// Every quantity is measured or model-derived; nothing here invents a reserve:
//
//   - occupancy comes from the backend's own ledger for this exact launch,
//   - the margin is measured runtime graph growth for this device, and a
//     device with no exact growth evidence is offered zero seats rather than a
//     guessed margin (invariant 4 -- estimated fit only advises),
//   - the entry cost a currently expert-free device must pay is charged at what
//     an expert-owning device was measured to pay on this same launch, so it is
//     an observation of this rig rather than a formula about GPUs in general.
func ComputeExpertSeats(cacheDir string, caps *detect.Capabilities, model *ModelProfile,
	s *Strategy, alloc MeasuredAllocation, parallel int, backendTag string,
) ExpertSeatReport {
	report := ExpertSeatReport{}
	if caps == nil || model == nil || len(caps.GPUs) == 0 {
		return report
	}
	report.ExpertLayerMB = LargestRoutedExpertLayerMB(model)
	if s != nil {
		report.ExpertsOnCPU = s.NCPUMoE
	}
	margins := marginByGPU(cacheDir, model, caps.GPUs, parallel, backendTag)
	entry := entryCostByGPU(caps.GPUs, alloc)

	for _, g := range caps.GPUs {
		occ := alloc.ModelByGPU[g.Index] + alloc.ContextByGPU[g.Index] + alloc.UnaccountedByGPU[g.Index]
		seat := ExpertSeat{
			GPUIndex:    g.Index,
			CapacityMB:  g.VRAMTotalMB,
			OccupiedMB:  occ,
			EntryCostMB: entry[g.Index],
		}
		margin, measured := margins[g.Index]
		seat.MarginMB, seat.MarginMeasured = margin, measured
		// Fail closed: an unmeasured margin buys no seats. The alternative --
		// a default margin -- is exactly the static fudge reserve invariant 8
		// forbids, and it is what a packer would lean on hardest.
		if measured && report.ExpertLayerMB > 0 {
			usable := seat.FreeMB() - seat.MarginMB - seat.EntryCostMB
			if usable > 0 {
				seat.Seats = usable / report.ExpertLayerMB
			}
		}
		report.CapacityMB += seat.CapacityMB
		report.IdleMB += seat.FreeMB()
		report.Seats = append(report.Seats, seat)
	}
	sort.Slice(report.Seats, func(i, j int) bool {
		return report.Seats[i].GPUIndex < report.Seats[j].GPUIndex
	})
	return report
}

// marginByGPU returns the measured runtime graph growth per device, plus
// whether each device actually has exact evidence.
//
// RelatedModelRuntimeGraphGrowth already refuses to carry estimated growth and
// already requires a matching GPU signature and slot count, so a value present
// here was observed on this rig at this parallelism. That is the only margin
// this file will spend, and it is why the report can be handed straight to a
// packer without re-deriving safety.
func marginByGPU(cacheDir string, model *ModelProfile, gpus []detect.GPU, parallel int, backendTag string) map[int]int {
	growth := RelatedModelRuntimeGraphGrowth(cacheDir, model, gpus, parallel, backendTag)
	out := make(map[int]int, len(gpus))
	for _, g := range gpus {
		if v, ok := growth[g.Index]; ok {
			out[g.Index] = v
		}
	}
	return out
}

// entryCostByGPU charges a device that owns no expert layers the compute
// buffer a real owner was measured to pay on this launch.
//
// UnaccountedByGPU is the backend's compute/graph residue. On the measured GLM
// launch it read 3115 MiB on CUDA0 and 2814 MiB on CUDA1 -- both layer owners --
// against 0.61 MiB on CUDA2, which owned nothing. Handing CUDA2 an expert layer
// therefore costs its share of a real graph before it holds a single byte of
// expert weight, and a packer that forgets this over-packs it into the same
// graph_reserve OOM that cost 46 minutes on 2026-09-02.
//
// A device already paying at or above the owner rate is charged nothing extra:
// it has bought the ticket already. That is CUDA0's case above, and it is why
// CUDA0 is the cheapest device on the rig to hand an expert layer to.
func entryCostByGPU(gpus []detect.GPU, alloc MeasuredAllocation) map[int]int {
	ownerRate := 0
	for _, g := range gpus {
		if v := alloc.UnaccountedByGPU[g.Index]; v > ownerRate {
			ownerRate = v
		}
	}
	out := make(map[int]int, len(gpus))
	for _, g := range gpus {
		if cost := ownerRate - alloc.UnaccountedByGPU[g.Index]; cost > 0 {
			out[g.Index] = cost
		}
	}
	return out
}
