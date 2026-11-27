package main

import (
	"fmt"
	"sort"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// validateInventoryNumbering rejects an --inventory whose card indexes are not
// the ones ggrun itself would give that set of cards.
//
// Saved host measurements (placement cache, probe records, system probe) are
// keyed by a hardware identity that sorts the cards and drops their index,
// because detection numbers cards by PCI bus id (CUDA_DEVICE_ORDER=PCI_BUS_ID),
// which makes the index a function of the identity. The records store their
// per-card values BY INDEX. An inventory that numbers the same cards some other
// way would match this host's records and put their values on the wrong cards,
// and `--gpus` plus the plan's per-card output would no longer mean the cards
// the file names. So the file is refused, never renumbered:
//
//   - indexes are exactly 0..N-1, each once, listed in index order (what
//     `ggrun detect` writes);
//   - no two cards share a pci_bus_id;
//   - if every card has a pci_bus_id, index order is ascending bus-id order,
//     compared as detect sorts them (plain string order);
//   - if any card lacks one, the order cannot be derived and the first rule is
//     all that applies.
func validateInventoryNumbering(gpus []detect.GPU) error {
	n := len(gpus)
	if n == 0 {
		return nil
	}
	want := fmt.Sprintf("number the cards 0..%d in the order they are listed", n-1)
	byIndex := make(map[int]int, n)
	for pos, g := range gpus {
		if g.Index < 0 {
			return fmt.Errorf("GPU %q has negative index %d; %s", g.Name, g.Index, want)
		}
		if prev, dup := byIndex[g.Index]; dup {
			return fmt.Errorf("index %d is given to two GPUs (%q and %q); %s", g.Index, gpus[prev].Name, g.Name, want)
		}
		byIndex[g.Index] = pos
	}
	for i := 0; i < n; i++ {
		if _, ok := byIndex[i]; !ok {
			return fmt.Errorf("index %d is missing (indexes must have no gap); %s", i, want)
		}
	}
	withBus := 0
	byBus := make(map[string]int, n)
	for pos, g := range gpus {
		if g.PCIBusID == "" {
			continue
		}
		withBus++
		if prev, dup := byBus[g.PCIBusID]; dup {
			return fmt.Errorf("pci_bus_id %s is given to two GPUs (index %d and %d); each card has its own bus id",
				g.PCIBusID, gpus[prev].Index, g.Index)
		}
		byBus[g.PCIBusID] = pos
	}
	if withBus == n {
		order := make([]int, n)
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return gpus[order[a]].PCIBusID < gpus[order[b]].PCIBusID })
		for i, pos := range order {
			if pos != i || gpus[pos].Index != i {
				var lines []string
				for j, p := range order {
					lines = append(lines, fmt.Sprintf("  index %d: %s (%s)", j, gpus[p].PCIBusID, gpus[p].Name))
				}
				return fmt.Errorf("the GPUs are not numbered the way ggrun numbers these cards "+
					"(by PCI bus id, as CUDA_DEVICE_ORDER=PCI_BUS_ID does); list them in this order with these indexes, "+
					"or capture the file with `ggrun detect` on that machine:\n%s", strings.Join(lines, "\n"))
			}
		}
		return nil
	}
	for pos, g := range gpus {
		if g.Index != pos {
			return fmt.Errorf("GPU %q has index %d but is listed at position %d; %s", g.Name, g.Index, pos, want)
		}
	}
	return nil
}
