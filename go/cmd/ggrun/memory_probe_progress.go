package main

import (
	"fmt"
	"io"

	"github.com/raketenkater/ggrun/pkg/placement"
)

// memoryProbeMaxAttempts bounds the probe's re-plan search. Like the launch
// budget it exists to stop churn rather than to stop progress -- but this is a
// preview command, where an extra round costs the user time and nothing else.
// A launch must refuse quickly because a real model load is waiting behind it;
// a probe has no such obligation.
//
// Raising this was tried on 2026-09-15 and REVERTED by its own measurement.
// Six looked too small because a cold-cache GLM-5.3-Flash search was still
// descending when it expired. At 16 the search made the same early progress and
// then stalled: rounds 5-9 each gave up exactly one 1,024-token granule against
// a ~9,300 MiB deficit that shrank by ~18 MiB per round, which needs on the
// order of 500 rounds. No budget fixes that.
//
// The per-round report added alongside this constant is what showed it: rounds
// with a usable deficit-to-token conversion step by 135,168 and 30,720 tokens,
// while rounds without one fall back to a single granule. The defect is that
// contextReclaimTokens yields nothing for those later rounds, not that the
// search is short of attempts. See PROBESTALL in
// docs/core-standard-launch-todos.md.
const memoryProbeMaxAttempts = 6

// probeSearchRound is one re-plan: what was tried, what it cost, and where the
// search went next.
type probeSearchRound struct {
	fromContext   int
	toContext     int
	deficitMB     int
	reclaimTokens int
}

// probeSearchProgress records the shape of the re-plan search so an exhausted
// budget can say *why* it was exhausted.
//
// "did not reach a fixed point after 6 attempts" is true but useless on its own:
// it cannot distinguish a search oscillating between two shapes from one
// descending correctly but far too slowly to arrive. Those have opposite fixes,
// and telling them apart on GLM-5.3-Flash 2026-09-15 required extracting the
// per-round argv by hand. Observed there: the context fell 530,432 -> 529,408 ->
// 528,384, one 1,024-token granule per round, against deficits of 2,267, 4,590
// and 4,571 MiB -- 3,072 tokens reclaimed against a gap needing roughly thirty
// times that.
type probeSearchProgress struct {
	startContext int
	rounds       []probeSearchRound
}

func newProbeSearchProgress(start *placement.Strategy) *probeSearchProgress {
	p := &probeSearchProgress{}
	if start != nil {
		p.startContext = start.ContextSize
	}
	return p
}

func (p *probeSearchProgress) record(from, to *placement.Strategy, deficitMB, reclaimTokens int) {
	if p == nil || from == nil || to == nil {
		return
	}
	p.rounds = append(p.rounds, probeSearchRound{
		fromContext:   from.ContextSize,
		toContext:     to.ContextSize,
		deficitMB:     deficitMB,
		reclaimTokens: reclaimTokens,
	})
}

// moved is the total context the search gave up across every round.
func (p *probeSearchProgress) moved() int {
	if p == nil || len(p.rounds) == 0 {
		return 0
	}
	last := p.rounds[len(p.rounds)-1]
	if p.startContext <= 0 || last.toContext <= 0 {
		return 0
	}
	return p.startContext - last.toContext
}

// needed is the largest per-round reclaim the measurements asked for. It is the
// scale the search had to cover, so comparing it with moved() separates "stuck"
// from "too slow".
func (p *probeSearchProgress) needed() int {
	if p == nil {
		return 0
	}
	worst := 0
	for _, r := range p.rounds {
		if r.reclaimTokens > worst {
			worst = r.reclaimTokens
		}
	}
	return worst
}

func (p *probeSearchProgress) report(w io.Writer) {
	if p == nil || len(p.rounds) == 0 {
		fmt.Fprintln(w, "  the search recorded no re-plan rounds; the first plan never produced a measurement")
		return
	}

	fmt.Fprintf(w, "  context search: %d tokens", p.startContext)
	for _, r := range p.rounds {
		fmt.Fprintf(w, " -> %d", r.toContext)
	}
	fmt.Fprintln(w)

	for i, r := range p.rounds {
		step := r.fromContext - r.toContext
		fmt.Fprintf(w, "    round %d: gave up %d tokens against a %d MiB deficit", i+1, step, r.deficitMB)
		if r.reclaimTokens > 0 {
			fmt.Fprintf(w, " (deficit is worth ~%d tokens)", r.reclaimTokens)
		}
		fmt.Fprintln(w)
	}

	moved, needed := p.moved(), p.needed()
	switch {
	case moved <= 0:
		fmt.Fprintln(w, "  the context never moved: the search is cycling between shapes, not descending.")
		fmt.Fprintln(w, "  Re-run with an explicit -ctx to bypass the automatic search.")
	case needed > 0 && moved < needed:
		fmt.Fprintf(w, "  the search moved %d tokens but a single round's deficit was worth ~%d.\n", moved, needed)
		fmt.Fprintln(w, "  It is descending correctly and far too slowly to arrive within the budget.")
		fmt.Fprintf(w, "  Naming a smaller context (-ctx %d or below) will preview immediately.\n", p.suggestedContext())
	default:
		fmt.Fprintf(w, "  the search moved %d tokens across %d rounds without reaching a stable plan.\n", moved, len(p.rounds))
	}
}

// suggestedContext is a starting point below the deepest measurement the search
// reached. It is advice for the user's next invocation, never a plan: nothing
// here has been proven to fit.
func (p *probeSearchProgress) suggestedContext() int {
	if p == nil || len(p.rounds) == 0 {
		return 0
	}
	deepest := p.rounds[len(p.rounds)-1].toContext
	if needed := p.needed(); needed > 0 && deepest > needed {
		return deepest - needed
	}
	// No usable conversion: fall back to a half step, which at least leaves the
	// range the search was crawling through.
	return deepest / 2
}
