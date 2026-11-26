package main

import "github.com/raketenkater/ggrun/pkg/placement"

// dryRunFitRefusal returns the refusal a dry run prints when the plan's own
// per-device ledger says KV on a GPU does not fit there, or "" when it does.
func dryRunFitRefusal(strategy *placement.Strategy) string {
	if strategy == nil {
		return ""
	}
	over := placement.DenseOffloadKVOverflow(strategy, strategy.ResourceLedger)
	if len(over) == 0 {
		return ""
	}
	return "[dry-run] plan does not fit as estimated: " + placement.FormatKVOverflow(over) +
		". Lower --ctx-size or use --kv-placement cpu."
}
