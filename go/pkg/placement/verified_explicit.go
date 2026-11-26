package placement

import (
	"fmt"
	"strings"
)

// verifiedConfigExplicitConflict names the first explicit request coordinate a
// saved verified config would silently override, or returns "" when reuse
// honours every one. It covers only coordinates the verified scope key leaves
// out: ctx, parallel, KV quality, batch sizes, threads, mmap, the GPU set and
// the memory budgets are in the key (a different value is a clean miss), and
// threads, host, mmap and --ctx-checkpoints are re-applied on reuse.
//
// An explicit choice is a constraint (core contract invariant 3). A record
// planned under another value is not evidence for this launch, so the caller
// re-plans instead of keeping the saved argv.
func verifiedConfigExplicitConflict(vc *VerifiedConfig, opts Options) string {
	if vc == nil {
		return ""
	}
	if opts.CPUMode {
		if vc.StrategyType != CPUOnly {
			return fmt.Sprintf("--cpu (the saved config is %s)", vc.StrategyType)
		}
	} else if kv := strings.TrimSpace(opts.KVPlacement); kv != "" && !strings.EqualFold(kv, "auto") &&
		!strings.EqualFold(kv, strings.TrimSpace(vc.KVPlacement)) {
		return fmt.Sprintf("--kv-placement %s (the saved config has %s)", kv, verifiedSavedValue(vc.KVPlacement))
	}
	// A legacy record without CRAMSet and CRAM 0 re-derives the budget, which
	// honours the flag; any other saved value replaces it.
	if opts.CacheRAMMB > 0 && vc.CRAM != opts.CacheRAMMB && (vc.CRAMSet || vc.CRAM != 0) {
		return fmt.Sprintf("--cache-ram %d (the saved config has %d)", opts.CacheRAMMB, vc.CRAM)
	}
	if opts.MMProjPath != "" && vc.MMProjPath != opts.MMProjPath {
		return fmt.Sprintf("--mmproj %s (the saved config has %s)", opts.MMProjPath, verifiedSavedValue(vc.MMProjPath))
	}
	return ""
}

func verifiedSavedValue(v string) string {
	if strings.TrimSpace(v) == "" {
		return "none"
	}
	return v
}
