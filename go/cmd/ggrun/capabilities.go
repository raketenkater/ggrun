package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// ggrun degrades quietly by design: a missing optional dependency makes a
// feature skip rather than fail a launch. That is the right behaviour and it
// has one bad consequence -- a rig can run for months in a degraded mode nobody
// can see.
//
// The clearest case: llama-fit-params. Both backend build paths compiled only
// the llama-server target, so the launch preflight's no-alloc oracle was never
// installed. findFitParamsBin returned "", the preflight skipped exactly as
// documented, and every candidate placement silently cost a real model load --
// measured at ~5 minutes each on GLM 5.3 Flash -- instead of a ~1 s dry run.
// Nothing anywhere said so.
//
// This report is the fix for the class rather than the instance: it names each
// optional capability, whether it is present, and what its absence costs.

// capability is one optional dependency and the consequence of losing it.
type capability struct {
	Name string
	OK   bool
	// Detail is where it was found, or why it was not.
	Detail string
	// Cost is what degrades when OK is false. Empty when the capability is
	// present, since there is nothing to warn about.
	Cost string
}

// collectCapabilities probes every optional dependency ggrun can run without.
func collectCapabilities(cfg *config.Config) []capability {
	var out []capability

	// The placement oracle, per registered backend: it must sit beside the
	// server binary it predicts for, or the preflight predicts against
	// different graph code than the launch runs.
	for _, be := range backends.Load() {
		if be.Path == "" {
			continue
		}
		dir := filepath.Dir(be.Path)
		if resolved, err := filepath.EvalSymlinks(be.Path); err == nil {
			dir = filepath.Dir(resolved)
		}
		oracle := filepath.Join(dir, backends.FitParamsBinaryName)
		c := capability{Name: fmt.Sprintf("placement oracle [%s]", be.Tag)}
		if fi, err := os.Stat(oracle); err == nil && !fi.IsDir() {
			c.OK, c.Detail = true, oracle
		} else {
			c.Detail = "not built beside " + be.Path
			c.Cost = "launch preflight is skipped; every candidate placement costs a real model load"
		}
		out = append(out, c)
	}

	out = append(out, lookPathCapability("systemd-run",
		"memory admission cannot scope a child process; a runaway load can OOM the host"))
	out = append(out, lookPathCapability("nvcc",
		"CUDA architecture is guessed rather than detected when building a backend"))
	out = append(out, lookPathCapability("claude",
		"claude-code mode cannot start"))
	out = append(out, lookPathCapability("python3",
		"GGUF metadata and bandwidth helpers cannot run"))

	// Today's two measured inputs to placement. Both are earned by running, so
	// "absent" is normal on a fresh install and is reported as such rather than
	// as a fault.
	cacheDir := ""
	if cfg != nil {
		cacheDir = cfg.CacheDir
	}
	demand := placement.MeasuredAgentContextDemand(cacheDir)
	if demand.Samples == 0 && cfg != nil {
		// Nothing recorded yet: show what the next launch would learn, so the
		// operator can see their own workload profile without launching. Read
		// only -- recording stays on the launch path.
		if observed, err := observeAgentContextDemand(cfg.LogDir); err == nil {
			demand = observed
		}
	}
	dc := capability{Name: "agent context demand"}
	if ceiling := placement.AgentContextCeiling(demand, 1); ceiling > 0 {
		dc.OK = true
		dc.Detail = fmt.Sprintf("%d requests: p50 %d, p99 %d, max %d -> context ceiling %d",
			demand.Samples, demand.P50Tokens, demand.P99Tokens, demand.MaxTokens, ceiling)
	} else {
		dc.Detail = fmt.Sprintf("%d requests observed (needs %d before it may size a deployment)",
			demand.Samples, placement.MinAgentContextSamples)
		dc.Cost = "automatic context is sized from the model's trained maximum, not from real agent traffic"
	}
	out = append(out, dc)

	return out
}

func lookPathCapability(bin, cost string) capability {
	c := capability{Name: bin}
	if p, err := exec.LookPath(bin); err == nil {
		c.OK, c.Detail = true, p
		return c
	}
	c.Detail, c.Cost = "not on PATH", cost
	return c
}

// printCapabilities renders the report. Degraded entries are listed with what
// they cost, so the output is actionable rather than a checklist.
func printCapabilities(w io.Writer, caps []capability) {
	degraded := 0
	fmt.Fprintln(w, "ggrun optional capabilities:")
	for _, c := range caps {
		mark := "ok     "
		if !c.OK {
			mark = "MISSING"
			degraded++
		}
		fmt.Fprintf(w, "  [%s] %-28s %s\n", mark, c.Name, c.Detail)
	}
	if degraded == 0 {
		fmt.Fprintln(w, "\nNothing is degraded.")
		return
	}
	fmt.Fprintf(w, "\n%d capability/capabilities degraded:\n", degraded)
	for _, c := range caps {
		if c.OK || c.Cost == "" {
			continue
		}
		fmt.Fprintf(w, "  %s\n    -> %s\n", c.Name, c.Cost)
	}
	fmt.Fprintln(w, "\n  `ggrun backend update <tag>` rebuilds a backend and its placement oracle.")
}

// capabilitySummaryLine is a one-line form for a launch banner: silent
// degradation is the problem, so a launch says what it is missing without
// printing the whole table.
func capabilitySummaryLine(caps []capability) string {
	var missing []string
	for _, c := range caps {
		if !c.OK {
			missing = append(missing, c.Name)
		}
	}
	if len(missing) == 0 {
		return ""
	}
	return fmt.Sprintf("degraded capabilities: %s (run `ggrun status --capabilities` for what each costs)",
		strings.Join(missing, ", "))
}
