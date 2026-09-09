package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

// The safe floor is the last tier of the launch ladder: when no ordinary
// placement can be proven serviceable, ggrun serves a deliberately conservative
// one rather than exiting. It is a SERVICEABILITY mechanism, not an evidence
// one -- see the package comment in pkg/placement/safefloor.go for why a floor
// launch teaches the requested plan almost nothing, and which single quantity
// (runtime graph growth, at the same slot count) it does contribute.
//
// It must never quietly become the steady state. Every activation prints the
// original failure in full, names each surrendered coordinate, and is recorded;
// a second activation for the same launch scope is reported as a defect rather
// than as normal operation.

// safeFloorConstraintsFromRequest maps the request's explicit-choice markers.
// Contract invariant 3 makes moving an AUTOMATIC coordinate legal with no
// opt-in; surrendering one the user pinned needs --safe-floor.
func safeFloorConstraintsFromRequest(req *launchRequest) placement.SafeFloorConstraints {
	if req == nil {
		return placement.SafeFloorConstraints{}
	}
	// These coordinates carry their explicitness in their value, the same test
	// the rest of the planner uses (autoContextQualityCandidates: a request is
	// automatic when it is empty or "auto"). A numeric --ctx-size is pinned;
	// "fit"/"max"/auto are not.
	automatic := func(v string) bool {
		v = strings.ToLower(strings.TrimSpace(v))
		return v == "" || v == "auto"
	}
	ctxPinned := !automatic(req.CtxFlag) &&
		!strings.EqualFold(strings.TrimSpace(req.CtxFlag), "fit") &&
		!strings.EqualFold(strings.TrimSpace(req.CtxFlag), "max")
	return placement.SafeFloorConstraints{
		ContextExplicit:     ctxPinned,
		KVQualityExplicit:   !automatic(req.KVQuality),
		KVPlacementExplicit: !automatic(req.KVPlacement),
		UBatchExplicit:      req.UBatchSizeSet,
		BatchExplicit:       req.BatchSizeSet,
		HotExpertsExplicit: req.HotExpertsSet &&
			!strings.EqualFold(strings.TrimSpace(req.HotExperts), "off"),
		// "off" is the value the floor wants anyway: a user turning a feature OFF
		// never blocks a tier that also turns it off.
		SpecExplicit:    !automatic(req.SpecMode) && !strings.EqualFold(strings.TrimSpace(req.SpecMode), "off"),
		SWAFullExplicit: hasArg(req.ExtraArgs, "--swa-full"),
		AllowExplicit:   req.SafeFloor,
	}
}

// terminalSafeFloorConstraints is the constraint set for a caller that is about
// to give up entirely. Invariant 3 stops an optimiser trading a user's explicit
// choice for speed; it was never meant to make ggrun refuse to serve. At a
// terminal stage the honest trade is to surrender the pin, name it loudly, and
// launch -- somebody who asked for hot experts wants a server without them far
// more than they want no server. Parallel is still never surrendered.
func terminalSafeFloorConstraints(req *launchRequest) placement.SafeFloorConstraints {
	cons := safeFloorConstraintsFromRequest(req)
	cons.AllowExplicit = true
	return cons
}

// launchSafeFloor computes and starts the most conservative placement that
// resolves, after `cause` made every ordinary path fail. It performs exactly one
// attempt and inherits the caller's memory-recovery state, so it can neither
// recurse nor resurrect an argv this launch already disproved (invariant 10).
func launchSafeFloor(
	req *launchRequest, cfg *config.Config, model *placement.ModelProfile,
	be *backendInfo, caps *detect.Capabilities, timeout time.Duration,
	launchRecovery *launchMemoryRecovery, cause error,
) (*server.Process, *placement.Strategy, []string, error) {
	if req == nil || cfg == nil || model == nil || be == nil {
		return nil, nil, nil, fmt.Errorf("safe floor needs a complete launch request")
	}
	if req.SafeFloorAttempted {
		return nil, nil, nil, fmt.Errorf("safe floor already attempted for this launch")
	}
	req.SafeFloorAttempted = true

	opts := placementOptionsFromRequestCaps(req, model, be, cfg.CacheDir, caps)
	opts.VerifiedConfigScopeKey = ""
	strategy, rung, err := placement.ComputeSafeFloor(caps, model, opts, terminalSafeFloorConstraints(req))
	if err != nil || strategy == nil {
		if err == nil {
			err = fmt.Errorf("safe floor produced no placement")
		}
		return nil, nil, nil, err
	}

	scope := calibrationScopeKey(req, model, be, caps, strategy)
	prior := recordSafeFloorActivation(cfg.CacheDir, scope, rung, cause)

	// The original failure is printed in full FIRST. A floor that hides why it
	// was needed turns a diagnosable defect into folklore.
	fmt.Fprintf(os.Stderr, "\n[safe-floor] the requested configuration could not be served: %v\n", cause)
	fmt.Fprintf(os.Stderr, "[safe-floor] falling back to the %q tier; surrendering: %s\n",
		rung.Name, strings.Join(rung.Surrenders, ", "))
	fmt.Fprintln(os.Stderr, "[safe-floor] this is a serviceability fallback, not a tuned configuration; it is never promoted or cached as a winner")
	if prior > 0 {
		fmt.Fprintf(os.Stderr,
			"[safe-floor] DEFECT: this launch scope has now needed the safe floor %d time(s). "+
				"A repeated floor is an optimizer bug, not normal operation - please report it with %s\n",
			prior+1, safeFloorLedgerPath(cfg.CacheDir))
	}

	claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
	serverArgs := buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
	fmt.Printf("[launch] %s\n", formatCommand(serverArgs))
	if launchRecovery == nil {
		launchRecovery = newLaunchMemoryRecovery()
	}
	return startLaunchWithCUDAOOMRecoveryStateFloor(req, cfg, model, strategy, be, caps, serverArgs, timeout, launchRecovery)
}

// startLaunchWithCUDAOOMRecoveryStateFloor exists so the floor's own start still
// gets the ordinary startup OOM ladder without the floor being re-entered from
// inside it (req.SafeFloorAttempted already guards that).
func startLaunchWithCUDAOOMRecoveryStateFloor(
	req *launchRequest, cfg *config.Config, model *placement.ModelProfile,
	strategy *placement.Strategy, be *backendInfo, caps *detect.Capabilities,
	serverArgs []string, timeout time.Duration, launchRecovery *launchMemoryRecovery,
) (*server.Process, *placement.Strategy, []string, error) {
	return startLaunchWithCUDAOOMRecoveryState(req, cfg, model, strategy, be, caps, serverArgs, timeout, launchRecovery)
}

// safeFloorActivation is one recorded fallback, kept so a recurring floor is
// visible as a defect signal instead of being mistaken for how ggrun works.
type safeFloorActivation struct {
	ScopeKey string `json:"scope_key"`
	At       string `json:"at"`
	Rung     string `json:"rung"`
	Cause    string `json:"cause"`
}

func safeFloorLedgerPath(cacheDir string) string {
	if cacheDir == "" {
		return ""
	}
	return filepath.Join(cacheDir, "safe-floor-activations.json")
}

// recordSafeFloorActivation appends this activation and returns how many prior
// ones exist for the same scope. A failure to record never fails the launch --
// serving is the point -- but it is reported, because a silent ledger defeats
// the purpose.
func recordSafeFloorActivation(cacheDir, scope string, rung placement.SafeFloorRung, cause error) int {
	path := safeFloorLedgerPath(cacheDir)
	if path == "" || scope == "" {
		return 0
	}
	var entries []safeFloorActivation
	if data, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(data, &entries)
	}
	prior := 0
	for _, e := range entries {
		if e.ScopeKey == scope {
			prior++
		}
	}
	reason := ""
	if cause != nil {
		reason = cause.Error()
	}
	entries = append(entries, safeFloorActivation{
		ScopeKey: scope,
		At:       time.Now().UTC().Format(time.RFC3339),
		Rung:     rung.String(),
		Cause:    reason,
	})
	// Bound the ledger; keep the newest.
	const maxSafeFloorActivations = 64
	if len(entries) > maxSafeFloorActivations {
		entries = entries[len(entries)-maxSafeFloorActivations:]
	}
	sort.SliceStable(entries, func(i, j int) bool { return entries[i].At < entries[j].At })
	if data, err := json.MarshalIndent(entries, "", "  "); err == nil {
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err == nil {
			if err := atomicWriteLocalFile(path, append(data, '\n')); err != nil {
				fmt.Fprintf(os.Stderr, "[safe-floor] warning: could not record this activation: %v\n", err)
			}
		}
	}
	return prior
}

// ClearSafeFloorActivations drops the ledger for one scope after an ordinary
// (non-floor) launch of that scope succeeds: the defect it recorded is gone.
func clearSafeFloorActivations(cacheDir, scope string) {
	path := safeFloorLedgerPath(cacheDir)
	if path == "" || scope == "" {
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var entries []safeFloorActivation
	if json.Unmarshal(data, &entries) != nil {
		return
	}
	kept := entries[:0]
	for _, e := range entries {
		if e.ScopeKey != scope {
			kept = append(kept, e)
		}
	}
	if len(kept) == len(entries) {
		return
	}
	if out, err := json.MarshalIndent(kept, "", "  "); err == nil {
		_ = atomicWriteLocalFile(path, append(out, '\n'))
	}
}

func atomicWriteLocalFile(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
