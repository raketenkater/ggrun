package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

// Every start boundary call owns one admission: the contained allocation
// probes, production starts and CUDA OOM retries it performs to reach one
// running server. Before this bound existed each nested probe took a fresh
// size-scaled timeout of its own, so a 131.7 GiB MoE without a memory oracle
// could spend hours in successive 30-minute probes that no caller had budgeted
// (STARTUP-MIMO-20260924).
const (
	// An ordinary start may begin at most this many weight-reading backend
	// processes: contained probes plus production starts, including retries.
	maxStartAdmissionLoads = 4
	// An optimizer challenger or a restoration needs at most one contained
	// probe and one production start.
	maxBoundedAdmissionLoads = 2
	// A remaining window shorter than this cannot load a model.
	minUsefulLoadWindow = time.Second
	// The first start may use this many per-process ceilings in total, which
	// leaves room for a probe, a corrected probe and the production load of a
	// legitimately slow first launch while still ending in bounded time.
	startupAdmissionWindowLoads = 3
	// Promotion, restoration and recovery restarts get two ceilings: at most a
	// probe and the production load.
	restartAdmissionWindowLoads = 2
)

func startupAdmissionWindow(model *placement.ModelProfile) time.Duration {
	return startupAdmissionWindowLoads * autoStartupTimeout(model)
}

func restartAdmissionWindow(model *placement.ModelProfile) time.Duration {
	return restartAdmissionWindowLoads * autoStartupTimeout(model)
}

// admissionBudgetError ends an admission that ran out of time or of
// weight-loading starts. It is neither a memory result nor a compatibility
// verdict: callers must not cache it as a rejected configuration.
type admissionBudgetError struct {
	Phase   string
	Reason  string
	Loads   int
	Elapsed time.Duration
}

func (e *admissionBudgetError) Error() string {
	return fmt.Sprintf("%s admission budget exhausted after %d weight-loading start(s) in %s: %s",
		e.Phase, e.Loads, e.Elapsed.Round(time.Second), e.Reason)
}

func isAdmissionBudgetError(err error) bool {
	var budget *admissionBudgetError
	return errors.As(err, &budget)
}

// admissionWork is the budget and cost record of one admission. A nil value is
// unbounded, which keeps unit tests of individual gates simple; every
// production start boundary creates one.
type admissionWork struct {
	phase    string
	started  time.Time
	deadline time.Time
	ceiling  time.Duration
	maxLoads int
	loads    int
	// oracleRuns counts no-allocation oracle runs; each may record measured
	// compute/context rows that a later plan can use.
	oracleRuns int
	cacheDir   string
	now        func() time.Time
	// lastProductionLoad is the elapsed time of this admission's successful
	// production start, the observed cost of loading this configuration.
	lastProductionLoad time.Duration
}

func newAdmissionWork(phase string, window, ceiling time.Duration, maxLoads int, cacheDir string, now func() time.Time) *admissionWork {
	if now == nil {
		now = time.Now
	}
	start := now()
	if ceiling <= 0 || ceiling > window {
		ceiling = window
	}
	return &admissionWork{
		phase: phase, started: start, deadline: start.Add(window), ceiling: ceiling,
		maxLoads: maxLoads, cacheDir: cacheDir, now: now,
	}
}

// loadedWeights reports whether this admission started any backend process
// that reads model weights. Unknown work is never reported as cheap: callers
// without a work record treat the failure as expensive.
func (w *admissionWork) loadedWeights() bool {
	return w != nil && w.loads > 0
}

func (w *admissionWork) remaining() time.Duration {
	if w == nil {
		return 0
	}
	return w.deadline.Sub(w.now())
}

// beginLoad reserves one weight-reading backend start and returns the timeout
// it may use: its own ceiling, never past the admission's deadline. It refuses
// once the load allowance or the window is spent, before a process starts.
func (w *admissionWork) beginLoad(configured time.Duration) (time.Duration, error) {
	if w == nil {
		return configured, nil
	}
	elapsed := w.now().Sub(w.started)
	if w.loads >= w.maxLoads {
		return 0, &admissionBudgetError{Phase: w.phase, Loads: w.loads, Elapsed: elapsed,
			Reason: fmt.Sprintf("the admission allows %d weight-loading starts", w.maxLoads)}
	}
	remaining := w.remaining()
	if remaining < minUsefulLoadWindow {
		return 0, &admissionBudgetError{Phase: w.phase, Loads: w.loads, Elapsed: elapsed,
			Reason: "the admission window has no time left for another model load"}
	}
	timeout := remaining
	if w.ceiling > 0 && w.ceiling < timeout {
		timeout = w.ceiling
	}
	if configured > 0 && configured < timeout {
		timeout = configured
	}
	w.loads++
	return timeout, nil
}

// boundCheap limits a step that reads no weights (the no-allocation oracle) to
// the admission's remaining window without charging a load.
func (w *admissionWork) boundCheap(configured time.Duration) time.Duration {
	if w == nil {
		return configured
	}
	if remaining := w.remaining(); remaining < configured {
		if remaining < time.Second {
			return time.Second
		}
		return remaining
	}
	return configured
}

// launchWorkRecord is one line of the local launch-work ledger. Every
// weight-reading start is recorded whatever its outcome, so the cost of a
// launch can be reconstructed without trusting printed progress.
type launchWorkRecord struct {
	Time          string  `json:"time"`
	Phase         string  `json:"phase"`
	Kind          string  `json:"kind"` // oracle, cached-evidence, contained-probe, production, budget-refusal, emergency-restore
	LoadedWeights bool    `json:"loaded_weights"`
	Model         string  `json:"model,omitempty"`
	Backend       string  `json:"backend,omitempty"`
	ArgvHash      string  `json:"argv_hash,omitempty"`
	EvidenceKey   string  `json:"evidence_key,omitempty"`
	ElapsedSec    float64 `json:"elapsed_sec"`
	TimeoutSec    float64 `json:"timeout_sec,omitempty"`
	Outcome       string  `json:"outcome"`
	Reason        string  `json:"reason,omitempty"`
	LastOutputAge float64 `json:"last_output_age_sec,omitempty"`
	CgroupPeakMiB int     `json:"cgroup_peak_mib,omitempty"`
	OOMKills      uint64  `json:"oom_kills,omitempty"`
	AdmissionLoad int     `json:"admission_load,omitempty"`
}

const launchWorkLedgerMaxBytes = 8 << 20

var launchWorkLedgerMu sync.Mutex

func launchWorkLedgerPath(cacheDir string) string {
	return filepath.Join(cacheDir, "launch-work.jsonl")
}

// record appends one ledger line. Best effort: the ledger explains a launch,
// it never decides one.
func (w *admissionWork) record(rec launchWorkRecord) {
	if w == nil || w.cacheDir == "" {
		return
	}
	if rec.Phase == "" {
		rec.Phase = w.phase
	}
	if rec.Kind == "oracle" {
		w.oracleRuns++
	}
	if rec.LoadedWeights {
		rec.AdmissionLoad = w.loads
	}
	appendLaunchWork(w.cacheDir, rec)
}

func appendLaunchWork(cacheDir string, rec launchWorkRecord) {
	if cacheDir == "" {
		return
	}
	if rec.Time == "" {
		rec.Time = time.Now().UTC().Format(time.RFC3339Nano)
	}
	if len(rec.Reason) > 600 {
		rec.Reason = rec.Reason[:600] + "..."
	}
	line, err := json.Marshal(rec)
	if err != nil {
		return
	}
	launchWorkLedgerMu.Lock()
	defer launchWorkLedgerMu.Unlock()
	path := launchWorkLedgerPath(cacheDir)
	if info, err := os.Stat(path); err == nil && info.Size() > launchWorkLedgerMaxBytes {
		_ = os.Rename(path, path+".1")
	}
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.Write(append(line, '\n'))
}

func argvHash(args []string) string {
	sum := sha256.Sum256([]byte(formatCommand(args)))
	return hex.EncodeToString(sum[:8])
}

// describeStartOutcome turns a start result into the ledger's outcome, reason
// and silence age. A supervisor timeout is kept distinct from a process exit.
func describeStartOutcome(err error) (outcome, reason string, lastOutputAge float64) {
	if err == nil {
		return "ready", "", 0
	}
	var failure *server.StartupFailure
	if errors.As(err, &failure) {
		age := 0.0
		if failure.LastOutputAge >= 0 {
			age = failure.LastOutputAge.Seconds()
		}
		return failure.Outcome, err.Error(), age
	}
	return "error", err.Error(), 0
}

func modelBasename(model *placement.ModelProfile) string {
	if model == nil {
		return ""
	}
	return model.Basename
}

func backendIdentity(be *backendInfo) string {
	if be == nil {
		return ""
	}
	if be.Identity != "" {
		return be.Identity
	}
	return be.Path
}

// preflightUnresolvedError is an admission that memory planning could not
// resolve. It never started a production process for the failing plan.
type preflightUnresolvedError struct{ err error }

func (e *preflightUnresolvedError) Error() string { return e.err.Error() }
func (e *preflightUnresolvedError) Unwrap() error { return e.err }

// learnedReplanEnv marks a launch that already re-planned once from the
// evidence its first admission measured, so the re-plan cannot repeat.
const learnedReplanEnv = "GGRUN_LEARNED_REPLAN"

// shouldReplanFromLearnedEvidence decides whether a failed first admission may
// plan once more. The first plan of a never-seen model is computed before any
// backend measurement exists; the oracle rounds that refused it record exact
// compute and context rows, and a plan computed from them can fit where the
// in-lifecycle ladder (which may only derate the first plan) cannot. Observed:
// MiMo-V2.6-Flash failed closed on first use and fitted at full context on the
// next launch. Allowed only when no weights were loaded, the oracle measured
// something, and this launch has not already re-planned.
func shouldReplanFromLearnedEvidence(err error, recovery *launchMemoryRecovery, alreadyReplanned bool) bool {
	var unresolved *preflightUnresolvedError
	if alreadyReplanned || recovery == nil || !errors.As(err, &unresolved) {
		return false
	}
	return recovery.weightLoads == 0 && recovery.oracleRuns > 0
}
