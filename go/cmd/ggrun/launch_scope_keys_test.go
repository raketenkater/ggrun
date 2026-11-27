package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/controller"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// Scope-key pinning (review W11, step 3, tests only).
//
// The launch path derives identity keys at specific moments: some before the
// request is edited by backend compatibility, some after. Saved state on a
// user's machine (a verified config, a calibration decision, a recoverable
// server log, a last-known-good profile) is found again only if the same
// launch computes the same key. Moving the planning prefix of cmdLaunch or
// cmdDryRun into a shared function can change WHEN a key is computed without
// changing any argv, which the argv goldens cannot see. This file pins, per
// case, the literal value of every key the launch path derives before start,
// and the property each key must keep.
//
// Keys, and where the launch computes them (main.go):
//   - ProfilePolicyIdentity: frozen in cmdLaunch right after parseModel,
//     BEFORE resolveLaunchBackend edits the request (arch KV rule, -khad,
//     --swa-full). Used after start only, by launchProfileScope.
//   - verified-config scope key and shape: memoized on the request by the
//     first caps-bearing placementOptionsFromRequestCaps (computeStrategy),
//     AFTER the backend-compatibility edits and the reviewer reservation.
//     Every later lookup, save and delete reuses the memo.
//   - launchProfileScope: after start (verifyAndActivateLaunch and the
//     runtime-OOM paths), from the frozen identity, the model and backend
//     families, the runtime hardware view and the companion lifecycle.
//   - claudeLaunchLogScope: from the built argv; recoverPreviousClaudeRuntimeOOM
//     uses the pre-repair argv, startLaunchProcess the final one.
//   - calibration scope key: applyCalibrationDecision (inside computeStrategy),
//     on the slot-adjusted strategy and the request as it is at that moment.
//   - controller.HashArgs(argv): the exact-argv half of the profile store
//     lookups, after start.
//
// The dry run computes the log scope (JSON only) and the calibration key, and
// never the other keys: it does not freeze ProfilePolicyIdentity and its
// placement options carry no caps, so no verified key is memoized.
//
// Replays reuse launchOrderReplay, so the fakes are the launch-order golden's.
// The recorded file is testdata/launch_scope_keys_golden.json. Regenerate on
// Linux with
//
//	go test ./cmd/ggrun -run TestLaunchScopeKeyPinning -update-scope-keys
//
// Without a local build a missing or differing file is logged as numbered
// gzip+base64 lines:
//
//	grep -o 'scope-keys-golden-b64 [0-9]* [A-Za-z0-9+/=]*' log | sort -k2,2n |
//	  awk '{printf "%s", $3}' | base64 -d | gunzip > testdata/launch_scope_keys_golden.json
//
// The replay runs under two different temporary homes. The reviewer profile
// carries an absolute path under the app home, which launchProfileScope
// hashes; the test replaces the home prefix with a fixed token before hashing
// so the literal does not depend on the temporary directory, and the second
// home proves that nothing else leaks one into a key.

var updateScopeKeys = flag.Bool("update-scope-keys", false, "rewrite testdata/launch_scope_keys_golden.json")

const scopeKeysGoldenPath = "testdata/launch_scope_keys_golden.json"

// scopeKeyHomeToken stands in for the temporary home in reviewer paths.
const scopeKeyHomeToken = "<home>"

type scopeKeyDryRun struct {
	ProfilePolicyIdentity string `json:"profile_policy_identity"`
	VerifiedScopeKey      string `json:"verified_scope_key"`
	LaunchLogScope        string `json:"launch_log_scope,omitempty"`
	CalibrationScopeKey   string `json:"calibration_scope_key,omitempty"`
	ArgvHash              string `json:"argv_hash,omitempty"`
}

type scopeKeyReviewerFree struct {
	VerifiedScopeKey   string `json:"verified_scope_key"`
	LaunchProfileScope string `json:"launch_profile_scope,omitempty"`
}

// scopeKeyRecord is one case: the keys of the launch order, the keys the dry
// order computes, and the reviewer-free candidate when a reviewer is seated.
type scopeKeyRecord struct {
	Case                  string                `json:"case"`
	Outcome               string                `json:"outcome"`
	Error                 string                `json:"error,omitempty"`
	ProfilePolicyIdentity string                `json:"profile_policy_identity,omitempty"`
	VerifiedScopeKey      string                `json:"verified_scope_key,omitempty"`
	VerifiedScopeShape    string                `json:"verified_scope_shape,omitempty"`
	LaunchProfileScope    string                `json:"launch_profile_scope,omitempty"`
	LaunchLogScope        string                `json:"launch_log_scope,omitempty"`
	CalibrationScopeKey   string                `json:"calibration_scope_key,omitempty"`
	CalibrationRuntimeKey string                `json:"calibration_scope_key_runtime_view,omitempty"`
	ArgvHash              string                `json:"argv_hash,omitempty"`
	ReviewerFree          *scopeKeyReviewerFree `json:"reviewer_free_candidate,omitempty"`
	DryRun                scopeKeyDryRun        `json:"dry_run_order"`
}

// scopeKeyCase is a replay case plus what the property checks need to know.
type scopeKeyCase struct {
	launchOrderCase
	// frozenIdentityDiffers: a request edit after the identity freeze changes
	// what the identity would be if it were computed later.
	frozenIdentityDiffers bool
}

func scopeKeyCases() []scopeKeyCase {
	mk := func(name, hw, model string, args ...string) scopeKeyCase {
		return scopeKeyCase{launchOrderCase: launchOrderCase{goldenPlanCase: goldenPlanCase{name: name, hw: hw, model: model, args: args}}}
	}
	swa := map[string]string{"LLM_SWA_FULL": "true"}
	selfReview := map[string]string{"GGRUN_CLAUDE_AUTO_SELF_REVIEW": "on"}

	golden := mk("mixed3/moe/claude-code", "mixed3", "moe", "--claude-code")
	reviewerOff := mk("mixed3/moe/claude-code+reviewer-off", "mixed3", "moe", "--claude-code", "--claude-reviewer", "off")
	archKV := mk("gpu24/deepseek4-moe/arch-kv-promotion", "gpu24", "deepseek4-moe")
	archKV.frozenIdentityDiffers = true
	plain := mk("gpu24/dense-small/auto", "gpu24", "dense-small")
	swaPlain := mk("gpu24/dense-small/swa-full-withdrawn", "gpu24", "dense-small")
	swaPlain.env = swa
	claude := mk("gpu24/dense-small/claude-code", "gpu24", "dense-small", "--claude-code")
	swaClaude := mk("gpu24/dense-small/claude-code+swa-full-withdrawn", "gpu24", "dense-small", "--claude-code")
	swaClaude.env = swa
	swaClaude.frozenIdentityDiffers = true
	gpus := mk("mixed3/moe/claude-code-gpus-1,2", "mixed3", "moe", "--claude-code", "--gpus", "1,2")
	dropped := mk("gpu24/dense-small/claude-code+self-review-on", "gpu24", "dense-small", "--claude-code")
	dropped.env = selfReview
	droppedOff := mk("gpu24/dense-small/claude-code+self-review-on+reviewer-off", "gpu24", "dense-small", "--claude-code", "--claude-reviewer", "off")
	droppedOff.env = selfReview
	return []scopeKeyCase{golden, reviewerOff, archKV, plain, swaPlain, claude, swaClaude, gpus, dropped, droppedOff}
}

// scopeKeyDeepSeek4MoE is the golden MoE model with the architecture whose KV
// rule forbids the default (auto = q8_0) K-cache, so backend compatibility
// promotes the request to f16 after the identity is frozen.
func scopeKeyDeepSeek4MoE() *placement.ModelProfile {
	m := goldenPlanMoE()
	m.ModelArch = "deepseek4"
	m.Path = "synthetic/deepseek4-moe.gguf"
	m.Basename = "deepseek4-moe"
	return m
}

func scopeKeyExpectSame(t *testing.T, name, what, got, want string) {
	t.Helper()
	if got != want {
		t.Errorf("%s: %s: want %q, got %q", name, what, want, got)
	}
}

func scopeKeyExpectDiff(t *testing.T, name, what, a, b string) {
	t.Helper()
	if a == b {
		t.Errorf("%s: %s: both %q, want different keys", name, what, a)
	}
}

// scopeKeyProfileScope is launchProfileScope as verifyAndActivateLaunch calls
// it (the runtime hardware view), with the temporary home replaced in the
// reviewer profile's paths.
func scopeKeyProfileScope(req *launchRequest, model *placement.ModelProfile, be *backendInfo, caps *detect.Capabilities, home string) string {
	r := *req
	if p := req.ReviewerProfile; p != nil {
		q := *p
		q.ModelPath = strings.ReplaceAll(q.ModelPath, home, scopeKeyHomeToken)
		q.BackendPath = strings.ReplaceAll(q.BackendPath, home, scopeKeyHomeToken)
		q.ModelMarker = strings.ReplaceAll(q.ModelMarker, home, scopeKeyHomeToken)
		r.ReviewerProfile = &q
	}
	runtimeCaps, _ := runtimeGPUCapabilities(caps, req)
	return launchProfileScope(&r, model, be, runtimeCaps)
}

func scopeKeyArgv(run launchOrderRun) []string {
	if run.record.Plan == nil {
		return nil
	}
	return run.record.Plan.ServerArgv
}

// scopeKeyFlagValueEdited returns argv with the value after flag changed.
func scopeKeyFlagValueEdited(argv []string, flag string) ([]string, bool) {
	for i := 1; i+1 < len(argv); i++ {
		if argv[i] == flag {
			out := append([]string(nil), argv...)
			out[i+1] += "9"
			return out, true
		}
	}
	return nil, false
}

func scopeKeyCalibration(run launchOrderRun) string {
	if run.req == nil || run.model == nil || run.be == nil || run.caps == nil || run.strategy == nil {
		return ""
	}
	return calibrationScopeKey(run.req, run.model, run.be, run.caps, run.strategy)
}

// scopeKeyRecordOf computes the record of one case from its two replays.
func scopeKeyRecordOf(c scopeKeyCase, dry, launch launchOrderRun, home string) scopeKeyRecord {
	rec := scopeKeyRecord{Case: c.name, Outcome: launch.record.Outcome, Error: launch.record.Error}
	if launch.req != nil {
		rec.ProfilePolicyIdentity = launch.req.ProfilePolicyIdentity
		rec.VerifiedScopeKey = launch.req.verifiedScopeKey
		rec.VerifiedScopeShape = launch.req.verifiedScopeShape
	}
	if launch.record.Plan != nil {
		argv := scopeKeyArgv(launch)
		rec.LaunchLogScope = claudeLaunchLogScope(launch.req, launch.model, launch.be, argv)
		rec.ArgvHash = controller.HashArgs(argv)
		rec.LaunchProfileScope = scopeKeyProfileScope(launch.req, launch.model, launch.be, launch.caps, home)
		rec.CalibrationScopeKey = scopeKeyCalibration(launch)
		if runtimeCaps, _ := runtimeGPUCapabilities(launch.caps, launch.req); runtimeCaps != launch.caps {
			rec.CalibrationRuntimeKey = calibrationScopeKey(launch.req, launch.model, launch.be, runtimeCaps, launch.strategy)
		}
		if launch.req.ReviewerReservation != nil {
			cand := reviewerFreeCandidate(launch.req)
			rec.ReviewerFree = &scopeKeyReviewerFree{
				VerifiedScopeKey:   verifiedConfigScopeKey(&cand, launch.model, launch.be, launch.caps),
				LaunchProfileScope: scopeKeyProfileScope(&cand, launch.model, launch.be, launch.caps, home),
			}
		}
	}
	if dry.req != nil {
		rec.DryRun.ProfilePolicyIdentity = dry.req.ProfilePolicyIdentity
		rec.DryRun.VerifiedScopeKey = dry.req.verifiedScopeKey
	}
	if dry.record.Plan != nil {
		rec.DryRun.LaunchLogScope = dry.record.Plan.LaunchScope
		rec.DryRun.ArgvHash = controller.HashArgs(scopeKeyArgv(dry))
		rec.DryRun.CalibrationScopeKey = scopeKeyCalibration(dry)
	}
	return rec
}

// scopeKeyProperties asserts, for one launch-order replay, what each key must
// and must not depend on. Edits are applied to copies of the request.
func scopeKeyProperties(t *testing.T, c scopeKeyCase, run launchOrderRun, home string) {
	t.Helper()
	name := c.name
	req, model, be, caps := run.req, run.model, run.be, run.caps
	if req == nil || model == nil || be == nil || caps == nil {
		t.Errorf("%s: the replay stopped before the request, model, backend and hardware were set (%s %s)", name, run.record.Outcome, run.record.Error)
		return
	}
	runtimeCaps, _ := runtimeGPUCapabilities(caps, req)
	clone := func(edit func(*launchRequest)) *launchRequest {
		r := *req
		if edit != nil {
			edit(&r)
		}
		return &r
	}

	// ---- verified-config scope key -------------------------------------
	memo := req.verifiedScopeKey
	if memo == "" || req.verifiedScopeShape == "" {
		t.Errorf("%s: the first caps-bearing placement options did not memoize the verified key (%q, %q)", name, memo, req.verifiedScopeShape)
		return
	}
	scopeKeyExpectSame(t, name, "verified shape is the model and backend identity", req.verifiedScopeShape, verifiedScopeShape(model, be, caps))
	scopeKeyExpectSame(t, name, "a memoized key is returned as is", verifiedConfigScopeKey(clone(nil), model, be, caps), memo)
	scopeKeyExpectSame(t, name, "the memo survives a narrowed hardware view (--gpus re-indexes it)", verifiedConfigScopeKey(clone(nil), model, be, runtimeCaps), memo)
	fresh := func(edit func(*launchRequest)) string {
		r := clone(func(r *launchRequest) {
			r.verifiedScopeKey, r.verifiedScopeShape = "", ""
			if edit != nil {
				edit(r)
			}
		})
		return verifiedConfigScopeKey(r, model, be, caps)
	}
	freshNow := fresh(nil)
	if req.PlacementHostReserveMB == 0 {
		scopeKeyExpectSame(t, name, "no request edit followed the memo, so a fresh key equals it", freshNow, memo)
	}
	ctx := "32768"
	if req.CtxFlag == ctx {
		ctx = "16384"
	}
	seated := req.ReviewerReservation != nil
	edits := []struct {
		what   string
		edit   func(*launchRequest)
		rekeys bool
	}{
		{"KV quality (the arch KV promotion)", func(r *launchRequest) { r.KVQuality = "q4_0" }, true},
		{"--swa-full in the passthrough (a generated --swa-full withdrawn or restored)", func(r *launchRequest) { r.ExtraArgs = append(append([]string(nil), r.ExtraArgs...), "--swa-full") }, true},
		{"containment host reserve", func(r *launchRequest) { r.PlacementHostReserveMB += 512 }, true},
		{"context", func(r *launchRequest) { r.CtxFlag = ctx }, true},
		{"the seated reviewer", func(r *launchRequest) { r.ReviewerReservation = nil }, seated},
		{"ProfilePolicyIdentity", func(r *launchRequest) { r.ProfilePolicyIdentity += "x" }, false},
		{"port", func(r *launchRequest) { r.Port++ }, false},
		{"SWAFullWithdrawnForCheckpoints", func(r *launchRequest) { r.SWAFullWithdrawnForCheckpoints = !r.SWAFullWithdrawnForCheckpoints }, false},
	}
	for _, e := range edits {
		scopeKeyExpectSame(t, name, "memoized key after an edit of "+e.what, verifiedConfigScopeKey(clone(e.edit), model, be, caps), memo)
		got := fresh(e.edit)
		if e.rekeys {
			scopeKeyExpectDiff(t, name, "an unmemoized key must follow "+e.what+" (so the memo is what keeps one launch on one key)", got, freshNow)
		} else {
			scopeKeyExpectSame(t, name, "the key must not depend on "+e.what, got, freshNow)
		}
	}
	other := *be
	other.Identity += "-other-build"
	scopeKeyExpectDiff(t, name, "a different backend identity re-keys the memo", verifiedConfigScopeKey(clone(nil), model, &other, caps), memo)
	if req.GPUsFlag != "" {
		scopeKeyExpectDiff(t, name, "the --gpus restriction is part of the key", fresh(func(r *launchRequest) { r.GPUsFlag = "" }), freshNow)
		hw := clone(func(r *launchRequest) { r.verifiedScopeKey, r.verifiedScopeShape = "", "" })
		scopeKeyExpectDiff(t, name, "hardware is part of an unmemoized key (the runtime view is not the detected one)", verifiedConfigScopeKey(hw, model, be, runtimeCaps), memo)
	}

	// ---- profile policy identity --------------------------------------
	frozen := req.ProfilePolicyIdentity
	if frozen == "" {
		t.Errorf("%s: the launch order did not freeze ProfilePolicyIdentity", name)
		return
	}
	now := requestedLaunchPolicyIdentity(req, model)
	if c.frozenIdentityDiffers {
		scopeKeyExpectDiff(t, name, "the request was edited after the freeze, so recomputing would give another identity", frozen, now)
	} else {
		scopeKeyExpectSame(t, name, "no edit followed the freeze", frozen, now)
	}
	if strings.Contains(c.name, "arch-kv") {
		pre := clone(func(r *launchRequest) { r.KVQuality = "auto" })
		scopeKeyExpectSame(t, name, "the frozen identity is the pre-promotion request's", requestedLaunchPolicyIdentity(pre, model), frozen)
		if req.KVQuality == "auto" {
			t.Errorf("%s: the arch KV rule did not promote the request (kv quality still %q)", name, req.KVQuality)
		}
	}
	mustChange := []struct {
		what string
		edit func(*launchRequest)
	}{
		{"context", func(r *launchRequest) { r.CtxFlag += "1" }},
		{"KV quality", func(r *launchRequest) { r.KVQuality += "1" }},
		{"KV placement", func(r *launchRequest) { r.KVPlacement += "1" }},
		{"KV K type", func(r *launchRequest) { r.KVTypeK += "1" }},
		{"--cpu", func(r *launchRequest) { r.CPUMode = !r.CPUMode }},
		{"--gpus", func(r *launchRequest) { r.GPUsFlag += "1" }},
		{"spec mode", func(r *launchRequest) { r.SpecMode += "1" }},
		{"RAM budget", func(r *launchRequest) { r.RamBudgetMB += 7 }},
		{"VRAM headroom", func(r *launchRequest) { r.VRAMHeadroomMB += 7 }},
		{"RAM headroom", func(r *launchRequest) { r.RAMHeadroomMB += 7 }},
		{"--no-mmap", func(r *launchRequest) { r.NoMMap = !r.NoMMap }},
		{"--mmap", func(r *launchRequest) { r.ForceMMap = !r.ForceMMap }},
		{"slots", func(r *launchRequest) { r.Parallel += 3 }},
		{"threads", func(r *launchRequest) { r.Threads += 3 }},
		{"prompt cache", func(r *launchRequest) { r.CacheRAMMB += 3 }},
		{"batch", func(r *launchRequest) { r.BatchSize += 3 }},
		{"ubatch", func(r *launchRequest) { r.UBatchSize += 3 }},
		{"benchmark", func(r *launchRequest) { r.Benchmark = !r.Benchmark }},
	}
	for _, e := range mustChange {
		scopeKeyExpectDiff(t, name, "the policy identity must follow "+e.what, requestedLaunchPolicyIdentity(clone(e.edit), model), now)
	}
	mustKeep := []struct {
		what string
		edit func(*launchRequest)
	}{
		{"port", func(r *launchRequest) { r.Port++ }},
		{"host", func(r *launchRequest) { r.Host += "x" }},
		{"the seated reviewer", func(r *launchRequest) { r.ReviewerReservation = nil }},
		{"the reviewer being disabled", func(r *launchRequest) { r.ClaudeReviewerDisabled = !r.ClaudeReviewerDisabled }},
		{"the verified memo", func(r *launchRequest) { r.verifiedScopeKey = "other" }},
		{"the frozen identity itself", func(r *launchRequest) { r.ProfilePolicyIdentity = "other" }},
		{"SWAFullWithdrawnForCheckpoints", func(r *launchRequest) { r.SWAFullWithdrawnForCheckpoints = !r.SWAFullWithdrawnForCheckpoints }},
	}
	for _, e := range mustKeep {
		scopeKeyExpectSame(t, name, "the policy identity must not depend on "+e.what, requestedLaunchPolicyIdentity(clone(e.edit), model), now)
	}
	// The passthrough reaches the identity only through the Claude workload
	// profile (a hash of the passthrough), so a generated --swa-full edit moves
	// the identity of a Claude launch and not of an ordinary one. Freezing
	// before the edit is what keeps a Claude launch's family stable.
	swaEdit := func(r *launchRequest) { r.ExtraArgs = append(append([]string(nil), r.ExtraArgs...), "--swa-full") }
	if requestWorkloadProfile(req, model) != "" {
		scopeKeyExpectDiff(t, name, "a Claude launch's identity follows the passthrough", requestedLaunchPolicyIdentity(clone(swaEdit), model), now)
	} else {
		scopeKeyExpectSame(t, name, "an ordinary launch's identity ignores the passthrough", requestedLaunchPolicyIdentity(clone(swaEdit), model), now)
	}

	// ---- launch profile scope -----------------------------------------
	scope := scopeKeyProfileScope(req, model, be, caps, home)
	scopeKeyExpectDiff(t, name, "the profile scope follows the frozen identity", scopeKeyProfileScope(clone(func(r *launchRequest) { r.ProfilePolicyIdentity = "other" }), model, be, caps, home), scope)
	scopeKeyExpectSame(t, name, "an empty frozen identity falls back to the request as it is now",
		scopeKeyProfileScope(clone(func(r *launchRequest) { r.ProfilePolicyIdentity = "" }), model, be, caps, home),
		scopeKeyProfileScope(clone(func(r *launchRequest) { r.ProfilePolicyIdentity = now }), model, be, caps, home))
	toggled := scopeKeyProfileScope(clone(func(r *launchRequest) { r.ClaudeReviewerDisabled = !r.ClaudeReviewerDisabled }), model, be, caps, home)
	if req.ClaudeCode {
		scopeKeyExpectDiff(t, name, "the companion lifecycle is part of a Claude launch's profile scope", toggled, scope)
	} else {
		scopeKeyExpectSame(t, name, "an ordinary launch has no companion in its profile scope", toggled, scope)
	}
	scopeKeyExpectSame(t, name, "the profile scope ignores the verified memo", scopeKeyProfileScope(clone(func(r *launchRequest) { r.verifiedScopeKey = "other" }), model, be, caps, home), scope)

	if run.record.Plan == nil {
		return
	}
	argv := run.record.Plan.ServerArgv

	// ---- launch log scope ---------------------------------------------
	logScope := claudeLaunchLogScope(req, model, be, argv)
	scopeKeyExpectSame(t, name, "the plan's launch_scope is the log scope of the built argv", run.record.Plan.LaunchScope, logScope)
	derivedSeen, shapeSeen := 0, 0
	for _, flag := range []string{"--port", "--host", "-ot", "--override-tensor", "--n-cpu-moe", "--tensor-split", "--split-mode", "-ngl", "--n-gpu-layers", "--gpu-layers", "-cram", "--cache-ram", "--ctx-checkpoints"} {
		if edited, ok := scopeKeyFlagValueEdited(argv, flag); ok {
			derivedSeen++
			scopeKeyExpectSame(t, name, "the log scope must not follow the plan-derived "+flag, claudeLaunchLogScope(req, model, be, edited), logScope)
			scopeKeyExpectDiff(t, name, "the exact argv hash follows "+flag, controller.HashArgs(edited), controller.HashArgs(argv))
		}
	}
	for _, flag := range []string{"-c", "--ctx-size", "-np", "--parallel", "-b", "--batch-size", "-ub", "--ubatch-size", "-ctk", "--cache-type-k", "-ctv", "--cache-type-v", "-t", "--threads"} {
		if edited, ok := scopeKeyFlagValueEdited(argv, flag); ok {
			shapeSeen++
			scopeKeyExpectDiff(t, name, "the log scope must follow the launch-shape "+flag, claudeLaunchLogScope(req, model, be, edited), logScope)
			scopeKeyExpectDiff(t, name, "the exact argv hash follows "+flag, controller.HashArgs(edited), controller.HashArgs(argv))
		}
	}
	if derivedSeen == 0 || shapeSeen == 0 {
		t.Errorf("%s: the argv carries %d plan-derived and %d launch-shape flags to edit, want both; argv %q", name, derivedSeen, shapeSeen, argv)
	}
	rebinary := append([]string(nil), argv...)
	rebinary[0] += "-moved"
	scopeKeyExpectSame(t, name, "the log scope ignores the backend binary path (the build identity carries it)", claudeLaunchLogScope(req, model, be, rebinary), logScope)
	scopeKeyExpectDiff(t, name, "the exact argv hash includes the binary path", controller.HashArgs(rebinary), controller.HashArgs(argv))
	scopeKeyExpectDiff(t, name, "the log scope follows the backend build identity", claudeLaunchLogScope(req, model, &other, argv), logScope)
	scopeKeyExpectSame(t, name, "the argv hash is the scope key of the argv", controller.HashArgs(argv), controller.ScopeKey(argv...))
	if req.ClaudeCode {
		scopeKeyExpectDiff(t, name, "the log scope follows the Claude workload profile", claudeLaunchLogScope(clone(func(r *launchRequest) { r.ClaudeCode = false }), model, be, argv), logScope)
	}

	// ---- calibration scope key ----------------------------------------
	if run.strategy == nil {
		return
	}
	cal := calibrationScopeKey(req, model, be, caps, run.strategy)
	scopeKeyExpectSame(t, name, "computing the calibration key leaves the verified memo alone", req.verifiedScopeKey, memo)
	calOf := func(edit func(*launchRequest)) string {
		return calibrationScopeKey(clone(edit), model, be, caps, run.strategy)
	}
	scopeKeyExpectSame(t, name, "the calibration key is a pure function of request, hardware and strategy", calOf(nil), cal)
	for _, e := range []struct {
		what   string
		edit   func(*launchRequest)
		rekeys bool
	}{
		{"KV quality", func(r *launchRequest) { r.KVQuality = "q4_0" }, true},
		{"--swa-full in the passthrough", func(r *launchRequest) { r.ExtraArgs = append(append([]string(nil), r.ExtraArgs...), "--swa-full") }, true},
		{"the seated reviewer", func(r *launchRequest) { r.ReviewerReservation = nil }, seated},
		{"ProfilePolicyIdentity", func(r *launchRequest) { r.ProfilePolicyIdentity += "x" }, false},
		{"the verified memo", func(r *launchRequest) { r.verifiedScopeKey, r.verifiedScopeShape = "", "" }, false},
		{"port", func(r *launchRequest) { r.Port++ }, false},
	} {
		if e.rekeys {
			scopeKeyExpectDiff(t, name, "the calibration key must follow "+e.what, calOf(e.edit), cal)
		} else {
			scopeKeyExpectSame(t, name, "the calibration key must not depend on "+e.what, calOf(e.edit), cal)
		}
	}
	if req.GPUsFlag != "" {
		scopeKeyExpectDiff(t, name, "the --gpus restriction is part of the calibration key", calOf(func(r *launchRequest) { r.GPUsFlag = "" }), cal)
	}
	grown := *run.strategy
	grown.ContextSize += 1024
	scopeKeyExpectDiff(t, name, "the calibration key follows the strategy's context", calibrationScopeKey(req, model, be, caps, &grown), cal)
	grown = *run.strategy
	grown.Parallel += 1
	scopeKeyExpectDiff(t, name, "the calibration key follows the strategy's slots", calibrationScopeKey(req, model, be, caps, &grown), cal)
	if runtimeCaps != caps {
		scopeKeyExpectDiff(t, name, "the runtime view of a --gpus launch hashes to another calibration key than the detected one (invalidateRuntimeOOMLaunch passes the runtime view)", calibrationScopeKey(req, model, be, runtimeCaps, run.strategy), cal)
	}

	// ---- reviewer-free candidate --------------------------------------
	if !seated {
		return
	}
	cand := reviewerFreeCandidate(req)
	if cand.verifiedScopeKey != "" || cand.verifiedScopeShape != "" || cand.ReviewerReservation != nil || !cand.ClaudeReviewerDisabled {
		t.Errorf("%s: the reviewer-free candidate must start without a memo and without a reviewer: %q %q %v %v", name, cand.verifiedScopeKey, cand.verifiedScopeShape, cand.ReviewerReservation, cand.ClaudeReviewerDisabled)
	}
	if req.verifiedScopeKey != memo || req.ReviewerReservation == nil || req.ClaudeReviewerDisabled {
		t.Errorf("%s: building the candidate changed the request", name)
	}
	candKey := verifiedConfigScopeKey(&cand, model, be, caps)
	scopeKeyExpectDiff(t, name, "the reviewer-free candidate's verified key must differ from the with-reviewer key", candKey, memo)
	control := clone(func(r *launchRequest) {
		r.ReviewerReservation, r.ClaudeReviewerDisabled = nil, true
	})
	scopeKeyExpectSame(t, name, "without clearing the memo the reviewer-free request would inherit the with-reviewer key (why the candidate clears it)", verifiedConfigScopeKey(control, model, be, caps), memo)
	adopted := clone(nil)
	adoptReviewerFreeCandidate(adopted, &cand)
	if adopted.ReviewerReservation != nil || !adopted.ClaudeReviewerDisabled {
		t.Errorf("%s: adopting the candidate did not drop the reviewer", name)
	}
	scopeKeyExpectSame(t, name, "an adopted candidate carries the candidate's key", adopted.verifiedScopeKey, candKey)
	scopeKeyExpectSame(t, name, "the identity is not recomputed for a candidate", cand.ProfilePolicyIdentity, frozen)
	scopeKeyExpectDiff(t, name, "the candidate's calibration key differs under the same strategy", calibrationScopeKey(&cand, model, be, caps, run.strategy), cal)
	scopeKeyExpectDiff(t, name, "the candidate's profile scope differs (companion lifecycle)", scopeKeyProfileScope(&cand, model, be, caps, home), scope)
}

// scopeKeyCrossChecks compares cases with each other and the two orders.
func scopeKeyCrossChecks(t *testing.T, runs map[string]struct{ dry, launch launchOrderRun }, recs map[string]scopeKeyRecord) {
	t.Helper()
	get := func(name string) (scopeKeyRecord, bool) {
		r, ok := recs[name]
		if !ok || r.Outcome != "plan" {
			t.Errorf("%s: no plan to compare (outcome %q %s)", name, r.Outcome, r.Error)
			return r, false
		}
		return r, true
	}
	// A reviewer-free candidate is the request a relaunch with the reviewer
	// off would plan, so their keys must agree (reviewerFreeCandidate's comment).
	if a, ok := get("mixed3/moe/claude-code"); ok {
		if off, ok := get("mixed3/moe/claude-code+reviewer-off"); ok && a.ReviewerFree != nil {
			scopeKeyExpectSame(t, a.Case, "the reviewer-free candidate's verified key equals a relaunch with --claude-reviewer off", a.ReviewerFree.VerifiedScopeKey, off.VerifiedScopeKey)
			scopeKeyExpectDiff(t, a.Case, "reviewer off is a different calibration scope", a.CalibrationScopeKey, off.CalibrationScopeKey)
		}
	}
	// The proactive reviewer drop adopts the candidate: the key it saves under
	// is the one a reviewer-off relaunch looks up.
	if d, ok := get("gpu24/dense-small/claude-code+self-review-on"); ok {
		if off, ok := get("gpu24/dense-small/claude-code+self-review-on+reviewer-off"); ok {
			scopeKeyExpectSame(t, d.Case, "the adopted reviewer drop files under the reviewer-off key", d.VerifiedScopeKey, off.VerifiedScopeKey)
		}
	}
	// A withdrawn configured --swa-full leaves the request as if it had never
	// been configured, so every key that is computed after the withdrawal
	// matches the plain launch. Only the frozen identity (and so the profile
	// scope) can tell a Claude launch apart: the passthrough was part of the
	// request when the identity was frozen.
	same := func(withdrawn, plain string, identityDiffers bool) {
		w, ok1 := get(withdrawn)
		p, ok2 := get(plain)
		if !ok1 || !ok2 {
			return
		}
		scopeKeyExpectSame(t, w.Case, "verified key equals the plain launch's", w.VerifiedScopeKey, p.VerifiedScopeKey)
		scopeKeyExpectSame(t, w.Case, "verified shape equals the plain launch's", w.VerifiedScopeShape, p.VerifiedScopeShape)
		scopeKeyExpectSame(t, w.Case, "calibration key equals the plain launch's", w.CalibrationScopeKey, p.CalibrationScopeKey)
		scopeKeyExpectSame(t, w.Case, "log scope equals the plain launch's", w.LaunchLogScope, p.LaunchLogScope)
		scopeKeyExpectSame(t, w.Case, "argv hash equals the plain launch's", w.ArgvHash, p.ArgvHash)
		if identityDiffers {
			scopeKeyExpectDiff(t, w.Case, "a Claude launch's frozen identity still carries the configured --swa-full", w.ProfilePolicyIdentity, p.ProfilePolicyIdentity)
			scopeKeyExpectDiff(t, w.Case, "so does its profile scope", w.LaunchProfileScope, p.LaunchProfileScope)
		} else {
			scopeKeyExpectSame(t, w.Case, "an ordinary launch's frozen identity equals the plain launch's", w.ProfilePolicyIdentity, p.ProfilePolicyIdentity)
			scopeKeyExpectSame(t, w.Case, "so does its profile scope", w.LaunchProfileScope, p.LaunchProfileScope)
		}
	}
	same("gpu24/dense-small/swa-full-withdrawn", "gpu24/dense-small/auto", false)
	same("gpu24/dense-small/claude-code+swa-full-withdrawn", "gpu24/dense-small/claude-code", true)

	// Where the two orders plan the same argv and outcome, the keys both
	// compute are equal; the dry run leaves the other two keys unset.
	for name, pair := range runs {
		if d := launchOrderDivergenceOf(pair.dry, pair.launch); len(d) != 0 {
			t.Logf("%s: the orders diverge (%v); their keys are not compared", name, d)
			continue
		}
		rec := recs[name]
		if rec.Outcome != "plan" {
			continue
		}
		scopeKeyExpectSame(t, name, "log scope, dry-run order vs launch order", rec.DryRun.LaunchLogScope, rec.LaunchLogScope)
		scopeKeyExpectSame(t, name, "argv hash, dry-run order vs launch order", rec.DryRun.ArgvHash, rec.ArgvHash)
		scopeKeyExpectSame(t, name, "calibration key, dry-run order vs launch order", rec.DryRun.CalibrationScopeKey, rec.CalibrationScopeKey)
	}
	for name, rec := range recs {
		if rec.DryRun.ProfilePolicyIdentity != "" || rec.DryRun.VerifiedScopeKey != "" {
			t.Errorf("%s: the dry run now computes the profile identity %q or the verified key %q; if planLaunch does this on purpose, record it and update this check",
				name, rec.DryRun.ProfilePolicyIdentity, rec.DryRun.VerifiedScopeKey)
		}
	}
}

// TestLaunchScopeKeyPinning replays the cases twice (two temporary homes),
// asserts the properties, and compares the literals with the recorded file.
func TestLaunchScopeKeyPinning(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("golden launch plans are recorded on Linux")
	}
	goldenPlanModels["deepseek4-moe"] = scopeKeyDeepSeek4MoE
	t.Cleanup(func() { delete(goldenPlanModels, "deepseek4-moe") })

	cases := scopeKeyCases()
	var passes [2][]scopeKeyRecord
	for pass := range passes {
		t.Run(fmt.Sprintf("home-%d", pass), func(t *testing.T) {
			home := launchOrderTestEnv(t)
			runs := map[string]struct{ dry, launch launchOrderRun }{}
			recs := map[string]scopeKeyRecord{}
			for _, c := range cases {
				var dry, launch, again launchOrderRun
				launchOrderRunCase(t, c.launchOrderCase, func(t *testing.T) {
					dry = launchOrderReplay(t, c.launchOrderCase, false, nil)
					launch = launchOrderReplay(t, c.launchOrderCase, true, nil)
					again = launchOrderReplay(t, c.launchOrderCase, true, nil)
				})
				rec := scopeKeyRecordOf(c, dry, launch, home)
				if second := scopeKeyRecordOf(c, dry, again, home); !bytes.Equal(scopeKeyMarshal(t, rec), scopeKeyMarshal(t, second)) {
					t.Errorf("%s: two identical launch-order replays computed different keys\nfirst:  %s\nsecond: %s", c.name, scopeKeyMarshal(t, rec), scopeKeyMarshal(t, second))
				}
				passes[pass] = append(passes[pass], rec)
				runs[c.name] = struct{ dry, launch launchOrderRun }{dry, launch}
				recs[c.name] = rec
				if pass == 0 {
					scopeKeyProperties(t, c, launch, home)
				}
			}
			if pass == 0 {
				scopeKeyCrossChecks(t, runs, recs)
			}
		})
	}
	first, second := scopeKeyMarshalAll(t, passes[0]), scopeKeyMarshalAll(t, passes[1])
	if !bytes.Equal(first, second) {
		t.Errorf("the keys depend on the temporary home or directory the replay ran in\nhome-0: %s\nhome-1: %s", first, second)
	}
	scopeKeysCompareFile(t, first)
}

func scopeKeyMarshal(t *testing.T, rec scopeKeyRecord) []byte {
	t.Helper()
	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func scopeKeyMarshalAll(t *testing.T, recs []scopeKeyRecord) []byte {
	t.Helper()
	b, err := json.MarshalIndent(recs, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return append(b, '\n')
}

// scopeKeysCompareFile compares the computed file with the recorded one. A
// missing or differing file is logged in a recoverable form and fails.
func scopeKeysCompareFile(t *testing.T, got []byte) {
	t.Helper()
	if *updateScopeKeys {
		if err := os.WriteFile(scopeKeysGoldenPath, got, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(scopeKeysGoldenPath)
	if os.IsNotExist(err) {
		launchOrderLogEncoded(t, "scope-keys-golden", got)
		t.Errorf("%s is missing; recover it from the scope-keys-golden lines above or regenerate with -update-scope-keys", scopeKeysGoldenPath)
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(got, want) {
		return
	}
	launchOrderLogEncoded(t, "scope-keys-golden", got)
	gotLines, wantLines := strings.Split(string(got), "\n"), strings.Split(string(want), "\n")
	for i := 0; i < max(len(gotLines), len(wantLines)); i++ {
		var g, w string
		if i < len(gotLines) {
			g = gotLines[i]
		}
		if i < len(wantLines) {
			w = wantLines[i]
		}
		if g != w {
			t.Errorf("%s line %d changed\nwant: %s\ngot:  %s", scopeKeysGoldenPath, i+1, w, g)
		}
	}
	t.Errorf("scope keys differ from %s; a key that moved strands saved state on users' machines, so change it only on purpose and regenerate with -update-scope-keys", scopeKeysGoldenPath)
}
