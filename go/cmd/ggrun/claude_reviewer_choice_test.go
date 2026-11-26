package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

func choiceTestScope() (*placement.ModelProfile, *detect.Capabilities) {
	model := &placement.ModelProfile{Path: "/models/choice-test.gguf", TotalSizeMB: 20000, SizeBytes: 20000 << 20, ModelArch: "llama", NumLayers: 40}
	caps := &detect.Capabilities{
		OS: "linux", Arch: "amd64",
		GPUs: []detect.GPU{{Index: 0, Name: "gpu-a", VRAMTotalMB: 24564, PCIBusID: "0000:01:00.0"}},
	}
	caps.RAM.TotalMB = 65536
	return model, caps
}

func choiceTestReservation() *placement.CompanionReservation {
	return &placement.CompanionReservation{Name: claudeReviewerCompanionName, VRAMMB: 2600}
}

// approveSelfClassifyInteractively answers yes to the reactive fallback and
// persists the answer the way cmdLaunch does.
func approveSelfClassifyInteractively(t *testing.T, cacheDir string, model *placement.ModelProfile, caps *detect.Capabilities) {
	t.Helper()
	req := &launchRequest{ClaudeCode: true, NoMMap: true, ReviewerReservation: choiceTestReservation()}
	compute := func(*launchRequest) (*placement.Strategy, error) {
		return &placement.Strategy{Type: placement.MoEOffload}, nil
	}
	if _, err := tryResidentWithoutClaudeReviewer(req, fmt.Errorf("no fit"), strings.NewReader("yes\n"), io.Discard, true, compute); err != nil {
		t.Fatal(err)
	}
	if !req.claudeSelfClassifyApproved {
		t.Fatal("interactive yes was not marked as an approval to remember")
	}
	var out bytes.Buffer
	rememberClaudeSelfClassify(req, cacheDir, model, caps, &out)
	if !strings.Contains(out.String(), "saved for this launch scope") {
		t.Fatalf("approval was not saved: %q", out.String())
	}
}

// Invariant: a self-classify answer given once is reused by a relaunch of the
// same scope, before the reviewer reservation exists, and without asking.
func TestRememberedSelfClassifyIsReusedOnRelaunch(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	cacheDir := t.TempDir()
	model, caps := choiceTestScope()
	approveSelfClassifyInteractively(t, cacheDir, model, caps)

	relaunch := &launchRequest{ClaudeCode: true, NoMMap: true}
	var out bytes.Buffer
	if !applyRememberedClaudeSelfClassify(relaunch, cacheDir, model, caps, &out) {
		t.Fatalf("saved answer was not reused: %q", out.String())
	}
	if !relaunch.ClaudeReviewerDisabled || !relaunch.claudeSelfClassifyRemembered {
		t.Fatalf("relaunch still plans a reviewer: %#v", relaunch)
	}
	if got := claudeReviewerReservation(relaunch, caps, cacheDir); got != nil {
		t.Fatalf("relaunch reserved a reviewer seat %#v after a saved self-classify answer", got)
	}
	if !strings.Contains(out.String(), "saved choice") {
		t.Fatalf("reuse is not announced: %q", out.String())
	}
	// A reused answer is not written again.
	before, err := os.ReadFile(claudeReviewerChoicePath(cacheDir, claudeReviewerChoiceScope(relaunch, model, caps)))
	if err != nil {
		t.Fatal(err)
	}
	rememberClaudeSelfClassify(relaunch, cacheDir, model, caps, io.Discard)
	after, _ := os.ReadFile(claudeReviewerChoicePath(cacheDir, claudeReviewerChoiceScope(relaunch, model, caps)))
	if !bytes.Equal(before, after) {
		t.Fatal("a reused answer rewrote the record")
	}
}

// Invariant: the proactive question persists only an interactive yes; the
// non-interactive env opt-in and an interactive decline leave no record.
func TestProactiveSelfReviewPersistsOnlyInteractiveYes(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	// The same 24 GB card and 22 GB dense model as
	// TestProactivelyDropReviewerSelfReviewRequiresConsent: no room for a
	// separate reviewer, so the gate asks.
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24564}}}
	tightModel := &placement.ModelProfile{Path: "/models/tight.gguf", TotalSizeMB: 22000, SizeBytes: 22000 * 1024 * 1024}
	plan := &placement.Strategy{Type: placement.SingleGPU, MainGPU: 0, ContextSize: 65536, KVType: "q4_0", KVPlacement: "gpu"}
	compute := func(*launchRequest) (*placement.Strategy, error) { return plan, nil }
	reservation := &placement.CompanionReservation{Name: claudeReviewerCompanionName, VRAMMB: 4600}
	run := func(answer string, interactive bool) (*launchRequest, string) {
		cacheDir := t.TempDir()
		req := &launchRequest{ClaudeCode: true, ReviewerReservation: reservation}
		proactivelyDropReviewerForVRAMModel(req, caps, tightModel, plan, compute, strings.NewReader(answer), io.Discard, interactive, "")
		rememberClaudeSelfClassify(req, cacheDir, tightModel, caps, io.Discard)
		return req, cacheDir
	}
	saved := func(cacheDir string) bool {
		_, err := os.Stat(claudeReviewerChoicePath(cacheDir, claudeReviewerChoiceScope(&launchRequest{}, tightModel, caps)))
		return err == nil
	}

	req, dir := run("yes\n", true)
	if !req.ClaudeReviewerDisabled || !saved(dir) {
		t.Fatalf("interactive yes was not remembered: %#v", req)
	}
	req, dir = run("n\n", true)
	if req.ClaudeReviewerDisabled || saved(dir) {
		t.Fatalf("interactive decline was remembered as self-classify: %#v", req)
	}
	t.Setenv("GGRUN_CLAUDE_AUTO_SELF_REVIEW", "on")
	req, dir = run("", false)
	if !req.ClaudeReviewerDisabled || saved(dir) {
		t.Fatalf("env opt-in must apply to this launch only: disabled=%t saved=%t", req.ClaudeReviewerDisabled, saved(dir))
	}
}

// Invariant: --claude-reviewer overrides the saved answer and replaces it;
// --claude-reviewer off keeps it; a dry-run style apply never deletes it.
func TestExplicitReviewerFlagOverridesSavedSelfClassify(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	cacheDir := t.TempDir()
	model, caps := choiceTestScope()
	approveSelfClassifyInteractively(t, cacheDir, model, caps)
	path := claudeReviewerChoicePath(cacheDir, claudeReviewerChoiceScope(&launchRequest{}, model, caps))

	for _, value := range []string{claudeReviewerAuto, claudeReviewerQwen2B} {
		req := &launchRequest{ClaudeCode: true, ClaudeReviewerOverride: value}
		if applyRememberedClaudeSelfClassify(req, cacheDir, model, caps, io.Discard) || req.ClaudeReviewerDisabled {
			t.Fatalf("--claude-reviewer %s did not override the saved answer: %#v", value, req)
		}
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("apply alone (dry-run) deleted the saved answer: %v", err)
		}
	}

	off := &launchRequest{ClaudeCode: true, ClaudeReviewerOverride: claudeReviewerOff, ClaudeReviewerDisabled: true}
	forgetClaudeSelfClassifyForExplicitReviewer(off, cacheDir, model, caps, io.Discard)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("--claude-reviewer off removed the saved self-classify answer: %v", err)
	}

	explicit := &launchRequest{ClaudeCode: true, ClaudeReviewerOverride: claudeReviewerAuto}
	var out bytes.Buffer
	forgetClaudeSelfClassifyForExplicitReviewer(explicit, cacheDir, model, caps, &out)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("an explicit reviewer launch kept the saved answer: %v", err)
	}
	if !strings.Contains(out.String(), "replaces the saved self-classify choice") {
		t.Fatalf("replacement is not announced: %q", out.String())
	}
	plain := &launchRequest{ClaudeCode: true}
	if applyRememberedClaudeSelfClassify(plain, cacheDir, model, caps, io.Discard) || plain.ClaudeReviewerDisabled {
		t.Fatalf("a replaced answer still applied: %#v", plain)
	}
}

// Invariant: GGRUN_CLAUDE_AUTO_REVIEWER overrides the saved answer for the
// launch it is set on and never erases it.
func TestReviewerEnvOverridesSavedSelfClassify(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	cacheDir := t.TempDir()
	model, caps := choiceTestScope()
	approveSelfClassifyInteractively(t, cacheDir, model, caps)

	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "on")
	req := &launchRequest{ClaudeCode: true}
	if applyRememberedClaudeSelfClassify(req, cacheDir, model, caps, io.Discard) || req.ClaudeReviewerDisabled {
		t.Fatalf("GGRUN_CLAUDE_AUTO_REVIEWER=on did not override the saved answer: %#v", req)
	}
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	req = &launchRequest{ClaudeCode: true}
	if !applyRememberedClaudeSelfClassify(req, cacheDir, model, caps, io.Discard) {
		t.Fatal("the env override erased the saved answer")
	}
}

// Invariant: an answer is scoped; another model, card set or GPU restriction
// does not inherit it.
func TestSavedSelfClassifyIsScoped(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	cacheDir := t.TempDir()
	model, caps := choiceTestScope()
	approveSelfClassifyInteractively(t, cacheDir, model, caps)

	otherModel := *model
	otherModel.Path = "/models/other.gguf"
	if applyRememberedClaudeSelfClassify(&launchRequest{ClaudeCode: true}, cacheDir, &otherModel, caps, io.Discard) {
		t.Fatal("another model inherited the answer")
	}
	otherCaps := *caps
	otherCaps.GPUs = append([]detect.GPU{}, caps.GPUs...)
	otherCaps.GPUs = append(otherCaps.GPUs, detect.GPU{Index: 1, Name: "gpu-b", VRAMTotalMB: 12288, PCIBusID: "0000:02:00.0"})
	if applyRememberedClaudeSelfClassify(&launchRequest{ClaudeCode: true}, cacheDir, model, &otherCaps, io.Discard) {
		t.Fatal("other hardware inherited the answer")
	}
	if applyRememberedClaudeSelfClassify(&launchRequest{ClaudeCode: true, GPUsFlag: "0"}, cacheDir, model, caps, io.Discard) {
		t.Fatal("a GPU-restricted launch inherited the unrestricted answer")
	}
	if applyRememberedClaudeSelfClassify(&launchRequest{}, cacheDir, model, caps, io.Discard) {
		t.Fatal("a non-Claude launch applied the answer")
	}
}

// Invariant: absent, corrupt, foreign-scope, foreign-schema or unknown records
// never disable the reviewer; the launch falls back to asking (interactive)
// or the fail-closed default (non-interactive).
func TestSavedSelfClassifyFailurePathsFallBack(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	model, caps := choiceTestScope()
	scope := claudeReviewerChoiceScope(&launchRequest{}, model, caps)
	records := map[string]string{
		"absent":       "",
		"corrupt":      "{not json",
		"foreignScope": `{"schema_version":1,"scope":"` + strings.Repeat("0", 64) + `","choice":"self-classify"}`,
		"futureSchema": `{"schema_version":2,"scope":"` + scope + `","choice":"self-classify"}`,
		"unknown":      `{"schema_version":1,"scope":"` + scope + `","choice":"maybe"}`,
	}
	for name, body := range records {
		t.Run(name, func(t *testing.T) {
			cacheDir := t.TempDir()
			if body != "" {
				path := claudeReviewerChoicePath(cacheDir, scope)
				if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			req := &launchRequest{ClaudeCode: true, NoMMap: true}
			var out bytes.Buffer
			if applyRememberedClaudeSelfClassify(req, cacheDir, model, caps, &out) || req.ClaudeReviewerDisabled {
				t.Fatalf("%s record disabled the reviewer: %#v", name, req)
			}
			if body != "" && !strings.Contains(out.String(), "ignoring the saved reviewer choice") {
				t.Fatalf("%s record was ignored silently: %q", name, out.String())
			}
			req.ReviewerReservation = choiceTestReservation()
			compute := func(*launchRequest) (*placement.Strategy, error) {
				return &placement.Strategy{Type: placement.MoEOffload}, nil
			}
			_, err := tryResidentWithoutClaudeReviewer(req, fmt.Errorf("no fit"), strings.NewReader(""), io.Discard, false, compute)
			if err == nil || !strings.Contains(err.Error(), "GGRUN_CLAUDE_AUTO_REVIEWER=off") {
				t.Fatalf("non-interactive fallback lost its fail-closed default: %v", err)
			}
			var prompt bytes.Buffer
			if _, err := tryResidentWithoutClaudeReviewer(req, fmt.Errorf("no fit"), strings.NewReader("no\n"), &prompt, true, compute); err == nil {
				t.Fatal("interactive decline adopted self-classify")
			}
			if !strings.Contains(prompt.String(), "[y/N]") {
				t.Fatalf("interactive launch did not ask: %q", prompt.String())
			}
		})
	}
}

// Invariant: an unwritable store degrades to a warning; the launch keeps the
// answer it was given.
func TestSelfClassifySaveFailureOnlyWarns(t *testing.T) {
	model, caps := choiceTestScope()
	blocker := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(blocker, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	req := &launchRequest{ClaudeCode: true, ClaudeReviewerDisabled: true, claudeSelfClassifyApproved: true}
	var out bytes.Buffer
	rememberClaudeSelfClassify(req, blocker, model, caps, &out)
	if !strings.Contains(out.String(), "could not be saved") || !req.ClaudeReviewerDisabled {
		t.Fatalf("save failure: out=%q req=%#v", out.String(), req)
	}
}

// Invariant: a launch that adopts self-classify files its verified config under
// the reviewer-free key a remembered relaunch looks up, never under the
// with-reviewer key the first plan memoized (a relaunch would miss it and
// re-plan, and a reviewer launch would replay a plan with no reviewer seat).
func TestSelfClassifyDropFilesVerifiedConfigUnderRelaunchKey(t *testing.T) {
	t.Setenv("GGRUN_CLAUDE_AUTO_REVIEWER", "")
	t.Setenv("GGRUN_CLAUDE_AUTO_SELF_REVIEW", "")
	be := fitTestBackend()
	caps := fitTestCaps(24564)

	relaunchKey := func(t *testing.T, cacheDir string, model *placement.ModelProfile, compute func(*launchRequest) (*placement.Strategy, error)) string {
		t.Helper()
		relaunch := &launchRequest{CtxFlag: "fit", ClaudeCode: true, NoMMap: true}
		if !applyRememberedClaudeSelfClassify(relaunch, cacheDir, model, caps, io.Discard) {
			t.Fatal("the answer was not remembered")
		}
		relaunch.ReviewerReservation = claudeReviewerReservation(relaunch, caps, cacheDir)
		if _, err := compute(relaunch); err != nil {
			t.Fatal(err)
		}
		return relaunch.verifiedScopeKey
	}

	t.Run("reactive", func(t *testing.T) {
		cacheDir := t.TempDir()
		model := fitTestModel(65536, 4000)
		compute := func(r *launchRequest) (*placement.Strategy, error) {
			placementOptionsFromRequestCaps(r, model, be, cacheDir, caps)
			if r.ReviewerReservation != nil {
				return nil, fmt.Errorf("no resident fit with the reviewer")
			}
			return &placement.Strategy{Type: placement.MoEOffload}, nil
		}
		req := &launchRequest{CtxFlag: "fit", ClaudeCode: true, NoMMap: true, ReviewerReservation: choiceTestReservation()}
		_, firstErr := compute(req)
		withReviewer := req.verifiedScopeKey
		if withReviewer == "" || firstErr == nil {
			t.Fatalf("first plan did not memoize a with-reviewer key: key=%q err=%v", withReviewer, firstErr)
		}
		if _, err := tryResidentWithoutClaudeReviewer(req, firstErr, strings.NewReader("yes\n"), io.Discard, true, compute); err != nil {
			t.Fatal(err)
		}
		rememberClaudeSelfClassify(req, cacheDir, model, caps, io.Discard)
		saved := verifiedConfigScopeKey(req, model, be, caps)
		if saved == withReviewer {
			t.Fatal("the reviewer-free plan is filed under the with-reviewer key")
		}
		if lookup := relaunchKey(t, cacheDir, model, compute); saved != lookup {
			t.Fatalf("saved under %q, relaunch looks up %q", saved, lookup)
		}
	})

	t.Run("proactive", func(t *testing.T) {
		cacheDir := t.TempDir()
		model := fitTestModel(65536, 22000) // no room for a 4600 MiB reviewer beside it on 24 GB
		plan := &placement.Strategy{Type: placement.SingleGPU, MainGPU: 0, ContextSize: 65536, KVType: "q4_0", KVPlacement: "gpu"}
		compute := func(r *launchRequest) (*placement.Strategy, error) {
			placementOptionsFromRequestCaps(r, model, be, cacheDir, caps)
			return plan, nil
		}
		req := &launchRequest{CtxFlag: "fit", ClaudeCode: true, ReviewerReservation: &placement.CompanionReservation{Name: claudeReviewerCompanionName, VRAMMB: 4600}}
		if _, err := compute(req); err != nil {
			t.Fatal(err)
		}
		withReviewer := req.verifiedScopeKey
		proactivelyDropReviewerForVRAMModel(req, caps, model, plan, compute, strings.NewReader("yes\n"), io.Discard, true, "")
		if !req.ClaudeReviewerDisabled {
			t.Fatal("the proactive gate did not drop the reviewer")
		}
		rememberClaudeSelfClassify(req, cacheDir, model, caps, io.Discard)
		saved := verifiedConfigScopeKey(req, model, be, caps)
		if withReviewer == "" || saved == withReviewer {
			t.Fatalf("the reviewer-free plan is filed under the with-reviewer key %q", withReviewer)
		}
		relaunch := &launchRequest{CtxFlag: "fit", ClaudeCode: true}
		if !applyRememberedClaudeSelfClassify(relaunch, cacheDir, model, caps, io.Discard) {
			t.Fatal("the answer was not remembered")
		}
		if _, err := compute(relaunch); err != nil {
			t.Fatal(err)
		}
		if saved != relaunch.verifiedScopeKey {
			t.Fatalf("saved under %q, relaunch looks up %q", saved, relaunch.verifiedScopeKey)
		}
	})
}
