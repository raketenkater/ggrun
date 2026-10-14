package main

import (
	"errors"
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

func stubResumeTokens(t *testing.T, tokens int, ok bool) {
	t.Helper()
	saved := resumeSessionTokens
	resumeSessionTokens = func(string, string) (string, int, bool) { return "s1", tokens, ok }
	t.Cleanup(func() { resumeSessionTokens = saved })
}

func TestResumeContextNeedLeavesRoomForTheNextTurn(t *testing.T) {
	if got := resumeContextNeed(155588); got != 188416 {
		t.Fatalf("need = %d, want 188416 (155588 + 32768 rounded up to 4096)", got)
	}
	if resumeContextNeed(0) != 0 {
		t.Fatal("an unmeasured session must not request context")
	}
}

// Observed 2026-10-14: a 155,588-token session resumed into a launch that had
// learned a GPU reserve and planned 144,384; every turn was rejected.
func TestResumeReplansAnAutomaticContextThatIsTooSmall(t *testing.T) {
	stubResumeTokens(t, 155588, true)
	req := &launchRequest{ClaudeCode: true, ClaudeResume: "s1", CtxFlag: "fit"}
	model := &placement.ModelProfile{CTXTrain: 262144}
	var asked []string
	compute := func(r *launchRequest) (*placement.Strategy, error) {
		asked = append(asked, r.CtxFlag)
		return &placement.Strategy{ContextSize: 188416, Parallel: 1}, nil
	}
	got := ensureResumeContext(req, "", model, &placement.Strategy{ContextSize: 144384, Parallel: 1}, compute)
	if len(asked) != 1 || asked[0] != "188416" || got.ContextSize != 188416 || req.CtxFlag != "188416" {
		t.Fatalf("asked %v, got ctx %d, req ctx %q", asked, got.ContextSize, req.CtxFlag)
	}
}

func TestResumeContextLeavesFittingAndUnmeasuredPlansAlone(t *testing.T) {
	model := &placement.ModelProfile{CTXTrain: 262144}
	fail := func(*launchRequest) (*placement.Strategy, error) {
		t.Fatal("re-planned a launch that needed no change")
		return nil, nil
	}
	plan := &placement.Strategy{ContextSize: 262144, Parallel: 1}

	stubResumeTokens(t, 155588, true)
	if got := ensureResumeContext(&launchRequest{ClaudeCode: true, ClaudeResume: "s1"}, "", model, plan, fail); got != plan {
		t.Fatal("a plan that holds the session was changed")
	}
	if got := ensureResumeContext(&launchRequest{ClaudeCode: true}, "", model, &placement.Strategy{ContextSize: 1000, Parallel: 1}, fail); got.ContextSize != 1000 {
		t.Fatal("a fresh session was re-planned")
	}
	if got := ensureResumeContext(&launchRequest{ClaudeCode: true, ClaudeResume: "s1", ClaudeResumeForce: true}, "", model, &placement.Strategy{ContextSize: 1000, Parallel: 1}, fail); got.ContextSize != 1000 {
		t.Fatal("--claude-resume-force was re-planned")
	}
	stubResumeTokens(t, 0, false)
	if got := ensureResumeContext(&launchRequest{ClaudeCode: true, ClaudeResume: "s1"}, "", model, &placement.Strategy{ContextSize: 1000, Parallel: 1}, fail); got.ContextSize != 1000 {
		t.Fatal("an unreadable transcript was re-planned")
	}
}

func TestResumeContextKeepsAnExplicitContextAndTheModelLimit(t *testing.T) {
	fail := func(*launchRequest) (*placement.Strategy, error) {
		t.Fatal("re-planned against an explicit choice or the model limit")
		return nil, nil
	}
	stubResumeTokens(t, 155588, true)
	req := &launchRequest{ClaudeCode: true, ClaudeResume: "s1", CtxFlag: "131072"}
	if got := ensureResumeContext(req, "", &placement.ModelProfile{CTXTrain: 262144}, &placement.Strategy{ContextSize: 131072, Parallel: 1}, fail); got.ContextSize != 131072 || req.CtxFlag != "131072" {
		t.Fatal("an explicit --ctx was overridden")
	}
	stubResumeTokens(t, 250000, true)
	if got := ensureResumeContext(&launchRequest{ClaudeCode: true, ClaudeResume: "s1"}, "", &placement.ModelProfile{CTXTrain: 262144}, &placement.Strategy{ContextSize: 200000, Parallel: 1}, fail); got.ContextSize != 200000 {
		t.Fatal("planned beyond the model's trained context")
	}
}

// With several automatic slots, the whole conversation needs one slot of its
// size; when the multi-slot plan does not fit, one slot is tried.
func TestResumeContextFallsBackToOneSlot(t *testing.T) {
	stubResumeTokens(t, 155588, true)
	req := &launchRequest{ClaudeCode: true, ClaudeResume: "s1"}
	var asked []string
	compute := func(r *launchRequest) (*placement.Strategy, error) {
		asked = append(asked, r.CtxFlag)
		if r.Parallel != 1 {
			return nil, errors.New("does not fit")
		}
		return &placement.Strategy{ContextSize: 188416, Parallel: 1}, nil
	}
	got := ensureResumeContext(req, "", &placement.ModelProfile{CTXTrain: 262144}, &placement.Strategy{ContextSize: 262144, Parallel: 2}, compute)
	if len(asked) != 2 || asked[0] != "376832" || asked[1] != "188416" || got.Parallel != 1 || !req.ParallelSet {
		t.Fatalf("asked %v, got %+v, req parallel set %v", asked, got, req.ParallelSet)
	}
}
