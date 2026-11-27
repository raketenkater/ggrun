package main

import "testing"

// planLaunch runs the launch's resume-context step: a dry run with
// --claude-resume plans the context the launch serves for the same session,
// and a dry run without it never measures a session.
func TestPlanLaunchResumeContextMatchesTheLaunch(t *testing.T) {
	launchOrderTestEnv(t) // resumeSessionTokens: a 200000-token session
	mk := func(name string, args ...string) launchOrderCase {
		return launchOrderCase{goldenPlanCase: goldenPlanCase{name: name, hw: "mixed3", model: "moe", args: args}}
	}
	resume := mk("mixed3/moe/claude-resume", "--claude-resume", "golden-session")
	var dry, launch launchOrderRun
	launchOrderRunCase(t, resume, func(t *testing.T) {
		dry = launchOrderReplay(t, resume, false, nil)
		launch = launchOrderReplay(t, resume, true, nil)
	})
	fresh := mk("mixed3/moe/claude-code", "--claude-code")
	var plain launchOrderRun
	launchOrderRunCase(t, fresh, func(t *testing.T) {
		saved := resumeSessionTokens
		resumeSessionTokens = func(string, string) (string, int, bool) {
			t.Errorf("a dry run without --claude-resume measured a session")
			return "", 0, false
		}
		t.Cleanup(func() { resumeSessionTokens = saved })
		plain = launchOrderReplay(t, fresh, false, nil)
	})
	for _, r := range []launchOrderRun{dry, launch, plain} {
		if r.record.Outcome != "plan" || r.record.Strategy == nil {
			t.Fatalf("%s (%s): %s %s", r.record.Case, r.record.Order, r.record.Outcome, r.record.Error)
		}
	}
	got, want := dry.record.Strategy, launch.record.Strategy
	if got.ContextSize != want.ContextSize || got.Parallel != want.Parallel {
		t.Errorf("dry run plans ctx %d np %d, the launch ctx %d np %d", got.ContextSize, got.Parallel, want.ContextSize, want.Parallel)
	}
	if need := resumeContextNeed(200000); got.ContextSize/max(got.Parallel, 1) < need {
		t.Errorf("dry run gives %d per slot, the session needs %d", got.ContextSize/max(got.Parallel, 1), need)
	}
	if plain.record.Strategy.ContextSize >= got.ContextSize {
		t.Errorf("the resume did not raise the context: %d without it, %d with it", plain.record.Strategy.ContextSize, got.ContextSize)
	}
	if dry.record.Plan == nil || launch.record.Plan == nil || dry.record.Plan.LaunchScope != launch.record.Plan.LaunchScope {
		t.Errorf("dry run and launch disagree on the launch scope")
	}
}
