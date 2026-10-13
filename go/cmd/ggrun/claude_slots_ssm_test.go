package main

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/placement"
)

// Default Claude mode serves an SSM-layout model on one slot; an explicit
// count or the parallel profile is kept (with a warning); other models and
// plain launches are unchanged.
func TestClaudeHybridSSMSlots(t *testing.T) {
	warned := 0
	prev := claudeHybridSSMWarn
	claudeHybridSSMWarn = func(int) { warned++ }
	t.Cleanup(func() { claudeHybridSSMWarn = prev })
	ssm := &placement.ModelProfile{Name: "Qwen3.8-27B", HasSSM: 1}
	dense := &placement.ModelProfile{Name: "Gemma 4", HasSSM: 0}
	cases := []struct {
		name     string
		model    *placement.ModelProfile
		claude   bool
		explicit bool
		profile  string
		in, want int
	}{
		{"default claude mode, SSM", ssm, true, false, "", 4, 1},
		{"explicit --parallel 4, SSM", ssm, true, true, "", 4, 4},
		{"parallel profile, SSM", ssm, true, false, claudeProfileParallel, 4, 4},
		{"default claude mode, no SSM", dense, true, false, "", 4, 4},
		{"plain launch, SSM", ssm, false, false, "", 4, 4},
		{"one slot already", ssm, true, false, "", 1, 1},
	}
	for _, c := range cases {
		if got := claudeHybridSSMSlots(c.in, c.model, c.claude, c.explicit, c.profile); got != c.want {
			t.Errorf("%s: %d slots, want %d", c.name, got, c.want)
		}
	}
	if warned != 2 {
		t.Errorf("warned %d times, want once per honored multi-slot request (2)", warned)
	}
	// The default request reaching this policy is the 4-slot Claude default.
	if got := claudeHybridSSMSlots(claudeCodeParallel(1, true, false, ""), ssm, true, false, ""); got != 1 {
		t.Errorf("default Claude request for an SSM model: %d slots, want 1", got)
	}
}
