package clientkit

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func capacityProfile(t *testing.T) Profile {
	t.Helper()
	b, err := os.ReadFile("testdata/profile-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var p Profile
	if err = json.Unmarshal(b, &p); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestUtilityCapacityIsOptionalAndConservative(t *testing.T) {
	p := capacityProfile(t)
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name  string
		route UtilityRoute
		valid bool
	}{
		{"shared", UtilityRoute{ContextPerSlot: p.ContextPerSlot, Slots: p.Slots}, true},
		{"reviewer", UtilityRoute{ContextPerSlot: 1024, Slots: 1, SeparateBackend: true, FallbackToMain: true}, true},
		{"overflow", UtilityRoute{ContextPerSlot: p.ContextPerSlot + 1, Slots: 1, SeparateBackend: true}, false},
		{"unknown context", UtilityRoute{Slots: 1, SeparateBackend: true}, false},
		{"unknown slots", UtilityRoute{ContextPerSlot: 1024, SeparateBackend: true}, false},
		{"shared mismatch", UtilityRoute{ContextPerSlot: 1, Slots: p.Slots}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p.UtilityRoute = &tc.route
			if (p.Validate() == nil) != tc.valid {
				t.Fatalf("unexpected validity: %+v", tc.route)
			}
		})
	}
}

func TestWorkflowCapUsesExecutorSlots(t *testing.T) {
	p := capacityProfile(t)
	p.Slots = 2
	p.SeparateWorker = true
	p.UtilityRoute = &UtilityRoute{ContextPerSlot: 1024, Slots: 8, SeparateBackend: true, FallbackToMain: true}
	for _, tc := range []struct{ input, want string }{{"", "2"}, {"1", "1"}, {"8", "2"}, {"0", "2"}, {"bad", "2"}} {
		env := p.Environment("http://127.0.0.1", "secret", []string{"CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS=" + tc.input})
		if !strings.Contains(strings.Join(env, "\n"), "CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS="+tc.want+"\n") {
			t.Fatalf("unexpected executor cap for %q", tc.input)
		}
	}
}
