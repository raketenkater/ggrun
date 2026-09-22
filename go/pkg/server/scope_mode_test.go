//go:build linux

package server

import (
	"os"
	"strings"
	"testing"
)

// Issue #64: as root, `ggrun <mode> --ai-tune` failed with systemd's raw
//
//	Failed to connect to user scope bus via local transport:
//	$DBUS_SESSION_BUS_ADDRESS and $XDG_RUNTIME_DIR not defined
//
// because every scope command was built with `--user`, and a root session has no
// per-user systemd manager or session bus. The failure also surfaced late, from
// cmd.Start, wrapped as "start server: ...", so support.go classified it as
// unclassified_launch_failure rather than a memory problem.
//
// These tests pin the three properties that make the fix correct: the mode is
// chosen by capability, argv follows the mode, and an unusable instance refuses
// explicitly instead of launching uncontained.

// withScopeMode forces a mode via the env seam so the root branch is testable
// without being root and without a real systemd.
func withScopeMode(t *testing.T, mode string) {
	t.Helper()
	prev, had := os.LookupEnv(scopeModeEnvOverride)
	if err := os.Setenv(scopeModeEnvOverride, mode); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if had {
			_ = os.Setenv(scopeModeEnvOverride, prev)
		} else {
			_ = os.Unsetenv(scopeModeEnvOverride)
		}
	})
}

// The start path must create the scope in the instance the mode names. Before the
// fix this was hardcoded `--user`, which is the whole of issue #64.
func TestScopedCommandArgsUsesTheResolvedScopeInstance(t *testing.T) {
	withScopeMode(t, "user")
	userArgs, err := scopedCommandArgsWithUnit([]string{"llama-server"}, 4096, "ggrun-test.scope")
	if err != nil {
		t.Fatalf("user mode refused: %v", err)
	}
	if !containsArg(userArgs, "--user") || !containsArg(userArgs, "--scope") {
		t.Errorf("user mode did not produce both --user and --scope: %v", userArgs)
	}

	withScopeMode(t, "system")
	sysArgs, err := scopedCommandArgsWithUnit([]string{"llama-server"}, 4096, "ggrun-test.scope")
	if err != nil {
		t.Fatalf("system mode refused: %v", err)
	}
	if containsArg(sysArgs, "--user") {
		t.Errorf("system mode still passed --user, which is the root failure: %v", sysArgs)
	}
	if !containsArg(sysArgs, "--scope") {
		t.Errorf("system mode did not produce a scope argv: %v", sysArgs)
	}
	// Containment must survive: the same limits, whichever instance owns it.
	for _, want := range []string{"MemoryMax=4096M", "MemorySwapMax=0", "OOMPolicy=kill"} {
		if !containsArg(sysArgs, want) {
			t.Errorf("system mode dropped containment property %q: %v", want, sysArgs)
		}
	}
}

// The three-way contract: every scope operation must address the SAME instance.
// Creating a system scope and reaping it with `systemctl --user` would leave the
// transient unit running, which is the leak the cleanup exists to prevent.
func TestSystemctlArgsMatchesTheScopeInstance(t *testing.T) {
	cases := []struct {
		mode     scopeMode
		wantUser bool
	}{
		{scopeUser, true},
		{scopeSystem, false},
	}
	for _, c := range cases {
		for _, verb := range []string{"stop", "reset-failed", "is-active", "show", "set-property"} {
			got := systemctlArgs(c.mode, verb, "ggrun-test.scope")
			hasUser := false
			for _, a := range got[1:] {
				if a == "--user" {
					hasUser = true
				}
			}
			if hasUser != c.wantUser {
				t.Errorf("mode %v verb %q: --user=%v, want %v (argv %v)",
					c.mode, verb, hasUser, c.wantUser, got)
			}
			if got[0] != "systemctl" {
				t.Errorf("verb %q: argv does not start with systemctl: %v", verb, got)
			}
		}
	}
}

// An instance that cannot own a scope must refuse BEFORE any argv exists, with a
// ggrun-authored message, rather than letting systemd's dbus error surface from
// cmd.Start. The message must also be one the classifier can route.
func TestUnsupportedScopeModeRefusesExplicitly(t *testing.T) {
	withScopeMode(t, "unsupported")
	args, err := scopedCommandArgsWithUnit([]string{"llama-server"}, 4096, "ggrun-test.scope")
	if err == nil {
		t.Fatal("an unusable systemd instance was accepted; the backend would " +
			"launch UNCONTAINED, which invariant 4 forbids")
	}
	if len(args) != 0 {
		t.Errorf("a refusal still produced argv: %v", args)
	}
	msg := err.Error()
	// The classifier in cmd/ggrun/support.go matches this phrase; without it the
	// root failure lands in unclassified_launch_failure.
	if !strings.Contains(msg, "backend memory containment unavailable") {
		t.Errorf("refusal is not classifiable as memory_cgroup_limit: %q", msg)
	}
	// It must name the way out, since a user hitting this needs an action.
	if !strings.Contains(msg, "--ram-budget") {
		t.Errorf("refusal does not tell the user how to proceed: %q", msg)
	}
	// And it must not be a raw systemd/dbus string.
	if strings.Contains(msg, "DBUS_SESSION_BUS_ADDRESS") {
		t.Errorf("refusal leaks the raw systemd diagnostic: %q", msg)
	}
}

// Containment off (memoryMaxMB <= 0) must stay a no-op: the scope is optional,
// and a user who did not ask for a limit must not be blocked by systemd's state.
func TestNoLimitNeedsNoScopeInstance(t *testing.T) {
	withScopeMode(t, "unsupported")
	args, err := scopedCommandArgsWithUnit([]string{"llama-server"}, 0, "")
	if err != nil {
		t.Fatalf("an unlimited launch was refused: %v", err)
	}
	if len(args) != 1 || args[0] != "llama-server" {
		t.Errorf("unlimited launch was rewrapped: %v", args)
	}
}

func containsArg(args []string, want string) bool {
	for _, a := range args {
		if a == want {
			return true
		}
	}
	return false
}
