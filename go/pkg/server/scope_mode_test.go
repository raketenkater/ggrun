//go:build linux

package server

import (
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
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
	if !strings.Contains(msg, "systemd login session") || strings.Contains(msg, "--ram-budget") {
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

// A host without systemd (Docker) must have one explicit way to launch, and
// every refusal must name it. The opt-out is never chosen implicitly: without
// it the same host still refuses (invariant 4 stays fail-closed by default).
func TestScopeModeNoneIsTheOnlyUncontainedPath(t *testing.T) {
	// No systemd-run and no systemctl anywhere on PATH, as in a container.
	t.Setenv("PATH", t.TempDir())
	for _, mode := range []string{"", "unsupported", "user"} {
		withScopeMode(t, mode)
		args, err := scopedCommandArgsWithUnit([]string{"llama-server"}, 4096, "ggrun-test.scope")
		if err == nil || len(args) != 0 {
			t.Fatalf("mode %q launched without systemd: args=%v", mode, args)
		}
		if !strings.Contains(err.Error(), "GGRUN_SCOPE_MODE=none") {
			t.Fatalf("mode %q refusal does not name the opt-out: %q", mode, err)
		}
	}
	withScopeMode(t, "none")
	args, owner, err := scopedCommandArgsWithLimits([]string{"llama-server", "-m", "m.gguf"}, 2048, 4096, "ggrun-test.scope")
	if err != nil {
		t.Fatalf("explicit opt-out refused: %v", err)
	}
	if owner != scopeNone {
		t.Fatalf("opt-out owner = %v, want scopeNone", owner)
	}
	if strings.Join(args, " ") != "llama-server -m m.gguf" {
		t.Fatalf("opt-out argv was wrapped: %v", args)
	}
}

// Under the opt-out the started process carries no scope, so every scope
// observation reports "no memory scope" instead of addressing a unit that was
// never created, and Stop still terminates the backend.
func TestScopeModeNoneStartsWithoutAScope(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not installed")
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()
	script := filepath.Join(t.TempDir(), "srv.py")
	content := "import http.server, socketserver, os\n" +
		"class H(http.server.BaseHTTPRequestHandler):\n" +
		"    def do_GET(self):\n" +
		"        self.send_response(200); self.end_headers(); self.wfile.write(os.environ.get('GGRUN_TEST_ENV', 'unset').encode())\n" +
		"    def log_message(self, *a): pass\n" +
		"with socketserver.TCPServer(('127.0.0.1', " + strconv.Itoa(port) + "), H) as httpd:\n" +
		"    httpd.serve_forever()\n"
	if err := os.WriteFile(script, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	withScopeMode(t, "none")
	p, err := StartWithTimeoutToOptions([]string{python, script}, port, 15*time.Second, io.Discard, io.Discard,
		StartOptions{MemoryHighMB: 512, MemoryMaxMB: 1024, EnvOverrides: []string{"GGRUN_TEST_ENV=applied"}})
	if err != nil {
		t.Fatalf("opt-out start: %v", err)
	}
	defer func() { _ = p.Stop() }()
	if p.scopeUnit != "" {
		t.Fatalf("opt-out created scope %q", p.scopeUnit)
	}
	if _, err := p.MemoryPeakBytes(); err == nil || !strings.Contains(err.Error(), "no memory scope") {
		t.Fatalf("peak on an uncontained backend: %v", err)
	}
	if err := p.SetMemoryMaxMB(2048); err == nil {
		t.Fatal("resize of an uncontained backend reported success")
	}
	resp, err := http.Get("http://127.0.0.1:" + strconv.Itoa(port) + "/")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if string(body) != "applied" {
		t.Fatalf("launch environment overrides lost without a scope: %q", body)
	}
	if err := p.Stop(); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if p.IsRunning() {
		t.Fatal("uncontained backend survived Stop")
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

func installScopeControlFixture(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "systemctl")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+body), 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	return dir
}

func TestScopeManagerRequiresAReachableInstance(t *testing.T) {
	dir := installScopeControlFixture(t, `printf '%s\n' "$*" > "$GGRUN_TEST_SCOPE_CALL"
case "$GGRUN_TEST_MANAGER" in
  reachable) printf '257\n' ;;
  empty) exit 0 ;;
  unavailable) exit 1 ;;
  hung) exec sleep 30 ;;
esac
`)
	call := filepath.Join(dir, "call")
	t.Setenv("GGRUN_TEST_SCOPE_CALL", call)
	// A stale runtime directory must not be treated as capability evidence.
	t.Setenv("XDG_RUNTIME_DIR", dir)
	for _, uid := range []int{1000, 0} {
		for _, state := range []string{"reachable", "empty", "unavailable"} {
			t.Setenv("GGRUN_TEST_MANAGER", state)
			got := scopeModeForUID(uid)
			want := scopeUnsupported
			if state == "reachable" {
				want = scopeUser
				if uid == 0 {
					want = scopeSystem
				}
			}
			if got != want {
				t.Fatalf("uid=%d state=%s: mode=%v, want %v", uid, state, got, want)
			}
			args, err := os.ReadFile(call)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(args), "--user") != (uid != 0) {
				t.Fatalf("uid=%d addressed the wrong manager: %s", uid, args)
			}
		}
	}
	t.Setenv("GGRUN_TEST_MANAGER", "hung")
	started := time.Now()
	if got := scopeModeForUID(1000); got != scopeUnsupported {
		t.Fatalf("hung manager accepted: %v", got)
	}
	if elapsed := time.Since(started); elapsed > 5*time.Second {
		t.Fatalf("manager probe was not bounded: %s", elapsed)
	}
}

func TestScopeOperationsRetainTheirLaunchOwner(t *testing.T) {
	for _, owner := range []string{"user", "system"} {
		t.Run(owner, func(t *testing.T) {
			dir := installScopeControlFixture(t, `printf '%s\n' "$*" >> "$GGRUN_TEST_SCOPE_CALL"
if [ "$GGRUN_TEST_SCOPE_OWNER" = user ]; then
  [ "$1" = --user ] || exit 1
  shift
else
  [ "$1" != --user ] || exit 1
fi
case "$1" in
  is-active) exit 3 ;;
  show)
    case "$*" in
      *ControlGroup*) printf '/ggrun-test-nonexistent-cgroup\n' ;;
      *MemoryPeak*) printf '2048\n' ;;
      *Result*) printf 'success\n' ;;
    esac ;;
  stop|reset-failed|set-property) exit 0 ;;
  *) exit 1 ;;
esac
`)
			logPath := filepath.Join(dir, "calls")
			t.Setenv("GGRUN_TEST_SCOPE_CALL", logPath)
			t.Setenv("GGRUN_TEST_SCOPE_OWNER", owner)
			t.Setenv(scopeModeEnvOverride, owner)
			_, mode, err := scopedCommandArgsWithLimits([]string{"backend"}, 0, 64, "test.scope")
			if err != nil {
				t.Fatal(err)
			}
			p := &Process{scopeUnit: "test.scope", scopeMode: mode, done: make(chan struct{}), stopDone: make(chan struct{})}
			close(p.done)
			// Re-resolving after launch would now choose a different instance.
			other := "system"
			if owner == "system" {
				other = "user"
			}
			t.Setenv(scopeModeEnvOverride, other)
			if peak, err := p.MemoryPeakBytes(); err != nil || peak != 2048 {
				t.Fatalf("peak=%d: %v", peak, err)
			}
			if err := p.SetMemoryMaxMB(64); err != nil {
				t.Fatal(err)
			}
			if err := p.SetMemoryHighMB(32); err != nil {
				t.Fatal(err)
			}
			if high, max := p.MemoryLimitsMB(); high != 32 || max != 64 {
				t.Fatalf("recorded limits high=%d max=%d, want 32/64", high, max)
			}
			if err := p.Stop(); err != nil {
				t.Fatal(err)
			}
			p.Kill()
			data, err := os.ReadFile(logPath)
			if err != nil {
				t.Fatal(err)
			}
			for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
				if strings.HasPrefix(line, "--user ") != (owner == "user") {
					t.Fatalf("scope owner changed: %s", line)
				}
			}
			for _, verb := range []string{"show", "stop", "is-active", "reset-failed", "set-property"} {
				if !strings.Contains(string(data), verb) {
					t.Errorf("lifecycle did not exercise %s", verb)
				}
			}
		})
	}
}

func TestScopeBusFailureIsNotSuccessfulCleanup(t *testing.T) {
	installScopeControlFixture(t, "exit 1\n")
	if err := stopScopeUnit("test.scope", scopeUser); err == nil {
		t.Fatal("failed bus contact was treated as stopped")
	}
	if err := waitScopeUnitStopped("test.scope", scopeUser, time.Second); err == nil {
		t.Fatal("failed bus contact was treated as inactive")
	}
	// A limit write that failed must not be reported as the scope's limit: the
	// post-canary reclaim threshold is clamped against it.
	p := &Process{scopeUnit: "test.scope", scopeMode: scopeUser, memoryHighMB: 10, memoryMaxMB: 20}
	if p.SetMemoryMaxMB(30) == nil || p.SetMemoryHighMB(15) == nil {
		t.Fatal("failed limit write reported success")
	}
	if high, max := p.MemoryLimitsMB(); high != 10 || max != 20 {
		t.Fatalf("failed writes changed recorded limits: high=%d max=%d", high, max)
	}
}
