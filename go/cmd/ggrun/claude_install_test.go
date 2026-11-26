package main

import (
	"bytes"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// missingClaude makes an empty HOME and PATH with no claude, and fakes the
// installer. When place is true the fake writes ~/.local/bin/claude (not on
// PATH), like the real installer; instErr is returned after that.
func missingClaude(t *testing.T, place bool, instErr error) *int {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("PATH", t.TempDir())
	t.Setenv(installClaudeEnv, "")
	calls := new(int)
	old := runClaudeInstaller
	t.Cleanup(func() { runClaudeInstaller = old })
	runClaudeInstaller = func(goos string, out io.Writer) error {
		*calls++
		io.WriteString(out, "fake installer output\n")
		if place {
			dir := filepath.Join(home, ".local", "bin")
			os.MkdirAll(dir, 0o755)
			name := "claude"
			if runtime.GOOS == "windows" {
				name += ".exe"
			}
			os.WriteFile(filepath.Join(dir, name), []byte("#!/bin/sh\n"), 0o755)
		}
		return instErr
	}
	return calls
}

func TestClaudeInstallCommandIsOfficialAndSudoFree(t *testing.T) {
	for _, goos := range []string{"linux", "darwin", "windows"} {
		shown, argv := claudeInstallCommand(goos)
		if !strings.Contains(shown, "claude.ai/install") || strings.Contains(strings.Join(argv, " "), "sudo") {
			t.Fatalf("%s: %q %v", goos, shown, argv)
		}
	}
	if s, _ := claudeInstallCommand("linux"); s != "curl -fsSL https://claude.ai/install.sh | bash" {
		t.Fatal(s)
	}
}

func TestEnsureClaudeInstallsOnYesAndFindsLocalBin(t *testing.T) {
	calls := missingClaude(t, true, nil)
	var asked string
	var out bytes.Buffer
	if err := ensureClaude(&out, func(q string) bool { asked = q; return true }, false); err != nil {
		t.Fatalf("%v\n%s", err, out.String())
	}
	if *calls != 1 || !strings.Contains(asked, "claude.ai/install") || !strings.Contains(asked, "no sudo") {
		t.Fatalf("calls=%d prompt=%q", *calls, asked)
	}
	if !strings.Contains(out.String(), "fake installer output") {
		t.Fatal("installer output not streamed")
	}
	if p, err := exec.LookPath("claude"); err != nil || !strings.Contains(p, ".local") {
		t.Fatalf("~/.local/bin not added to PATH: %q %v", p, err)
	}
}

func TestEnsureClaudeDeclineStopsWithCommand(t *testing.T) {
	calls := missingClaude(t, true, nil)
	err := ensureClaude(io.Discard, func(string) bool { return false }, false)
	if err == nil || *calls != 0 || !strings.Contains(err.Error(), claudeSetupDoc) || !strings.Contains(err.Error(), "claude.ai/install") {
		t.Fatalf("%v %d", err, *calls)
	}
}

func TestEnsureClaudeNoTerminalNeverInstalls(t *testing.T) {
	calls := missingClaude(t, true, nil)
	err := ensureClaude(io.Discard, nil, false)
	if err == nil || *calls != 0 || !strings.Contains(err.Error(), installClaudeEnv) || !strings.Contains(err.Error(), "--install-claude") {
		t.Fatalf("%v %d", err, *calls)
	}
}

func TestEnsureClaudeOptInInstallsWithoutPrompt(t *testing.T) {
	for _, via := range []string{"flag", "env"} {
		calls := missingClaude(t, true, nil)
		flag := via == "flag"
		if !flag {
			t.Setenv(installClaudeEnv, "1")
		}
		if !installClaudeOptIn(flag) {
			t.Fatalf("%s: opt-in not seen", via)
		}
		if err := ensureClaude(io.Discard, nil, true); err != nil || *calls != 1 {
			t.Fatalf("%s: %v %d", via, err, *calls)
		}
	}
	missingClaude(t, false, nil)
	if installClaudeOptIn(false) {
		t.Fatal("opt-in without flag or env")
	}
}

func TestEnsureClaudeInstallerFailure(t *testing.T) {
	missingClaude(t, false, errors.New("boom"))
	err := ensureClaude(io.Discard, func(string) bool { return true }, false)
	if err == nil || !strings.Contains(err.Error(), "installer failed") || !strings.Contains(err.Error(), claudeSetupDoc) {
		t.Fatal(err)
	}
}

func TestEnsureClaudeNotFoundAfterInstall(t *testing.T) {
	missingClaude(t, false, nil)
	err := ensureClaude(io.Discard, func(string) bool { return true }, false)
	if err == nil || !strings.Contains(err.Error(), "was not found") {
		t.Fatal(err)
	}
}

func TestEnsureClaudePresentSkipsEverything(t *testing.T) {
	calls := missingClaude(t, true, nil)
	dir := t.TempDir()
	name := "claude"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	os.WriteFile(filepath.Join(dir, name), []byte("#!/bin/sh\n"), 0o755)
	t.Setenv("PATH", dir)
	if err := ensureClaude(io.Discard, func(string) bool { t.Fatal("asked"); return false }, false); err != nil || *calls != 0 {
		t.Fatalf("%v %d", err, *calls)
	}
}

// Invariant: the check never fires for launches that do not start Claude Code
// here, and a missing claude with no way to ask stops before any model work.
func TestEnsureClaudeBeforeLaunchScope(t *testing.T) {
	calls := missingClaude(t, true, nil)
	for _, req := range []*launchRequest{
		nil,
		{},
		{ClaudeCode: true, RemoteClaude: remoteClaudeOptions{Headless: true}},
	} {
		if err := ensureClaudeBeforeLaunch(req); err != nil {
			t.Fatalf("%+v: %v", req, err)
		}
	}
	if *calls != 0 {
		t.Fatal("installer ran for a launch that does not start Claude Code")
	}
	// Local --claude-code with the test's non-terminal stdin and no opt-in.
	if stdinIsTerminal() {
		t.Skip("stdin is a terminal")
	}
	if err := ensureClaudeBeforeLaunch(&launchRequest{ClaudeCode: true}); err == nil || *calls != 0 {
		t.Fatalf("%v %d", err, *calls)
	}
	if err := ensureClaudeBeforeLaunch(&launchRequest{ClaudeCode: true, InstallClaude: true}); err != nil || *calls != 1 {
		t.Fatalf("%v %d", err, *calls)
	}
}

func TestInstallClaudeFlagParses(t *testing.T) {
	for _, args := range [][]string{{"m.gguf", "--claude-code", "--install-claude"}, {"m.gguf", "--claude-code", "--install-claude=true"}} {
		req, err := parseLaunchArgs(args)
		if err != nil || !req.InstallClaude || len(req.ExtraArgs) != 0 {
			t.Fatalf("%v: %+v %v", args, req, err)
		}
	}
}
