package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// Official Claude Code installer, per Anthropic's setup page
// (https://code.claude.com/docs/en/setup):
//
//	macOS, Linux, WSL: curl -fsSL https://claude.ai/install.sh | bash
//	Windows PowerShell: & ([scriptblock]::Create((irm https://claude.ai/install.ps1)))
//
// ggrun does not pin a Claude Code version, so no version argument is passed
// and the installer picks its current release. It needs no administrator
// rights and places the launcher in ~/.local/bin; ggrun never uses sudo.
const (
	claudeSetupDoc       = "https://code.claude.com/docs/en/setup"
	installClaudeEnv     = "GGRUN_INSTALL_CLAUDE"
	claudeInstallTimeout = 10 * time.Minute
)

// claudeInstallCommand returns the command line to show the user and the argv
// that runs it for goos.
func claudeInstallCommand(goos string) (shown string, argv []string) {
	if goos == "windows" {
		shown = "& ([scriptblock]::Create((irm https://claude.ai/install.ps1)))"
		return "powershell -Command \"" + shown + "\"", []string{"powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", shown}
	}
	shown = "curl -fsSL https://claude.ai/install.sh | bash"
	return shown, []string{"bash", "-o", "pipefail", "-c", shown}
}

// runClaudeInstaller runs the official installer, streaming its output. Tests
// replace it so nothing touches the network.
var runClaudeInstaller = func(goos string, out io.Writer) error {
	_, argv := claudeInstallCommand(goos)
	ctx, cancel := context.WithTimeout(context.Background(), claudeInstallTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Stdout, cmd.Stderr = out, out
	err := cmd.Run()
	if ctx.Err() != nil {
		return fmt.Errorf("installer timed out after %s", claudeInstallTimeout)
	}
	return err
}

func installClaudeOptIn(flag bool) bool {
	v := strings.ToLower(os.Getenv(installClaudeEnv))
	return flag || v == "1" || v == "true" || v == "yes"
}

// findClaude locates claude on PATH, or in the installer's ~/.local/bin, which
// may not be on PATH yet for this process; in that case the directory is added
// to PATH so the later launch finds it too.
func findClaude() (string, bool) {
	if p, err := exec.LookPath("claude"); err == nil {
		return p, true
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", false
	}
	dir := filepath.Join(home, ".local", "bin")
	name := "claude"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	p := filepath.Join(dir, name)
	if st, err := os.Stat(p); err != nil || st.IsDir() {
		return "", false
	}
	os.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	return p, true
}

// ensureClaude makes sure a claude executable exists, installing it with the
// official installer when it is missing. confirm asks the user (nil when there
// is no terminal); optIn skips the question. Without either it never installs:
// it names the exact command and stops.
func ensureClaude(out io.Writer, confirm func(string) bool, optIn bool) error {
	if _, ok := findClaude(); ok {
		return nil
	}
	shown, _ := claudeInstallCommand(runtime.GOOS)
	manual := fmt.Sprintf("Claude Code is not installed. Install it from %s, for example:\n  %s", claudeSetupDoc, shown)
	if !optIn {
		if confirm == nil {
			return fmt.Errorf("%s\nthen run ggrun again (or set %s=1 / pass --install-claude to let ggrun run it)", manual, installClaudeEnv)
		}
		if !confirm("Claude Code is not installed. Install it now with Anthropic's official installer (latest release, from claude.ai, no sudo): " + shown) {
			return errors.New(manual + "\nthen run ggrun again")
		}
	}
	fmt.Fprintf(out, "Installing Claude Code: %s\n", shown)
	if err := runClaudeInstaller(runtime.GOOS, out); err != nil {
		var ee *exec.ExitError
		status := err.Error()
		if errors.As(err, &ee) {
			status = fmt.Sprintf("exit status %d", ee.ExitCode())
		}
		return fmt.Errorf("the Claude Code installer failed (%s). Install it manually from %s:\n  %s", status, claudeSetupDoc, shown)
	}
	if _, ok := findClaude(); !ok {
		return fmt.Errorf("the installer finished but claude was not found on PATH or in ~/.local/bin; open a new terminal and run claude --version, or see %s", claudeSetupDoc)
	}
	fmt.Fprintln(out, "Claude Code installed.")
	return nil
}

// ensureClaudeForLaunch asks on a terminal, otherwise installs only with the
// explicit opt-in (--install-claude or GGRUN_INSTALL_CLAUDE=1).
func ensureClaudeForLaunch(flag bool) error {
	var confirm func(string) bool
	if stdinIsTerminal() {
		confirm = func(q string) bool {
			fmt.Fprint(os.Stderr, q+" (y/N) ")
			var a string
			fmt.Scanln(&a)
			a = strings.ToLower(strings.TrimSpace(a))
			return a == "y" || a == "yes"
		}
	}
	return ensureClaude(os.Stderr, confirm, installClaudeOptIn(flag))
}

// ensureClaudeBeforeLaunch runs before any model work, so declining costs no
// model load. Headless launches never start Claude Code here and are skipped.
func ensureClaudeBeforeLaunch(req *launchRequest) error {
	if req == nil || !req.ClaudeCode || req.RemoteClaude.Headless {
		return nil
	}
	return ensureClaudeForLaunch(req.InstallClaude)
}
