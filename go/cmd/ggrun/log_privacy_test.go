package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestHeadlessLogVerbosity(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		env  string
		bad  bool
		want string
	}{
		{"lv5", []string{"-lv", "5"}, "", true, "-lv"},
		{"long5", []string{"--log-verbosity", "5"}, "", true, "--log-verbosity"},
		{"eq9", []string{"--verbosity=9"}, "", true, "--verbosity"},
		{"env5", nil, "5", true, "LLAMA_ARG_LOG_VERBOSITY"},
		{"verbose", []string{"-v"}, "", true, "-v"},
		{"junk", []string{"-lv", "x"}, "", true, "-lv"},
		{"lv4", []string{"-lv", "4"}, "", false, ""},
		{"env4", nil, "4", false, ""},
		{"none", []string{"-c", "5"}, "", false, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("LLAMA_ARG_LOG_VERBOSITY", tc.env)
			err := validateHeadlessLogVerbosity(tc.args)
			if tc.bad != (err != nil) {
				t.Fatalf("err=%v want bad=%v", err, tc.bad)
			}
			if tc.bad && !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q lacks %q", err, tc.want)
			}
		})
	}
}

func TestValidateRemoteClaudeRefusesVerboseOnlyHeadless(t *testing.T) {
	t.Setenv("LLAMA_ARG_LOG_VERBOSITY", "")
	req := &launchRequest{ClaudeCode: true, Port: 8080, ExtraArgs: []string{"-lv", "5"}}
	req.RemoteClaude = remoteClaudeOptions{Headless: true, Port: 8081, Host: "0.0.0.0"}
	if err := validateRemoteClaude(req); err == nil || !strings.Contains(err.Error(), "request bodies") {
		t.Fatalf("headless -lv 5 not refused: %v", err)
	}
	req.ExtraArgs = []string{"-lv", "4"}
	if err := validateRemoteClaude(req); err != nil && strings.Contains(err.Error(), "verbosity") {
		t.Fatalf("headless -lv 4 refused: %v", err)
	}
	if err := validateRemoteClaude(&launchRequest{ExtraArgs: []string{"-lv", "5"}}); err != nil {
		t.Fatalf("non-headless changed: %v", err)
	}
}

func TestOpenPrivateLogMode0600(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX modes")
	}
	p := filepath.Join(t.TempDir(), "ggrun-claude-server-v2-1-x.log")
	if err := os.WriteFile(p, []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := openPrivateLog(p)
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	if fi, _ := os.Stat(p); fi.Mode().Perm() != 0o600 {
		t.Fatalf("existing mode %v", fi.Mode().Perm())
	}
	q := filepath.Join(filepath.Dir(p), "new.log")
	f, _ = openPrivateLog(q)
	f.Close()
	if fi, _ := os.Stat(q); fi.Mode().Perm() != 0o600 {
		t.Fatalf("new mode %v", fi.Mode().Perm())
	}
}

// Files beyond the newest N that were written recently are kept: a long-running
// server's open log must survive a burst of newer launches.
func TestRotateGgrunLogsKeepsRecentlyWrittenBeyondN(t *testing.T) {
	dir := t.TempDir()
	for i := 0; i < 5; i++ {
		p := filepath.Join(dir, "ggrun-claude-server-v2-"+string(rune('a'+i))+".log")
		if err := os.WriteFile(p, []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
		ts := time.Now().Add(-time.Duration(i) * time.Hour)
		_ = os.Chtimes(p, ts, ts)
	}
	rotateGgrunLogs(dir, 2)
	if left, _ := filepath.Glob(filepath.Join(dir, "ggrun-claude-server-v2-*.log")); len(left) != 5 {
		t.Fatalf("recently written logs were rotated: %d left", len(left))
	}
}

func TestRotateGgrunLogsKeepsNewestAndSparesOthers(t *testing.T) {
	dir := t.TempDir()
	base := time.Now().Add(-logRetainMinIdle - time.Hour)
	mk := func(name string, age int) {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
		ts := base.Add(time.Duration(-age) * time.Minute)
		_ = os.Chtimes(p, ts, ts)
	}
	for i := 0; i < 5; i++ {
		mk("ggrun-claude-server-v2-"+string(rune('a'+i))+".log", i)
		mk("ggrun-claude-requests-"+string(rune('a'+i))+".jsonl", i)
	}
	mk("notes.log", 99)
	mk("other-server.log", 99)
	rotateGgrunLogs(dir, 2)
	for _, n := range []string{"ggrun-claude-server-v2-a.log", "ggrun-claude-server-v2-b.log", "ggrun-claude-requests-a.jsonl", "ggrun-claude-requests-b.jsonl", "notes.log", "other-server.log"} {
		if _, err := os.Stat(filepath.Join(dir, n)); err != nil {
			t.Fatalf("%s should remain: %v", n, err)
		}
	}
	for _, n := range []string{"ggrun-claude-server-v2-c.log", "ggrun-claude-server-v2-e.log", "ggrun-claude-requests-d.jsonl"} {
		if _, err := os.Stat(filepath.Join(dir, n)); err == nil {
			t.Fatalf("%s should be rotated", n)
		}
	}
}
