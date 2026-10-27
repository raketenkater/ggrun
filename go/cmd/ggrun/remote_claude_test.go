package main

import (
	"github.com/raketenkater/ggrun/pkg/config"
	"io"
	"net"
	"net/http"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestRemoteClaudeFlagsDoNotReachBackend(t *testing.T) {
	t.Setenv("LLM_CONFIG_FILE", "/nonexistent/ggrun-remote-test.env")
	args := []string{"model.gguf", "--claude-code", "--headless", "--gateway-host=127.0.0.1", "--gateway-port", "18080", "--gateway-url", "https://models.example/inference", "--gateway-token-file", "/tmp/private-key"}
	r, err := parseLaunchArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if !r.ClaudeCode || !r.RemoteClaude.Headless || r.RemoteClaude.Port != 18080 || r.Host != "127.0.0.1" {
		t.Fatalf("bad headless request %+v", r)
	}
	if !reflect.DeepEqual(args, r.OriginalArgs) {
		t.Fatal("original launch identity lost")
	}
	for _, arg := range r.ExtraArgs {
		if strings.Contains(arg, "gateway") || arg == "--headless" {
			t.Fatalf("gateway flag leaked to backend: %v", r.ExtraArgs)
		}
	}
	local, err := parseLaunchArgs([]string{"model.gguf", "--claude-code"})
	if err != nil {
		t.Fatal(err)
	}
	if local.RemoteClaude.Headless {
		t.Fatal("ordinary Claude launch became remote")
	}
}

func TestRemoteClaudeRejectsContradictoryClientOwnership(t *testing.T) {
	for _, args := range [][]string{
		{"model.gguf", "--headless"},
		{"model.gguf", "--claude-code", "--headless", "--claude-resume", "latest"},
		{"model.gguf", "--gateway-url", "https://models.example"},
		{"model.gguf", "--gateway-host", "127.0.0.1"},
		{"model.gguf", "--claude-code", "--headless=wrong"},
		{"model.gguf", "--claude-code", "--headless", "--", "--host", "0.0.0.0"},
		{"model.gguf", "--claude-code", "--headless", "--gateway-port", "8081", "--port", "8081"},
		{"model.gguf", "--claude-code", "--headless", "--gateway-url", "http://0.0.0.0:8080"},
	} {
		if _, err := parseLaunchArgs(args); err == nil {
			t.Errorf("accepted %v", args)
		}
	}
}

func TestRemoteLoadingAndRecoveryAreAuthenticatedWithoutClaude(t *testing.T) {
	t.Setenv("PATH", t.TempDir()) // No claude, Tailscale, model or backend process.
	req := &launchRequest{RemoteClaude: remoteClaudeOptions{Headless: true, Host: "127.0.0.1", TokenFile: filepath.Join(t.TempDir(), "key")}}
	s, err := prepareRemoteClaude(req, &config.Config{})
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	for _, recovering := range []bool{false, true} {
		if recovering {
			s.recovering()
		}
		for _, authenticated := range []bool{false, true} {
			r, _ := http.NewRequest("GET", "http://"+s.listener.Addr().String()+"/ggrun/client", nil)
			if authenticated {
				r.Header.Set("Authorization", "Bearer "+s.key)
			}
			resp, err := http.DefaultClient.Do(r)
			if err != nil {
				t.Fatal(err)
			}
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			want := 401
			if authenticated {
				want = 503
			}
			if resp.StatusCode != want {
				t.Fatalf("loading auth=%t got %d want %d", authenticated, resp.StatusCode, want)
			}
		}
	}
	host, port, _ := net.SplitHostPort(s.listener.Addr().String())
	req.RemoteClaude.Host = host
	req.RemoteClaude.Port, _ = strconv.Atoi(port)
	if duplicate, err := prepareRemoteClaude(req, &config.Config{}); err == nil {
		duplicate.close()
		t.Fatal("occupied gateway port was accepted")
	}
}

func TestRemoteClaudeLeavesExplicitPassthroughUntouched(t *testing.T) {
	rest, o, err := extractRemoteClaudeArgs([]string{"model.gguf", "--", "--gateway-port", "7"})
	if err != nil || o.Headless || !reflect.DeepEqual(rest, []string{"model.gguf", "--", "--gateway-port", "7"}) {
		t.Fatalf("%v %+v %v", rest, o, err)
	}
}

func TestRemoteAndLocalPolicyUseResolvedSlotContext(t *testing.T) {
	t.Setenv("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "")
	t.Setenv("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "")
	args := []string{"--ctx-size", "1048576", "--parallel", "2"}
	p := claudeClientProfile(args, 262144)
	if p.ContextPerSlot != 131072 || p.Policy.CompactWindow != 131072 {
		t.Fatalf("policy used requested context instead of resolved slot: %+v", p)
	}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	env := strings.Join(claudeCodeEnv("127.0.0.1", 8080, args, 262144), "\n")
	for _, want := range []string{"CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072", "CLAUDE_CODE_AUTO_COMPACT_WINDOW=131072", "CLAUDE_CODE_SUBAGENT_MODEL=local"} {
		if !strings.Contains(env, want) {
			t.Fatal("local launcher disagrees with remote policy:", want)
		}
	}
	capture := captureStdout(t)
	printClaudeCodeRecipe("127.0.0.1", 8080, args, 262144)
	out := capture()
	if !strings.Contains(out, "CLAUDE_CODE_MAX_CONTEXT_TOKENS") || !strings.Contains(out, "131072") || strings.Contains(out, "1048576") {
		t.Fatal("recipe disagrees with resolved policy", out)
	}
}

func TestRemoteDryRunDoesNotPromiseALocalClaudeProcess(t *testing.T) {
	capture := captureStdout(t)
	printDryRunRemoteClaudeNotice(&launchRequest{RemoteClaude: remoteClaudeOptions{Headless: true, Host: "127.0.0.1", Port: 8080}})
	out := capture()
	if strings.Contains(out, "opens Claude Code") || !strings.Contains(out, "without starting Claude Code") || !strings.Contains(out, "127.0.0.1:8080") {
		t.Fatal(out)
	}
}
