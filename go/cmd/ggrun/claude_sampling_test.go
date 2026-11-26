package main

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// stubSamplingReader replaces the GGUF reader for one test.
func stubSamplingReader(t *testing.T, keys map[string]string, err error) {
	t.Helper()
	old := samplingDefaultsReader
	samplingDefaultsReader = func(string) (map[string]string, error) { return keys, err }
	t.Cleanup(func() { samplingDefaultsReader = old })
}

var testSamplingModel = &placement.ModelProfile{Path: "/models/m.gguf", ModelArch: "qwen35"}

func samplingValue(args []string, flag string) string { return valueAfter(args, flag) }

func TestClaudePenaltyModes(t *testing.T) {
	stubSamplingReader(t, map[string]string{}, nil)
	base := []string{"-m", "m.gguf"}

	late := claudeCodeSamplingArgs(base, true, testSamplingModel, "late", true)
	if samplingValue(late, "--repeat-penalty") != "1.05" || samplingValue(late, "--repeat-last-n") != "512" ||
		samplingValue(late, "--presence-penalty") != "0" || samplingValue(late, "--samplers") != claudeLateSamplers {
		t.Fatalf("late = %v", late)
	}

	off := claudeCodeSamplingArgs(base, true, testSamplingModel, "off", true)
	if samplingValue(off, "--repeat-penalty") != "1" || samplingValue(off, "--presence-penalty") != "0" {
		t.Fatalf("off must send neutral penalties, got %v", off)
	}
	if hasArg(off, "--samplers") || hasArg(off, "--repeat-last-n") {
		t.Fatalf("off must not reorder samplers or set a window, got %v", off)
	}

	full := claudeCodeSamplingArgs(base, true, testSamplingModel, "full", true)
	if samplingValue(full, "--presence-penalty") != "1.0" || samplingValue(full, "--repeat-penalty") != "1.05" ||
		samplingValue(full, "--repeat-last-n") != "512" || hasArg(full, "--samplers") {
		t.Fatalf("full = %v", full)
	}

	// An unknown value behaves as the built-in default, which sends no active penalty.
	if got := claudeCodeSamplingArgs(base, true, testSamplingModel, "bogus", true); !reflect.DeepEqual(got, off) {
		t.Fatalf("unknown mode = %v, want the default (%s) = %v", got, config.DefaultClaudePenalties, off)
	}
	if config.DefaultClaudePenalties != "off" {
		t.Fatalf("default changed to %q; a repeat penalty broke GLM-4.7-Flash tool calls in E-10, update this test with new evidence", config.DefaultClaudePenalties)
	}
}

func TestClaudeLateFallsBackToFullWithoutSamplersSupport(t *testing.T) {
	stubSamplingReader(t, map[string]string{}, nil)
	got, summary := claudeCodeSampling([]string{"-m", "m.gguf"}, true, testSamplingModel, "late", false)
	if samplingValue(got, "--presence-penalty") != "1.0" || hasArg(got, "--samplers") {
		t.Fatalf("late without --samplers must keep today's flags, got %v", got)
	}
	if !strings.Contains(summary, "penalties full") || !strings.Contains(summary, "late needs --samplers") {
		t.Fatalf("summary = %q", summary)
	}
	if claudeSamplersSupported(&backendInfo{Help: "--samplers SAMPLERS", IsIK: true}) ||
		claudeSamplersSupported(&backendInfo{Help: "--top-k"}) || claudeSamplersSupported(nil) ||
		!claudeSamplersSupported(&backendInfo{Help: "  --samplers SAMPLERS  order"}) {
		t.Fatal("claudeSamplersSupported misjudged a backend")
	}
}

func TestClaudeSamplingFollowsGGUFKeys(t *testing.T) {
	base := []string{"-m", "m.gguf"}

	// File carries all three: ggrun sends none of them and never a temperature.
	stubSamplingReader(t, map[string]string{"top_k": "64", "top_p": "0.95", "min_p": "0.0", "temp": "1.0"}, nil)
	got, summary := claudeCodeSampling(base, true, testSamplingModel, "late", true)
	for _, f := range []string{"--top-k", "--top-p", "--min-p", "--temp", "--temperature"} {
		if hasArg(got, f) {
			t.Errorf("GGUF has the key; %s must not be passed: %v", f, got)
		}
	}
	if !strings.Contains(summary, "top_k=GGUF") || !strings.Contains(summary, "top_p=GGUF") || !strings.Contains(summary, "min_p=GGUF") {
		t.Errorf("summary = %q", summary)
	}

	// File carries only top_k: the other two come from ggrun.
	stubSamplingReader(t, map[string]string{"top_k": "64"}, nil)
	got, summary = claudeCodeSampling(base, true, testSamplingModel, "late", true)
	if hasArg(got, "--top-k") || samplingValue(got, "--top-p") != "0.95" || samplingValue(got, "--min-p") != "0.0" {
		t.Fatalf("partial GGUF keys = %v", got)
	}
	if !strings.Contains(summary, "top_k=GGUF") || !strings.Contains(summary, "top_p=ggrun") {
		t.Errorf("summary = %q", summary)
	}

	// File without sampling keys: ggrun's values.
	stubSamplingReader(t, map[string]string{"temp": "0.7"}, nil)
	got = claudeCodeSamplingArgs(base, true, testSamplingModel, "late", true)
	if samplingValue(got, "--top-k") != "20" || samplingValue(got, "--top-p") != "0.95" || samplingValue(got, "--min-p") != "0.0" || hasArg(got, "--temp") {
		t.Fatalf("no sampling keys = %v", got)
	}
}

func TestClaudeSamplingGGUFReadFailureFallsBack(t *testing.T) {
	stubSamplingReader(t, nil, errors.New("corrupt"))
	got, summary := claudeCodeSampling([]string{"-m", "m.gguf"}, true, testSamplingModel, "full", true)
	if samplingValue(got, "--top-k") != "20" || samplingValue(got, "--top-p") != "0.95" || samplingValue(got, "--min-p") != "0.0" {
		t.Fatalf("read failure must fall back to ggrun values, got %v", got)
	}
	if !strings.Contains(summary, "GGUF unreadable") {
		t.Errorf("summary = %q", summary)
	}
	// No model at all is the same fallback.
	stubSamplingReader(t, map[string]string{"top_k": "64"}, nil)
	if got := claudeCodeSamplingArgs([]string{"-m", "m.gguf"}, true, nil, "full", true); samplingValue(got, "--top-k") != "20" {
		t.Fatalf("nil model = %v", got)
	}
}

func TestClaudeSamplingUserFlagsWin(t *testing.T) {
	stubSamplingReader(t, map[string]string{}, nil)
	user := []string{"-m", "m.gguf", "--samplers", "top_k;temperature", "--top-k", "7", "--repeat-penalty", "1.2", "--presence-penalty", "0.5"}
	got := claudeCodeSamplingArgs(user, true, testSamplingModel, "late", true)
	count := func(flag string) int {
		n := 0
		for _, a := range got {
			if a == flag {
				n++
			}
		}
		return n
	}
	for flag, want := range map[string]string{"--samplers": "top_k;temperature", "--top-k": "7", "--repeat-penalty": "1.2", "--presence-penalty": "0.5"} {
		if count(flag) != 1 || samplingValue(got, flag) != want {
			t.Errorf("%s = %q (x%d), want user value %q once: %v", flag, samplingValue(got, flag), count(flag), want, got)
		}
	}
	if samplingValue(got, "--repeat-last-n") != "512" || samplingValue(got, "--top-p") != "0.95" {
		t.Errorf("unset flags must still default: %v", got)
	}
	// The --flag=value spelling and the sequence shorthand also count as the user's own sampler order.
	for _, in := range [][]string{{"--samplers=top_k;temperature"}, {"--sampling-seq", "kt"}, {"--sampler-seq", "kt"}} {
		if g := claudeCodeSamplingArgs(append([]string{"-m", "m.gguf"}, in...), true, testSamplingModel, "late", true); hasArg(g, "--samplers") {
			t.Errorf("user sampler order %v must suppress --samplers: %v", in, g)
		}
	}
	// A user-named flag is reported as the user's.
	_, summary := claudeCodeSampling(user, true, testSamplingModel, "late", true)
	if !strings.Contains(summary, "top_k=you") {
		t.Errorf("summary = %q", summary)
	}
}

func TestClaudeSamplingDeepSeek4IgnoresPenaltyPolicyAndGGUF(t *testing.T) {
	stubSamplingReader(t, map[string]string{"top_k": "64"}, nil)
	model := &placement.ModelProfile{Path: "/models/ds.gguf", ModelArch: "deepseek4"}
	want := []string{"-m", "m.gguf",
		"--presence-penalty", "1.0", "--repeat-penalty", "1.05", "--repeat-last-n", "512",
		"--temp", "0.7", "--top-k", "40", "--top-p", "0.95", "--min-p", "0.05", "--reasoning-budget", "0"}
	for _, mode := range []string{"off", "late", "full"} {
		if got := claudeCodeSamplingArgs([]string{"-m", "m.gguf"}, true, model, mode, true); !reflect.DeepEqual(got, want) {
			t.Errorf("deepseek4 under %s = %v, want %v", mode, got, want)
		}
	}
}

func TestClaudeSamplingNoticePrintsOncePerValue(t *testing.T) {
	claudeSamplingNoticeMu.Lock()
	claudeSamplingNoticeLast = ""
	claudeSamplingNoticeMu.Unlock()
	read := captureStdout(t)
	announceClaudeSampling("penalties late, top_k=GGUF")
	announceClaudeSampling("penalties late, top_k=GGUF")
	announceClaudeSampling("")
	announceClaudeSampling("penalties off, top_k=ggrun")
	out := read()
	if got := strings.Count(out, "[claude-code] sampling:"); got != 2 {
		t.Fatalf("notice printed %d times, want 2:\n%s", got, out)
	}
}

func TestClaudePenaltiesFlagAndConfig(t *testing.T) {
	t.Setenv("LLM_CONFIG", filepath.Join(t.TempDir(), "config"))
	req, err := parseLaunchArgs([]string{"--claude-code"})
	if err != nil || effectiveClaudePenalties(req) != config.DefaultClaudePenalties {
		t.Fatalf("default = %q, %v", effectiveClaudePenalties(req), err)
	}
	for _, args := range [][]string{
		{"--claude-code", "--claude-penalties", "off"},
		{"--claude-code", "--claude-penalties=off"},
	} {
		req, err := parseLaunchArgs(args)
		if err != nil || effectiveClaudePenalties(req) != "off" || len(req.ExtraArgs) != 0 {
			t.Fatalf("%v: mode %q extra %v err %v", args, effectiveClaudePenalties(req), req.ExtraArgs, err)
		}
	}
	if _, err := parseLaunchArgs([]string{"--claude-code", "--claude-penalties", "heavy"}); err == nil {
		t.Error("an unknown mode must be rejected")
	}
	if _, err := parseLaunchArgs([]string{"--claude-penalties", "off"}); err == nil || !strings.Contains(err.Error(), "--claude-code") {
		t.Errorf("flag outside --claude-code: %v", err)
	}
	// The config key sets the mode; the flag overrides it.
	cfg := config.Defaults()
	cfg.ClaudePenalties = "full"
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}
	req, err = parseLaunchArgs([]string{"--claude-code"})
	if err != nil || effectiveClaudePenalties(req) != "full" {
		t.Fatalf("config full = %q, %v", effectiveClaudePenalties(req), err)
	}
	req, err = parseLaunchArgs([]string{"--claude-code", "--claude-penalties", "late"})
	if err != nil || effectiveClaudePenalties(req) != "late" {
		t.Fatalf("flag over config = %q, %v", effectiveClaudePenalties(req), err)
	}
}

// A real GGUF file goes through the cached reader end to end.
func TestCachedGGUFSamplingDefaultsReadsFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.gguf")
	var b []byte
	le := func(v uint64, n int) {
		for i := 0; i < n; i++ {
			b = append(b, byte(v>>(8*i)))
		}
	}
	b = append(b, "GGUF"...)
	le(3, 4)
	le(0, 8)
	le(1, 8)
	key := "general.sampling.top_k"
	le(uint64(len(key)), 8)
	b = append(b, key...)
	le(5, 4)
	le(64, 4)
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ { // second call is served from the cache
		got, err := cachedGGUFSamplingDefaults(path)
		if err != nil || got["top_k"] != "64" {
			t.Fatalf("call %d: %v, %v", i, got, err)
		}
	}
	if _, err := cachedGGUFSamplingDefaults(filepath.Join(t.TempDir(), "missing.gguf")); err == nil {
		t.Error("missing file must error")
	}
}
