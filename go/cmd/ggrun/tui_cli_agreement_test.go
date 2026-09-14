package main

import (
	"reflect"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/tui"
)

// The TUI does not plan anything itself: cmdGUI turns the user's selections
// into argv with tuiLaunchArgs and then calls the same cmdLaunch the command
// line calls. So "the TUI and the CLI agree" reduces to one property — every
// selection the TUI can express survives the argv round trip and arrives in the
// launch request unchanged. A field that stops being emitted, or a flag the
// parser no longer accepts, silently discards a user's explicit choice, which
// the change contract treats as a constraint rather than a hint.
func TestTUISelectionsSurviveTheArgvRoundTrip(t *testing.T) {
	req := &tui.LaunchRequest{
		ModelPath:      "/models/agent.gguf",
		Port:           18999,
		CtxFlag:        "fit",
		KVPlacement:    "gpu",
		KVQuality:      "high",
		KVQualitySet:   true,
		SWAFull:        true,
		SWAFullSet:     true,
		Parallel:       3,
		ParallelSet:    true,
		Vision:         true,
		Backend:        "cuda",
		TuneCache:      "off",
		Benchmark:      true,
		SupportExpert:  "on",
		SupportOnline:  true,
		SupportSet:     true,
		NoCachedConfig: true,
		// A catalog entry name, which is the only thing the TUI's template row
		// can produce. Any other value is a backend passthrough by design.
		ChatTemplate:           "qwen3.8-27b",
		ClaudeCode:             true,
		ClaudeProfile:          "agent-interactive",
		ClaudeReviewerOverride: "qwen2b",
		ResumeSession:          "latest",
	}

	args := tuiLaunchArgs(req, nil)
	parsed, err := parseLaunchArgs(args)
	if err != nil {
		t.Fatalf("the CLI rejected argv the TUI produced: %v\nargv: %s", err, strings.Join(args, " "))
	}

	checks := []struct {
		field string
		got   any
		want  any
	}{
		{"ModelPath", parsed.ModelPath, req.ModelPath},
		{"Port", parsed.Port, req.Port},
		{"CtxFlag", parsed.CtxFlag, req.CtxFlag},
		{"KVPlacement", parsed.KVPlacement, req.KVPlacement},
		{"KVQuality", parsed.KVQuality, req.KVQuality},
		{"Parallel", parsed.Parallel, req.Parallel},
		{"ParallelSet", parsed.ParallelSet, true},
		{"VisionAuto", parsed.VisionAuto, true},
		{"Backend", parsed.Backend, req.Backend},
		{"TuneCache", parsed.TuneCache, req.TuneCache},
		{"Benchmark", parsed.Benchmark, true},
		{"SupportExpert", parsed.SupportExpert, req.SupportExpert},
		{"SupportOnline", parsed.SupportOnline, true},
		{"SupportOnlineSet", parsed.SupportOnlineSet, true},
		{"NoCachedConfig", parsed.NoCachedConfig, true},
		{"ChatTemplateOverride", parsed.ChatTemplateOverride, req.ChatTemplate},
		{"ClaudeCode", parsed.ClaudeCode, true},
		{"ClaudeProfile", parsed.ClaudeProfile, req.ClaudeProfile},
		{"ClaudeReviewerOverride", parsed.ClaudeReviewerOverride, req.ClaudeReviewerOverride},
		{"ClaudeResume", parsed.ClaudeResume, req.ResumeSession},
	}
	for _, c := range checks {
		if !reflect.DeepEqual(c.got, c.want) {
			t.Errorf("%s: TUI chose %v, the CLI parsed %v", c.field, c.want, c.got)
		}
	}

	// An explicit SWA choice is a backend passthrough rather than a request
	// field, so it is checked where it actually lands.
	if joined := strings.Join(parsed.ExtraArgs, " "); !strings.Contains(joined, "--swa-full") {
		t.Errorf("an explicit SWA choice did not reach the backend args: %q", joined)
	}

	// The backend must be recorded as user-typed. userExplicitBackendFlag reads
	// the original argv, so a selection that arrives without it is subject to
	// automatic withdrawal the user never asked for.
	if !parsed.BackendExplicit {
		t.Error("a backend chosen in the TUI did not arrive as an explicit choice")
	}
}

// An untouched TUI row is a preference, not an instruction, and must not become
// a typed flag: ggrun treats typed flags as inviolable, which locks the recovery
// ladder out of the very levers it needs. This is the inverse of the test above
// and the reason KVQualitySet/SWAFullSet/SupportSet exist at all.
func TestUntouchedTUIRowsDoNotBecomeTypedFlags(t *testing.T) {
	req := &tui.LaunchRequest{ModelPath: "/models/agent.gguf", KVQuality: "mid", SWAFull: true, SupportOnline: true}
	joined := strings.Join(tuiLaunchArgs(req, nil), " ")
	for _, flag := range []string{"--kv-quality", "--swa-full", "--no-swa-full", "--support-online", "--no-support-online"} {
		if strings.Contains(joined, flag) {
			t.Errorf("an untouched row emitted %s as a typed flag: %q", flag, joined)
		}
	}
}

// Nothing fails when a new LaunchRequest field is added and never wired into
// LaunchArgs — the selection simply disappears between the screen and the
// launch. This pins each field to a deliberate disposition so that adding one
// forces a decision here.
func TestEveryTUILaunchFieldHasADisposition(t *testing.T) {
	// Fields that legitimately never become launch argv.
	notLaunchArgv := map[string]string{
		"Update":        "routes to cmdUpdate instead of a launch",
		"BackendArgs":   "routes to cmdBackend instead of a launch",
		"DownloadRepo":  "download path; never launches",
		"DownloadQuant": "download path; never launches",
		"DownloadDir":   "download path; never launches",
		"AITune":        "selects cmdTune over cmdLaunch; only gates --rounds",
		"FlashAttn":     "hardcoded true in buildLaunchRequest and never emitted; vestigial",
	}

	typ := reflect.TypeOf(tui.LaunchRequest{})
	for i := 0; i < typ.NumField(); i++ {
		name := typ.Field(i).Name
		if _, ok := notLaunchArgv[name]; ok {
			continue
		}
		if !fieldReachesArgv(t, name) {
			t.Errorf("LaunchRequest.%s never reaches launch argv and has no recorded reason; "+
				"either emit it from LaunchArgs or record why it is not a launch flag", name)
		}
	}
}

// gatedBy pairs a field with the companion it needs before LaunchArgs will emit
// it. The gates exist so an untouched preference does not become a typed flag;
// see TestUntouchedTUIRowsDoNotBecomeTypedFlags.
var gatedBy = map[string][]string{
	"KVQuality":     {"KVQualitySet"},
	"KVQualitySet":  {"KVQuality"},
	"Parallel":      {"ParallelSet"},
	"ParallelSet":   {"Parallel"},
	"SupportOnline": {"SupportSet"},
	"SWAFull":       {"SWAFullSet"},
	"AITuneRounds":  {"AITune"},
}

// fieldReachesArgv sets one field (with any companion it is gated behind) to a
// distinctive value and reports whether the emitted argv changes as a result.
func fieldReachesArgv(t *testing.T, name string) bool {
	t.Helper()
	// Claude-only rows are gated behind --claude-code, so the baseline turns it
	// on; ModelPath is always present and anchors the argv.
	base := func() *tui.LaunchRequest {
		return &tui.LaunchRequest{ModelPath: "/models/agent.gguf", ClaudeCode: true}
	}
	before := strings.Join(base().LaunchArgs(), " ")

	req := base()
	for _, f := range append([]string{name}, gatedBy[name]...) {
		v := reflect.ValueOf(req).Elem().FieldByName(f)
		switch v.Kind() {
		case reflect.String:
			v.SetString("roundtrip-probe")
		case reflect.Int:
			v.SetInt(4321)
		case reflect.Bool:
			v.SetBool(!v.Bool())
		default:
			t.Fatalf("LaunchRequest.%s has kind %s, which this guard cannot probe", f, v.Kind())
		}
	}
	return strings.Join(req.LaunchArgs(), " ") != before
}
