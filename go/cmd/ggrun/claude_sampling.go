package main

import (
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/gguf"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// claudeLateSamplers is llama.cpp's default sampler chain with top_k moved to
// the front. The penalties sampler does one hash lookup per vocabulary entry
// per token whenever a penalty is active (about 2.2 ms/token on a 248k-token
// vocabulary, measured live: Qwen MoE decode 105 vs 136 tok/s); after top_k it
// only sees the surviving candidates (131 tok/s measured).
const claudeLateSamplers = "top_k;penalties;dry;top_n_sigma;typ_p;top_p;min_p;xtc;temperature"

// samplingDefaultsReader reads a model's general.sampling.* keys. Tests
// replace it; the default memoizes per file identity because launch argv is
// rebuilt many times per run (calibration candidates, recovery relaunches).
var samplingDefaultsReader = cachedGGUFSamplingDefaults

type samplingCacheKey struct {
	path    string
	size    int64
	modNano int64
}

type samplingCacheEntry struct {
	keys map[string]string
	err  error
}

var samplingCache sync.Map // samplingCacheKey -> samplingCacheEntry

func cachedGGUFSamplingDefaults(path string) (map[string]string, error) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	key := samplingCacheKey{path, st.Size(), st.ModTime().UnixNano()}
	if v, ok := samplingCache.Load(key); ok {
		e := v.(samplingCacheEntry)
		return e.keys, e.err
	}
	keys, err := gguf.SamplingDefaults(path)
	samplingCache.Store(key, samplingCacheEntry{keys, err})
	return keys, err
}

// effectiveClaudePenalties is the penalty policy for this request: the flag or
// config value, else the built-in default.
func effectiveClaudePenalties(req *launchRequest) string {
	if req != nil {
		if mode, err := config.NormalizeClaudePenalties(req.ClaudePenalties); err == nil {
			return mode
		}
	}
	return config.DefaultClaudePenalties
}

// claudeSamplersSupported reports whether the backend takes llama.cpp's
// --samplers chain. ik_llama names its samplers differently, so it keeps the
// full-vocabulary penalties instead of risking an unknown-sampler failure.
func claudeSamplersSupported(be *backendInfo) bool {
	return be != nil && !be.IsIK && strings.Contains(be.Help, "--samplers")
}

// argPresent reports whether the user passed flag, as "--flag value" or
// "--flag=value".
func argPresent(args []string, flag string) bool {
	for _, a := range args {
		if a == flag || strings.HasPrefix(a, flag+"=") {
			return true
		}
	}
	return false
}

// claudeCodeSamplingArgs appends sampling defaults in Claude Code mode and
// returns the argv. See claudeCodeSampling for the rules.
func claudeCodeSamplingArgs(args []string, claudeCode bool, model *placement.ModelProfile, penalties string, samplersOK bool) []string {
	out, _ := claudeCodeSampling(args, claudeCode, model, penalties, samplersOK)
	return out
}

// claudeCodeSampling appends sampling defaults in Claude Code mode and returns
// a one-line summary of what it chose.
//
// The Anthropic /v1/messages conversion only forwards temperature/top_p/top_k from
// the client (server-chat.cpp), and the Anthropic API has no penalty fields at all,
// so repetition control can only come from server-side flags. Penalties are a
// per-model trade-off, not a safe default: in live W1 runs (E-10) repeat 1.05
// broke GLM-4.7-Flash's think and tool-call markup in either sampler order (0 of
// 12 passed, 10 of 12 without it), Qwen3.5-35B-A3B and Gemma 4 did not loop
// without them, and only Qwen3.5-4B looped more often (8 of 27 runs against 3 of
// 39). Hence the default "off" and the opt-in "late" for models that loop.
//
// Penalties follow the policy (penalties): "full" is presence 1.0 + repeat 1.05
// over the last 512 tokens in llama.cpp's default order; "late" keeps repeat 1.05
// but drops presence and runs penalties after top_k (see claudeLateSamplers);
// "off" sends neutral values so a penalty embedded in the GGUF cannot switch
// them back on. "late" without backend --samplers support falls back to "full".
//
// Top-k 20 / top-p 0.95 / min-p 0 (Qwen thinking-mode recommendation; also
// softens the client's greedy temperature-0 classifier calls) are sent only when
// the GGUF carries no general.sampling.<key>; otherwise llama.cpp applies the
// file's value. A temperature is never sent. If the GGUF cannot be read the
// ggrun values are sent. The deepseek4 recipe is fixed and ignores the policy.
// Any flag the user already passed (ExtraArgs) wins — we skip it here.
func claudeCodeSampling(args []string, claudeCode bool, model *placement.ModelProfile, penalties string, samplersOK bool) ([]string, string) {
	if !claudeCode {
		return args, ""
	}
	if model != nil && strings.EqualFold(model.ModelArch, "deepseek4") {
		// V4 template starts assistant turns inside <think>. The validated
		// server recipe closes that immediately; leaving the budget unlimited
		// made Claude Code requests wander in malformed thinking output and the
		// Anthropic parser returned 500s before any useful tool call/content.
		for _, d := range [][2]string{
			{"--presence-penalty", "1.0"},
			{"--repeat-penalty", "1.05"},
			{"--repeat-last-n", "512"},
			{"--temp", "0.7"},
			{"--top-k", "40"},
			{"--top-p", "0.95"},
			{"--min-p", "0.05"},
			{"--reasoning-budget", "0"},
		} {
			if !hasArg(args, d[0]) {
				args = append(args, d[0], d[1])
			}
		}
		return args, "deepseek4 recipe"
	}

	mode, err := config.NormalizeClaudePenalties(penalties)
	if err != nil {
		mode = config.DefaultClaudePenalties
	}
	note := ""
	if mode == "late" && !samplersOK {
		mode, note = "full", " (late needs --samplers; backend lacks it)"
	}
	var defaults [][2]string
	switch mode {
	case "off":
		defaults = [][2]string{{"--repeat-penalty", "1"}, {"--presence-penalty", "0"}}
	case "late":
		defaults = [][2]string{{"--repeat-penalty", "1.05"}, {"--repeat-last-n", "512"}, {"--presence-penalty", "0"}}
	default:
		defaults = [][2]string{{"--presence-penalty", "1.0"}, {"--repeat-penalty", "1.05"}, {"--repeat-last-n", "512"}}
	}

	var fromFile map[string]string
	var readErr error
	switch {
	case model == nil || strings.TrimSpace(model.Path) == "":
		readErr = fmt.Errorf("no model path")
	default:
		fromFile, readErr = samplingDefaultsReader(model.Path)
	}
	sources := make([]string, 0, 3)
	for _, s := range []struct{ flag, key, value string }{
		{"--top-k", "top_k", "20"},
		{"--top-p", "top_p", "0.95"},
		{"--min-p", "min_p", "0.0"},
	} {
		src := "ggrun"
		switch {
		case hasArg(args, s.flag):
			src = "you"
		case readErr == nil && hasKey(fromFile, s.key):
			src = "GGUF"
		default:
			defaults = append(defaults, [2]string{s.flag, s.value})
		}
		sources = append(sources, s.key+"="+src)
	}
	if mode == "late" && !argPresent(args, "--samplers") && !hasArg(args, "--sampling-seq") && !hasArg(args, "--sampler-seq") {
		defaults = append(defaults, [2]string{"--samplers", claudeLateSamplers})
	}
	for _, d := range defaults {
		if !hasArg(args, d[0]) {
			args = append(args, d[0], d[1])
		}
	}
	if readErr != nil {
		note += " (GGUF unreadable)"
	}
	return args, "penalties " + mode + ", " + strings.Join(sources, " ") + note
}

func hasKey(m map[string]string, key string) bool {
	_, ok := m[key]
	return ok
}

var (
	claudeSamplingNoticeMu   sync.Mutex
	claudeSamplingNoticeLast string
)

// announceClaudeSampling prints the sampling summary once per distinct value:
// launch argv is rebuilt for every calibration candidate and relaunch.
func announceClaudeSampling(summary string) {
	if summary == "" {
		return
	}
	claudeSamplingNoticeMu.Lock()
	defer claudeSamplingNoticeMu.Unlock()
	if summary == claudeSamplingNoticeLast {
		return
	}
	claudeSamplingNoticeLast = summary
	fmt.Printf("[claude-code] sampling: %s\n", summary)
}
