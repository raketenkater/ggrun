package main

import (
	"fmt"
	"os"
	"strings"
)

// launchFlag describes one of ggrun's own launch flags. value: it takes an
// argument. equalsCase: parseLaunchArgs has a dedicated --flag=value case for
// it; every other flag's --flag=value form is rewritten to the space form, so
// both spellings mean the same and neither reaches the backend (review C-14).
type launchFlag struct {
	value, equalsCase bool
}

// launchFlags is every ggrun launch flag. TestLaunchFlagTableMatchesParser
// fails when parseLaunchArgs gains or loses a case this table does not list.
var launchFlags = map[string]launchFlag{
	"--allow-live-memory-probe": {value: false, equalsCase: true},
	"--support-online":          {value: false, equalsCase: true},
	"--no-support-online":       {value: false, equalsCase: true},
	"--support-expert":          {value: true, equalsCase: true},
	"--claude-penalties":        {value: true, equalsCase: true},
	"--benchmark":               {value: false, equalsCase: false},
	"--worker-benchmark":        {value: false, equalsCase: false},
	"--calibrate":               {value: true, equalsCase: false},
	"--dry-run":                 {value: false, equalsCase: false},
	"--emit-server-argv-json":   {value: false, equalsCase: false},
	"--ai-tune":                 {value: false, equalsCase: false},
	"--retune":                  {value: false, equalsCase: false},
	"--download":                {value: false, equalsCase: false},
	"--show-configs":            {value: false, equalsCase: false},
	"--keep-alive":              {value: false, equalsCase: false},
	"--inventory":               {value: true, equalsCase: true},
	"--model":                   {value: true, equalsCase: true},
	"-m":                        {value: true, equalsCase: true},
	"--port":                    {value: true, equalsCase: true},
	"-port":                     {value: true, equalsCase: true},
	"--ctx":                     {value: true, equalsCase: true},
	"-ctx":                      {value: true, equalsCase: true},
	"--ctx-size":                {value: true, equalsCase: true},
	"-c":                        {value: true, equalsCase: true},
	"--kv":                      {value: true, equalsCase: true},
	"-kv":                       {value: true, equalsCase: true},
	"--kv-placement":            {value: true, equalsCase: true},
	"--kv-quality":              {value: true, equalsCase: true},
	"-kv-quality":               {value: true, equalsCase: true},
	"--cache-type-k":            {value: true, equalsCase: true},
	"-ctk":                      {value: true, equalsCase: true},
	"--cache-type-v":            {value: true, equalsCase: true},
	"-ctv":                      {value: true, equalsCase: true},
	"--cpu":                     {value: false, equalsCase: false},
	"-cpu":                      {value: false, equalsCase: false},
	"--gpus":                    {value: true, equalsCase: true},
	"-gpus":                     {value: true, equalsCase: true},
	"--host":                    {value: true, equalsCase: true},
	"--vision":                  {value: false, equalsCase: false},
	"-vision":                   {value: false, equalsCase: false},
	"--claude-code":             {value: false, equalsCase: false},
	"--claude-reviewer":         {value: true, equalsCase: true},
	"--chat-template":           {value: true, equalsCase: true},
	"--claude-resume":           {value: true, equalsCase: true},
	"--claude-resume-force":     {value: false, equalsCase: false},
	"--no-cached-config":        {value: false, equalsCase: false},
	"--claude-profile":          {value: true, equalsCase: true},
	"--no-mmap":                 {value: false, equalsCase: true},
	"--mmap":                    {value: false, equalsCase: true},
	"--swa-full":                {value: false, equalsCase: true},
	"--no-swa-full":             {value: false, equalsCase: true},
	"--mmproj":                  {value: true, equalsCase: true},
	"--server-bin":              {value: true, equalsCase: true},
	"--backend":                 {value: true, equalsCase: true},
	"--tune-cache":              {value: true, equalsCase: true},
	"--rounds":                  {value: true, equalsCase: true},
	"--ram-budget":              {value: true, equalsCase: true},
	"--ram-limit-percent":       {value: true, equalsCase: true},
	"--vram-headroom":           {value: true, equalsCase: true},
	"--ram-headroom":            {value: true, equalsCase: true},
	"--cgroup-headroom":         {value: true, equalsCase: false},
	"--spec":                    {value: true, equalsCase: true},
	"--parallel":                {value: true, equalsCase: true},
	"--threads":                 {value: true, equalsCase: true},
	"-t":                        {value: true, equalsCase: true},
	"--cache-ram":               {value: true, equalsCase: true},
	"-cram":                     {value: true, equalsCase: true},
	"--ctx-checkpoints":         {value: true, equalsCase: true},
	"-ctxcp":                    {value: true, equalsCase: true},
	"--claude-max-active":       {value: true, equalsCase: true},
	"--batch-size":              {value: true, equalsCase: true},
	"-b":                        {value: true, equalsCase: true},
	"--ubatch-size":             {value: true, equalsCase: true},
	"-ub":                       {value: true, equalsCase: true},
	"--force-spec-moe":          {value: false, equalsCase: false},
}

// normalizeLaunchFlagForms rewrites --flag=value for ggrun flags without a
// dedicated equals case into the space form the main switch reads: a value
// flag becomes "--flag value"; a switch becomes "--flag" when true and is
// dropped when false. Unknown flags and everything after "--" pass unchanged.
func normalizeLaunchFlagForms(args []string) ([]string, error) {
	out := make([]string, 0, len(args))
	for i, a := range args {
		if a == "--" {
			return append(out, args[i:]...), nil
		}
		key, val, ok := strings.Cut(a, "=")
		spec, known := launchFlags[key]
		if !ok || !strings.HasPrefix(key, "-") || !known || spec.equalsCase {
			out = append(out, a)
			continue
		}
		if spec.value {
			out = append(out, key, val)
			continue
		}
		on, err := parseSwitchValue(val)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", key, err)
		}
		if on {
			out = append(out, key)
		}
	}
	return out, nil
}

func parseSwitchValue(v string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "", "1", "true", "yes", "on":
		return true, nil
	case "0", "false", "no", "off":
		return false, nil
	}
	return false, fmt.Errorf("want true or false, got %q", v)
}

// looksLikeModelArg is a positional that names a model: a .gguf file, an
// existing file, or a Hugging Face owner/name repository.
func looksLikeModelArg(a string) bool {
	if a == "" || strings.HasPrefix(a, "-") {
		return false
	}
	if strings.HasSuffix(strings.ToLower(a), ".gguf") {
		return true
	}
	if st, err := os.Stat(a); err == nil && st.Mode().IsRegular() {
		return true
	}
	owner, name, ok := strings.Cut(a, "/")
	return ok && owner != "" && name != "" && !strings.ContainsAny(name, "/ ") && !strings.ContainsAny(owner, " .")
}

// unknownFlagTakesNext decides whether a backend flag ggrun does not know
// takes the next token as its value. It must not take the only model path:
// "ggrun --some-switch model.gguf" lost the model. When a later positional
// also names a model, the next token is this flag's value as before
// ("--lora adapter.gguf model.gguf").
func unknownFlagTakesNext(args []string, i int, haveModel bool) bool {
	if i+1 >= len(args) || strings.HasPrefix(args[i+1], "-") {
		return false
	}
	if haveModel || !looksLikeModelArg(args[i+1]) {
		return true
	}
	for _, later := range args[i+2:] {
		if later == "--" {
			break
		}
		if looksLikeModelArg(later) {
			return true
		}
	}
	return false
}
