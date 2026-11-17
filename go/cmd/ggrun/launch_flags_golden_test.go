package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

var updateLaunchFlagGolden = flag.Bool("update-launch-flags", false, "rewrite testdata/launch_flags_golden.json")

// goldenLaunchFlags lists every ggrun launch flag and whether it takes a value,
// as on origin/main 8dd7a52 before the flag table (review W10).
var goldenLaunchFlags = []struct {
	name  string
	value bool
}{
	{"--allow-live-memory-probe", false},
	{"--support-online", false},
	{"--no-support-online", false},
	{"--support-expert", true},
	{"--benchmark", false},
	{"--worker-benchmark", false},
	{"--calibrate", true},
	{"--dry-run", false},
	{"--emit-server-argv-json", false},
	{"--ai-tune", false},
	{"--retune", false},
	{"--download", false},
	{"--show-configs", false},
	{"--keep-alive", false},
	{"--inventory", true},
	{"--model", true},
	{"-m", true},
	{"--port", true},
	{"-port", true},
	{"--ctx", true},
	{"-ctx", true},
	{"--ctx-size", true},
	{"-c", true},
	{"--kv", true},
	{"-kv", true},
	{"--kv-placement", true},
	{"--kv-quality", true},
	{"-kv-quality", true},
	{"--cache-type-k", true},
	{"-ctk", true},
	{"--cache-type-v", true},
	{"-ctv", true},
	{"--cpu", false},
	{"-cpu", false},
	{"--gpus", true},
	{"-gpus", true},
	{"--host", true},
	{"--vision", false},
	{"-vision", false},
	{"--claude-code", false},
	{"--claude-reviewer", true},
	{"--chat-template", true},
	{"--claude-resume", true},
	{"--claude-resume-force", false},
	{"--no-cached-config", false},
	{"--claude-profile", true},
	{"--no-mmap", false},
	{"--mmap", false},
	{"--swa-full", false},
	{"--no-swa-full", false},
	{"--mmproj", true},
	{"--server-bin", true},
	{"--backend", true},
	{"--tune-cache", true},
	{"--rounds", true},
	{"--ram-budget", true},
	{"--ram-limit-percent", true},
	{"--vram-headroom", true},
	{"--ram-headroom", true},
	{"--cgroup-headroom", true},
	{"--spec", true},
	{"--parallel", true},
	{"--threads", true},
	{"-t", true},
	{"--cache-ram", true},
	{"-cram", true},
	{"--ctx-checkpoints", true},
	{"-ctxcp", true},
	{"--claude-max-active", true},
	{"--batch-size", true},
	{"-b", true},
	{"--ubatch-size", true},
	{"-ub", true},
	{"--force-spec-moe", false},
}

// sampleValues are tried in order until the space form accepts one.
var sampleValues = []string{"8", "auto", "on", "off", "512", "1", "4096", "local", "q8_0", "127.0.0.1", "0", "dense", "fast", "always", "8GB", "/tmp/x.gguf", "org/repo", "never", "full", "native", "smart"}

func requestJSON(t *testing.T, args []string) (string, error) {
	t.Helper()
	req, err := parseLaunchArgs(args)
	if err != nil {
		return "", err
	}
	// Func fields (callbacks) are not request data; zero them to encode.
	b, jerr := json.Marshal(requestData(reflect.ValueOf(req)))
	if jerr != nil {
		t.Fatalf("marshal request: %v", jerr)
	}
	return string(b), nil
}

// requestData is the request as plain data: exported fields, callbacks
// omitted (encoding/json refuses func types even when nil).
func requestData(v reflect.Value) any {
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if v.IsNil() {
			return nil
		}
		return requestData(v.Elem())
	case reflect.Struct:
		out := map[string]any{}
		for i := 0; i < v.NumField(); i++ {
			f := v.Type().Field(i)
			if !f.IsExported() || f.Type.Kind() == reflect.Func || f.Type.Kind() == reflect.Chan {
				continue
			}
			out[f.Name] = requestData(v.Field(i))
		}
		return out
	case reflect.Slice:
		if v.IsNil() {
			return nil
		}
		out := make([]any, v.Len())
		for i := range out {
			out[i] = requestData(v.Index(i))
		}
		return out
	case reflect.Map:
		if v.IsNil() {
			return nil
		}
		out := map[string]any{}
		for _, k := range v.MapKeys() {
			out[fmt.Sprint(k.Interface())] = requestData(v.MapIndex(k))
		}
		return out
	case reflect.Func, reflect.Chan:
		return nil
	default:
		return v.Interface()
	}
}

type goldenRow struct {
	Flag    string `json:"flag"`
	Sample  string `json:"sample,omitempty"`
	Request string `json:"request"`
}

func spaceFormSample(t *testing.T, name string, value bool) (goldenRow, bool) {
	if !value {
		got, err := requestJSON(t, []string{name, "m.gguf"})
		if err != nil {
			return goldenRow{}, false
		}
		return goldenRow{Flag: name, Request: got}, true
	}
	for _, v := range sampleValues {
		if got, err := requestJSON(t, []string{name, v, "m.gguf"}); err == nil {
			return goldenRow{Flag: name, Sample: v, Request: got}, true
		}
	}
	return goldenRow{}, false
}

// Today's space-form behaviour, captured before the refactor, must not change.
func TestLaunchFlagSpaceFormGolden(t *testing.T) {
	path := filepath.Join("testdata", "launch_flags_golden.json")
	var rows []goldenRow
	for _, f := range goldenLaunchFlags {
		row, ok := spaceFormSample(t, f.name, f.value)
		if !ok {
			row = goldenRow{Flag: f.name, Request: "rejected"}
		}
		rows = append(rows, row)
	}
	if *updateLaunchFlagGolden {
		b, _ := json.MarshalIndent(rows, "", "  ")
		if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var want []goldenRow
	if err := json.Unmarshal(b, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(rows, want) {
		for i := range want {
			if i >= len(rows) || !reflect.DeepEqual(rows[i], want[i]) {
				got := goldenRow{}
				if i < len(rows) {
					got = rows[i]
				}
				t.Errorf("%s changed:\n got %s\nwant %s", want[i].Flag, strings.TrimSpace(got.Request), strings.TrimSpace(want[i].Request))
			}
		}
	}
}
