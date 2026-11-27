package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Golden console output of the real `ggrun dry-run` (review W11).
//
// The plan goldens (launch_plan_golden_test.go and friends) replay the planning
// steps by hand. This test runs cmdDryRun itself, in a child copy of the test
// binary, and pins what it prints on stdout, on stderr, and its exit code, so a
// refactor that turns prints and os.Exit calls into values cannot change a
// stream, the order of notes or an exit status unnoticed.
//
// Hermetic inputs, all built by the test, twice (two fixture roots, so a value
// derived from where the fixtures live shows up as a difference):
//   - synthetic dense GGUFs (real header and tensor table, sparse data), parsed
//     by the embedded parse_gguf.py with the host's python3;
//   - a fake backend, a /bin/sh script that prints fixed help and version text,
//     named with --server-bin or (one case) found in the app home;
//   - --inventory hardware files;
//   - a fresh HOME, cache, config path and TMPDIR per run, and a PATH holding
//     only python3. detect.Detect runs even with --inventory; with no
//     nvidia-smi or vulkaninfo on PATH it, and the live VRAM readings after it,
//     find nothing. Nothing in the child is stubbed.
// The child runs in the fixture root and every path argument is relative.
//
// Normalised: the fixture root becomes $ROOT and the per-run directory $RUN.
// Nothing else is rewritten.
//
// Not reached: dryRunFitRefusal. With these fixtures a dense CPU-offload plan
// never puts KV on a GPU ledger, with or without backend --fit.
//
// Left out: a "Model does not fit" placement error. placementErrorMessage then
// scans /proc for running llama-server processes (activeLlamaServerMemoryHint),
// which another package's test can be running at the same moment.
//
// Regenerate (Linux only):
//
//	go test ./cmd/ggrun -run TestDryRunConsoleGolden -update-dry-run-console
//
// On a difference the computed file is logged as gzip+base64 lines:
//
//	grep -o 'dry-run-console-golden-b64 [0-9]* [A-Za-z0-9+/=]*' log | sort -k2,2n |
//	  awk '{printf "%s", $3}' | base64 -d | gunzip > testdata/dry_run_console_golden.json

var updateDryRunConsole = flag.Bool("update-dry-run-console", false, "rewrite testdata/dry_run_console_golden.json")

const (
	dryRunConsoleGoldenPath = "testdata/dry_run_console_golden.json"
	dryRunConsoleHelperEnv  = "GGRUN_DRY_RUN_CONSOLE_HELPER"
	dryRunConsoleArgsEnv    = "GGRUN_DRY_RUN_CONSOLE_ARGS"
	dryRunConsoleMainEnv    = "GGRUN_DRY_RUN_CONSOLE_MAIN"
)

type dryRunConsoleCase struct {
	name string
	args []string // relative to the fixture root, the child's working directory
	env  []string // extra child environment
	// autoBackend installs the fake backend as <app home>/.bin/llama-server-cuda
	// for automatic selection instead of naming it with --server-bin.
	// "literal" appends the NUL-bracketed architecture name the loader probe
	// looks for (backends.BackendSupportsArch) after the script's exit.
	autoBackend string
	// viaMain runs the child through main with args as its command line
	// (args[0] is the command), so the real argument dispatch is covered.
	viaMain bool
}

// jsonMode reports whether the case asks for --emit-server-argv-json: the
// record then says whether stdout is one JSON document and the contract is
// checked.
func (c dryRunConsoleCase) jsonMode() bool {
	return hasArg(c.args, "--emit-server-argv-json")
}

type dryRunConsoleRecord struct {
	Case          string   `json:"case"`
	Args          []string `json:"args"`
	Env           []string `json:"env,omitempty"`
	ExitCode      int      `json:"exit_code"`
	Stdout        string   `json:"stdout"`
	Stderr        string   `json:"stderr"`
	StdoutOneJSON *bool    `json:"stdout_is_one_json_document,omitempty"`
}

const (
	dryRunConsoleDense     = "models/Synthetic-Dense-8B-Q4_K_M.gguf"
	dryRunConsoleDeepSeek4 = "models/Synthetic-DeepSeek4-Q4_K_M.gguf"
	dryRunConsoleBackend   = "backend/llama-server"
	dryRunConsoleNoFit     = "backend-nofit/llama-server"
)

func dryRunConsoleCases() []dryRunConsoleCase {
	on := func(model, hw string, extra ...string) []string {
		return append([]string{model, "--server-bin", dryRunConsoleBackend, "--inventory", "inventories/" + hw + ".json"}, extra...)
	}
	dense := func(hw string, extra ...string) []string { return on(dryRunConsoleDense, hw, extra...) }
	return []dryRunConsoleCase{
		{name: "plain/gpu12", args: dense("gpu12")},
		{name: "plain/mixed3", args: dense("mixed3")},
		{name: "emit-json/gpu12", args: dense("gpu12", "--emit-server-argv-json")},
		{name: "claude-code/gpu12", args: dense("gpu12", "--claude-code")},
		{name: "claude-code/gpu12/emit-json", args: dense("gpu12", "--claude-code", "--emit-server-argv-json")},
		// claudeCodeSlotAdjust lowers --parallel and warns about the slot size.
		{name: "claude-code/gpu12/ctx16384/emit-json", args: dense("gpu12", "--claude-code", "--ctx", "16384", "--emit-server-argv-json")},
		// The deepseek4 KV rule promotes a quantised K-cache on mainline.
		{name: "arch-kv-rule/deepseek4/gpu12/emit-json", args: on(dryRunConsoleDeepSeek4, "gpu12", "--kv-quality", "q8_0", "--emit-server-argv-json")},
		{name: "trace-placement/gpu12/emit-json", args: dense("gpu12", "--emit-server-argv-json"), env: []string{"GGRUN_TRACE_PLACEMENT=1"}},
		{name: "exposed-host/gpu12", args: dense("gpu12", "--host", "0.0.0.0")},
		// Automatic backend selection: no --server-bin, the backend in the app home.
		// The loader probe reads the script, finds no architecture literal and
		// refuses it.
		{name: "auto-backend/gpu12", args: []string{dryRunConsoleDense, "--inventory", "inventories/gpu12.json"}, autoBackend: "plain"},
		{name: "auto-backend/arch-literal/gpu12", args: []string{dryRunConsoleDense, "--inventory", "inventories/gpu12.json"}, autoBackend: "literal"},
		// A 4 GB dense model on a 2 GB card: CPU offload, an explicit context
		// and GPU KV. A backend with --fit plans no KV on the card, so the dry
		// run's own fit check does not fire.
		{name: "dense-offload-gpu-kv/gpu2", args: dense("gpu2", "--ctx", "131072", "--kv-placement", "gpu")},
		{name: "dense-offload-gpu-kv/gpu2/emit-json", args: dense("gpu2", "--ctx", "131072", "--kv-placement", "gpu", "--emit-server-argv-json")},
		// The same on a backend without --fit. The ledger still carries no KV on
		// the card, so dryRunFitRefusal does not fire here either, and the argv
		// keeps -ngl 999 for a model larger than the card (the known latent
		// defect the plan golden's nofit case records).
		{name: "dense-offload-gpu-kv/nofit-backend/gpu2", args: append([]string{dryRunConsoleDense, "--server-bin", dryRunConsoleNoFit, "--inventory", "inventories/gpu2.json"}, "--ctx", "131072", "--kv-placement", "gpu")},
		// planLaunch rejects a --gpus index the hardware view lacks, with the
		// launch's own message, before the model is read.
		{name: "gpus-index-missing/gpu12", args: dense("gpu12", "--gpus", "9")},
		{name: "gpus-index-missing/mixed3/emit-json", args: dense("mixed3", "--gpus", "0,5", "--emit-server-argv-json")},
		// parseLaunchArgs rejects these first (exit 2), launch and dry run alike.
		{name: "gpus-duplicate/mixed3", args: dense("mixed3", "--gpus", "1,1")},
		{name: "gpus-malformed/mixed3", args: dense("mixed3", "--gpus", "1,x")},
		// A valid selection of a card that is not index 0 plans as before.
		{name: "gpus-valid/mixed3", args: dense("mixed3", "--gpus", "1")},
		{name: "unknown-flag/gpu12", args: dense("gpu12", "--no-such-ggrun-flag")},
		{name: "flag-missing-value", args: dense("gpu12", "--ctx")},
		{name: "no-model", args: []string{"--inventory", "inventories/gpu12.json"}},
		{name: "model-file-missing", args: on("models/missing.gguf", "gpu12")},
		{name: "inventory-file-missing", args: dense("missing")},
		// JSON mode with notes from several callees, and a JSON-mode error.
		{name: "trace-placement/claude-code/gpu12/emit-json", args: dense("gpu12", "--claude-code", "--emit-server-argv-json"), env: []string{"GGRUN_TRACE_PLACEMENT=1"}},
		{name: "model-file-missing/emit-json", args: on("models/missing.gguf", "gpu12", "--emit-server-argv-json")},
		// The same inventory plan as emit-json/gpu12, entered through main's
		// dispatch of "dry-run".
		{name: "main/dry-run/gpu12/emit-json", args: append([]string{"dry-run"}, dense("gpu12", "--emit-server-argv-json")...), viaMain: true},
	}
}

// TestDryRunConsoleHelper is the child: it runs cmdDryRun with the arguments
// the parent passes, or main with them as its command line, and exits with its
// status. It does nothing in a normal run.
func TestDryRunConsoleHelper(t *testing.T) {
	if os.Getenv(dryRunConsoleHelperEnv) != "1" {
		return
	}
	var args []string
	if err := json.Unmarshal([]byte(os.Getenv(dryRunConsoleArgsEnv)), &args); err != nil {
		fmt.Fprintf(os.Stderr, "helper: bad args: %v\n", err)
		os.Exit(97)
	}
	if os.Getenv(dryRunConsoleMainEnv) == "1" {
		os.Args = append([]string{"ggrun"}, args...)
		main()
	} else {
		cmdDryRun(args)
	}
	os.Exit(0)
}

func TestDryRunConsoleGolden(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("dry-run console golden is recorded on Linux")
	}
	python, err := exec.LookPath("python3")
	if err != nil {
		if os.Getenv("CI") != "" {
			t.Fatalf("python3 is required to parse the synthetic GGUF: %v", err)
		}
		t.Skip("python3 not found")
	}
	if resolved, err := filepath.EvalSymlinks(python); err == nil {
		python = resolved
	}
	// Two independent fixture roots: a value derived from where the fixtures
	// live (a hashed path, an mtime) differs between the two runs of a case.
	roots := [2]string{dryRunConsoleFixture(t, python), dryRunConsoleFixture(t, python)}

	var records []dryRunConsoleRecord
	// The JSON interface's contract, checked for every JSON-mode case and
	// reported after the golden comparison (so a recording run still logs the
	// file): a successful run prints exactly one JSON document on stdout and
	// nothing else; a failed run prints nothing on stdout.
	var contract []string
	defer func() {
		for _, msg := range contract {
			t.Error(msg)
		}
	}()
	for _, c := range dryRunConsoleCases() {
		var runs [2]dryRunConsoleRecord
		for i := range runs {
			runs[i] = dryRunConsoleRun(t, c, roots[i])
		}
		if c.jsonMode() {
			if r := runs[0]; r.ExitCode == 0 && !*r.StdoutOneJSON {
				contract = append(contract, fmt.Sprintf("%s: exit 0 but stdout is not exactly one JSON document: %q", c.name, r.Stdout))
			} else if r.ExitCode != 0 && r.Stdout != "" {
				contract = append(contract, fmt.Sprintf("%s: exit %d but stdout is not empty: %q", c.name, r.ExitCode, r.Stdout))
			}
		}
		first, second := dryRunConsoleMarshal(t, runs[0]), dryRunConsoleMarshal(t, runs[1])
		if !bytes.Equal(first, second) {
			t.Errorf("%s: two identical dry runs disagree\nfirst:  %s\nsecond: %s", c.name, first, second)
		}
		if bytes.Contains(first, []byte("TestDryRunConsoleGolden")) {
			t.Errorf("%s: output carries an unnormalised temporary path: %s", c.name, first)
		}
		records = append(records, runs[0])
	}
	if t.Failed() {
		return
	}
	got, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	if *updateDryRunConsole {
		if err := os.WriteFile(dryRunConsoleGoldenPath, got, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(dryRunConsoleGoldenPath)
	if errors.Is(err, os.ErrNotExist) {
		_ = os.WriteFile(dryRunConsoleGoldenPath, got, 0o644)
		dryRunConsoleLogEncoded(t, got)
		t.Fatalf("%s was missing; wrote it from this run. Review and commit it, or regenerate with -update-dry-run-console", dryRunConsoleGoldenPath)
	}
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(got, want) {
		return
	}
	var wantRecords []dryRunConsoleRecord
	if err := json.Unmarshal(want, &wantRecords); err != nil {
		dryRunConsoleLogEncoded(t, got)
		t.Fatalf("%s is not a dry-run console list: %v", dryRunConsoleGoldenPath, err)
	}
	for i := 0; i < max(len(records), len(wantRecords)); i++ {
		var g, w []byte
		if i < len(records) {
			g = dryRunConsoleMarshal(t, records[i])
		}
		if i < len(wantRecords) {
			w = dryRunConsoleMarshal(t, wantRecords[i])
		}
		if !bytes.Equal(g, w) {
			t.Errorf("dry-run console %d changed\nwant: %s\ngot:  %s", i, w, g)
		}
	}
	dryRunConsoleLogEncoded(t, got)
	t.Fatalf("dry-run console output differs from %s; if the change is intended, regenerate with -update-dry-run-console", dryRunConsoleGoldenPath)
}

// dryRunConsoleFixture builds one fixture root: PATH with python3 only, the
// fake backend, the two models and the inventories.
func dryRunConsoleFixture(t *testing.T, python string) string {
	t.Helper()
	root := t.TempDir()
	for _, dir := range []string{"bin", "backend", "backend-nofit", "models", "inventories"} {
		if err := os.MkdirAll(filepath.Join(root, dir), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink(python, filepath.Join(root, "bin", "python3")); err != nil {
		t.Fatal(err)
	}
	dryRunConsoleWriteBackend(t, filepath.Join(root, "backend"), "llama-server", goldenPlanMainlineHelp+goldenPlanMainlineFitHelp, "0000000", "")
	dryRunConsoleWriteBackend(t, filepath.Join(root, "backend-nofit"), "llama-server", goldenPlanMainlineHelp, "0000001", "")
	dryRunConsoleWriteDenseGGUF(t, filepath.Join(root, dryRunConsoleDense), "llama", "Synthetic Dense 8B")
	dryRunConsoleWriteDenseGGUF(t, filepath.Join(root, dryRunConsoleDeepSeek4), "deepseek4", "Synthetic DeepSeek4")
	inventories := map[string]string{
		"gpu12":  goldenPlanInventories["gpu12"],
		"mixed3": goldenPlanInventories["mixed3"],
		"gpu2":   goldenPlanInventory(goldenPlanGPU(0, 2048, 16, 224000)),
	}
	for name, body := range inventories {
		if err := os.WriteFile(filepath.Join(root, "inventories", name+".json"), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// dryRunConsoleRun runs one case in a child process, in root, with a fresh
// home.
func dryRunConsoleRun(t *testing.T, c dryRunConsoleCase, root string) dryRunConsoleRecord {
	t.Helper()
	run := t.TempDir()
	home := filepath.Join(run, "home")
	for _, dir := range []string{home, filepath.Join(run, "tmp")} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	appHome := filepath.Join(home, "app")
	if c.autoBackend != "" {
		dir := filepath.Join(appHome, ".bin")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		tail := ""
		if c.autoBackend == "literal" {
			tail = "\x00llama\x00"
		}
		dryRunConsoleWriteBackend(t, dir, "llama-server-cuda", goldenPlanMainlineHelp+goldenPlanMainlineFitHelp, "0000000", tail)
	}
	encodedArgs, err := json.Marshal(c.args)
	if err != nil {
		t.Fatal(err)
	}
	env := []string{
		dryRunConsoleHelperEnv + "=1",
		dryRunConsoleArgsEnv + "=" + string(encodedArgs),
		"PATH=" + filepath.Join(root, "bin"),
		"HOME=" + home,
		"TMPDIR=" + filepath.Join(run, "tmp"),
		"XDG_CACHE_HOME=" + filepath.Join(home, ".cache"),
		"XDG_CONFIG_HOME=" + filepath.Join(home, ".config"),
		"LLM_CONFIG=" + filepath.Join(home, "missing-config"),
		"LLM_APP_HOME=" + appHome,
		"LLM_CACHE_DIR=" + filepath.Join(home, "cache"),
		"LLM_SERVER_SKIP_DRAFT_DOWNLOAD=1",
	}
	if c.viaMain {
		env = append(env, dryRunConsoleMainEnv+"=1")
	}
	env = append(env, c.env...)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestDryRunConsoleHelper$", "-test.count=1")
	cmd.Env = env
	cmd.Dir = root
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	exit := 0
	if err := cmd.Run(); err != nil {
		var exitErr *exec.ExitError
		if !errors.As(err, &exitErr) {
			t.Fatalf("%s: run helper: %v", c.name, err)
		}
		exit = exitErr.ExitCode()
	}
	normalise := func(s string) string {
		s = strings.ReplaceAll(s, run, "$RUN")
		return strings.ReplaceAll(s, root, "$ROOT")
	}
	record := dryRunConsoleRecord{Case: c.name, Args: c.args, Env: c.env, ExitCode: exit, Stdout: normalise(stdout.String()), Stderr: normalise(stderr.String())}
	if c.jsonMode() {
		one := dryRunConsoleOneJSON(stdout.Bytes())
		record.StdoutOneJSON = &one
	}
	return record
}

// dryRunConsoleOneJSON reports whether out is exactly one JSON document and
// nothing else (one trailing newline allowed).
func dryRunConsoleOneJSON(out []byte) bool {
	dec := json.NewDecoder(bytes.NewReader(out))
	var v any
	if err := dec.Decode(&v); err != nil {
		return false
	}
	return len(bytes.TrimSpace(out[dec.InputOffset():])) == 0 && len(bytes.TrimLeft(out, " \t\r\n")) == len(out)
}

// dryRunConsoleWriteBackend writes a fake llama-server: a shell script that
// prints help for --help and a fixed version for --version, using shell
// builtins only (PATH holds nothing but python3). Its size and mtime are fixed,
// so its build identity is the same in every run. A non-empty tail follows an
// exit the shell never reads past.
func dryRunConsoleWriteBackend(t *testing.T, dir, name, help, commit, tail string) {
	t.Helper()
	script := "#!/bin/sh\n" +
		"d=${0%/*}\n" +
		"f=\"$d/help.txt\"\n" +
		"[ \"$1\" = --version ] && f=\"$d/version.txt\"\n" +
		"while IFS= read -r line || [ -n \"$line\" ]; do printf '%s\\n' \"$line\"; done < \"$f\"\n"
	if tail != "" {
		script += "exit 0\n" + tail
	}
	files := map[string]string{
		"help.txt":    help,
		"version.txt": "version: 9999 (" + commit + ")\nbuilt with synthetic golden toolchain for x86_64-linux-gnu\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	fixed := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	if err := os.Chtimes(path, fixed, fixed); err != nil {
		t.Fatal(err)
	}
}

// dryRunConsoleWriteDenseGGUF writes a dense GGUF of about 4 GB with the given
// architecture key prefix:
// 32 layers, 4096 hidden, 32 heads (8 KV), 14336 FF, 32000 tokens, Q4_K
// weights and a Q6_K head. The header and tensor table are real; the tensor
// data is a sparse hole, which parse_gguf.py never reads (it sizes tensors by
// their offsets and the file size).
func dryRunConsoleWriteDenseGGUF(t *testing.T, path, arch, name string) {
	t.Helper()
	const (
		layers, embd, heads, headsKV, ff, vocab, ctx = 32, 4096, 32, 8, 14336, 32000, 131072
		typeF32, typeQ4K, typeQ6K                    = 0, 12, 14
		align                                        = 32
	)
	type tensor struct {
		name  string
		dims  []uint64
		ttype uint32
	}
	tensorBytes := func(tt tensor) uint64 {
		n := uint64(1)
		for _, d := range tt.dims {
			n *= d
		}
		switch tt.ttype {
		case typeQ4K:
			return n / 256 * 144
		case typeQ6K:
			return n / 256 * 210
		}
		return n * 4
	}
	kvDim := uint64(embd / heads * headsKV)
	tensors := []tensor{{"token_embd.weight", []uint64{embd, vocab}, typeQ4K}}
	for il := 0; il < layers; il++ {
		p := fmt.Sprintf("blk.%d.", il)
		tensors = append(tensors,
			tensor{p + "attn_norm.weight", []uint64{embd}, typeF32},
			tensor{p + "attn_q.weight", []uint64{embd, embd}, typeQ4K},
			tensor{p + "attn_k.weight", []uint64{embd, kvDim}, typeQ4K},
			tensor{p + "attn_v.weight", []uint64{embd, kvDim}, typeQ4K},
			tensor{p + "attn_output.weight", []uint64{embd, embd}, typeQ4K},
			tensor{p + "ffn_norm.weight", []uint64{embd}, typeF32},
			tensor{p + "ffn_gate.weight", []uint64{embd, ff}, typeQ4K},
			tensor{p + "ffn_up.weight", []uint64{embd, ff}, typeQ4K},
			tensor{p + "ffn_down.weight", []uint64{ff, embd}, typeQ4K},
		)
	}
	tensors = append(tensors,
		tensor{"output_norm.weight", []uint64{embd}, typeF32},
		tensor{"output.weight", []uint64{embd, vocab}, typeQ6K},
	)

	buf := new(bytes.Buffer)
	le := func(v any) { _ = binary.Write(buf, binary.LittleEndian, v) }
	str := func(s string) { le(uint64(len(s))); buf.WriteString(s) }
	u32 := func(key string, v uint32) { str(key); le(uint32(4)); le(v) }
	text := func(key, v string) { str(key); le(uint32(8)); str(v) }
	buf.WriteString("GGUF")
	le(uint32(3))
	le(uint64(len(tensors)))
	le(uint64(11)) // metadata count
	text("general.architecture", arch)
	text("general.name", name)
	u32(arch+".block_count", layers)
	u32(arch+".context_length", ctx)
	u32(arch+".embedding_length", embd)
	u32(arch+".feed_forward_length", ff)
	u32(arch+".attention.head_count", heads)
	u32(arch+".attention.head_count_kv", headsKV)
	u32(arch+".rope.dimension_count", embd/heads)
	text("tokenizer.ggml.model", "llama")
	str("tokenizer.ggml.tokens")
	le(uint32(9)) // array
	le(uint32(8)) // of strings
	le(uint64(vocab))
	for i := 0; i < vocab; i++ {
		str(fmt.Sprintf("t%d", i))
	}
	var offset uint64
	for _, tt := range tensors {
		str(tt.name)
		le(uint32(len(tt.dims)))
		for _, d := range tt.dims {
			le(d)
		}
		le(tt.ttype)
		le(offset)
		offset += (tensorBytes(tt) + align - 1) / align * align
	}
	dataStart := (uint64(buf.Len()) + align - 1) / align * align
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(buf.Bytes()); err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(int64(dataStart + offset)); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	fixed := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	if err := os.Chtimes(path, fixed, fixed); err != nil {
		t.Fatal(err)
	}
}

func dryRunConsoleMarshal(t *testing.T, record dryRunConsoleRecord) []byte {
	t.Helper()
	b, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// dryRunConsoleLogEncoded logs data as numbered gzip+base64 lines with its
// sha256, so a CI log yields the exact file (see the comment at the top).
func dryRunConsoleLogEncoded(t *testing.T, data []byte) {
	t.Helper()
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	if _, err := zw.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	encoded := base64.StdEncoding.EncodeToString(buf.Bytes())
	const width = 1000
	chunks := (len(encoded) + width - 1) / width
	t.Logf("dry-run-console-golden sha256 %x bytes %d chunks %d", sha256.Sum256(data), len(data), chunks)
	for i := 0; i < chunks; i++ {
		t.Logf("dry-run-console-golden-b64 %04d %s", i, encoded[i*width:min((i+1)*width, len(encoded))])
	}
}
