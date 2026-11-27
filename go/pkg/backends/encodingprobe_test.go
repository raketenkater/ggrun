package backends

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"testing/iotest"
)

// lits builds a blob of NUL-terminated C literals, as a string table is.
func lits(parts ...string) []byte {
	var b bytes.Buffer
	b.WriteByte(0)
	for _, p := range parts {
		b.WriteString(p)
		b.WriteByte(0)
	}
	return b.Bytes()
}

func TestEncodingsInHelpMatchesWholeWordsIgnoringCase(t *testing.T) {
	help := "  -ctk, --cache-type-k TYPE   (default: f16; allowed: f32, f16, q8_0, stq1_0, q1_0_g128)\n" +
		"  --note  see notestq2_0 and xq3_0\n"
	cases := []struct {
		names []string
		want  []string
	}{
		{[]string{"STQ1_0", "Q8_0", "F16"}, []string{"F16", "Q8_0", "STQ1_0"}},
		{[]string{"Q1_0"}, nil},                          // inside stq1_0 and a prefix of q1_0_g128
		{[]string{"Q1_0_G128"}, []string{"Q1_0_G128"}},   // the whole word is there
		{[]string{"STQ2_0", "Q3_0", "Q2_0"}, nil},        // glued to other letters
		{[]string{"", "Q8_0", "Q8_0"}, []string{"Q8_0"}}, // blanks and repeats collapse
		{nil, nil},
	}
	for _, tc := range cases {
		if got := EncodingsInHelp(help, tc.names); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("EncodingsInHelp(%v) = %v, want %v", tc.names, got, tc.want)
		}
	}
	if got := EncodingsInHelp("", []string{"Q8_0"}); got != nil {
		t.Errorf("empty help found %v", got)
	}
}

// The type-name strings below are cut from real builds (read-only byte dumps,
// 2026-11-27).
//
// mainlineTable: libggml-base.so.0.29.0 of a mainline llama.cpp CUDA build, 733 kB
// into the file, from the end of the op-name table through the type names.
// "f16" is not a string of its own there: the linker merged it into "bf16\0",
// as it did "q2_0" into "tq2_0\0" and "q1_0" into "tq1_0\0".
const mainlineTable = "\x00OPT_STEP_SGD\x00f32\x00q4_0\x00q4_1\x00DEPRECATED\x00q5_0\x00q5_1\x00q8_0\x00q8_1\x00" +
	"q2_K\x00q3_K\x00q4_K\x00q5_K\x00q6_K\x00q8_K\x00iq2_xxs\x00iq2_xs\x00iq3_xxs\x00iq1_s\x00iq4_nl\x00" +
	"iq3_s\x00iq2_s\x00iq4_xs\x00i8\x00i16\x00i32\x00i64\x00f64\x00iq1_m\x00bf16\x00tq1_0\x00tq2_0\x00" +
	"mxfp4\x00nvfp4\x00f8_e4m3\x00parent_size >= node_size\x00buffer_id >= 0\x00"

// ikTable: libggml.so of an ik_llama.cpp build, 42.7 MB into the file; the table
// starts without f32 and f16 and carries the fork's own types (q1_0_g128, the
// _r4 and _kt families). Shortened after "q8_k_r8".
const ikTable = "\x00LATENT_ATTN\x00DS4_COMP\x00q4_0\x00q4_1\x00DEPRECATED\x00q5_0\x00q5_1\x00q8_0\x00q8_1\x00" +
	"q2_K\x00q3_K\x00q4_K\x00q5_K\x00q6_K\x00q8_K\x00iq2_xxs\x00iq2_xs\x00iq3_xxs\x00iq1_s\x00iq4_nl\x00" +
	"iq3_s\x00iq2_s\x00iq4_xs\x00iq1_m\x00bf16\x00q4_0_4x4\x00q4_0_4x8\x00q4_0_8x8\x00i2_s\x00q1_0_g128\x00" +
	"q8_0_x4\x00q8_1_x4\x00q8_2_x4\x00q6_0\x00iq1_bn\x00iq2_bn\x00q8_K64\x00iq2_k\x00iq3_k\x00iq4_k\x00" +
	"iq5_k\x00iq6_k\x00iq4_ks\x00iq2_ks\x00iq4_kss\x00q8_K16\x00q8_K32\x00q8_KR8\x00q8_K128\x00q8_KV\x00" +
	"iq5_ks\x00iq2_kt\x00iq3_kt\x00iq4_kt\x00iq3_ks\x00iq2_kl\x00iq1_kt\x00q4_0_r8\x00q8_k_r8\x00galloc != NULL\x00"

// The first pair alone also occurs outside the table: the ik_llama.cpp server
// binary holds "\0q4_0\0q4_1\0" once and no second pair.
const pairOnly = "\x00ggml_row_size\x00q4_0\x00q4_1\x00some_other_list\x00"

func tableOf(t *testing.T, blob string, chunk int, pad int) (names map[string]bool) {
	t.Helper()
	data := append(bytes.Repeat([]byte("x\x00"), pad), blob...)
	region, _, complete, err := scanTypeTable(bytes.NewReader(data), chunk, 0)
	if err != nil || !complete || region == nil {
		t.Fatalf("chunk %d pad %d: region %d bytes complete %v err %v", chunk, pad, len(region), complete, err)
	}
	names = map[string]bool{}
	for _, tok := range typeTableTokens(region) {
		names[tok] = true
	}
	return names
}

func TestTypeTableIsFoundWhereverItFallsAgainstTheChunks(t *testing.T) {
	for _, blob := range []string{mainlineTable, ikTable} {
		for chunk := 1; chunk <= 70; chunk += 3 {
			for pad := 0; pad < 40; pad += 7 {
				names := tableOf(t, blob, chunk, pad)
				if !names["q4_K"] || !names["bf16"] || !names["iq4_xs"] {
					t.Fatalf("chunk %d pad %d: tokens %v", chunk, pad, names)
				}
			}
		}
	}
	// A large first block: the table sits well past the first reads.
	big := append(bytes.Repeat([]byte("abc\x00"), 5000), mainlineTable...)
	if region, _, complete, err := scanTypeTable(bytes.NewReader(big), 1000, 0); err != nil || !complete || region == nil {
		t.Fatalf("big: %d %v %v", len(region), complete, err)
	}
}

func TestClassifyAgainstTheRealTables(t *testing.T) {
	asked := []string{"Q4_K", "IQ4_XS", "Q8_0", "F16", "BF16", "TQ1_0", "TQ2_0", "Q2_0", "Q1_0", "STQ1_0", "NO_SUCH_TYPE_XYZ", "q4_k", "mxfp4"}
	cases := []struct {
		label        string
		blob         string
		found        []string
		undetermined map[string]string
	}{
		// f16, q2_0 and q1_0 exist only as tails of strings the linker kept.
		{"mainline", mainlineTable,
			[]string{"Q4_K", "IQ4_XS", "Q8_0", "BF16", "TQ1_0", "TQ2_0", "q4_k", "mxfp4"},
			map[string]string{"F16": "bf16", "Q2_0": "tq2_0", "Q1_0": "tq1_0"}},
		// ik: no tq types, no mxfp4 in this cut; q1_0_g128 is not a q1_0.
		{"ik", ikTable,
			[]string{"Q4_K", "IQ4_XS", "Q8_0", "BF16", "q4_k"},
			map[string]string{"F16": "bf16"}},
	}
	for _, tc := range cases {
		names := tableOf(t, tc.blob, 64, 3)
		var tokens []string
		for n := range names {
			tokens = append(tokens, n)
		}
		sort.Strings(tokens)
		found, undetermined := classifyTypeNames(asked, tokens)
		if !reflect.DeepEqual(found, tc.found) || !reflect.DeepEqual(undetermined, tc.undetermined) {
			t.Errorf("%s: found %v undetermined %v\n want %v %v", tc.label, found, undetermined, tc.found, tc.undetermined)
		}
	}
}

func TestTypeTableNeedsBothPairs(t *testing.T) {
	region, _, complete, err := scanTypeTable(bytes.NewReader([]byte(pairOnly)), 8, 0)
	if err != nil || !complete || region != nil {
		t.Fatalf("one pair is no table: %d bytes complete %v err %v", len(region), complete, err)
	}
	// The second pair too far from the first.
	far := "\x00q4_0\x00q4_1\x00" + strings.Repeat("padding\x00", 20) + "q8_0\x00q8_1\x00"
	if region, _, _, _ := scanTypeTable(bytes.NewReader([]byte(far)), 8, 0); region != nil {
		t.Fatal("a second pair beyond the gap counted")
	}
	// The first occurrence is a decoy, the second a real table.
	both := append([]byte(pairOnly), mainlineTable...)
	if region, _, _, _ := scanTypeTable(bytes.NewReader(both), 16, 0); region == nil {
		t.Fatal("a decoy hid the table")
	}
}

func TestScanTypeTableIsBounded(t *testing.T) {
	blob := append(bytes.Repeat([]byte{'x'}, 1000), mainlineTable...)
	region, read, complete, err := scanTypeTable(bytes.NewReader(blob), 64, 500)
	if err != nil || complete || region != nil || read != 500 {
		t.Fatalf("over the limit: %d bytes, read %d, complete %v, err %v", len(region), read, complete, err)
	}
	if _, _, complete, err = scanTypeTable(zeroReader{}, 64, 0); err == nil || complete {
		t.Fatalf("stalled reader: complete=%v err=%v", complete, err)
	}
	_, _, complete, err = scanTypeTable(io.MultiReader(bytes.NewReader([]byte("abc")), iotest.ErrReader(io.ErrClosedPipe)), 64, 0)
	if err == nil || complete {
		t.Fatalf("failing reader: complete=%v err=%v", complete, err)
	}
	if got, _, _, err := scanTypeTable(iotest.OneByteReader(bytes.NewReader([]byte(mainlineTable))), 64, 0); err != nil || got == nil {
		t.Fatalf("one-byte reads: %v", err)
	}
}

type zeroReader struct{}

func (zeroReader) Read([]byte) (int, error) { return 0, nil }

func writeEncodingFixture(t *testing.T, dir, name string, data []byte) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

var probeNames = []string{"Q4_K", "F16", "Q2_0", "STQ1_0", "NO_SUCH_TYPE_XYZ"}

func TestProbeReadsTheTableBesideTheBinary(t *testing.T) {
	dir := t.TempDir()
	bin := writeEncodingFixture(t, dir, "llama-server", lits("hy_v4", "/build/q4_K/x"))
	writeEncodingFixture(t, dir, "libllama.so", lits("q4_K", "f16")) // an unrelated table: never evidence
	writeEncodingFixture(t, dir, "libggml-base.so.0", []byte(mainlineTable))
	writeEncodingFixture(t, dir, "notes.txt", []byte(mainlineTable))
	got := ProbeBackendEncodings(bin, "", probeNames)
	if !got.TableSeen || !got.Complete {
		t.Fatalf("table not seen: %+v", got)
	}
	if !reflect.DeepEqual(got.Found, map[string]string{"Q4_K": "libggml-base.so.0"}) ||
		!reflect.DeepEqual(got.Undetermined, map[string]string{"F16": "bf16", "Q2_0": "tq2_0"}) {
		t.Fatalf("found %v undetermined %v", got.Found, got.Undetermined)
	}
	for name, want := range map[string]bool{"STQ1_0": true, "NO_SUCH_TYPE_XYZ": true, "Q4_K": false, "F16": false, "Q2_0": false} {
		if got.Absent(name) != want {
			t.Errorf("Absent(%q) = %v, want %v", name, got.Absent(name), want)
		}
	}
}

// A static build keeps the table in the binary itself.
func TestProbeReadsTheTableInTheBinary(t *testing.T) {
	bin := writeEncodingFixture(t, t.TempDir(), "llama-server", append([]byte("ELFish\x00"), ikTable...))
	got := ProbeBackendEncodings(bin, "allowed: f16", []string{"Q8_0", "F16", "TQ1_0"})
	if !got.Complete || got.Found["Q8_0"] != "llama-server" || got.Found["F16"] != "help" || !got.Absent("TQ1_0") {
		t.Fatalf("%+v", got)
	}
}

// Without a table nothing is known: a table-less binary, one pair only, no
// libraries located, no build given, a missing file.
func TestProbeWithoutTheTableProvesNothing(t *testing.T) {
	for label, bin := range map[string]string{
		"table-less":  writeEncodingFixture(t, t.TempDir(), "llama-server", lits("hy_v4", "q8_0", "f16")),
		"one pair":    writeEncodingFixture(t, t.TempDir(), "llama-server", []byte(pairOnly)),
		"missing":     filepath.Join(t.TempDir(), "missing"),
		"empty named": "",
	} {
		got := ProbeBackendEncodings(bin, "", []string{"STQ1_0", "Q8_0"})
		if got.Complete || got.TableSeen || len(got.Found) != 0 || len(got.Undetermined) != 0 || got.Absent("STQ1_0") {
			t.Errorf("%s: %+v", label, got)
		}
	}
	// Help alone is partial evidence: a hit is found, the rest stays open.
	got := ProbeBackendEncodings("", "allowed: f16", []string{"F16", "STQ1_0"})
	if !reflect.DeepEqual(got.Found, map[string]string{"F16": "help"}) || got.Complete || got.Absent("STQ1_0") {
		t.Fatalf("help only: %+v", got)
	}
	if got = ProbeBackendEncodings("", "", nil); !got.Complete {
		t.Fatalf("no names: %+v", got)
	}
	if got = ProbeBackendEncodings("", "q8_0", []string{"Q8_0"}); !got.Complete || got.Found["Q8_0"] != "help" {
		t.Fatalf("all in help: %+v", got)
	}
}

// The libraries of a build tree sit outside the binary's directory; its RUNPATH
// says where. Directories are confined to the build tree.
func TestProbeFollowsRunpathInsideTheBuildTree(t *testing.T) {
	base := t.TempDir()
	bin := writeEncodingFixture(t, filepath.Join(base, "build", "bin"), "llama-server", lits("hy_v4"))
	writeEncodingFixture(t, filepath.Join(base, "build", "ggml", "src"), "libggml.so", []byte(ikTable))
	writeEncodingFixture(t, filepath.Join(base, "outside"), "libggml.so", []byte(mainlineTable))
	old := runpathEntries
	defer func() { runpathEntries = old }()

	runpathEntries = func(string) []string { return []string{"$ORIGIN/../ggml/src:$ORIGIN/../../outside:"} }
	got := ProbeBackendEncodings(bin, "", []string{"Q4_K", "TQ1_0"})
	if !got.TableSeen || got.Found["Q4_K"] != "libggml.so" || !got.Absent("TQ1_0") {
		t.Fatalf("runpath inside the tree: %+v", got)
	}

	// Only the outside directory is named: it is not read.
	runpathEntries = func(string) []string { return []string{"${ORIGIN}/../../outside"} }
	got = ProbeBackendEncodings(bin, "", []string{"Q4_K"})
	if got.TableSeen || got.Complete || len(got.Found) != 0 {
		t.Fatalf("a directory outside the build tree was read: %+v", got)
	}

	// No runpath: the libraries are not located.
	runpathEntries = func(string) []string { return nil }
	if got = ProbeBackendEncodings(bin, "", []string{"Q4_K"}); got.Complete || got.TableSeen {
		t.Fatalf("no runpath: %+v", got)
	}
}

func TestProbeSkipsSymlinksLeavingTheBuildTree(t *testing.T) {
	base := t.TempDir()
	target := writeEncodingFixture(t, filepath.Join(base, "outside"), "real.so", []byte(mainlineTable))
	bin := writeEncodingFixture(t, filepath.Join(base, "build", "bin"), "llama-server", lits("hy_v4"))
	if err := os.Symlink(target, filepath.Join(base, "build", "bin", "libggml-evil.so")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	got := ProbeBackendEncodings(bin, "", []string{"Q4_K"})
	if got.TableSeen || len(got.Found) != 0 {
		t.Fatalf("followed a symlink out of the tree: %+v", got)
	}
	// A symlink to a file inside the tree is the same file, read once.
	writeEncodingFixture(t, filepath.Join(base, "build", "bin"), "libggml-base.so.0.1", []byte(mainlineTable))
	if err := os.Symlink("libggml-base.so.0.1", filepath.Join(base, "build", "bin", "libggml-base.so")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	files := encodingProbeFiles(bin)
	n := 0
	for _, f := range files {
		if filepath.Base(f) == "libggml-base.so.0.1" {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("libggml-base listed %d times: %v", n, files)
	}
}

func TestExpandRunpath(t *testing.T) {
	origin := filepath.Join(t.TempDir(), "build", "bin")
	got := expandRunpath(origin, []string{"$ORIGIN/../src:${ORIGIN}:relative/dir::  ", "/abs/lib"})
	want := []string{filepath.Join(filepath.Dir(origin), "src"), origin}
	if abs := "/abs/lib"; filepath.IsAbs(abs) {
		want = append(want, filepath.Clean(abs))
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
	root := filepath.Dir(origin)
	for path, ok := range map[string]bool{root: true, origin: true, filepath.Dir(root): false, filepath.Join(root, "..", "x"): false} {
		if within(root, path) != ok {
			t.Errorf("within(%q, %q) = %v", root, path, !ok)
		}
	}
}

// The byte budget bounds the whole probe: no file is read once it is spent.
func TestProbeStopsAtTheByteBudget(t *testing.T) {
	dir := t.TempDir()
	filler := bytes.Repeat([]byte{'x'}, 60)
	bin := writeEncodingFixture(t, dir, "llama-server", filler)
	writeEncodingFixture(t, dir, "libggml-a.so", filler)
	writeEncodingFixture(t, dir, "libggml-b.so", filler)
	writeEncodingFixture(t, dir, "libggml-c.so", []byte(mainlineTable))
	// 150 bytes: the binary takes 60, libggml-a 60, libggml-b gets 30, libggml-c nothing.
	got := probeBackendEncodings(bin, "", []string{"Q4_K"}, 100, 150, 48)
	if got.TableSeen || got.Complete || len(got.Found) != 0 {
		t.Fatalf("budget spent: %+v", got)
	}
	got = probeBackendEncodings(bin, "", []string{"Q4_K"}, 100000, 1<<20, 48)
	if got.Found["Q4_K"] != "libggml-c.so" || !got.Complete {
		t.Fatalf("budget not spent: %+v", got)
	}
	// A cap on the file count keeps libggml-c out of reach.
	got = probeBackendEncodings(bin, "", []string{"Q4_K"}, 100000, 1<<20, 2)
	if got.Complete || got.TableSeen {
		t.Fatalf("file cap: %+v", got)
	}
}
