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
	region, _, _, complete, err := scanTypeTable(bytes.NewReader(data), chunk, 0, nil)
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
	if region, _, _, complete, err := scanTypeTable(bytes.NewReader(big), 1000, 0, nil); err != nil || !complete || region == nil {
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
		found, undetermined := classifyTypeNames(asked, tokens, nil)
		if !reflect.DeepEqual(found, tc.found) || !reflect.DeepEqual(undetermined, tc.undetermined) {
			t.Errorf("%s: found %v undetermined %v\n want %v %v", tc.label, found, undetermined, tc.found, tc.undetermined)
		}
	}
}

func TestTypeTableNeedsBothPairs(t *testing.T) {
	region, _, _, complete, err := scanTypeTable(bytes.NewReader([]byte(pairOnly)), 8, 0, nil)
	if err != nil || !complete || region != nil {
		t.Fatalf("one pair is no table: %d bytes complete %v err %v", len(region), complete, err)
	}
	// The second pair too far from the first.
	far := "\x00q4_0\x00q4_1\x00" + strings.Repeat("padding\x00", 20) + "q8_0\x00q8_1\x00"
	if region, _, _, _, _ := scanTypeTable(bytes.NewReader([]byte(far)), 8, 0, nil); region != nil {
		t.Fatal("a second pair beyond the gap counted")
	}
	// The first occurrence is a decoy, the second a real table.
	both := append([]byte(pairOnly), mainlineTable...)
	if region, _, _, _, _ := scanTypeTable(bytes.NewReader(both), 16, 0, nil); region == nil {
		t.Fatal("a decoy hid the table")
	}
}

func TestScanTypeTableIsBounded(t *testing.T) {
	blob := append(bytes.Repeat([]byte{'x'}, 1000), mainlineTable...)
	region, _, read, complete, err := scanTypeTable(bytes.NewReader(blob), 64, 500, nil)
	if err != nil || complete || region != nil || read != 500 {
		t.Fatalf("over the limit: %d bytes, read %d, complete %v, err %v", len(region), read, complete, err)
	}
	if _, _, _, complete, err = scanTypeTable(zeroReader{}, 64, 0, nil); err == nil || complete {
		t.Fatalf("stalled reader: complete=%v err=%v", complete, err)
	}
	_, _, _, complete, err = scanTypeTable(io.MultiReader(bytes.NewReader([]byte("abc")), iotest.ErrReader(io.ErrClosedPipe)), 64, 0, nil)
	if err == nil || complete {
		t.Fatalf("failing reader: complete=%v err=%v", complete, err)
	}
	if got, _, _, _, err := scanTypeTable(iotest.OneByteReader(bytes.NewReader([]byte(mainlineTable))), 64, 0, nil); err != nil || got == nil {
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
	if !got.Complete || got.Found["Q8_0"] != "llama-server" || !got.Absent("TQ1_0") || !got.Absent("tq1_0") {
		t.Fatalf("%+v", got)
	}
	// F16 is only a tail of bf16 in this table: undetermined, whatever the help says.
	if _, found := got.Found["F16"]; found || got.Undetermined["F16"] != "bf16" || got.Absent("F16") {
		t.Fatalf("F16: %+v", got)
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
	// Help alone proves nothing: its words are only hints.
	got := ProbeBackendEncodings("", "allowed: f16", []string{"F16", "STQ1_0"})
	if len(got.Found) != 0 || got.Complete || got.TableSeen || got.Absent("STQ1_0") || got.Absent("F16") ||
		!reflect.DeepEqual(got.HelpHints, []string{"F16"}) {
		t.Fatalf("help only: %+v", got)
	}
	if got = ProbeBackendEncodings("", "", nil); got.Complete || got.Absent("anything") {
		t.Fatalf("no names: %+v", got)
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

// ikScalarNames: the ik build's library keeps the short literals "i8", "i16",
// "i32", "f32", "i64" and "f64" in a string table of the gguf value types, 2.3 kB
// before the type-name anchor, and the type table points at those copies. Cut
// from the ik build's library (op names shortened).
const ikScalarNames = "\x00GEGLU_ERF\x00GEGLU_QUICK\x00u8\x00i8\x00u16\x00i16\x00u32\x00i32\x00f32\x00bool\x00arr\x00u64\x00i64\x00f64\x00ABS\x00SGN\x00NEG\x00"

// ikLibrary is that layout: the scalar names, 2.3 kB of other strings, the table.
func ikLibrary() []byte {
	return append(append([]byte(ikScalarNames), bytes.Repeat([]byte("ADD\x00"), 575)...), ikTable...)
}

// A name the table region lacks but the file carries as a NUL-terminated string
// is undetermined, never absent.
func TestProbeNamesOutsideTheTableRegionAreUndetermined(t *testing.T) {
	bin := writeEncodingFixture(t, t.TempDir(), "llama-server", ikLibrary())
	scalars := []string{"F32", "F64", "I8", "I16", "I32", "I64"}
	names := append([]string{"Q4_K", "TQ1_0", "NO_SUCH_TYPE_XYZ"}, scalars...)
	got := ProbeBackendEncodings(bin, "", names)
	if !got.TableSeen || !got.Complete {
		t.Fatalf("%+v", got)
	}
	for _, n := range scalars {
		if got.Undetermined[n] != outsideTable || got.Absent(n) {
			t.Errorf("%s: undetermined %q absent %v", n, got.Undetermined[n], got.Absent(n))
		}
	}
	if got.Found["Q4_K"] == "" || !got.Absent("TQ1_0") || !got.Absent("NO_SUCH_TYPE_XYZ") {
		t.Fatalf("found %v undetermined %v", got.Found, got.Undetermined)
	}
}

// Property: no name present as "name\0" anywhere in the scanned bytes, in any
// case and with any left boundary, comes back absent; whole table strings come
// back found.
func TestProbeNeverReportsAnAbsentNameThatTheFileCarries(t *testing.T) {
	blob := append(ikLibrary(), mainlineTable...)
	blob = append(blob, "\x00Some_MIXED_Case\x00ggml_f16\x00"...)
	bin := writeEncodingFixture(t, t.TempDir(), "llama-server", blob)
	seen := map[string]bool{}
	var names []string
	for _, str := range bytes.Split(blob, []byte{0}) {
		for i := 0; i < len(str); i++ {
			tail := str[i:]
			ok := len(tail) >= 2 && len(tail) <= 32
			for _, c := range tail {
				ok = ok && encodingWordByte(c)
			}
			if ok && !seen[string(tail)] {
				seen[string(tail)] = true
				names = append(names, string(tail), strings.ToUpper(string(tail)))
			}
		}
	}
	got := ProbeBackendEncodings(bin, "", names)
	if !got.Complete {
		t.Fatalf("%+v", got)
	}
	for _, n := range names {
		if got.Absent(n) {
			t.Errorf("%q occurs in the file as a NUL-terminated string but is reported absent", n)
		}
	}
}

// When a read bound cuts the file holding the table short, names found stay
// found but nothing is proven absent: Complete is false.
func TestProbeFileCutShortIsNotComplete(t *testing.T) {
	blob := append(append([]byte(mainlineTable), bytes.Repeat([]byte{'x'}, 8000)...), "\x00q1_0\x00"...)
	bin := writeEncodingFixture(t, t.TempDir(), "llama-server", blob)
	got := probeBackendEncodings(bin, "", []string{"Q4_K", "STQ1_0"}, int64(len(mainlineTable)+4500), 1<<20, 48)
	if !got.TableSeen || got.Complete || got.Absent("STQ1_0") {
		t.Fatalf("cut short: %+v", got)
	}
	if got.Found["Q4_K"] != "llama-server" {
		t.Fatalf("a name in the table region is still found: %+v", got)
	}
	got = probeBackendEncodings(bin, "", []string{"Q4_K", "STQ1_0"}, 1<<20, 1<<20, 48)
	if !got.Complete || !got.Absent("STQ1_0") {
		t.Fatalf("read in full: %+v", got)
	}
}

// Real help output lists KV-cache types, not tensor encodings. These lines are
// the --cache-type-k/-v entries of goldenPlanMainlineHelp and goldenPlanIKHelp
// (go/cmd/ggrun/launch_plan_golden_test.go) plus the "allowed values" line that
// common/arg.cpp prints under them (docs/usage.md lists the same set).
const (
	mainlineKVHelp = "-ctk,  --cache-type-k TYPE              KV cache data type for K (default: f16)\n" +
		"-ctv,  --cache-type-v TYPE              KV cache data type for V (default: f16)\n" +
		"                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1\n"
	ikKVHelp = "-ctk,  --cache-type-k TYPE      KV cache data type for K (default: f16)\n" +
		"-ctv,  --cache-type-v TYPE      KV cache data type for V (default: f16)\n"
)

// Help is never a source of Found or Absent: a name in the cache-type list is a
// hint, and a table that lacks a name stays authoritative.
func TestHelpTextIsOnlyAHint(t *testing.T) {
	names := []string{"F16", "BF16", "Q8_0", "IQ4_NL", "Q4_K", "STQ1_0"}
	table := writeEncodingFixture(t, t.TempDir(), "llama-server", []byte(ikTable))
	for label, help := range map[string]string{"mainline": mainlineKVHelp, "ik": ikKVHelp, "empty": ""} {
		got := ProbeBackendEncodings(table, help, names)
		if !got.Complete || got.Found["Q8_0"] != "llama-server" || got.Found["IQ4_NL"] != "llama-server" ||
			got.Found["Q4_K"] != "llama-server" || got.Found["BF16"] != "llama-server" {
			t.Errorf("%s: table evidence changed: %+v", label, got)
		}
		if !got.Absent("STQ1_0") {
			t.Errorf("%s: a help line must not rescue a name the table lacks: %+v", label, got)
		}
		if _, ok := got.Found["F16"]; ok || got.Undetermined["F16"] != "bf16" {
			t.Errorf("%s: F16 %+v", label, got)
		}
	}
	got := ProbeBackendEncodings(table, mainlineKVHelp, names)
	if want := []string{"BF16", "F16", "IQ4_NL", "Q8_0"}; !reflect.DeepEqual(got.HelpHints, want) {
		t.Errorf("mainline hints %v, want %v", got.HelpHints, want)
	}
	if got = ProbeBackendEncodings(table, ikKVHelp, names); !reflect.DeepEqual(got.HelpHints, []string{"F16"}) {
		t.Errorf("ik hints %v", got.HelpHints)
	}
	// Without a table the whole cache-type list gives no Found and no Absent.
	none := writeEncodingFixture(t, t.TempDir(), "llama-server", lits("hy_v4"))
	got = ProbeBackendEncodings(none, mainlineKVHelp, names)
	if len(got.Found) != 0 || got.Complete {
		t.Fatalf("help without a table: %+v", got)
	}
	for _, n := range names {
		if got.Absent(n) {
			t.Errorf("Absent(%q) from help alone", n)
		}
	}
}

// Absent answers only for names that were asked about, and only after the type
// table was read in full.
func TestAbsentNeedsAskedNamesAndTheTable(t *testing.T) {
	table := writeEncodingFixture(t, t.TempDir(), "llama-server", []byte(mainlineTable))
	got := ProbeBackendEncodings(table, "", []string{"Q4_K", "STQ1_0"})
	if !got.Complete || !got.Absent("STQ1_0") || !got.Absent("stq1_0") {
		t.Fatalf("asked, table read: %+v", got)
	}
	for _, other := range []string{"TQ1_0", "NO_SUCH_TYPE_XYZ", "", "q4_0"} {
		if got.Absent(other) {
			t.Errorf("Absent(%q) is true for a name that was not asked about", other)
		}
	}
	if got.Absent("Q4_K") {
		t.Error("a found name is not absent")
	}
	// Everything asked about is in the help text: the old "all found in help"
	// completeness is gone, nothing is known about any name.
	all := ProbeBackendEncodings("", mainlineKVHelp, []string{"F16", "Q8_0"})
	if all.Complete || all.TableSeen || all.Absent("F16") || all.Absent("Q8_0") || all.Absent("STQ1_0") {
		t.Fatalf("help-only completeness: %+v", all)
	}
	// A zero value probes nothing and answers nothing.
	var zero EncodingProbe
	if zero.Absent("Q4_K") {
		t.Error("zero probe answered")
	}
	// Table seen but the file cut short: not Complete, so not Absent.
	blob := append(append([]byte(mainlineTable), bytes.Repeat([]byte{'x'}, 8000)...), 0)
	bin := writeEncodingFixture(t, t.TempDir(), "llama-server", blob)
	cut := probeBackendEncodings(bin, "", []string{"STQ1_0"}, int64(len(mainlineTable)+4500), 1<<20, 48)
	if !cut.TableSeen || cut.Complete || cut.Absent("STQ1_0") {
		t.Fatalf("cut short: %+v", cut)
	}
}
