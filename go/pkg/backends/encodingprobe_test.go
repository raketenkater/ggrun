package backends

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"reflect"
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

// Q1_0 is a substring of STQ1_0 and the prefix of Q1_0_G128, and build paths
// embedded in a binary hold names too; only a whole NUL-bracketed literal
// counts, in the case the catalog spells.
func TestScanEncodingLiteralsNeedsWholeLiterals(t *testing.T) {
	decoys := lits("STQ1_0", "Q1_0_G128", "q1_0_g128", "/home/build/Q1_0/ggml.c", "xQ1_0", "Q1_0x", "stq1_0x")
	got, _, complete, err := scanEncodingLiterals(bytes.NewReader(decoys), []string{"Q1_0", "STQ1_0"}, 5, 0)
	if err != nil || !complete {
		t.Fatalf("complete=%v err=%v", complete, err)
	}
	if want := []string{"STQ1_0"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("found %v, want %v: Q1_0 appears only inside other literals", got, want)
	}
	withIt := lits("STQ1_0", "Q1_0_G128", "Q1_0")
	got, _, _, err = scanEncodingLiterals(bytes.NewReader(withIt), []string{"Q1_0", "STQ1_0", "Q2_0"}, 5, 0)
	if err != nil || !reflect.DeepEqual(got, []string{"Q1_0", "STQ1_0"}) {
		t.Fatalf("found %v err %v", got, err)
	}
}

// ggml spells type names in lower case; a catalog name matches either spelling,
// and is reported as the caller wrote it.
func TestScanEncodingLiteralsMatchesLowerCaseSpelling(t *testing.T) {
	blob := lits("q8_0", "stq1_0")
	got, _, _, err := scanEncodingLiterals(bytes.NewReader(blob), []string{"STQ1_0", "Q8_0", "q4_0"}, 5, 0)
	if err != nil || !reflect.DeepEqual(got, []string{"Q8_0", "STQ1_0"}) {
		t.Fatalf("found %v err %v", got, err)
	}
}

// A literal is found wherever it falls against the read chunks.
func TestScanEncodingLiteralsAcrossChunkBoundaries(t *testing.T) {
	for chunk := 1; chunk <= 9; chunk++ {
		for pad := 0; pad < 12; pad++ {
			blob := append(bytes.Repeat([]byte("x"), pad), lits("Q8_0", "STQ1_0")...)
			got, read, complete, err := scanEncodingLiterals(bytes.NewReader(blob), []string{"STQ1_0", "Q8_0"}, chunk, 0)
			if err != nil || !reflect.DeepEqual(got, []string{"Q8_0", "STQ1_0"}) {
				t.Fatalf("chunk %d pad %d: found %v err %v", chunk, pad, got, err)
			}
			if !complete || read > int64(len(blob)) {
				t.Fatalf("chunk %d pad %d: complete=%v read=%d of %d", chunk, pad, complete, read, len(blob))
			}
		}
	}
	// Reads of one byte at a time behave the same.
	got, _, _, err := scanEncodingLiterals(iotest.OneByteReader(bytes.NewReader(lits("STQ1_0"))), []string{"STQ1_0"}, 64, 0)
	if err != nil || !reflect.DeepEqual(got, []string{"STQ1_0"}) {
		t.Fatalf("one-byte reads: %v %v", got, err)
	}
}

func TestScanEncodingLiteralsIsBounded(t *testing.T) {
	blob := append(bytes.Repeat([]byte{'x'}, 1000), lits("STQ1_0")...)
	got, read, complete, err := scanEncodingLiterals(bytes.NewReader(blob), []string{"STQ1_0"}, 64, 500)
	if err != nil || complete || len(got) != 0 || read != 500 {
		t.Fatalf("over the limit: found %v read %d complete %v err %v", got, read, complete, err)
	}
	// A reader that never makes progress ends in an error, not a hang.
	_, _, complete, err = scanEncodingLiterals(zeroReader{}, []string{"STQ1_0"}, 64, 0)
	if err == nil || complete {
		t.Fatalf("stalled reader: complete=%v err=%v", complete, err)
	}
	// A failing reader reports what it found before failing.
	got, _, complete, err = scanEncodingLiterals(io.MultiReader(bytes.NewReader(lits("Q8_0")), iotest.ErrReader(io.ErrClosedPipe)),
		[]string{"Q8_0", "STQ1_0"}, 64, 0)
	if err == nil || complete || !reflect.DeepEqual(got, []string{"Q8_0"}) {
		t.Fatalf("failing reader: found %v complete %v err %v", got, complete, err)
	}
}

type zeroReader struct{}

func (zeroReader) Read([]byte) (int, error) { return 0, nil }

func writeEncodingFixture(t *testing.T, dir, name string, data []byte) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestProbeBackendEncodingsReadsTheGgmlLibraryBesideTheBinary(t *testing.T) {
	dir := t.TempDir()
	bin := writeEncodingFixture(t, dir, "llama-server", lits("hy_v4", "/build/Q1_0/x"))
	writeEncodingFixture(t, dir, "libggml-base.so.0", lits("q8_0", "STQ1_0", "Q1_0_G128"))
	writeEncodingFixture(t, dir, "ggml-cuda.dll", lits("Q4_0_CUDA"))
	writeEncodingFixture(t, dir, "notes.txt", lits("ZZZ_9")) // not a library: never read
	names := []string{"STQ1_0", "Q1_0", "ZZZ_9"}

	got := ProbeBackendEncodings(bin, "", names)
	if want := map[string]string{"STQ1_0": "libggml-base.so.0"}; !reflect.DeepEqual(got.Found, want) {
		t.Fatalf("found %v, want %v", got.Found, want)
	}
	if !got.Complete {
		t.Fatal("every file was read in full, so absence is evidence")
	}

	// The binary itself counts (a static build), and help text counts. Its type
	// table is shown to be there by the canary literal every table carries.
	static := writeEncodingFixture(t, t.TempDir(), "llama-server", lits("f16", "STQ1_0", "Q1_0"))
	got = ProbeBackendEncodings(static, "allowed: q8_0", []string{"Q8_0", "STQ1_0", "Q1_0", "ZZZ_9"})
	want := map[string]string{"Q8_0": "help", "STQ1_0": "llama-server", "Q1_0": "llama-server"}
	if !reflect.DeepEqual(got.Found, want) || !got.Complete {
		t.Fatalf("static build: %+v, want %v", got, want)
	}
}

// A binary read in full is no proof of absence when the type table was not
// among the files read (a system install: libraries not located), unless the
// canary literal shows it was; a ggml library read in full is enough. A name
// spelled in lower case in the table is found.
func TestProbeBackendEncodingsCompleteNeedsTheTypeTable(t *testing.T) {
	noTable := writeEncodingFixture(t, t.TempDir(), "llama-server", lits("hy_v4", "q8_0"))
	got := ProbeBackendEncodings(noTable, "", []string{"STQ1_0"})
	if len(got.Found) != 0 || got.Complete {
		t.Fatalf("no ggml library and no canary: %+v", got)
	}
	withCanary := writeEncodingFixture(t, t.TempDir(), "llama-server", lits("hy_v4", "f16"))
	got = ProbeBackendEncodings(withCanary, "", []string{"STQ1_0"})
	if len(got.Found) != 0 || !got.Complete {
		t.Fatalf("canary present, name absent: %+v", got)
	}
	dir := t.TempDir()
	bin := writeEncodingFixture(t, dir, "llama-server", lits("hy_v4"))
	writeEncodingFixture(t, dir, "libggml-base.so", lits("q8_0", "stq1_0"))
	got = ProbeBackendEncodings(bin, "", []string{"STQ1_0", "ZZZ_9"})
	if !reflect.DeepEqual(got.Found, map[string]string{"STQ1_0": "libggml-base.so"}) || !got.Complete {
		t.Fatalf("lower-case spelling in a ggml library: %+v", got)
	}
	// A ggml library that could not be read in full leaves it open.
	dir = t.TempDir()
	bin = writeEncodingFixture(t, dir, "llama-server", lits("hy_v4"))
	lib := writeEncodingFixture(t, dir, "libggml-base.so", lits("q8_0"))
	if err := os.Chmod(lib, 0); err == nil {
		if f, openErr := os.Open(lib); openErr != nil { // not readable (not root, not Windows)
			got = ProbeBackendEncodings(bin, "", []string{"STQ1_0"})
			if got.Complete {
				t.Fatalf("unreadable ggml library: %+v", got)
			}
		} else {
			f.Close()
		}
	}
}

// Once the byte budget is spent no further file is read, and absence proves
// nothing: a literal in a file past the budget is not found.
func TestProbeBackendEncodingsStopsAtTheByteBudget(t *testing.T) {
	dir := t.TempDir()
	filler := bytes.Repeat([]byte{'x'}, 60)
	bin := writeEncodingFixture(t, dir, "llama-server", filler)
	writeEncodingFixture(t, dir, "libggml-a.so", filler)
	writeEncodingFixture(t, dir, "libggml-b.so", filler)
	writeEncodingFixture(t, dir, "libggml-c.so", lits("STQ1_0"))
	// 150 bytes: the binary takes 60, libggml-a 60, libggml-b gets 30, libggml-c nothing.
	got := probeBackendEncodings(bin, "", []string{"STQ1_0"}, 100, 150, 48)
	if len(got.Found) != 0 || got.Complete {
		t.Fatalf("budget spent: %+v", got)
	}
	// With room for everything the literal is found.
	got = probeBackendEncodings(bin, "", []string{"STQ1_0"}, 100, 1000, 48)
	if got.Found["STQ1_0"] != "libggml-c.so" || !got.Complete {
		t.Fatalf("budget not spent: %+v", got)
	}
	// A cap on the file count leaves it open as well.
	got = probeBackendEncodings(bin, "", []string{"ZZZ_9"}, 100, 1000, 2)
	if got.Complete {
		t.Fatalf("file cap: %+v", got)
	}
}

func TestProbeBackendEncodingsSaysWhenAbsenceProvesNothing(t *testing.T) {
	// No build given: only the help text speaks, and it is partial evidence.
	got := ProbeBackendEncodings("", "allowed: f16", []string{"F16", "STQ1_0"})
	if !reflect.DeepEqual(got.Found, map[string]string{"F16": "help"}) || got.Complete {
		t.Fatalf("help only: %+v", got)
	}
	// An unreadable binary is not an answer either.
	got = ProbeBackendEncodings(filepath.Join(t.TempDir(), "missing"), "", []string{"STQ1_0"})
	if len(got.Found) != 0 || got.Complete {
		t.Fatalf("missing binary: %+v", got)
	}
	// Nothing asked, nothing to prove.
	if got = ProbeBackendEncodings("", "", nil); !got.Complete || len(got.Found) != 0 {
		t.Fatalf("no names: %+v", got)
	}
	// Everything found in the help text needs no file at all.
	if got = ProbeBackendEncodings("", "q8_0", []string{"Q8_0"}); !got.Complete || got.Found["Q8_0"] != "help" {
		t.Fatalf("all in help: %+v", got)
	}
}
