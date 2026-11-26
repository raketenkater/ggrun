package gguf

import (
	"os"
	"path/filepath"
	"testing"
)

// Two F32 tensors: 8 elems = 32 bytes at offset 0, 16 elems = 64 bytes at 32.
// The data section is therefore 96 bytes and the file is complete at
// ExpectedSize.
func writeCompleteFixture(t *testing.T, path string, align int, lastType uint32) {
	t.Helper()
	writeGGUFFixture(t, path, []fixtureTensor{
		{name: "a", elems: 8, ttype: 0, offset: 0},
		{name: "b", elems: 16, ttype: lastType, offset: 32},
	}, align, 96)
}

func TestExpectedSizeMatchesCompleteFile(t *testing.T) {
	for _, align := range []int{32, 64} {
		path := filepath.Join(t.TempDir(), "m.gguf")
		writeCompleteFixture(t, path, align, 0)
		st, _ := os.Stat(path)
		got, ok := ExpectedSize(path)
		if !ok || got != st.Size() {
			t.Fatalf("align %d: ExpectedSize = %d, %v; file is %d", align, got, ok, st.Size())
		}
	}
}

func TestExpectedSizeExceedsTruncatedFileByOneByte(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.gguf")
	writeCompleteFixture(t, path, 32, 0)
	st, _ := os.Stat(path)
	if err := os.Truncate(path, st.Size()-1); err != nil {
		t.Fatal(err)
	}
	got, ok := ExpectedSize(path)
	if !ok || got != st.Size() {
		t.Fatalf("ExpectedSize = %d, %v; want %d", got, ok, st.Size())
	}
}

func TestExpectedSizeUnknownTensorTypeIsNoVerdict(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.gguf")
	writeCompleteFixture(t, path, 32, 9999)
	if got, ok := ExpectedSize(path); ok {
		t.Fatalf("unknown tensor type gave a verdict: %d", got)
	}
}

func TestExpectedSizeCorruptHeaderIsNoVerdict(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.gguf")
	if err := os.WriteFile(path, []byte("GGUF\x03\x00\x00\x00"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, ok := ExpectedSize(path); ok {
		t.Fatal("truncated header gave a verdict")
	}
}
