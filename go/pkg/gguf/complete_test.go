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

// A metadata-only shard (no tensors) is written unpadded by split tools, so
// its expected size is the unpadded header end.
func TestExpectedSizeMetadataOnlyShardIsUnpadded(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m-00001-of-00002.gguf")
	writeGGUFFixture(t, path, nil, 32, 0)
	// The fixture pads the header to the alignment; cut the file back to the
	// bare header end, which is what ExpectedSize reports for no tensors.
	hdr, ok := ExpectedSize(path)
	if !ok || hdr <= 0 {
		t.Fatalf("ExpectedSize = %d, %v", hdr, ok)
	}
	if hdr != 106 {
		t.Fatalf("bare header end = %d, want 106", hdr)
	}
	if hdr%32 == 0 {
		t.Fatalf("fixture header %d happens to be aligned; test needs an unaligned one", hdr)
	}
	if err := os.Truncate(path, hdr); err != nil {
		t.Fatal(err)
	}
	got, ok := ExpectedSize(path)
	if !ok || got != hdr {
		t.Fatalf("ExpectedSize = %d, %v; file is %d", got, ok, hdr)
	}
	// One byte short cuts into the header itself, which cannot be parsed:
	// no verdict rather than a guess.
	if err := os.Truncate(path, hdr-1); err != nil {
		t.Fatal(err)
	}
	if got, ok := ExpectedSize(path); ok {
		t.Fatalf("truncated header gave a verdict: %d", got)
	}
}

// With tensor data following, the header end is still rounded up to the
// alignment.
func TestExpectedSizeWithTensorKeepsAlignment(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.gguf")
	writeGGUFFixture(t, path, []fixtureTensor{{name: "a", elems: 8, ttype: 0, offset: 0}}, 32, 32)
	st, _ := os.Stat(path)
	got, ok := ExpectedSize(path)
	if !ok || got != st.Size() || got%32 != 0 {
		t.Fatalf("ExpectedSize = %d, %v; file is %d", got, ok, st.Size())
	}
}
