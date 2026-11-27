package gguf

import (
	"os"
	"path/filepath"
	"testing"
)

// syntheticFixture is a 267,200-byte file written by tools/perf/make_synthetic_gguf.py:
//
//	--arch dense --layers 2 --embd 64 --ff 128 --heads 2 --vocab 360 --ctx 512 \
//	--seed 1 --name ggrun-synth-fixture
//
// The ".fixture" suffix keeps it clear of the repository's *.gguf ignore rule.
// tests/test_synthetic_gguf.py regenerates it and requires the same bytes.
// Every tensor size is a multiple of the 32-byte alignment, so spans equal
// byte sizes and the file has no trailing padding.
const syntheticFixture = "synthetic-dense-tiny.gguf.fixture"

// The generator's offsets (each the running sum of 32-byte-padded sizes), data
// start and file size must satisfy this package's own readers.
func TestSyntheticGeneratorFixtureSatisfiesReaders(t *testing.T) {
	path := filepath.Join("testdata", syntheticFixture)
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if st.Size() > 1<<20 {
		t.Fatalf("fixture is %d bytes; keep it under 1 MiB", st.Size())
	}
	got, ok := ExpectedSize(path)
	if !ok || got != st.Size() {
		t.Fatalf("ExpectedSize = %d, %v; file is %d", got, ok, st.Size())
	}

	// The histogram walk fails on unusable offsets, so success also proves the
	// first offset is 0 and every span is positive and in order.
	types, err := ReadTypeHistogramModel(path)
	if err != nil {
		t.Fatal(err)
	}
	want := map[uint32]TypeUsage{
		0: {Type: 0, Tensors: 5, Elements: 320, Bytes: 1280},       // F32: 4 block norms + output_norm
		1: {Type: 1, Tensors: 16, Elements: 128000, Bytes: 256000}, // F16: embedding, output, 7 matrices per block
	}
	if len(types) != len(want) {
		t.Fatalf("histogram has %d types, want %d: %+v", len(types), len(want), types)
	}
	for _, u := range types {
		w, found := want[u.Type]
		if !found || u.Tensors != w.Tensors || u.Elements != w.Elements || u.Bytes != w.Bytes {
			t.Fatalf("type %d = %+v, want %+v", u.Type, u, w)
		}
	}

	// ggrun's own entry point: the Python helper's view of the same file.
	info, err := Parse(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if info.Architecture != "llama" || info.BlockCount != 2 || info.EmbeddingLength != 64 ||
		info.FeedForwardLength != 128 || info.VocabSize != 360 || info.ExpertBytes != 0 {
		t.Fatalf("parsed %+v", info)
	}
	if info.TokenEmbdBytes != 46080 || info.OutputBytes != 46080 || info.NonExpertBytes != 257280 {
		t.Fatalf("bytes: token_embd %d output %d non-expert %d", info.TokenEmbdBytes, info.OutputBytes, info.NonExpertBytes)
	}
}
