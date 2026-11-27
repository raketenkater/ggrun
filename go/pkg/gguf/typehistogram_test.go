package gguf

import (
	"bytes"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"testing"
)

// The same fixture as the helper's test_type_histogram_counts_spans_and_names:
// an empty tensor counts as an entry but owns no bytes, and an id the type
// table has never heard of is reported as the number it is.
func histogramFixture() []byte {
	return profileFile(nil, []profileTensor{
		{"blk.0.attn_q.weight", []uint64{256, 32}, 8},
		{"zero.weight", []uint64{0, 5}, 8},
		{"blk.1.attn_q.weight", []uint64{256, 32}, 8},
		{"blk.0.ffn_up_exps.weight", []uint64{256, 32, 2}, 99},
	}, map[string]uint64{"blk.0.ffn_up_exps.weight": 4096})
}

var histogramFixtureWant = []TypeUsage{
	{Type: 8, Tensors: 3, Elements: 16384, Bytes: 17408, Names: []string{"blk.*.attn_q.weight", "zero.weight"}},
	{Type: 99, Tensors: 1, Elements: 16384, Bytes: 4096, Names: []string{"blk.*.ffn_up_exps.weight"}},
}

func TestTypeHistogramCountsSpansAndNames(t *testing.T) {
	data := histogramFixture()
	got, err := ReadTypeHistogram(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, histogramFixtureWant) {
		t.Fatalf("histogram\n got  %+v\n want %+v", got, histogramFixtureWant)
	}
	// The profile walk is the same one: its totals match the histogram's.
	p, err := readProfileBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	var bytesSum int64
	for _, u := range got {
		bytesSum += u.Bytes
	}
	if p.Bytes != bytesSum {
		t.Fatalf("profile bytes %d, histogram bytes %d", p.Bytes, bytesSum)
	}
}

// One id, one element count, two encodings: only the spans differ (2.25 and
// 1.3125 bits per weight). A size table keyed by id could not tell them apart.
func TestTypeHistogramCollidingIDDiffersByBytes(t *testing.T) {
	read := func(nbytes uint64) TypeUsage {
		data := profileFile(nil, []profileTensor{{"blk.0.ffn_down_exps.weight", []uint64{256, 64}, 42}},
			map[string]uint64{"blk.0.ffn_down_exps.weight": nbytes})
		got, err := ReadTypeHistogram(bytes.NewReader(data), int64(len(data)))
		if err != nil || len(got) != 1 {
			t.Fatalf("%d bytes: %+v %v", nbytes, got, err)
		}
		return got[0]
	}
	a, b := read(4608), read(2688)
	if a.Type != 42 || b.Type != 42 || a.Elements != b.Elements || a.Bytes != 4608 || b.Bytes != 2688 {
		t.Fatalf("a=%+v b=%+v", a, b)
	}
}

func TestTypeHistogramTruncatedTableGivesNoHistogram(t *testing.T) {
	good := profileHeader([]fixtureKV{kvString("general.architecture", "llama")},
		[]profileTensor{{"blk.0.attn_q.weight", []uint64{256, 256}, 12}})
	if _, err := ReadTypeHistogram(bytes.NewReader(good), int64(len(good))); err != nil {
		t.Fatalf("good header: %v", err)
	}
	for n := 0; n < len(good); n++ {
		if types, err := ReadTypeHistogram(bytes.NewReader(good[:n]), int64(n)); err == nil || types != nil {
			t.Fatalf("truncated at %d: %+v %v", n, types, err)
		}
	}
	// Offsets that do not fit the file: an interrupted download.
	data := histogramFixture()
	if _, err := ReadTypeHistogram(bytes.NewReader(data[:len(data)-20000]), int64(len(data)-20000)); !errors.Is(err, ErrTensorDataIncomplete) {
		t.Fatalf("incomplete: %v", err)
	}
}

func TestTypeHistogramBoundsDistinctTypesAndNames(t *testing.T) {
	var tensors []profileTensor
	spans := map[string]uint64{}
	for i := 0; i < maxTypeIDs+1; i++ {
		name := fmt.Sprintf("t%d.weight", i)
		tensors = append(tensors, profileTensor{name, []uint64{32}, uint32(1000 + i)})
		spans[name] = 34
	}
	data := profileFile(nil, tensors, spans)
	if _, err := ReadTypeHistogram(bytes.NewReader(data), int64(len(data))); !errors.Is(err, ErrTooManyTensorTypes) {
		t.Fatalf("too many ids: %v", err)
	}
	if _, err := readProfileBytes(data); err != nil {
		t.Fatalf("the profile must not fail on a type count only the histogram bounds: %v", err)
	}

	tensors, spans = nil, map[string]uint64{}
	for i := 0; i < maxNamesOut+2; i++ {
		name := fmt.Sprintf("x%03d.weight", i)
		tensors = append(tensors, profileTensor{name, []uint64{32}, 8})
		spans[name] = 34
	}
	data = profileFile(nil, tensors, spans)
	got, err := ReadTypeHistogram(bytes.NewReader(data), int64(len(data)))
	if err != nil || len(got) != 1 {
		t.Fatalf("%+v %v", got, err)
	}
	if got[0].Tensors != maxNamesOut+2 || len(got[0].Names) != maxNamesOut || !got[0].NamesTruncated ||
		got[0].Names[0] != "x000.weight" {
		t.Fatalf("names: %d listed, truncated %v, tensors %d", len(got[0].Names), got[0].NamesTruncated, got[0].Tensors)
	}
}

func TestTensorNamePattern(t *testing.T) {
	for _, tc := range []struct {
		in, want string
		ok       bool
	}{
		{"blk.12.ffn_down_exps.weight", "blk.*.ffn_down_exps.weight", true},
		{"token_embd.weight", "token_embd.weight", true},
		{"blk.0.attn_q2.weight", "blk.*.attn_q2.weight", true},
		{"a.1.2.b", "a.*.*.b", true},
		{"7", "*", true},
		{"", "", false},
		{"café.weight", "", false},
		{"bad\x01name", "", false},
	} {
		got, ok := tensorNamePattern(tc.in)
		if got != tc.want || ok != tc.ok {
			t.Errorf("tensorNamePattern(%q) = %q, %v; want %q, %v", tc.in, got, ok, tc.want, tc.ok)
		}
	}
}

func TestReadTypeHistogramModelSumsShardsAndRequiresThemAll(t *testing.T) {
	dir := t.TempDir()
	first := filepath.Join(dir, "m-00001-of-00002.gguf")
	second := filepath.Join(dir, "m-00002-of-00002.gguf")
	writeProfileFixture(t, first, nil, []profileTensor{{"blk.0.attn_q.weight", []uint64{256, 32}, 8}})
	if _, err := ReadTypeHistogramModel(first); err == nil {
		t.Fatal("a missing shard gave a histogram")
	}
	writeProfileFile(t, second, profileFile(nil, []profileTensor{
		{"blk.1.attn_q.weight", []uint64{256, 32}, 8},
		{"blk.1.ffn_up_exps.weight", []uint64{256, 64}, 42},
	}, map[string]uint64{"blk.1.ffn_up_exps.weight": 4608}))
	got, err := ReadTypeHistogramModel(first)
	if err != nil {
		t.Fatal(err)
	}
	want := []TypeUsage{
		{Type: 8, Tensors: 2, Elements: 16384, Bytes: 2 * 8704, Names: []string{"blk.*.attn_q.weight"}},
		{Type: 42, Tensors: 1, Elements: 16384, Bytes: 4608, Names: []string{"blk.*.ffn_up_exps.weight"}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("merged\n got  %+v\n want %+v", got, want)
	}
}

// Both readers must report the same histogram for the same file: the helper
// feeds Info on the launch path, the Go reader serves everything else.
func TestTypeHistogramGoAndHelperAgree(t *testing.T) {
	dir := t.TempDir()
	single := filepath.Join(dir, "single.gguf")
	writeProfileFile(t, single, histogramFixture())

	first := filepath.Join(dir, "m-00001-of-00002.gguf")
	writeProfileFile(t, first, profileFile(nil, []profileTensor{{"blk.0.attn_q.weight", []uint64{256, 32}, 8}}, nil))
	writeProfileFile(t, filepath.Join(dir, "m-00002-of-00002.gguf"), profileFile(nil, []profileTensor{
		{"blk.1.attn_q.weight", []uint64{256, 32}, 8},
		{"blk.1.ffn_up_exps.weight", []uint64{256, 64}, 42},
	}, map[string]uint64{"blk.1.ffn_up_exps.weight": 4608}))

	for _, path := range []string{single, first} {
		want, err := ReadTypeHistogramModel(path)
		if err != nil {
			t.Fatalf("%s: %v", filepath.Base(path), err)
		}
		info, err := Parse(path)
		if err != nil {
			t.Fatalf("%s: %v", filepath.Base(path), err)
		}
		if !reflect.DeepEqual(info.TensorTypes, want) {
			t.Errorf("%s: helper and Go reader disagree\n helper %+v\n go     %+v", filepath.Base(path), info.TensorTypes, want)
		}
	}

	// A file whose offsets cannot be trusted has no histogram in either reader.
	partial := filepath.Join(dir, "partial.gguf")
	data := histogramFixture()
	writeProfileFile(t, partial, data[:len(data)-20000])
	if _, err := ReadTypeHistogramModel(partial); err == nil {
		t.Error("Go reader gave a histogram for an interrupted download")
	}
	info, err := Parse(partial)
	if err != nil {
		t.Fatal(err)
	}
	if len(info.TensorTypes) != 0 {
		t.Errorf("helper gave a histogram for an interrupted download: %+v", info.TensorTypes)
	}
}
