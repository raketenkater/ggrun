package gguf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type profileTensor struct {
	name  string
	dims  []uint64
	ttype uint32
}

// profileFile encodes a whole GGUF v3 file: metadata, tensor table and a data
// section (at the header end rounded up to 32 bytes) in which each tensor
// occupies its type's layout size, or spans[name] bytes when given. Shapes
// too large to materialize get no data section; the reader rejects them first.
func profileFile(kvs []fixtureKV, tensors []profileTensor, spans map[string]uint64) []byte {
	sizes := make([]uint64, len(tensors))
	var total uint64
	for i, tn := range tensors {
		elems := uint64(1)
		for _, d := range tn.dims {
			elems *= d
		}
		if span, ok := spans[tn.name]; ok {
			sizes[i] = span
		} else if size, ok := ggmlTypeSize[tn.ttype]; ok {
			sizes[i] = (elems + size[1] - 1) / size[1] * size[0]
		}
		total += sizes[i]
	}
	buf := new(bytes.Buffer)
	buf.WriteString("GGUF")
	binary.Write(buf, binary.LittleEndian, uint32(3))
	binary.Write(buf, binary.LittleEndian, uint64(len(tensors)))
	binary.Write(buf, binary.LittleEndian, uint64(len(kvs)))
	for _, kv := range kvs {
		binary.Write(buf, binary.LittleEndian, uint64(len(kv.key)))
		buf.WriteString(kv.key)
		binary.Write(buf, binary.LittleEndian, kv.vtype)
		kv.write(buf)
	}
	var offset uint64
	for i, tn := range tensors {
		binary.Write(buf, binary.LittleEndian, uint64(len(tn.name)))
		buf.WriteString(tn.name)
		binary.Write(buf, binary.LittleEndian, uint32(len(tn.dims)))
		for _, d := range tn.dims {
			binary.Write(buf, binary.LittleEndian, d)
		}
		binary.Write(buf, binary.LittleEndian, tn.ttype)
		binary.Write(buf, binary.LittleEndian, offset)
		offset += sizes[i]
	}
	if len(tensors) == 0 || total > 1<<26 {
		return buf.Bytes()
	}
	pad := (32 - buf.Len()%32) % 32
	buf.Write(make([]byte, pad+int(total)))
	return buf.Bytes()
}

func profileHeader(kvs []fixtureKV, tensors []profileTensor) []byte {
	return profileFile(kvs, tensors, nil)
}

func writeProfileFixture(t *testing.T, path string, kvs []fixtureKV, tensors []profileTensor) {
	t.Helper()
	writeProfileFile(t, path, profileFile(kvs, tensors, nil))
}

func writeProfileFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func readProfileBytes(data []byte) (QuantProfile, error) {
	return ReadQuantProfile(bytes.NewReader(data), int64(len(data)))
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// The block sizes are ggml's; their bits per weight are the published values
// (Q4_K 4.5, Q6_K 6.5625, IQ2_XXS 2.0625, ...). A wrong entry would silently
// misstate every file that uses the type.
func TestGGMLTypeSizesGiveKnownBitsPerWeight(t *testing.T) {
	want := map[uint32]float64{
		0: 32, 1: 16, 30: 16, 8: 8.5, 2: 4.5, 12: 4.5, 13: 5.5, 14: 6.5625, 11: 3.4375, 10: 2.625,
		16: 2.0625, 17: 2.3125, 22: 2.5625, 18: 3.0625, 21: 3.4375, 23: 4.25, 20: 4.5,
		19: 1.5625, 29: 1.75, 39: 4.25,
	}
	for id, bpw := range want {
		size, ok := ggmlTypeSize[id]
		if !ok {
			t.Fatalf("type %d missing", id)
		}
		if got := float64(size[1]) * 8 / float64(size[0]); !near(got, bpw) {
			t.Errorf("type %d: %.4f bpw, want %.4f", id, got, bpw)
		}
	}
}

func TestQuantProfileMixedDenseFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dense-Q4_K_M.gguf")
	writeProfileFixture(t, path, []fixtureKV{kvString("general.architecture", "llama"), kvU32("general.file_type", 15)}, []profileTensor{
		{"blk.0.attn_q.weight", []uint64{4096, 4096}, 12},   // Q4_K
		{"blk.0.ffn_down.weight", []uint64{4096, 1024}, 14}, // Q6_K
		{"blk.0.attn_norm.weight", []uint64{4096}, 0},       // F32
	})
	p, err := ReadQuantProfileModel(path)
	if err != nil {
		t.Fatal(err)
	}
	wantParams := int64(4096*4096 + 4096*1024 + 4096)
	wantBytes := int64(4096*4096/256*144 + 4096*1024/256*210 + 4096*4)
	if p.Params != wantParams || p.Bytes != wantBytes {
		t.Fatalf("params %d bytes %d, want %d %d", p.Params, p.Bytes, wantParams, wantBytes)
	}
	if !near(p.BPW(), float64(wantBytes)*8/float64(wantParams)) || p.BPW() < 4.9 || p.BPW() > 4.95 {
		t.Fatalf("bpw %.4f", p.BPW())
	}
	if p.ExpertParams != 0 || p.ExpertBPW() != 0 || !near(p.NonExpertBPW(), p.BPW()) {
		t.Fatalf("dense file reported experts: %+v", p)
	}
	if p.Imatrix != "" || p.FileType != 15 || p.Shards != 1 || p.Version != QuantProfileVersion {
		t.Fatalf("metadata: %+v", p)
	}
}

func TestQuantProfileSplitsRoutedExpertsAndInfersImatrixFromTypes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "moe-IQ2_XXS.gguf")
	writeProfileFixture(t, path, []fixtureKV{kvString("general.architecture", "qwen3moe")}, []profileTensor{
		{"token_embd.weight", []uint64{512, 1000}, 8},             // Q8_0
		{"blk.0.attn_q.weight", []uint64{512, 512}, 8},            // Q8_0
		{"blk.0.ffn_gate_shexp.weight", []uint64{512, 256}, 8},    // shared: non-expert
		{"blk.0.ffn_gate_exps.weight", []uint64{512, 256, 8}, 16}, // IQ2_XXS
		{"blk.0.ffn_down_exps.weight", []uint64{256, 512, 8}, 16}, // IQ2_XXS
		{"blk.0.ffn_gate_inp.weight", []uint64{512, 8}, 0},        // router F32
	})
	p, err := ReadQuantProfileModel(path)
	if err != nil {
		t.Fatal(err)
	}
	if want := int64(2 * 512 * 256 * 8); p.ExpertParams != want {
		t.Fatalf("expert params %d, want %d", p.ExpertParams, want)
	}
	if !near(p.ExpertBPW(), 2.0625) {
		t.Fatalf("expert bpw %.4f, want 2.0625", p.ExpertBPW())
	}
	nonExpertParams := int64(512*1000 + 512*512 + 512*256 + 512*8)
	nonExpertBytes := int64((512*1000+512*512+512*256)/32*34 + 512*8*4)
	if !near(p.NonExpertBPW(), float64(nonExpertBytes)*8/float64(nonExpertParams)) {
		t.Fatalf("non-expert bpw %.4f", p.NonExpertBPW())
	}
	if p.BPW() <= p.ExpertBPW() || p.BPW() >= p.NonExpertBPW() {
		t.Fatalf("whole-file bpw %.4f outside its parts", p.BPW())
	}
	if p.Imatrix != ImatrixTypes {
		t.Fatalf("IQ2_XXS experts need an imatrix; got %q", p.Imatrix)
	}
}

func TestQuantProfileImatrixFromKeysAndNotFromEmbeddings(t *testing.T) {
	dir := t.TempDir()
	keys := filepath.Join(dir, "keys.gguf")
	writeProfileFixture(t, keys, []fixtureKV{kvString("quantize.imatrix.file", "imatrix.dat"), kvU32("quantize.imatrix.entries_count", 7)},
		[]profileTensor{{"blk.0.attn_q.weight", []uint64{256, 256}, 12}})
	if p, err := ReadQuantProfileModel(keys); err != nil || p.Imatrix != ImatrixKeys {
		t.Fatalf("keys: %+v %v", p, err)
	}
	// llama.cpp exempts token embeddings and the output head from the rule.
	embd := filepath.Join(dir, "embd.gguf")
	writeProfileFixture(t, embd, nil, []profileTensor{
		{"token_embd.weight", []uint64{256, 256}, 16},
		{"output.weight", []uint64{256, 256}, 16},
		{"blk.0.attn_q.weight", []uint64{256, 256}, 10},
	})
	if p, err := ReadQuantProfileModel(embd); err != nil || p.Imatrix != "" {
		t.Fatalf("embeddings: %+v %v", p, err)
	}
}

// Type ids from 40 up mean different layouts in different backends (upstream
// Q2_0 is a fork's F8_E4M3_B128), so a tensor is sized by its span, not by id.
func TestQuantProfileSizesUnknownTypesByOffsets(t *testing.T) {
	data := profileFile(nil, []profileTensor{
		{"blk.0.attn_q.weight", []uint64{256, 256}, 8},          // Q8_0
		{"blk.0.ffn_up_exps.weight", []uint64{256, 256, 4}, 42}, // 2.25 bpw on disk
		{"blk.0.ffn_down_exps.weight", []uint64{256, 256, 4}, 99},
	}, map[string]uint64{"blk.0.ffn_up_exps.weight": 256 * 256 * 4 / 64 * 18, "blk.0.ffn_down_exps.weight": 256 * 256 * 4 / 64 * 18})
	p, err := readProfileBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	if !near(p.ExpertBPW(), 2.25) || !near(p.NonExpertBPW(), 8.5) {
		t.Fatalf("expert %.4f rest %.4f", p.ExpertBPW(), p.NonExpertBPW())
	}
}

// Offsets and the type table agree for known types (no padding here), and a
// known type in a smaller span than its layout gives no profile.
func TestQuantProfileOffsetsCrossCheckTheTypeTable(t *testing.T) {
	tensors := []profileTensor{
		{"blk.0.attn_q.weight", []uint64{256, 256}, 12},
		{"blk.0.ffn_down.weight", []uint64{256, 256}, 14},
	}
	p, err := readProfileBytes(profileFile(nil, tensors, nil))
	if err != nil || p.Bytes != 256*256/256*(144+210) {
		t.Fatalf("%+v %v", p, err)
	}
	// Padding after a tensor counts as its bytes: that is what a loader maps.
	p, err = readProfileBytes(profileFile(nil, tensors, map[string]uint64{"blk.0.attn_q.weight": 256*144 + 32}))
	if err != nil || p.Bytes != 256*(144+210)+32 {
		t.Fatalf("padded %+v %v", p, err)
	}
	_, err = readProfileBytes(profileFile(nil, tensors, map[string]uint64{"blk.0.attn_q.weight": 256*144 - 1}))
	if !errors.Is(err, ErrTensorLayout) {
		t.Fatalf("short span: %v", err)
	}
}

// An interrupted download gives no profile: its offsets would read as a tiny
// width.
func TestQuantProfileTruncatedFileGivesNoProfile(t *testing.T) {
	data := profileFile(nil, []profileTensor{
		{"blk.0.attn_q.weight", []uint64{256, 256}, 99},
		{"blk.0.ffn_down.weight", []uint64{256, 256}, 99},
	}, map[string]uint64{"blk.0.attn_q.weight": 40000, "blk.0.ffn_down.weight": 40000})
	if _, err := readProfileBytes(data); err != nil {
		t.Fatal(err)
	}
	cut := data[:len(data)-40000-100] // the second tensor's data is missing
	if _, err := readProfileBytes(cut); !errors.Is(err, ErrTensorDataIncomplete) {
		t.Fatalf("truncated: %v", err)
	}
	path := filepath.Join(t.TempDir(), "partial.gguf")
	writeProfileFile(t, path, cut)
	if _, err := ReadQuantProfileModel(path); err == nil {
		t.Fatal("a partial file got a profile")
	}
}

func TestQuantProfileMissingShardGivesNoProfile(t *testing.T) {
	dir := t.TempDir()
	first := filepath.Join(dir, "m-00001-of-00003.gguf")
	writeProfileFixture(t, first, []fixtureKV{kvString("quantize.imatrix.file", "x")}, nil)
	writeProfileFixture(t, filepath.Join(dir, "m-00003-of-00003.gguf"), nil, []profileTensor{{"blk.0.attn_q.weight", []uint64{256, 256}, 12}})
	if p, err := ReadQuantProfileModel(first); err == nil {
		t.Fatalf("missing shard 2 got a profile: %+v", p)
	}
	if p, err := OpenHeaderCache(filepath.Join(dir, "cache")).QuantProfile(first); err == nil {
		t.Fatalf("missing shard 2 got a cached profile: %+v", p)
	}
}

func TestQuantProfileSumsShards(t *testing.T) {
	dir := t.TempDir()
	first := filepath.Join(dir, "m-00001-of-00002.gguf")
	writeProfileFixture(t, first, []fixtureKV{kvString("quantize.imatrix.dataset", "chat")}, []profileTensor{{"blk.0.attn_q.weight", []uint64{256, 256}, 12}})
	writeProfileFixture(t, filepath.Join(dir, "m-00002-of-00002.gguf"), nil, []profileTensor{{"blk.1.ffn_up_exps.weight", []uint64{256, 256, 4}, 10}})
	p, err := ReadQuantProfileModel(first)
	if err != nil {
		t.Fatal(err)
	}
	if p.Shards != 2 || p.Params != 256*256*5 || p.ExpertParams != 256*256*4 || p.Imatrix != ImatrixKeys {
		t.Fatalf("merged: %+v", p)
	}
	if !near(p.ExpertBPW(), 2.625) || !near(p.NonExpertBPW(), 4.5) {
		t.Fatalf("bpw %.4f / %.4f", p.ExpertBPW(), p.NonExpertBPW())
	}
	if _, err := ReadQuantProfileModel(filepath.Join(dir, "m-00002-of-00002.gguf")); err == nil {
		t.Fatal("a non-first shard is not a model entrypoint")
	}
}

func TestReadQuantProfileRejectsCorruptHeadersWithoutPanicking(t *testing.T) {
	good := profileHeader([]fixtureKV{kvString("general.architecture", "llama")},
		[]profileTensor{{"blk.0.attn_q.weight", []uint64{256, 256}, 12}})
	if _, err := readProfileBytes(good); err != nil {
		t.Fatalf("good header: %v", err)
	}
	for n := 0; n < len(good); n++ {
		if _, err := readProfileBytes(good[:n]); err == nil {
			t.Fatalf("truncated at %d accepted", n)
		}
	}
	tooManyDims := profileHeader(nil, []profileTensor{{"t", []uint64{1, 1, 1, 1, 1, 1, 1, 1, 1}, 0}})
	if _, err := readProfileBytes(tooManyDims); err == nil {
		t.Fatal("9 dimensions accepted")
	}
	huge := profileHeader(nil, []profileTensor{{"t", []uint64{1 << 30, 1 << 30, 1 << 30}, 0}})
	if _, err := readProfileBytes(huge); err == nil || !strings.Contains(err.Error(), "larger") {
		t.Fatalf("overflowing shape: %v", err)
	}
}

func TestHeaderCacheStoresQuantProfilePerFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	writeProfileFixture(t, path, nil, []profileTensor{{"blk.0.attn_q.weight", []uint64{256, 256}, 23}})
	cacheDir := filepath.Join(dir, "cache")
	c := OpenHeaderCache(cacheDir)
	p, err := c.QuantProfile(path)
	if err != nil || !near(p.BPW(), 4.25) {
		t.Fatalf("profile %+v %v", p, err)
	}
	if err := c.Save(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(cacheDir, headerCacheFile))
	if err != nil || !bytes.Contains(data, []byte(quantProfileCacheName)) {
		t.Fatalf("cache file lacks %s: %v", quantProfileCacheName, err)
	}
	again, err := OpenHeaderCache(cacheDir).QuantProfile(path)
	if err != nil || again != p {
		t.Fatalf("cached %+v %v, want %+v", again, err, p)
	}
}

// Shard 00001 of a real split model is often metadata only, and it alone
// carries quantize.imatrix.* and general.file_type.
func TestQuantProfileMetadataOnlyFirstShard(t *testing.T) {
	dir := t.TempDir()
	first := filepath.Join(dir, "m-00001-of-00003.gguf")
	writeProfileFixture(t, first, []fixtureKV{kvString("quantize.imatrix.file", "imatrix.gguf"), kvU32("general.file_type", 12)}, nil)
	writeProfileFixture(t, filepath.Join(dir, "m-00002-of-00003.gguf"), nil, []profileTensor{{"blk.0.attn_q.weight", []uint64{256, 256}, 12}})
	writeProfileFixture(t, filepath.Join(dir, "m-00003-of-00003.gguf"), nil, []profileTensor{{"blk.1.ffn_up_exps.weight", []uint64{256, 256, 4}, 11}})
	wantBPW := float64(256*256/256*144+256*256*4/256*110) * 8 / float64(256*256*5)
	check := func(name string, p QuantProfile, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if p.Shards != 3 || p.Imatrix != ImatrixKeys || p.FileType != 12 || !near(p.BPW(), wantBPW) || !near(p.ExpertBPW(), 3.4375) {
			t.Fatalf("%s: %+v (bpw %.4f, want %.4f)", name, p, p.BPW(), wantBPW)
		}
	}
	p, err := ReadQuantProfileModel(first)
	check("direct", p, err)
	cacheDir := filepath.Join(dir, "cache")
	c := OpenHeaderCache(cacheDir)
	p, err = c.QuantProfile(first)
	check("cache", p, err)
	if err := c.Save(); err != nil {
		t.Fatal(err)
	}
	p, err = OpenHeaderCache(cacheDir).QuantProfile(first)
	check("stored", p, err)

	// A single file without tensors has no profile.
	empty := filepath.Join(dir, "empty.gguf")
	writeProfileFixture(t, empty, []fixtureKV{kvU32("general.file_type", 12)}, nil)
	if _, err := ReadQuantProfileModel(empty); err == nil {
		t.Fatal("a tensor-less model got a profile")
	}
}
