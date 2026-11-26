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

// profileHeader encodes a GGUF v3 header (metadata and tensor table, no data),
// which is all ReadQuantProfile reads.
func profileHeader(kvs []fixtureKV, tensors []profileTensor) []byte {
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
	for _, tn := range tensors {
		binary.Write(buf, binary.LittleEndian, uint64(len(tn.name)))
		buf.WriteString(tn.name)
		binary.Write(buf, binary.LittleEndian, uint32(len(tn.dims)))
		for _, d := range tn.dims {
			binary.Write(buf, binary.LittleEndian, d)
		}
		binary.Write(buf, binary.LittleEndian, tn.ttype)
		binary.Write(buf, binary.LittleEndian, uint64(0))
	}
	return buf.Bytes()
}

func writeProfileFixture(t *testing.T, path string, kvs []fixtureKV, tensors []profileTensor) {
	t.Helper()
	if err := os.WriteFile(path, profileHeader(kvs, tensors), 0o644); err != nil {
		t.Fatal(err)
	}
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

func TestQuantProfileUnknownTypeGivesNoProfile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.gguf")
	writeProfileFixture(t, path, nil, []profileTensor{{"blk.0.attn_q.weight", []uint64{256, 256}, 99}})
	if _, err := ReadQuantProfileModel(path); !errors.Is(err, ErrUnknownTensorType) {
		t.Fatalf("err = %v", err)
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
	if _, err := ReadQuantProfile(bytes.NewReader(good)); err != nil {
		t.Fatalf("good header: %v", err)
	}
	for n := 0; n < len(good); n++ {
		if _, err := ReadQuantProfile(bytes.NewReader(good[:n])); err == nil {
			t.Fatalf("truncated at %d accepted", n)
		}
	}
	tooManyDims := profileHeader(nil, []profileTensor{{"t", []uint64{1, 1, 1, 1, 1, 1, 1, 1, 1}, 0}})
	if _, err := ReadQuantProfile(bytes.NewReader(tooManyDims)); err == nil {
		t.Fatal("9 dimensions accepted")
	}
	huge := profileHeader(nil, []profileTensor{{"t", []uint64{1 << 30, 1 << 30, 1 << 30}, 0}})
	if _, err := ReadQuantProfile(bytes.NewReader(huge)); err == nil || !strings.Contains(err.Error(), "larger") {
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
