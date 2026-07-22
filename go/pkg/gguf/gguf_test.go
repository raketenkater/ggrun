package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// ---------------------------------------------------------------------------
// Helpers: build minimal valid GGUF binary blobs in memory
// ---------------------------------------------------------------------------

func writeU32(buf *bytes.Buffer, v uint32) {
	binary.Write(buf, binary.LittleEndian, v)
}

func writeU64(buf *bytes.Buffer, v uint64) {
	binary.Write(buf, binary.LittleEndian, v)
}

func writeString(buf *bytes.Buffer, s string) {
	writeU64(buf, uint64(len(s)))
	buf.WriteString(s)
}

// buildMinimalGGUF creates a valid GGUF v3 file with the given metadata and tensors.
func buildMinimalGGUF(metadata map[string]interface{}, tensors []tensorInfo) []byte {
	var buf bytes.Buffer

	// Magic + version
	writeU32(&buf, ggufMagic)
	writeU32(&buf, ggufVersion3)

	// Tensor count + KV count
	writeU64(&buf, uint64(len(tensors)))
	writeU64(&buf, uint64(len(metadata)))

	// KV pairs (simplified: only string and uint32 values for testing)
	for key, val := range metadata {
		writeString(&buf, key)
		switch v := val.(type) {
		case string:
			writeU32(&buf, ggufTypeString)
			writeString(&buf, v)
		case uint32:
			writeU32(&buf, ggufTypeUint32)
			writeU32(&buf, v)
		case uint64:
			writeU32(&buf, ggufTypeUint64)
			writeU64(&buf, v)
		}
	}

	// Tensor info records
	for _, t := range tensors {
		writeString(&buf, t.name)
		writeU32(&buf, t.nDims)
		for d := uint32(0); d < t.nDims; d++ {
			writeU64(&buf, t.dims[d])
		}
		writeU32(&buf, t.typeID)
		writeU64(&buf, t.offset)
	}

	// Pad to alignment (32 bytes default)
	pos := buf.Len()
	aligned := (pos + 31) / 32 * 32
	for i := pos; i < aligned; i++ {
		buf.WriteByte(0)
	}

	// Fake tensor data (just zeros to fill file size)
	buf.Write(make([]byte, 1024))

	return buf.Bytes()
}

func writeTempGGUF(t *testing.T, data []byte) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "test-model.gguf")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("write temp gguf: %v", err)
	}
	return path
}

// ---------------------------------------------------------------------------
// Parse — pure-Go parser (no Python dependency)
// ---------------------------------------------------------------------------

func TestParse_MinimalValidGGUF(t *testing.T) {
	metadata := map[string]interface{}{
		"general.architecture":       "llama",
		"general.name":               "TestModel",
		"llama.block_count":          uint32(32),
		"llama.context_length":       uint32(4096),
		"llama.embedding_length":     uint32(4096),
		"llama.feed_forward_length":  uint32(11008),
		"llama.attention.head_count_kv": uint32(8),
	}
	tensors := []tensorInfo{
		{name: "token_embd.weight", nDims: 2, dims: [4]uint64{4096, 32000, 0, 0}, typeID: 2, offset: 0},
		{name: "blk.0.attn_q.weight", nDims: 2, dims: [4]uint64{4096, 4096, 0, 0}, typeID: 2, offset: 512},
	}

	data := buildMinimalGGUF(metadata, tensors)
	path := writeTempGGUF(t, data)

	info, err := Parse(path)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if info.Architecture != "llama" {
		t.Errorf("expected arch=llama, got %q", info.Architecture)
	}
	if info.Name != "TestModel" {
		t.Errorf("expected name=TestModel, got %q", info.Name)
	}
	if info.BlockCount != 32 {
		t.Errorf("expected BlockCount=32, got %d", info.BlockCount)
	}
	if info.ContextLength != 4096 {
		t.Errorf("expected ContextLength=4096, got %d", info.ContextLength)
	}
	if info.EmbeddingLength != 4096 {
		t.Errorf("expected EmbeddingLength=4096, got %d", info.EmbeddingLength)
	}
	if info.IsMoE {
		t.Errorf("expected IsMoE=false for non-MoE model")
	}
}

func TestParse_MoEDetection(t *testing.T) {
	metadata := map[string]interface{}{
		"general.architecture":  "qwen2moe",
		"qwen2moe.block_count":  uint32(60),
		"qwen2moe.expert_count": uint32(64),
		"qwen2moe.expert_used_count": uint32(4),
	}
	data := buildMinimalGGUF(metadata, nil)
	path := writeTempGGUF(t, data)

	info, err := Parse(path)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	if !info.IsMoE {
		t.Errorf("expected IsMoE=true when expert_count > 0")
	}
	if info.Experts != 64 {
		t.Errorf("expected Experts=64, got %d", info.Experts)
	}
	if info.ExpertUsed != 4 {
		t.Errorf("expected ExpertUsed=4, got %d", info.ExpertUsed)
	}
}

func TestParse_EmptyPath(t *testing.T) {
	_, err := Parse("")
	if err == nil {
		t.Errorf("expected error for empty path")
	}
}

func TestParse_NonexistentFile(t *testing.T) {
	_, err := Parse("/nonexistent/model.gguf")
	if err == nil {
		t.Errorf("expected error for nonexistent file")
	}
}

func TestParse_BadMagic(t *testing.T) {
	data := []byte{0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00}
	path := writeTempGGUF(t, data)
	_, err := Parse(path)
	if err == nil {
		t.Errorf("expected error for bad magic")
	}
}

// ---------------------------------------------------------------------------
// dataSectionOff — tracked for tensor span math
// ---------------------------------------------------------------------------

func TestDataSectionOff_Tracked(t *testing.T) {
	metadata := map[string]interface{}{
		"general.architecture": "llama",
	}
	tensors := []tensorInfo{
		{name: "token_embd.weight", nDims: 2, dims: [4]uint64{4096, 32000, 0, 0}, typeID: 2, offset: 0},
	}
	data := buildMinimalGGUF(metadata, tensors)
	path := writeTempGGUF(t, data)

	// Parse the shard directly to inspect dataSectionOff
	shard, err := parseShard(path)
	if err != nil {
		t.Fatalf("parseShard failed: %v", err)
	}

	// dataSectionOff must be > 0 (past the header) and aligned to 32
	if shard.dataSectionOff <= 0 {
		t.Errorf("expected dataSectionOff > 0, got %d", shard.dataSectionOff)
	}
	if shard.dataSectionOff%32 != 0 {
		t.Errorf("expected dataSectionOff aligned to 32, got %d", shard.dataSectionOff)
	}
}

// ---------------------------------------------------------------------------
// categorizeTensorSpan — routes tensors to correct byte categories
// ---------------------------------------------------------------------------

func TestCategorizeTensorSpan_TokenEmbd(t *testing.T) {
	info := &Info{}
	categorizeTensorSpan("token_embd.weight", 1024, info)

	if info.TokenEmbdBytes != 1024 {
		t.Errorf("expected TokenEmbdBytes=1024, got %d", info.TokenEmbdBytes)
	}
	if info.NonExpertBytes != 1024 {
		t.Errorf("expected NonExpertBytes=1024 (token_embd is non-expert), got %d", info.NonExpertBytes)
	}
}

func TestCategorizeTensorSpan_Output(t *testing.T) {
	info := &Info{}
	categorizeTensorSpan("output.weight", 2048, info)

	if info.OutputBytes != 2048 {
		t.Errorf("expected OutputBytes=2048, got %d", info.OutputBytes)
	}
	if info.NonExpertBytes != 2048 {
		t.Errorf("expected NonExpertBytes=2048, got %d", info.NonExpertBytes)
	}
}

func TestCategorizeTensorSpan_Shexp(t *testing.T) {
	info := &Info{}
	categorizeTensorSpan("blk.0.ffn_gate_shexp.weight", 512, info)

	if info.ShexpBytes != 512 {
		t.Errorf("expected ShexpBytes=512, got %d", info.ShexpBytes)
	}
	if info.NonExpertBytes != 512 {
		t.Errorf("expected NonExpertBytes=512 (shexp is non-expert), got %d", info.NonExpertBytes)
	}
}

func TestCategorizeTensorSpan_Expert(t *testing.T) {
	info := &Info{}
	categorizeTensorSpan("blk.0.ffn_up_exps.weight", 4096, info)

	if info.ExpertBytes != 4096 {
		t.Errorf("expected ExpertBytes=4096, got %d", info.ExpertBytes)
	}
	// Expert bytes should NOT be in NonExpertBytes
	if info.NonExpertBytes != 0 {
		t.Errorf("expected NonExpertBytes=0 for expert tensor, got %d", info.NonExpertBytes)
	}
}

func TestCategorizeTensorSpan_FusedExpert(t *testing.T) {
	info := &Info{}
	categorizeTensorSpan("blk.5.ffn_gate_up_exps.weight", 8192, info)

	if info.ExpertBytes != 8192 {
		t.Errorf("expected ExpertBytes=8192 for fused expert, got %d", info.ExpertBytes)
	}
}

func TestCategorizeTensorSpan_UnknownFallsToNonExpert(t *testing.T) {
	info := &Info{}
	categorizeTensorSpan("blk.0.attn_q.weight", 1024, info)

	if info.NonExpertBytes != 1024 {
		t.Errorf("expected NonExpertBytes=1024 for unknown tensor, got %d", info.NonExpertBytes)
	}
	if info.ExpertBytes != 0 {
		t.Errorf("expected ExpertBytes=0 for non-expert tensor, got %d", info.ExpertBytes)
	}
}

// ---------------------------------------------------------------------------
// calculateSpans — uses dataSectionOff for byte accounting
// ---------------------------------------------------------------------------

func TestCalculateSpans_UsesDataSectionOff(t *testing.T) {
	shards := []parsedShard{
		{
			path:           "test.gguf",
			size:           2048, // total file size
			dataSectionOff: 512,  // data starts at offset 512
			tensors: []tensorInfo{
				{name: "token_embd.weight", nDims: 2, dims: [4]uint64{4096, 100, 0, 0}, typeID: 2, offset: 0},
				{name: "blk.0.ffn_up_exps.weight", nDims: 2, dims: [4]uint64{4096, 100, 0, 0}, typeID: 2, offset: 512},
			},
		},
	}
	info := &Info{}
	calculateSpans(shards, 32, info)

	// First tensor: span = next.offset - this.offset = 512 - 0 = 512
	if info.TokenEmbdBytes != 512 {
		t.Errorf("expected TokenEmbdBytes=512, got %d", info.TokenEmbdBytes)
	}

	// Second (last) tensor: span = fileSize - (dataSectionOff + offset) = 2048 - (512 + 512) = 1024
	if info.ExpertBytes != 1024 {
		t.Errorf("expected ExpertBytes=1024, got %d", info.ExpertBytes)
	}
}

// ---------------------------------------------------------------------------
// discardN — handles large skips
// ---------------------------------------------------------------------------

func TestDiscardN_SmallSkip(t *testing.T) {
	data := make([]byte, 100)
	r := bytes.NewReader(data)
	err := discardN(r, 50)
	if err != nil {
		t.Fatalf("discardN(50) failed: %v", err)
	}
	if r.Len() != 50 {
		t.Errorf("expected 50 bytes remaining, got %d", r.Len())
	}
}

func TestDiscardN_ZeroSkip(t *testing.T) {
	data := make([]byte, 100)
	r := bytes.NewReader(data)
	err := discardN(r, 0)
	if err != nil {
		t.Fatalf("discardN(0) failed: %v", err)
	}
	if r.Len() != 100 {
		t.Errorf("expected 100 bytes remaining, got %d", r.Len())
	}
}

func TestDiscardN_TooLarge(t *testing.T) {
	data := make([]byte, 100)
	r := bytes.NewReader(data)
	err := discardN(r, 1<<63) // exceeds 1<<62 guard
	if err == nil {
		t.Errorf("expected error for oversized discardN")
	}
}

// ---------------------------------------------------------------------------
// EstimateParams — from tensors vs metadata fallback
// ---------------------------------------------------------------------------

func TestEstimateParams_FromTensors(t *testing.T) {
	info := &Info{
		tensors: []tensorInfo{
			{name: "a", nDims: 2, dims: [4]uint64{100, 200, 0, 0}},  // 20000
			{name: "b", nDims: 3, dims: [4]uint64{10, 20, 30, 0}},   // 6000
			{name: "c", nDims: 1, dims: [4]uint64{500, 0, 0, 0}},    // 500
		},
	}

	params := info.EstimateParams()
	expected := int64(20000 + 6000 + 500)
	if params != expected {
		t.Errorf("expected EstimateParams=%d, got %d", expected, params)
	}
}

func TestEstimateParams_FromMetadataFallback(t *testing.T) {
	info := &Info{
		BlockCount:        32,
		EmbeddingLength:   4096,
		FeedForwardLength: 11008,
		VocabSize:         32000,
		IsMoE:             false,
		// No tensors → falls back to metadata estimate
	}

	params := info.EstimateParams()
	if params <= 0 {
		t.Errorf("expected positive parameter count from metadata, got %d", params)
	}

	// Sanity: 32 layers * (4*4096*4096 + 3*4096*11008) + 2*32000*4096
	// = 32 * (67108864 + 135266304) + 262144000
	// = 32 * 202375168 + 262144000
	// = 6476005376 + 262144000 = 6738149376
	expected := int64(32*(4*4096*4096+3*4096*11008) + 2*32000*4096)
	if params != expected {
		t.Errorf("expected EstimateParams=%d, got %d", expected, params)
	}
}

func TestEstimateParams_ZeroDims(t *testing.T) {
	info := &Info{
		tensors: []tensorInfo{
			{name: "a", nDims: 2, dims: [4]uint64{0, 200, 0, 0}}, // zero dim → 0 params
		},
	}
	if info.EstimateParams() != 0 {
		t.Errorf("expected 0 params for zero-dim tensor")
	}
}

// ---------------------------------------------------------------------------
// detectShards — split-shard filename parsing
// ---------------------------------------------------------------------------

func TestDetectShards_SplitFile(t *testing.T) {
	ref, ok := detectShards("/models/model-00003-of-00010.gguf")
	if !ok {
		t.Fatal("expected split shard detection")
	}
	if ref.index != 3 {
		t.Errorf("expected index=3, got %d", ref.index)
	}
	if ref.total != 10 {
		t.Errorf("expected total=10, got %d", ref.total)
	}
}

func TestDetectShards_SingleFile(t *testing.T) {
	_, ok := detectShards("/models/model.gguf")
	if ok {
		t.Errorf("expected no split detection for single file")
	}
}

// ---------------------------------------------------------------------------
// ggufScalar — type-safe accessors
// ---------------------------------------------------------------------------

func TestGgufScalar_Accessors(t *testing.T) {
	s := ggufScalar{k: scalarUint, u: 42}
	if v, ok := s.asUint(); !ok || v != 42 {
		t.Errorf("asUint: expected (42, true), got (%d, %v)", v, ok)
	}
	if _, ok := s.asString(); ok {
		t.Errorf("asString on uint scalar should return false")
	}

	str := ggufScalar{k: scalarString, s: "hello"}
	if v, ok := str.asString(); !ok || v != "hello" {
		t.Errorf("asString: expected (hello, true), got (%q, %v)", v, ok)
	}

	b := ggufScalar{k: scalarBool, b: true}
	if v, ok := b.asBool(); !ok || !v {
		t.Errorf("asBool: expected (true, true), got (%v, %v)", v, ok)
	}

	arr := ggufScalar{k: scalarArrayLen, u: 32000}
	if v, ok := arr.asArrayLen(); !ok || v != 32000 {
		t.Errorf("asArrayLen: expected (32000, true), got (%d, %v)", v, ok)
	}
}