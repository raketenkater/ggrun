package gguf

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fixtureTensor struct {
	name   string
	elems  uint64
	ttype  uint32
	offset uint64 // relative to data section start
}

// writeGGUFFixture writes a minimal GGUF v3 file whose data section is
// dataSize bytes. Tensor byte sizes are therefore defined by offset deltas
// (the last tensor runs to end of file) — exactly what parse_gguf.py's
// span sizing must recover.
func writeGGUFFixture(t *testing.T, path string, tensors []fixtureTensor, align, dataSize int) {
	t.Helper()
	buf := new(bytes.Buffer)
	buf.WriteString("GGUF")
	binary.Write(buf, binary.LittleEndian, uint32(3))
	binary.Write(buf, binary.LittleEndian, uint64(len(tensors)))
	binary.Write(buf, binary.LittleEndian, uint64(2)) // kv count

	writeStr := func(s string) {
		binary.Write(buf, binary.LittleEndian, uint64(len(s)))
		buf.WriteString(s)
	}
	// general.architecture = "deepseek4" (string, type 8)
	writeStr("general.architecture")
	binary.Write(buf, binary.LittleEndian, uint32(8))
	writeStr("deepseek4")
	// general.alignment (uint32, type 4)
	writeStr("general.alignment")
	binary.Write(buf, binary.LittleEndian, uint32(4))
	binary.Write(buf, binary.LittleEndian, uint32(align))

	for _, tn := range tensors {
		writeStr(tn.name)
		binary.Write(buf, binary.LittleEndian, uint32(1)) // n_dims
		binary.Write(buf, binary.LittleEndian, tn.elems)
		binary.Write(buf, binary.LittleEndian, tn.ttype)
		binary.Write(buf, binary.LittleEndian, tn.offset)
	}
	headerEnd := buf.Len()
	dataStart := (headerEnd + align - 1) / align * align
	buf.Write(make([]byte, dataStart-headerEnd+dataSize))
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
}

// MXFP4 (ggml type 39) is 17 bytes per 32 elements, so 64 elems = 34 bytes by
// type-math. The fixture gives the first tensor a padded 96-byte span; span
// sizing must report 96, not 34. Under-sizing here once made MoE placement pin
// one expert layer too many and CUDA-OOM after a full model load.
func TestParseSizesTensorsByDiskSpan(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "span.gguf")
	writeGGUFFixture(t, path, []fixtureTensor{
		{name: "token_embd.weight", elems: 64, ttype: 39, offset: 0},
		{name: "blk.0.ffn_gate_exps.weight", elems: 64, ttype: 39, offset: 96},
		{name: "blk.0.ffn_gate_shexp.weight", elems: 64, ttype: 39, offset: 130},
		{name: "output.weight", elems: 64, ttype: 39, offset: 164},
	}, 32, 164+34)

	info, err := Parse(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if info.NonExpertBytes != 96+34 {
		t.Fatalf("expected non-expert span 130 (type-math would say 68), got %d", info.NonExpertBytes)
	}
	if info.ExpertBytes != 34+34 {
		t.Fatalf("expected expert spans 68, got %d", info.ExpertBytes)
	}
	if info.TokenEmbdBytes != 96 {
		t.Fatalf("expected token_embd bytes 96, got %d", info.TokenEmbdBytes)
	}
	if info.OutputBytes != 34 {
		t.Fatalf("expected output bytes 34, got %d", info.OutputBytes)
	}
	if info.ShexpBytes != 34 {
		t.Fatalf("expected shexp bytes 34, got %d", info.ShexpBytes)
	}
	if info.Architecture != "deepseek4" {
		t.Fatalf("expected arch deepseek4, got %q", info.Architecture)
	}
}

func TestParseCountsPerLayerTokenEmbdAsHostResident(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ple.gguf")
	writeGGUFFixture(t, path, []fixtureTensor{
		{name: "token_embd.weight", elems: 64, ttype: 39, offset: 0},
		{name: "per_layer_token_embd.weight", elems: 64, ttype: 39, offset: 96},
		{name: "blk.0.ffn_gate_exps.weight", elems: 64, ttype: 39, offset: 192},
		{name: "output.weight", elems: 64, ttype: 39, offset: 226},
	}, 32, 226+34)

	info, err := Parse(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if info.PerLayerTokenEmbdBytes != 96 {
		t.Fatalf("PLE bytes = %d, want 96", info.PerLayerTokenEmbdBytes)
	}
	if info.TokenEmbdBytes != 96+96 {
		t.Fatalf("host embeddings = %d, want token_embd+PLE 192", info.TokenEmbdBytes)
	}
}

func TestParseReportsPerLayerExpertTransferBytes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "layer-bytes.gguf")
	writeGGUFFixture(t, path, []fixtureTensor{
		{name: "token_embd.weight", elems: 64, ttype: 39, offset: 0},
		{name: "blk.0.ffn_gate_exps.weight", elems: 64, ttype: 39, offset: 64},
		{name: "blk.0.ffn_gate_shexp.weight", elems: 64, ttype: 39, offset: 128},
		{name: "blk.0.ffn_gate_inp.weight", elems: 64, ttype: 39, offset: 192},
		{name: "output.weight", elems: 64, ttype: 39, offset: 256},
	}, 32, 320)

	info, err := Parse(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(info.RoutedExpertLayerBytes) != 1 || info.RoutedExpertLayerBytes[0] != 64 {
		t.Fatalf("routed layer bytes = %v, want [64]", info.RoutedExpertLayerBytes)
	}
	if len(info.ShexpLayerBytes) != 1 || info.ShexpLayerBytes[0] != 64 {
		t.Fatalf("shared layer bytes = %v, want [64]", info.ShexpLayerBytes)
	}
	if len(info.ExpertAuxLayerBytes) != 1 || info.ExpertAuxLayerBytes[0] != 64 {
		t.Fatalf("routing layer bytes = %v, want [64]", info.ExpertAuxLayerBytes)
	}
	if info.ExpertAuxBytes != 64 {
		t.Fatalf("routing total = %d, want 64", info.ExpertAuxBytes)
	}
	if len(info.NonExpertLayerBytes) != 0 {
		t.Fatalf("router must not remain in base layer bytes: %v", info.NonExpertLayerBytes)
	}
}

// Split model where the first shard is metadata-only (0 tensors) — the
// DeepSeek-V4-Flash layout. Totals must come from the sibling shards.
func TestParseSplitShardsMetadataOnlyFirst(t *testing.T) {
	dir := t.TempDir()
	first := filepath.Join(dir, "m-00001-of-00002.gguf")
	second := filepath.Join(dir, "m-00002-of-00002.gguf")
	writeGGUFFixture(t, first, nil, 32, 0)
	writeGGUFFixture(t, second, []fixtureTensor{
		{name: "token_embd.weight", elems: 64, ttype: 39, offset: 0},
		{name: "blk.0.ffn_gate_exps.weight", elems: 64, ttype: 39, offset: 96},
	}, 32, 96+34)

	info, err := Parse(first)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if info.NonExpertBytes != 96 || info.ExpertBytes != 34 || info.TokenEmbdBytes != 96 {
		t.Fatalf("split totals wrong: non-expert=%d expert=%d token_embd=%d",
			info.NonExpertBytes, info.ExpertBytes, info.TokenEmbdBytes)
	}
}

func TestFindParseScriptPrefersAppHomeSourceCheckout(t *testing.T) {
	dir := t.TempDir()
	sourceParser := filepath.Join(dir, ".src", "llm-server", "tools", "gguf", "parse_gguf.py")
	installedParser := filepath.Join(dir, ".bin", "parse_gguf.py")
	if err := os.MkdirAll(filepath.Dir(sourceParser), 0755); err != nil {
		t.Fatalf("mkdir source parser dir: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(installedParser), 0755); err != nil {
		t.Fatalf("mkdir installed parser dir: %v", err)
	}
	if err := os.WriteFile(sourceParser, []byte("#!/usr/bin/env python3\nr['tensor_accounting_schema'] = 2\n"), 0755); err != nil {
		t.Fatalf("write source parser: %v", err)
	}
	if err := os.WriteFile(installedParser, []byte("#!/usr/bin/env python3\nr['tensor_accounting_schema'] = 2\n"), 0755); err != nil {
		t.Fatalf("write installed parser: %v", err)
	}

	t.Setenv("LLM_SCRIPT_DIR", "")
	t.Setenv("LLM_SERVER_HOME", "")
	t.Setenv("LLM_APP_HOME", dir)

	got := findParseScript()
	want, _ := filepath.Abs(sourceParser)
	if got != want {
		t.Fatalf("expected app-home source parser %s, got %s", want, got)
	}
}

func TestFindParseScriptSkipsStaleParserMissingHostPLE(t *testing.T) {
	dir := t.TempDir()
	stale := filepath.Join(dir, "stale", "parse_gguf.py")
	current := filepath.Join(dir, "current", "parse_gguf.py")
	if err := os.MkdirAll(filepath.Dir(stale), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(current), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stale, []byte("#!/usr/bin/env python3\nprint('old')\n"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(current, []byte("#!/usr/bin/env python3\nr['tensor_accounting_schema'] = 2\n"), 0755); err != nil {
		t.Fatal(err)
	}

	t.Setenv("LLM_SCRIPT_DIR", filepath.Dir(stale))
	t.Setenv("LLM_SERVER_HOME", "")
	t.Setenv("LLM_APP_HOME", filepath.Dir(current))

	got := findParseScript()
	want, _ := filepath.Abs(current)
	if got != want {
		t.Fatalf("stale PATH parser won: got %s, want %s", got, want)
	}
}

func TestValidateParserOutputRejectsQwen4ExpStaleTensorAccounting(t *testing.T) {
	info := &Info{
		Architecture:   "qwen4exp",
		TokenEmbdBytes: 675430400,
		NonExpertBytes: 33900597760,
	}
	err := validateParserOutput("/old/parse_gguf.py", "/models/Qwen3.8-Flash-Next.gguf", info)
	if err == nil {
		t.Fatal("expected stale parser output to fail")
	}
	if !strings.Contains(err.Error(), "tensor accounting schema 2") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateParserOutputAcceptsQwen4ExpWithoutPLE(t *testing.T) {
	info := &Info{
		TensorAccountingSchema: 2,
		Architecture:           "qwen4exp",
		TokenEmbdBytes:         675430400,
		NonExpertBytes:         5100000000,
	}
	if err := validateParserOutput("/current/parse_gguf.py", "/models/no-ple.gguf", info); err != nil {
		t.Fatalf("valid no-PLE qwen4exp was rejected: %v", err)
	}
}

func TestValidateParserOutputRejectsPresentButUnaccountedPLE(t *testing.T) {
	info := &Info{
		TensorAccountingSchema: 2,
		Architecture:           "qwen4exp",
		HasPerLayerTokenEmbd:   1,
	}
	err := validateParserOutput("/broken/parse_gguf.py", "/models/ple.gguf", info)
	if err == nil || !strings.Contains(err.Error(), "did not account") {
		t.Fatalf("inconsistent PLE accounting error = %v", err)
	}
}

func TestValidateParserOutputRejectsDeepSeek4MissingByteSplits(t *testing.T) {
	info := &Info{
		Architecture:   "deepseek4",
		HasShexp:       1,
		ExpertBytes:    100,
		NonExpertBytes: 10,
	}
	err := validateParserOutput("/old/parse_gguf.py", "/models/DeepSeek-V4.gguf", info)
	if err == nil {
		t.Fatal("expected stale parser output to fail")
	}
	if !strings.Contains(err.Error(), "missing required DeepSeek4 byte splits") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateParserOutputAcceptsDeepSeek4ByteSplits(t *testing.T) {
	info := &Info{
		Architecture:   "deepseek4",
		HasShexp:       1,
		ExpertBytes:    100,
		NonExpertBytes: 10,
		TokenEmbdBytes: 1,
		OutputBytes:    1,
		ShexpBytes:     1,
	}
	if err := validateParserOutput("/new/parse_gguf.py", "/models/DeepSeek-V4.gguf", info); err != nil {
		t.Fatalf("expected complete parser output to pass: %v", err)
	}
}

func TestParseMissingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.gguf")
	_, err := Parse(path)
	if err == nil {
		t.Fatal("expected missing model file to fail")
	}
	if !strings.Contains(err.Error(), "model file") {
		t.Fatalf("expected model-file error, got %v", err)
	}
}

func TestParse(t *testing.T) {
	// Exercise the parser against a real model if one is provided via
	// GGUF_TEST_MODEL=/path/to/model.gguf; otherwise the test skips.
	paths := []string{}
	if p := os.Getenv("GGUF_TEST_MODEL"); p != "" {
		paths = append(paths, p)
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			info, err := Parse(p)
			if err != nil {
				t.Fatalf("parse %s: %v", p, err)
			}
			if info.Architecture == "" {
				t.Skip("architecture empty in test model")
			}
			return
		}
	}
	t.Skip("no test model available")
}

func TestEstimateParams(t *testing.T) {
	info := &Info{
		VocabSize:         151936,
		EmbeddingLength:   1024,
		BlockCount:        28,
		FeedForwardLength: 3072,
	}
	got := info.EstimateParams()
	// Expected ~596M for Qwen3 0.6B
	if got < 500000000 || got > 700000000 {
		t.Fatalf("expected ~596M params, got %d", got)
	}
}

// writeChatTemplateFixture writes a minimal GGUF whose metadata section carries
// a tokenizer.chat_template string (type 8) plus a preceding architecture key.
func writeChatTemplateFixture(t *testing.T, path, arch, template string) {
	t.Helper()
	buf := new(bytes.Buffer)
	buf.WriteString("GGUF")
	binary.Write(buf, binary.LittleEndian, uint32(3))
	binary.Write(buf, binary.LittleEndian, uint64(0)) // tensor count
	binary.Write(buf, binary.LittleEndian, uint64(2)) // kv count

	writeStr := func(s string) {
		binary.Write(buf, binary.LittleEndian, uint64(len(s)))
		buf.WriteString(s)
	}
	writeStr("general.architecture")
	binary.Write(buf, binary.LittleEndian, uint32(8))
	writeStr(arch)

	writeStr("tokenizer.chat_template")
	binary.Write(buf, binary.LittleEndian, uint32(8))
	writeStr(template)

	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
}

func TestChatTemplateReadsEmbeddedTemplate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	writeChatTemplateFixture(t, path, "qwen35",
		"{%- if not messages %}{{- raise_exception('No messages provided.') }}{%- endif %}")
	if got := ChatTemplate(path); got == "" || !strings.Contains(got, "raise_exception") {
		t.Fatalf("ChatTemplate = %q, want embedded raise_exception template", got)
	}
}

func TestChatTemplateAbsentReturnsEmpty(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	writeChatTemplateFixture(t, path, "deepseek4", "")
	if got := ChatTemplate(path); got != "" {
		t.Fatalf("ChatTemplate with empty template = %q, want \"\"", got)
	}
	if got := ChatTemplate(filepath.Join(t.TempDir(), "missing.gguf")); got != "" {
		t.Fatalf("ChatTemplate of missing file = %q, want \"\"", got)
	}
	if got := ChatTemplate(""); got != "" {
		t.Fatalf("ChatTemplate of empty path = %q, want \"\"", got)
	}
}

// fixtureKV writes one GGUF metadata entry: key, value type, and payload.
type fixtureKV struct {
	key   string
	vtype uint32
	write func(*bytes.Buffer)
}

func kvU32(key string, v uint32) fixtureKV {
	return fixtureKV{key, 4, func(b *bytes.Buffer) { binary.Write(b, binary.LittleEndian, v) }}
}

func kvString(key, v string) fixtureKV {
	return fixtureKV{key, 8, func(b *bytes.Buffer) {
		binary.Write(b, binary.LittleEndian, uint64(len(v)))
		b.WriteString(v)
	}}
}

// kvArray encodes values with the given GGUF element type (0..12).
func kvArray(key string, elemType uint32, values []float64) fixtureKV {
	return fixtureKV{key, 9, func(b *bytes.Buffer) {
		binary.Write(b, binary.LittleEndian, elemType)
		binary.Write(b, binary.LittleEndian, uint64(len(values)))
		for _, v := range values {
			switch elemType {
			case 0, 7:
				b.WriteByte(uint8(v))
			case 1:
				binary.Write(b, binary.LittleEndian, int8(v))
			case 2:
				binary.Write(b, binary.LittleEndian, uint16(v))
			case 3:
				binary.Write(b, binary.LittleEndian, int16(v))
			case 4:
				binary.Write(b, binary.LittleEndian, uint32(v))
			case 5:
				binary.Write(b, binary.LittleEndian, int32(v))
			case 6:
				binary.Write(b, binary.LittleEndian, float32(v))
			case 10:
				binary.Write(b, binary.LittleEndian, uint64(v))
			case 11:
				binary.Write(b, binary.LittleEndian, int64(v))
			}
		}
	}}
}

// writeGGUFMetadataFixture writes a tensor-less GGUF v3 header carrying only
// the given metadata, which is all the attention-geometry parser reads.
func writeGGUFMetadataFixture(t *testing.T, path string, kvs []fixtureKV) {
	t.Helper()
	buf := new(bytes.Buffer)
	buf.WriteString("GGUF")
	binary.Write(buf, binary.LittleEndian, uint32(3))
	binary.Write(buf, binary.LittleEndian, uint64(0))
	binary.Write(buf, binary.LittleEndian, uint64(len(kvs)))
	for _, kv := range kvs {
		binary.Write(buf, binary.LittleEndian, uint64(len(kv.key)))
		buf.WriteString(kv.key)
		binary.Write(buf, binary.LittleEndian, kv.vtype)
		kv.write(buf)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
}

// mimoAttentionArrays loads the independently verified MiMo-V2 header arrays.
func mimoAttentionArrays(t *testing.T) (hkv, pattern []float64) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "placement", "testdata", "mimo2-kv-geometry.json"))
	if err != nil {
		t.Fatalf("read verified fixture: %v", err)
	}
	var fixture struct {
		Metadata map[string]json.RawMessage `json:"metadata"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(fixture.Metadata["mimo2.attention.head_count_kv"], &hkv); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(fixture.Metadata["mimo2.attention.sliding_window_pattern"], &pattern); err != nil {
		t.Fatal(err)
	}
	return hkv, pattern
}

func intsEqual(got []int, want []float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if float64(got[i]) != want[i] {
			return false
		}
	}
	return true
}

// A mixed-head model states its KV geometry only as per-layer arrays. The
// parser once skipped every numeric array, so this model reached placement
// with no KV width at all. Both signed and unsigned encodings are legal.
func TestParsePreservesMixedPerLayerAttentionArrays(t *testing.T) {
	hkv, pattern := mimoAttentionArrays(t)
	for _, enc := range []struct {
		name          string
		hkvType, swaT uint32
	}{{"uint32/bool", 4, 7}, {"int32/int32", 5, 5}, {"uint8/uint32", 0, 4}} {
		t.Run(enc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "mimo.gguf")
			writeGGUFMetadataFixture(t, path, []fixtureKV{
				kvString("general.architecture", "mimo2"),
				kvArray("mimo2.attention.head_count_kv", enc.hkvType, hkv),
				kvArray("mimo2.attention.sliding_window_pattern", enc.swaT, pattern),
				// Keys after the arrays prove the reader stayed aligned.
				kvU32("mimo2.block_count", 51),
				kvU32("mimo2.nextn_predict_layers", 3),
				kvU32("mimo2.attention.key_length", 192),
				kvU32("mimo2.attention.value_length", 128),
				kvU32("mimo2.attention.sliding_window", 128),
			})
			info, err := Parse(path)
			if err != nil {
				t.Fatal(err)
			}
			if !intsEqual(info.HeadCountKVByLayer, hkv) || !intsEqual(info.SlidingWindowPattern, pattern) {
				t.Fatalf("arrays lost: hkv=%v swa=%v", info.HeadCountKVByLayer, info.SlidingWindowPattern)
			}
			if info.HeadCountKV != 0 {
				t.Fatalf("mixed 4/8 head counts collapsed into scalar %d", info.HeadCountKV)
			}
			if info.BlockCount != 51 || info.NextNPredictLayers != 3 || info.KeyLength != 192 ||
				info.ValueLength != 128 || info.SlidingWindow != 128 {
				t.Fatalf("reader misaligned after arrays: %+v", info)
			}
		})
	}
}

func TestParseUniformHeadArrayKeepsScalarContract(t *testing.T) {
	path := filepath.Join(t.TempDir(), "uniform.gguf")
	writeGGUFMetadataFixture(t, path, []fixtureKV{
		kvArray("llama.attention.head_count_kv", 4, []float64{8, 8, 8, 8}),
		kvU32("llama.block_count", 4),
		kvU32("llama.attention.key_length_swa", 256),
		kvU32("llama.attention.value_length_swa", 128),
	})
	info, err := Parse(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.HeadCountKV != 8 || len(info.HeadCountKVByLayer) != 4 {
		t.Fatalf("uniform array must equal the scalar form: %+v", info)
	}
	if info.KeyLengthSWA != 256 || info.ValueLengthSWA != 128 || info.KeyLength != 0 {
		t.Fatalf("SWA head widths confused with full widths: %+v", info)
	}
}

// Invalid or foreign encodings are skipped, never guessed, and never leave the
// reader misaligned for the keys that follow.
func TestParseRejectsInvalidAttentionArrays(t *testing.T) {
	path := filepath.Join(t.TempDir(), "invalid.gguf")
	writeGGUFMetadataFixture(t, path, []fixtureKV{
		kvArray("x.attention.head_count_kv", 5, []float64{4, -1, 8}),
		kvArray("x.attention.sliding_window_pattern", 6, []float64{0, 1, 1}),
		kvU32("x.block_count", 3),
	})
	info, err := Parse(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(info.HeadCountKVByLayer) != 0 || info.HeadCountKV != 0 {
		t.Fatalf("negative head count accepted: %+v", info)
	}
	if len(info.SlidingWindowPattern) != 0 {
		t.Fatalf("float pattern accepted: %v", info.SlidingWindowPattern)
	}
	if info.BlockCount != 3 {
		t.Fatalf("reader misaligned after skipped arrays: %d", info.BlockCount)
	}
}

// Recurrent state is a cache fact, not only an SSM layout. GLM-5.3-Flash states
// ssm.conv_kernel and kda.* but no ssm.state_size, so it was served as a plain
// attention model and a branch restored 9 of 4,247 shared tokens.
func TestParseReportsRecurrentStateBeyondSSMStateSize(t *testing.T) {
	glmHeads := make([]float64, 46)
	for i := 3; i < len(glmHeads); i += 4 {
		glmHeads[i] = 1
	}
	glmHeads[45] = 1
	for _, tc := range []struct {
		name           string
		kvs            []fixtureKV
		ssm, recurrent int
	}{
		{"glm5next KDA", []fixtureKV{
			kvString("general.architecture", "glm5next"),
			kvArray("glm5next.attention.head_count_kv", 4, glmHeads),
			kvU32("glm5next.attention.kv_lora_rank", 512),
			kvU32("glm5next.ssm.conv_kernel", 4),
			kvU32("glm5next.kda.head_dim", 128),
			kvU32("glm5next.block_count", 46),
		}, 0, 1},
		{"inkling short convolution", []fixtureKV{
			kvString("general.architecture", "inkling"),
			kvU32("inkling.attention.sliding_window", 512),
			kvU32("inkling.shortconv_kernel", 4),
		}, 0, 1},
		{"unlisted architecture stating KDA", []fixtureKV{
			kvString("general.architecture", "future-linear"),
			kvU32("future-linear.kda.head_dim", 128),
		}, 0, 1},
		{"listed architecture without state keys", []fixtureKV{
			kvString("general.architecture", "lfm2"),
			kvU32("lfm2.block_count", 16),
		}, 0, 1},
		{"qwen35 SSM layout", []fixtureKV{
			kvString("general.architecture", "qwen35"),
			kvU32("qwen35.ssm.state_size", 128),
			kvU32("qwen35.full_attention_interval", 4),
		}, 1, 1},
		{"plain attention", []fixtureKV{
			kvString("general.architecture", "llama"),
			kvU32("llama.block_count", 32),
			kvU32("llama.attention.head_count_kv", 8),
		}, 0, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "model.gguf")
			writeGGUFMetadataFixture(t, path, tc.kvs)
			info, err := Parse(path)
			if err != nil {
				t.Fatal(err)
			}
			if info.SSM != tc.ssm || info.RecurrentState != tc.recurrent {
				t.Fatalf("ssm=%d recurrent=%d, want ssm=%d recurrent=%d", info.SSM, info.RecurrentState, tc.ssm, tc.recurrent)
			}
		})
	}
}
