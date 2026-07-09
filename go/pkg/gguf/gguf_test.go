package gguf

import (
	"bytes"
	"encoding/binary"
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
	if err := os.WriteFile(sourceParser, []byte("#!/usr/bin/env python3\n"), 0755); err != nil {
		t.Fatalf("write source parser: %v", err)
	}
	if err := os.WriteFile(installedParser, []byte("#!/usr/bin/env python3\n"), 0755); err != nil {
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
