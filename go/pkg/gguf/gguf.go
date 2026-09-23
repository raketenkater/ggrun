package gguf

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Info holds parsed GGUF metadata.
type Info struct {
	TensorAccountingSchema    int     `json:"tensor_accounting_schema"`
	Architecture              string  `json:"arch"`
	Name                      string  `json:"name"`
	Basename                  string  `json:"basename"`
	QuantizedBy               string  `json:"quantized_by"`
	BlockCount                int     `json:"layers"`
	ContextLength             int     `json:"ctx_train"`
	EmbeddingLength           int     `json:"embd"`
	FeedForwardLength         int     `json:"ff"`
	HeadCountKV               int     `json:"hkv"`
	HeadCount                 int     `json:"heads"` // attention.head_count; absent in some converts
	KeyLength                 int     `json:"kl"`
	ValueLength               int     `json:"vl"`
	VocabSize                 int     `json:"vocab_size"`
	TokenizerModel            string  `json:"tokenizer_model"`
	TokenizerPre              string  `json:"tokenizer_pre"`
	TokenizerHash             string  `json:"tokenizer_hash"`
	ExpertBytes               int64   `json:"expert_bytes"`
	NonExpertBytes            int64   `json:"non_expert_bytes"`
	TokenEmbdBytes            int64   `json:"token_embd_bytes"`           // host-resident embeddings: token_embd + per_layer_token_embd
	PerLayerTokenEmbdBytes    int64   `json:"per_layer_token_embd_bytes"` // PLE/n-gram table; subset of TokenEmbdBytes
	HasPerLayerTokenEmbd      int     `json:"has_per_layer_token_embd"`   // tensor-table presence; distinguishes no-PLE models from stale parsers
	OutputBytes               int64   `json:"output_bytes"`               // output head; lands whole on the last split device
	ShexpBytes                int64   `json:"shexp_bytes"`                // shared experts; stay on the layer's device even when routed experts offload to CPU
	ExpertAuxBytes            int64   `json:"expert_aux_bytes"`           // routing tensors moved by whole expert-layer overrides
	ExpertLayerBytes          []int64 `json:"expert_layer_bytes,omitempty"`
	RoutedExpertLayerBytes    []int64 `json:"routed_expert_layer_bytes,omitempty"`
	ShexpLayerBytes           []int64 `json:"shexp_layer_bytes,omitempty"`
	ExpertAuxLayerBytes       []int64 `json:"expert_aux_layer_bytes,omitempty"`
	NonExpertLayerBytes       []int64 `json:"non_expert_layer_bytes,omitempty"`
	Fused                     int     `json:"fused"`
	Experts                   int     `json:"experts"`                      // total number of experts (MoE)
	ExpertUsed                int     `json:"exp_used"`                     // experts used per token
	ExpFF                     int     `json:"exp_ff"`                       // expert feed-forward size
	ExpSharedFF               int     `json:"exp_shared_ff"`                // expert shared feed-forward size
	NRot                      int     `json:"n_rot"`                        // rope dimension
	SSM                       int     `json:"ssm"`                          // 1 if model uses SSM layers
	FullAttnInterval          int     `json:"full_interval"`                // full attention every N layers (hybrid SSM/SWA)
	SlidingWindow             int     `json:"swa"`                          // sliding window size (0 = no SWA)
	LeadingDense              int     `json:"leading_dense"`                // leading dense block count (MoE models)
	KVLoops                   int     `json:"kv_loops"`                     // looped-transformer passes, each with its own KV (0/1 = none)
	LeadingDenseInferred      int     `json:"leading_dense_inferred"`       // derived from tensor layout
	ExpertSharedCount         int     `json:"expert_shared_count"`          // shared experts per routed layer
	ExpertSharedCountInferred int     `json:"expert_shared_count_inferred"` // derived from tensor layout
	KVLoraRank                int     `json:"kv_lora"`                      // MLA KV lora rank
	QLoraRank                 int     `json:"q_lora"`                       // MLA Q lora rank
	KeyLengthMLA              int     `json:"kl_mla"`                       // MLA key length
	ValueLengthMLA            int     `json:"vl_mla"`                       // MLA value length
	HasShexp                  int     `json:"has_shexp"`                    // shared experts present
	NextNPredictLayers        int     `json:"nextn_predict_layers"`         // MTP/NextN prediction layers
	IsMoE                     bool    `json:"is_moe"`
}

// Parse calls the bundled GGUF metadata helper.
func Parse(path string) (*Info, error) {
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("model file path is empty")
	}
	fi, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("model file %q: %w", path, err)
	}
	if fi.IsDir() {
		return nil, fmt.Errorf("model file %q is a directory", path)
	}

	script := findParseScript()
	if script == "" {
		return nil, fmt.Errorf("parse_gguf.py not found")
	}

	cmd := exec.Command(pythonCommand(), script, path)
	out, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok && len(exitErr.Stderr) > 0 {
			return nil, fmt.Errorf("parse_gguf.py: %s", exitErr.Stderr)
		}
		return nil, fmt.Errorf("parse_gguf.py: %w", err)
	}

	var info Info
	if err := json.Unmarshal(out, &info); err != nil {
		return nil, fmt.Errorf("parse output: %w", err)
	}
	if err := validateParserOutput(script, path, &info); err != nil {
		return nil, err
	}

	// Derive MoE from architecture or filename
	if info.ExpertBytes > 0 || info.Fused > 0 {
		info.IsMoE = true
	}
	name := strings.ToLower(filepath.Base(path))
	if strings.Contains(name, "moe") || strings.Contains(name, "mixtral") ||
		strings.Contains(name, "a10b") || strings.Contains(name, "a20b") ||
		strings.Contains(name, "a40b") || strings.Contains(name, "a100b") {
		info.IsMoE = true
	}

	return &info, nil
}

func pythonCommand() string {
	for _, name := range []string{"python3", "python", "py"} {
		if path, err := exec.LookPath(name); err == nil {
			return path
		}
	}
	return "python3"
}

func findParseScript() string {
	candidates := []string{}
	if p := os.Getenv("LLM_SCRIPT_DIR"); p != "" {
		candidates = append(candidates,
			filepath.Join(p, "parse_gguf.py"),
			filepath.Join(p, "tools", "gguf", "parse_gguf.py"),
		)
	}
	if home := os.Getenv("LLM_SERVER_HOME"); home != "" {
		candidates = append(candidates,
			filepath.Join(home, "tools", "gguf", "parse_gguf.py"),
			filepath.Join(home, "parse_gguf.py"),
		)
	}
	if appHome := os.Getenv("LLM_APP_HOME"); appHome != "" {
		candidates = append(candidates,
			filepath.Join(appHome, ".src", "llm-server", "tools", "gguf", "parse_gguf.py"),
			filepath.Join(appHome, ".src", "ggrun", "tools", "gguf", "parse_gguf.py"),
			filepath.Join(appHome, ".bin", "parse_gguf.py"),
			filepath.Join(appHome, "bin", "parse_gguf.py"),
			filepath.Join(appHome, "parse_gguf.py"),
		)
	}
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		// A development or source-install binary commonly lives in
		// <app-home>/.bin while its authoritative scripts live in the adjacent
		// source checkout. Prefer that checkout just as an explicit
		// LLM_APP_HOME does; a release-only install simply falls through to the
		// bundled parser below.
		if base := filepath.Base(exeDir); base == ".bin" || base == "bin" {
			appHome := filepath.Dir(exeDir)
			candidates = append(candidates,
				filepath.Join(appHome, ".src", "llm-server", "tools", "gguf", "parse_gguf.py"),
				filepath.Join(appHome, ".src", "ggrun", "tools", "gguf", "parse_gguf.py"),
			)
		}
		candidates = append(candidates,
			filepath.Join(exeDir, "parse_gguf.py"),
			filepath.Join(exeDir, "..", "tools", "gguf", "parse_gguf.py"),
			filepath.Join(exeDir, "..", "..", "tools", "gguf", "parse_gguf.py"),
			filepath.Join(exeDir, "..", "..", "..", "tools", "gguf", "parse_gguf.py"),
		)
	}
	wd, _ := os.Getwd()
	candidates = append(candidates,
		filepath.Join(wd, "tools", "gguf", "parse_gguf.py"),
		filepath.Join(wd, "..", "tools", "gguf", "parse_gguf.py"),
		filepath.Join(wd, "..", "..", "tools", "gguf", "parse_gguf.py"),
		filepath.Join(wd, "..", "..", "..", "tools", "gguf", "parse_gguf.py"),
		filepath.Join(wd, "parse_gguf.py"),
		filepath.Join(wd, "..", "parse_gguf.py"),
	)
	if p, err := exec.LookPath("parse_gguf.py"); err == nil {
		candidates = append(candidates, p)
	}
	home, _ := os.UserHomeDir()
	candidates = append(candidates,
		filepath.Join(home, "ggrun", "bin", "parse_gguf.py"),
		filepath.Join(home, "ggrun", "tools", "gguf", "parse_gguf.py"),
		filepath.Join(home, "parse_gguf.py"),
	)
	var fallback string
	for _, p := range candidates {
		if _, err := os.Stat(p); err != nil {
			continue
		}
		abs, _ := filepath.Abs(p)
		if parserHasCurrentTensorAccounting(abs) {
			return abs
		}
		if fallback == "" {
			fallback = abs
		}
	}
	return fallback
}

// parserHasCurrentTensorAccounting is true when the helper exposes a versioned
// tensor-accounting contract. Looking only for a PLE byte field cannot tell a
// stale parser from a valid qwen4exp file that simply has no PLE tensor.
func parserHasCurrentTensorAccounting(path string) bool {
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	return bytes.Contains(data, []byte("tensor_accounting_schema"))
}

func validateParserOutput(script, modelPath string, info *Info) error {
	if info == nil {
		return nil
	}
	// VERIFICATION: Big-MoE stable-max placement depends on these file-anchored
	// byte splits. A stale installed helper can still return expert/non-expert
	// totals, but without token/output/shared-expert bytes the GPU ledger is not
	// complete and the no-cache path can overfill GPUs while looking "exact".
	if strings.EqualFold(info.Architecture, "deepseek4") && info.HasShexp == 1 {
		if info.TokenEmbdBytes <= 0 || info.OutputBytes <= 0 || info.ShexpBytes <= 0 {
			return fmt.Errorf(
				"parse_gguf.py at %s is missing required DeepSeek4 byte splits for %s; reinstall ggrun or set LLM_SCRIPT_DIR to the repo tools/gguf directory",
				script, filepath.Base(modelPath))
		}
	}
	// qwen4exp may legitimately omit the optional n-gram PLE table. Require a
	// parser contract new enough to report tensor presence separately, then
	// reject only an internally inconsistent result. This still catches the
	// dangerous stale-helper case without rejecting valid no-PLE fixtures.
	if strings.EqualFold(info.Architecture, "qwen4exp") {
		if info.TensorAccountingSchema < 2 {
			return fmt.Errorf(
				"parse_gguf.py at %s lacks tensor accounting schema 2 for %s; reinstall ggrun or set LLM_SCRIPT_DIR to the repo tools/gguf directory",
				script, filepath.Base(modelPath))
		}
		if info.HasPerLayerTokenEmbd != 0 && info.PerLayerTokenEmbdBytes <= 0 {
			return fmt.Errorf(
				"parse_gguf.py at %s reports per_layer_token_embd in %s but did not account its host-resident bytes",
				script, filepath.Base(modelPath))
		}
	}
	return nil
}

// EstimateParams returns a rough parameter count from metadata.
func (i *Info) EstimateParams() int64 {
	// Rough estimate: 2 * vocab * embed + layers * (4 * embed^2 + 3 * embed * ffn)
	vocab := int64(i.VocabSize)
	embed := int64(i.EmbeddingLength)
	layers := int64(i.BlockCount)
	ffn := int64(i.FeedForwardLength)
	return 2*vocab*embed + layers*(4*embed*embed+3*embed*ffn)
}

// ChatTemplate reads just the `tokenizer.chat_template` metadata string from a
// GGUF header, without shelling out to parse_gguf.py or walking the tensor
// table. It is a fast, pure-Go probe used by the chat-template catalog to
// detect a raise_exception guard in the model's embedded template. Returns ""
// when the key is absent, the file is unreadable, or the header is corrupt.
func ChatTemplate(path string) string {
	if strings.TrimSpace(path) == "" {
		return ""
	}
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()

	var buf [8]byte
	readFull := func(dst []byte) bool {
		_, err := io.ReadFull(f, dst)
		return err == nil
	}
	if !readFull(buf[:4]) || string(buf[:4]) != "GGUF" {
		return ""
	}
	if !readFull(buf[:4]) { // version
		return ""
	}
	if !readFull(buf[:8]) { // tensor_count
		return ""
	}
	var kvCount uint64
	if !readFull(buf[:8]) {
		return ""
	}
	kvCount = binary.LittleEndian.Uint64(buf[:8])

	for i := uint64(0); i < kvCount; i++ {
		if !readFull(buf[:8]) {
			return ""
		}
		keyLen := binary.LittleEndian.Uint64(buf[:8])
		key := make([]byte, keyLen)
		if !readFull(key) {
			return ""
		}
		if !readFull(buf[:4]) {
			return ""
		}
		valType := binary.LittleEndian.Uint32(buf[:4])
		switch valType {
		case 8: // string
			if !readFull(buf[:8]) {
				return ""
			}
			valLen := binary.LittleEndian.Uint64(buf[:8])
			val := make([]byte, valLen)
			if !readFull(val) {
				return ""
			}
			if string(key) == "tokenizer.chat_template" {
				return string(val)
			}
		case 9: // array — skip element type + count, then each element
			if !readFull(buf[:4]) {
				return ""
			}
			elemType := binary.LittleEndian.Uint32(buf[:4])
			if !readFull(buf[:8]) {
				return ""
			}
			arrLen := binary.LittleEndian.Uint64(buf[:8])
			for j := uint64(0); j < arrLen; j++ {
				if !skipGGUFValue(f, elemType) {
					return ""
				}
			}
		default:
			if !skipGGUFValue(f, valType) {
				return ""
			}
		}
	}
	return ""
}

// skipGGUFValue advances the reader past one GGUF metadata value of the given
// type id. Returns false on EOF/truncation.
func skipGGUFValue(r io.Reader, valType uint32) bool {
	fixed := map[uint32]uint64{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
	if n, ok := fixed[valType]; ok {
		_, err := io.CopyN(io.Discard, r, int64(n))
		return err == nil
	}
	if valType == 8 { // string
		var buf [8]byte
		if _, err := io.ReadFull(r, buf[:]); err != nil {
			return false
		}
		n := binary.LittleEndian.Uint64(buf[:])
		_, err := io.CopyN(io.Discard, r, int64(n))
		return err == nil
	}
	return true
}
