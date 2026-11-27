package gguf

import (
	"bufio"
	"bytes"
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/boundexec"
	"github.com/raketenkater/ggrun/pkg/embedscript"
	"github.com/raketenkater/ggrun/pkg/python"
)

// parseGGUFSource is tools/gguf/parse_gguf.py, embedded so a bare binary can
// parse models. TestEmbeddedParserMatchesTools keeps the two copies identical.
//
//go:generate cp ../../../tools/gguf/parse_gguf.py parse_gguf.py
//go:embed parse_gguf.py
var parseGGUFSource string

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
	SSMConvKernel             int     `json:"ssm_d_conv"`                   // ssm.conv_kernel
	SSMStateSize              int     `json:"ssm_d_state"`                  // ssm.state_size
	SSMGroupCount             int     `json:"ssm_n_group"`                  // ssm.group_count
	SSMInnerSize              int     `json:"ssm_d_inner"`                  // ssm.inner_size
	SSMTimeStepRank           int     `json:"ssm_dt_rank"`                  // ssm.time_step_rank
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

	// SparseLookupBytes is the arch-proven lookup-table tier (schema 3+). It is
	// included in NonExpertBytes for the file-size identity but absent from
	// NonExpertLayerBytes.
	SparseLookupBytes      int64   `json:"sparse_lookup_bytes"`
	SparseLookupLayerBytes []int64 `json:"sparse_lookup_layer_bytes,omitempty"`
	SparseLookupKind       string  `json:"sparse_lookup_kind,omitempty"`
	SparseLookupLayerIDs   []int   `json:"sparse_lookup_layer_ids,omitempty"`
	// SparseLookupRanges are the tables' absolute byte ranges per shard file,
	// present only when the shard's tensor offsets were usable.
	SparseLookupRanges []SparseLookupRange `json:"sparse_lookup_ranges,omitempty"`
	// HeadCountKVByLayer and SlidingWindowPattern are the per-block arrays a
	// mixed-attention model states instead of (or alongside) the scalars.
	// They are raw metadata; placement validates them against the block count
	// and never averages mixed head counts into HeadCountKV.
	HeadCountKVByLayer   []int `json:"hkv_arr,omitempty"`
	SlidingWindowPattern []int `json:"swa_pattern,omitempty"` // 1 = windowed layer
	KeyLengthSWA         int   `json:"kl_swa"`                // 0 = same as KeyLength
	ValueLengthSWA       int   `json:"vl_swa"`                // 0 = same as ValueLength

	// RecurrentState is 1 when the backend keeps untruncatable recurrent state
	// beside the KV cache (SSM, delta-net/KDA, short convolution, RWKV). It is
	// wider than SSM, which also describes the block layout: GLM-5.3-Flash and
	// Inkling state no ssm.state_size but still resume only from checkpoints.
	RecurrentState int `json:"recurrent"`

	// TensorTypes is the tensor type histogram (see TypeUsage), written by the
	// helper from its own tensor-table walk. It is absent for a helper that
	// predates it and for a file whose offsets are unusable (an interrupted
	// download, a truncated table, a missing shard), so an empty value means
	// "unknown", never "no tensors of any type".
	TensorTypes []TypeUsage `json:"tensor_types,omitempty"`
}

// UnmarshalJSON decodes the helper's output. The optional tensor_types field is
// decoded on its own: one of a shape this build does not know (an old or foreign
// helper) leaves TensorTypes nil, which means "unknown", instead of failing the
// whole parse and the launch with it.
func (i *Info) UnmarshalJSON(data []byte) error {
	type plain Info
	aux := struct {
		*plain
		TensorTypes json.RawMessage `json:"tensor_types"`
	}{plain: (*plain)(i)}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	i.TensorTypes = nil
	if len(aux.TensorTypes) > 0 {
		var types []TypeUsage
		if json.Unmarshal(aux.TensorTypes, &types) == nil {
			i.TensorTypes = types
		}
	}
	return nil
}

// Parse calls the bundled GGUF metadata helper.
// parseHelperTimeout bounds parse_gguf.py; tests shorten it.
var parseHelperTimeout = boundexec.HelperTimeout

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

	py, ok := python.Command()
	if !ok {
		return nil, fmt.Errorf("parse_gguf.py needs Python 3, but no working python interpreter was found on PATH.\n%s", python.InstallHint())
	}
	cmd, finish := boundexec.Command(parseHelperTimeout, py, script, path)
	out, err := cmd.Output()
	if err = finish(err); err != nil {
		if boundexec.IsTimeout(err) {
			return nil, fmt.Errorf("parse_gguf.py did not answer within %s for %s", parseHelperTimeout, filepath.Base(path))
		}
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
	explicit := len(candidates) // locations the user or installer named
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, exeParseScriptCandidates(filepath.Dir(exe))...)
	}
	// The working directory, PATH and the home directory are never searched:
	// a model directory or a cloned repository could plant a parse_gguf.py
	// there and ggrun would execute it. The embedded copy is the last resort,
	// so a binary with nothing beside it still parses models.
	if p, err := embedscript.Extract("parse_gguf.py", parseGGUFSource); err == nil {
		candidates = append(candidates, p)
	}
	// A parser with a tensor-accounting contract in a location named by
	// LLM_SCRIPT_DIR, LLM_SERVER_HOME or LLM_APP_HOME is an explicit choice.
	// Among the discovered locations the newest contract wins, so a stale
	// copy beside the binary cannot shadow the newer embedded parser; equal
	// versions keep the order above.
	var fallback, best string
	bestSchema := 0
	for i, p := range candidates {
		if _, err := os.Stat(p); err != nil {
			continue
		}
		abs, _ := filepath.Abs(p)
		schema := parserTensorAccountingSchema(abs)
		if schema > 0 && i < explicit {
			return abs
		}
		if schema > bestSchema {
			best, bestSchema = abs, schema
		}
		if fallback == "" {
			fallback = abs
		}
	}
	if best != "" {
		return best
	}
	return fallback
}

// exeParseScriptCandidates are the helper locations of the binary's own
// install: beside it, and for a binary in <app-home>/bin or .bin the source
// checkout the installer keeps in that app home. No walk climbs further, so a
// binary in ~/.local/bin never reaches files in the home directory.
func exeParseScriptCandidates(exeDir string) []string {
	candidates := []string{filepath.Join(exeDir, "parse_gguf.py")}
	if base := filepath.Base(exeDir); base == ".bin" || base == "bin" {
		appHome := filepath.Dir(exeDir)
		candidates = append(candidates,
			filepath.Join(appHome, ".src", "llm-server", "tools", "gguf", "parse_gguf.py"),
			filepath.Join(appHome, ".src", "ggrun", "tools", "gguf", "parse_gguf.py"),
		)
	}
	return candidates
}

var parserSchemaPattern = regexp.MustCompile(`tensor_accounting_schema['"]\]?\s*[:=]\s*(\d+)`)

// parserTensorAccountingSchema is the tensor-accounting contract version a
// helper declares: 0 without the contract, 1 when its version is unreadable.
func parserTensorAccountingSchema(path string) int {
	data, err := os.ReadFile(path)
	if err != nil || !bytes.Contains(data, []byte("tensor_accounting_schema")) {
		return 0
	}
	if m := parserSchemaPattern.FindSubmatch(data); m != nil {
		if v, err := strconv.Atoi(string(m[1])); err == nil && v > 0 {
			return v
		}
	}
	return 1
}

// SparseLookupRange is one lookup table's byte range inside a shard file.
type SparseLookupRange struct {
	Shard  string `json:"shard"` // shard file basename
	Offset int64  `json:"offset"`
	Bytes  int64  `json:"bytes"`
}

// sparseLookupArchitectures mirrors SPARSE_LOOKUP_ADAPTERS in parse_gguf.py:
// the architectures whose sparse lookup tables the parser classifies, and the
// kind it reports. A tier for any other architecture is a parser mismatch.
var sparseLookupArchitectures = map[string]string{"deepseek41": "engram"}

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
	// Sparse lookup tables can be hundreds of GiB (DeepSeek-V4.1 Engram:
	// ~195 GiB). A schema-2 helper folds them into the per-layer backbone, so
	// an adapter architecture needs schema 3.
	adapterKind, adapted := sparseLookupArchitectures[strings.ToLower(info.Architecture)]
	if adapted && info.TensorAccountingSchema < 3 {
		return fmt.Errorf(
			"parse_gguf.py at %s lacks tensor accounting schema 3 (sparse lookup tables) for %s; reinstall ggrun or set LLM_SCRIPT_DIR to the repo tools/gguf directory",
			script, filepath.Base(modelPath))
	}
	var layerSum, rangeSum int64
	for _, b := range info.SparseLookupLayerBytes {
		layerSum += b
	}
	for _, r := range info.SparseLookupRanges {
		if r.Offset < 0 || r.Bytes <= 0 || r.Shard == "" {
			rangeSum = -1
			break
		}
		rangeSum += r.Bytes
	}
	if info.SparseLookupBytes < 0 || info.SparseLookupBytes > info.NonExpertBytes ||
		(info.SparseLookupBytes > 0 && (!adapted || info.SparseLookupKind != adapterKind)) ||
		(len(info.SparseLookupLayerBytes) > 0 && layerSum != info.SparseLookupBytes) ||
		(len(info.SparseLookupRanges) > 0 && rangeSum != info.SparseLookupBytes) {
		return fmt.Errorf(
			"parse_gguf.py at %s reported an inconsistent sparse lookup tier for %s (%d of %d non-expert bytes, %d by layer, kind %q, architecture %q)",
			script, filepath.Base(modelPath), info.SparseLookupBytes, info.NonExpertBytes, layerSum, info.SparseLookupKind, info.Architecture)
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

// GGUF keys are at most 2^16-1 bytes by spec. A chat template is a few KiB to
// a few hundred KiB; the cap only stops a corrupt length from sizing a huge
// allocation. Nested arrays are legal but never deep in real files.
const (
	maxGGUFKeyLen       = 1<<16 - 1
	maxChatTemplateLen  = 16 << 20
	maxGGUFArrayNesting = 8
)

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
	// A vocabulary is hundreds of thousands of tiny values; unbuffered, each
	// one is its own read syscall (tens of seconds for one model on a FUSE
	// disk). The buffer turns them into memory reads.
	template, err := readChatTemplate(bufio.NewReaderSize(f, 1<<20))
	if err != nil {
		return ""
	}
	return template
}

// readChatTemplate scans GGUF metadata for tokenizer.chat_template. Every
// length it reads comes from the file, so each is bounded before it sizes an
// allocation or a skip: a corrupt header is an error, never a panic.
func readChatTemplate(r io.Reader) (string, error) {
	var buf [8]byte
	readFull := func(dst []byte) error {
		_, err := io.ReadFull(r, dst)
		return err
	}
	readU32 := func() (uint32, error) {
		if err := readFull(buf[:4]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint32(buf[:4]), nil
	}
	readU64 := func() (uint64, error) {
		if err := readFull(buf[:8]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint64(buf[:8]), nil
	}
	if err := readFull(buf[:4]); err != nil || string(buf[:4]) != "GGUF" {
		return "", fmt.Errorf("not a GGUF file")
	}
	if _, err := readU32(); err != nil { // version
		return "", err
	}
	if _, err := readU64(); err != nil { // tensor_count
		return "", err
	}
	kvCount, err := readU64()
	if err != nil {
		return "", err
	}

	for i := uint64(0); i < kvCount; i++ {
		keyLen, err := readU64()
		if err != nil {
			return "", err
		}
		if keyLen > maxGGUFKeyLen {
			return "", fmt.Errorf("metadata key %d: length %d exceeds %d", i, keyLen, maxGGUFKeyLen)
		}
		key := make([]byte, keyLen)
		if err := readFull(key); err != nil {
			return "", err
		}
		valType, err := readU32()
		if err != nil {
			return "", err
		}
		if valType == 8 && string(key) == "tokenizer.chat_template" {
			valLen, err := readU64()
			if err != nil {
				return "", err
			}
			if valLen > maxChatTemplateLen {
				return "", fmt.Errorf("chat template length %d exceeds %d", valLen, maxChatTemplateLen)
			}
			val := make([]byte, valLen)
			if err := readFull(val); err != nil {
				return "", err
			}
			return string(val), nil
		}
		if err := skipGGUFValue(r, valType, 0); err != nil {
			return "", fmt.Errorf("metadata %q: %w", key, err)
		}
	}
	return "", nil
}

// skipGGUFValue advances the reader past one GGUF metadata value of the given
// type id, including arrays. It fails on truncation, an unknown type, or a
// length no real file could hold.
func skipGGUFValue(r io.Reader, valType uint32, depth int) error {
	if n, ok := ggufFixedSize[valType]; ok {
		return discardGGUF(r, n)
	}
	var buf [12]byte
	switch valType {
	case 8: // string
		if _, err := io.ReadFull(r, buf[:8]); err != nil {
			return err
		}
		n := binary.LittleEndian.Uint64(buf[:8])
		if n > math.MaxInt64 {
			return fmt.Errorf("string length %d out of range", n)
		}
		return discardGGUF(r, int64(n))
	case 9: // array: element type + count, then each element
		if depth >= maxGGUFArrayNesting {
			return fmt.Errorf("arrays nested deeper than %d", maxGGUFArrayNesting)
		}
		if _, err := io.ReadFull(r, buf[:12]); err != nil {
			return err
		}
		elemType := binary.LittleEndian.Uint32(buf[:4])
		count := binary.LittleEndian.Uint64(buf[4:12])
		// Fixed-size elements (token types, scores) are skipped in one step.
		if size, ok := ggufFixedSize[elemType]; ok {
			if count > uint64(math.MaxInt64)/uint64(size) {
				return fmt.Errorf("array of %d elements out of range", count)
			}
			return discardGGUF(r, int64(count)*size)
		}
		for j := uint64(0); j < count; j++ {
			if err := skipGGUFValue(r, elemType, depth+1); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unknown value type %d", valType)
	}
}

// ggufFixedSize is the byte size of each fixed-width metadata value type.
var ggufFixedSize = map[uint32]int64{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

// discardGGUF skips n bytes. A buffered reader discards them in memory; any
// other reader falls back to io.CopyN, so a short read is still an error.
func discardGGUF(r io.Reader, n int64) error {
	if d, ok := r.(interface{ Discard(int) (int, error) }); ok {
		for n > 0 {
			step := int64(math.MaxInt32)
			if n < step {
				step = n
			}
			got, err := d.Discard(int(step))
			n -= int64(got)
			if err != nil {
				return err
			}
		}
		return nil
	}
	_, err := io.CopyN(io.Discard, r, n)
	return err
}

// samplingKeyPrefix is the GGUF metadata namespace llama.cpp reads sampler
// defaults from (general.sampling.top_k, .top_p, .min_p, .temp, ...).
const samplingKeyPrefix = "general.sampling."

// maxSamplingStringLen bounds a general.sampling.* string value (for example
// the sampler sequence); real ones are a few dozen bytes.
const maxSamplingStringLen = 1 << 12

// SamplingDefaults returns the scalar general.sampling.* metadata keys of a
// GGUF, keyed by the suffix after the prefix ("top_k" -> "64", "temp" ->
// "1.0"). Floats keep a decimal point so they read as floats. Array values and
// other namespaces are skipped. It reads only the metadata section, never
// tensor data, and returns an empty map when the file has no such keys.
func SamplingDefaults(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	// A tokenizer vocabulary is a quarter-million tiny strings; buffer so each
	// is not its own read syscall.
	return readSamplingDefaults(bufio.NewReaderSize(f, 1<<20))
}

func readSamplingDefaults(r io.Reader) (map[string]string, error) {
	var buf [8]byte
	readFull := func(dst []byte) error {
		_, err := io.ReadFull(r, dst)
		return err
	}
	readU32 := func() (uint32, error) {
		if err := readFull(buf[:4]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint32(buf[:4]), nil
	}
	readU64 := func() (uint64, error) {
		if err := readFull(buf[:8]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint64(buf[:8]), nil
	}
	if err := readFull(buf[:4]); err != nil || string(buf[:4]) != "GGUF" {
		return nil, fmt.Errorf("not a GGUF file")
	}
	if _, err := readU32(); err != nil { // version
		return nil, err
	}
	if _, err := readU64(); err != nil { // tensor_count
		return nil, err
	}
	kvCount, err := readU64()
	if err != nil {
		return nil, err
	}

	out := map[string]string{}
	for i := uint64(0); i < kvCount; i++ {
		keyLen, err := readU64()
		if err != nil {
			return nil, err
		}
		if keyLen > maxGGUFKeyLen {
			return nil, fmt.Errorf("metadata key %d: length %d exceeds %d", i, keyLen, maxGGUFKeyLen)
		}
		key := make([]byte, keyLen)
		if err := readFull(key); err != nil {
			return nil, err
		}
		valType, err := readU32()
		if err != nil {
			return nil, err
		}
		name, sampling := strings.CutPrefix(string(key), samplingKeyPrefix)
		if !sampling || name == "" || valType == 9 {
			if err := skipGGUFValue(r, valType, 0); err != nil {
				return nil, fmt.Errorf("metadata %q: %w", key, err)
			}
			continue
		}
		val, err := readSamplingScalar(r, valType)
		if err != nil {
			return nil, fmt.Errorf("metadata %q: %w", key, err)
		}
		out[name] = val
	}
	return out, nil
}

// readSamplingScalar decodes one non-array GGUF value as a string.
func readSamplingScalar(r io.Reader, valType uint32) (string, error) {
	var buf [8]byte
	size := map[uint32]int{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[valType]
	if valType == 8 {
		if _, err := io.ReadFull(r, buf[:8]); err != nil {
			return "", err
		}
		n := binary.LittleEndian.Uint64(buf[:8])
		if n > maxSamplingStringLen {
			return "", fmt.Errorf("string length %d exceeds %d", n, maxSamplingStringLen)
		}
		val := make([]byte, n)
		if _, err := io.ReadFull(r, val); err != nil {
			return "", err
		}
		return string(val), nil
	}
	if size == 0 {
		return "", fmt.Errorf("unknown value type %d", valType)
	}
	if _, err := io.ReadFull(r, buf[:size]); err != nil {
		return "", err
	}
	le := binary.LittleEndian
	switch valType {
	case 0:
		return strconv.FormatUint(uint64(buf[0]), 10), nil
	case 1:
		return strconv.FormatInt(int64(int8(buf[0])), 10), nil
	case 2:
		return strconv.FormatUint(uint64(le.Uint16(buf[:2])), 10), nil
	case 3:
		return strconv.FormatInt(int64(int16(le.Uint16(buf[:2]))), 10), nil
	case 4:
		return strconv.FormatUint(uint64(le.Uint32(buf[:4])), 10), nil
	case 5:
		return strconv.FormatInt(int64(int32(le.Uint32(buf[:4]))), 10), nil
	case 6:
		return formatSamplingFloat(float64(math.Float32frombits(le.Uint32(buf[:4]))), 32), nil
	case 7:
		return strconv.FormatBool(buf[0] != 0), nil
	case 10:
		return strconv.FormatUint(le.Uint64(buf[:8]), 10), nil
	case 11:
		return strconv.FormatInt(int64(le.Uint64(buf[:8])), 10), nil
	default: // 12: float64
		return formatSamplingFloat(math.Float64frombits(le.Uint64(buf[:8])), 64), nil
	}
}

func formatSamplingFloat(v float64, bits int) string {
	s := strconv.FormatFloat(v, 'g', -1, bits)
	if !strings.ContainsAny(s, ".eEIN") { // integral: keep it reading as a float
		s += ".0"
	}
	return s
}
