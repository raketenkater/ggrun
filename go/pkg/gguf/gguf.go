package gguf

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Info holds parsed GGUF metadata.
type Info struct {
	Architecture       string `json:"arch"`
	Name               string `json:"name"`
	Basename           string `json:"basename"`
	QuantizedBy        string `json:"quantized_by"`
	BlockCount         int    `json:"layers"`
	ContextLength      int    `json:"ctx_train"`
	EmbeddingLength    int    `json:"embd"`
	FeedForwardLength  int    `json:"ff"`
	HeadCountKV        int    `json:"hkv"`
	KeyLength          int    `json:"kl"`
	ValueLength        int    `json:"vl"`
	VocabSize          int    `json:"vocab_size"`
	ExpertBytes        int64  `json:"expert_bytes"`
	NonExpertBytes     int64  `json:"non_expert_bytes"`
	Fused              int    `json:"fused"`
	Experts            int    `json:"experts"`              // total number of experts (MoE)
	ExpertUsed         int    `json:"exp_used"`             // experts used per token
	ExpFF              int    `json:"exp_ff"`               // expert feed-forward size
	ExpSharedFF        int    `json:"exp_shared_ff"`        // expert shared feed-forward size
	NRot               int    `json:"n_rot"`                // rope dimension
	SSM                int    `json:"ssm"`                  // 1 if model uses SSM layers
	FullAttnInterval   int    `json:"full_interval"`        // full attention every N layers (hybrid SSM/SWA)
	SlidingWindow      int    `json:"swa"`                  // sliding window size (0 = no SWA)
	LeadingDense       int    `json:"leading_dense"`        // leading dense block count (MoE models)
	KVLoraRank         int    `json:"kv_lora"`              // MLA KV lora rank
	QLoraRank          int    `json:"q_lora"`               // MLA Q lora rank
	KeyLengthMLA       int    `json:"kl_mla"`               // MLA key length
	ValueLengthMLA     int    `json:"vl_mla"`               // MLA value length
	HasShexp           int    `json:"has_shexp"`            // shared experts present
	NextNPredictLayers int    `json:"nextn_predict_layers"` // MTP/NextN prediction layers
	IsMoE              bool   `json:"is_moe"`
}

// Parse calls the bundled GGUF metadata helper.
func Parse(path string) (*Info, error) {
	script := findParseScript()
	if script == "" {
		return nil, fmt.Errorf("parse_gguf.py not found")
	}

	cmd := exec.Command("python3", script, path)
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
			filepath.Join(appHome, ".bin", "parse_gguf.py"),
			filepath.Join(appHome, "bin", "parse_gguf.py"),
			filepath.Join(appHome, "parse_gguf.py"),
		)
	}
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		candidates = append(candidates,
			filepath.Join(exeDir, "parse_gguf.py"),
			filepath.Join(exeDir, "..", "tools", "gguf", "parse_gguf.py"),
			filepath.Join(exeDir, "..", "..", "tools", "gguf", "parse_gguf.py"),
		)
	}
	wd, _ := os.Getwd()
	candidates = append(candidates,
		filepath.Join(wd, "tools", "gguf", "parse_gguf.py"),
		filepath.Join(wd, "..", "tools", "gguf", "parse_gguf.py"),
		filepath.Join(wd, "parse_gguf.py"),
		filepath.Join(wd, "..", "parse_gguf.py"),
	)
	if p, err := exec.LookPath("parse_gguf.py"); err == nil {
		candidates = append(candidates, p)
	}
	home, _ := os.UserHomeDir()
	candidates = append(candidates,
		filepath.Join(home, "llm-server", "bin", "parse_gguf.py"),
		filepath.Join(home, "llm-server", "tools", "gguf", "parse_gguf.py"),
		filepath.Join(home, "parse_gguf.py"),
	)
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			abs, _ := filepath.Abs(p)
			return abs
		}
	}
	return ""
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
