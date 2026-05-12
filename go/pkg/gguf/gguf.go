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
	Architecture      string `json:"arch"`
	Name              string `json:"name"`
	Basename          string `json:"basename"`
	BlockCount        int    `json:"layers"`
	ContextLength     int    `json:"ctx_train"`
	EmbeddingLength   int    `json:"embd"`
	FeedForwardLength int    `json:"ff"`
	HeadCountKV       int    `json:"hkv"`
	KeyLength         int    `json:"kl"`
	ValueLength       int    `json:"vl"`
	VocabSize         int    `json:"vocab_size"`
	ExpertBytes       int64  `json:"expert_bytes"`
	NonExpertBytes    int64  `json:"non_expert_bytes"`
	Fused             int    `json:"fused"`
	IsMoE             bool   `json:"is_moe"`
}

// Parse calls parse_gguf.py to extract model metadata.
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
	// Check env var
	if p := os.Getenv("LLM_SCRIPT_DIR"); p != "" {
		candidate := filepath.Join(p, "parse_gguf.py")
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}

	// Check PATH
	if p, err := exec.LookPath("parse_gguf.py"); err == nil {
		return p
	}

	// Check repo root (where go/ is a subdirectory)
	wd, _ := os.Getwd()
	candidate := filepath.Join(wd, "..", "parse_gguf.py")
	if _, err := os.Stat(candidate); err == nil {
		abs, _ := filepath.Abs(candidate)
		return abs
	}

	// Check home directory
	home, _ := os.UserHomeDir()
	paths := []string{
		filepath.Join(home, "llm-server", "parse_gguf.py"),
		filepath.Join(home, "parse_gguf.py"),
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
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
