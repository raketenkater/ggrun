package config

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Config holds llm-server settings with precedence:
// CLI flag > env var > config file > built-in default.
type Config struct {
	Port        int    `json:"port"`
	CtxSize     int    `json:"ctx_size"`
	ModelDir    string `json:"model_dir"`
	CacheDir    string `json:"cache_dir"`
	Backend     string `json:"backend"`
	KVPlacement string `json:"kv_placement"`
	KVQuality   string `json:"kv_quality"`
	TuneRounds  int    `json:"tune_rounds"`
	Vision      bool   `json:"vision"`
	Parallel    int    `json:"parallel"`
	KeepAlive   int    `json:"keep_alive"`
	Host        string `json:"host"`
}

// Defaults returns the built-in defaults.
func Defaults() *Config {
	home, _ := os.UserHomeDir()
	return &Config{
		Port:        8081,
		CtxSize:     0, // auto
		ModelDir:    filepath.Join(home, "ai_models"),
		CacheDir:    filepath.Join(home, ".cache", "llm-server"),
		Backend:     "",
		KVPlacement: "auto",
		KVQuality:   "mid",
		TuneRounds:  5,
		Vision:      false,
		Parallel:    1,
		KeepAlive:   0,
		Host:        "127.0.0.1",
	}
}

// Load reads the config file and env vars, returning a merged config.
func Load() (*Config, error) {
	cfg := Defaults()

	// Load from config file
	cfgPath := filepath.Join(os.Getenv("HOME"), ".config", "llm-server", "config")
	if p := os.Getenv("LLM_CONFIG"); p != "" {
		cfgPath = p
	}
	if _, err := os.Stat(cfgPath); err == nil {
		if err := loadFile(cfgPath, cfg); err != nil {
			return nil, fmt.Errorf("load config: %w", err)
		}
	}

	// Override with env vars
	if v := os.Getenv("LLM_PORT"); v != "" {
		cfg.Port, _ = strconv.Atoi(v)
	}
	if v := os.Getenv("LLM_CTX_SIZE"); v != "" {
		cfg.CtxSize, _ = strconv.Atoi(v)
	}
	if v := os.Getenv("LLM_MODEL_DIR"); v != "" {
		cfg.ModelDir = v
	}
	if v := os.Getenv("LLM_BACKEND"); v != "" {
		cfg.Backend = v
	}
	if v := os.Getenv("LLM_KV_PLACEMENT"); v != "" {
		cfg.KVPlacement = v
	}
	if v := os.Getenv("LLM_KV_QUALITY"); v != "" {
		cfg.KVQuality = v
	}
	if v := os.Getenv("LLM_TUNE_ROUNDS"); v != "" {
		cfg.TuneRounds, _ = strconv.Atoi(v)
	}

	return cfg, nil
}

func loadFile(path string, cfg *Config) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.Trim(strings.TrimSpace(parts[1]), `"'`)

		switch key {
		case "PORT":
			cfg.Port, _ = strconv.Atoi(val)
		case "CTX_SIZE":
			cfg.CtxSize, _ = strconv.Atoi(val)
		case "MODEL_DIR":
			cfg.ModelDir = val
		case "CACHE_DIR":
			cfg.CacheDir = val
		case "BACKEND":
			cfg.Backend = val
		case "KV_PLACEMENT":
			cfg.KVPlacement = val
		case "KV_QUALITY":
			cfg.KVQuality = val
		case "TUNE_ROUNDS":
			cfg.TuneRounds, _ = strconv.Atoi(val)
		case "VISION":
			cfg.Vision = val == "true" || val == "1"
		case "PARALLEL":
			cfg.Parallel, _ = strconv.Atoi(val)
		case "KEEP_ALIVE":
			cfg.KeepAlive, _ = strconv.Atoi(val)
		case "HOST":
			cfg.Host = val
		}
	}
	return scanner.Err()
}

// Save writes the config to the default config file.
func (c *Config) Save() error {
	dir := filepath.Join(os.Getenv("HOME"), ".config", "llm-server")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	path := filepath.Join(dir, "config")
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	fmt.Fprintf(f, "# llm-server configuration\n")
	fmt.Fprintf(f, "PORT=%d\n", c.Port)
	fmt.Fprintf(f, "CTX_SIZE=%d\n", c.CtxSize)
	fmt.Fprintf(f, "MODEL_DIR=%q\n", c.ModelDir)
	fmt.Fprintf(f, "CACHE_DIR=%q\n", c.CacheDir)
	fmt.Fprintf(f, "BACKEND=%q\n", c.Backend)
	fmt.Fprintf(f, "KV_PLACEMENT=%q\n", c.KVPlacement)
	fmt.Fprintf(f, "KV_QUALITY=%q\n", c.KVQuality)
	fmt.Fprintf(f, "TUNE_ROUNDS=%d\n", c.TuneRounds)
	fmt.Fprintf(f, "VISION=%v\n", c.Vision)
	fmt.Fprintf(f, "PARALLEL=%d\n", c.Parallel)
	fmt.Fprintf(f, "KEEP_ALIVE=%d\n", c.KeepAlive)
	fmt.Fprintf(f, "HOST=%q\n", c.Host)
	return nil
}
