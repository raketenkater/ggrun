package config

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Config holds llm-server settings with precedence:
// CLI flag > env var > config file > built-in default.
// This is the single source of truth for all user-tunable settings.
type Config struct {
	Port          int    `json:"port"`
	CtxSize       int    `json:"ctx_size"`
	CtxMode       string `json:"ctx_mode"` // fit, max, manual
	MaxRestarts   int    `json:"max_restarts"`
	KeepAlive     int    `json:"keep_alive"`
	HealthTimeout int    `json:"health_timeout"`
	ModelDir      string `json:"model_dir"`
	CacheDir      string `json:"cache_dir"`
	LogDir        string `json:"log_dir"`
	RamBudget     string `json:"ram_budget"`
	KVPlacement   string `json:"kv_placement"`
	KVQuality     string `json:"kv_quality"`
	AssumeYes     bool   `json:"assume_yes"`
	Backend       string `json:"backend"`
	LlamaServer   string `json:"llama_server"`
	AppHome       string `json:"app_home"`
	TuneRounds    int    `json:"tune_rounds"`
	Vision        bool   `json:"vision"`
	Parallel      int    `json:"parallel"`
	Host          string `json:"host"`
	Spec          string `json:"spec"` // off, auto, draft, eagle3, ngram, ngram-mod, ngram-k4v, mtp
}

// DefaultKeys is the stable display order for config show / template generation.
var DefaultKeys = []string{
	"PORT", "CTX_SIZE", "MAX_RESTARTS", "KEEP_ALIVE", "HEALTH_TIMEOUT",
	"MODEL_DIR", "CACHE_DIR", "LOG_DIR",
	"RAM_BUDGET", "KV_PLACEMENT", "KV_QUALITY",
	"ASSUME_YES",
	"BACKEND", "LLAMA_SERVER", "APP_HOME",
	"TUNE_ROUNDS", "VISION", "PARALLEL", "HOST", "SPEC",
}

// Defaults returns the built-in defaults.
func Defaults() *Config {
	home, _ := os.UserHomeDir()
	return &Config{
		Port:          8081,
		CtxSize:       0,
		CtxMode:       "fit",
		MaxRestarts:   5,
		KeepAlive:     0,
		HealthTimeout: 0, // auto
		ModelDir:      filepath.Join(home, "ai_models"),
		CacheDir:      filepath.Join(home, ".cache", "llm-server"),
		LogDir:        "",
		RamBudget:     "",
		KVPlacement:   "auto",
		KVQuality:     "low",
		AssumeYes:     false,
		Backend:       "",
		LlamaServer:   "",
		AppHome:       "",
		TuneRounds:    8,
		Vision:        false,
		Parallel:      1,
		Host:          "0.0.0.0",
		Spec:          "off",
	}
}

// Path returns the canonical config file path.
func Path() string {
	if p := os.Getenv("LLM_CONFIG"); p != "" {
		return p
	}
	if p := os.Getenv("LLM_APP_HOME"); p != "" {
		if f := filepath.Join(p, ".config", "config"); fileExists(f) {
			return f
		}
		if f := filepath.Join(p, "config", "config"); fileExists(f) {
			return f
		}
	}
	home := os.Getenv("HOME")
	if home == "" {
		home, _ = os.UserHomeDir()
	}
	return filepath.Join(home, ".config", "llm-server", "config")
}

// Load reads the config file and env vars, returning a merged config.
// Precedence: env var > config file > built-in default.
func Load() (*Config, error) {
	cfg := Defaults()

	// Snapshot env-set values BEFORE loading file, so env wins.
	envSnapshot := snapshotEnv()

	// Migrate legacy config.sh if needed
	cfgPath := Path()
	if err := migrateLegacyConfig(cfgPath); err != nil {
		// non-fatal
	}

	if fileExists(cfgPath) {
		if err := loadFile(cfgPath, cfg); err != nil {
			return nil, fmt.Errorf("load config: %w", err)
		}
	}

	// Re-apply env snapshot so env wins over file
	applyEnvSnapshot(cfg, envSnapshot)

	// Fill remaining unset keys from defaults (already in cfg)
	return cfg, nil
}

func snapshotEnv() map[string]string {
	m := make(map[string]string)
	for _, k := range []string{
		"LLM_PORT", "LLM_CTX_SIZE", "LLM_MAX_RESTARTS", "LLM_KEEP_ALIVE",
		"LLM_HEALTH_TIMEOUT", "LLM_MODEL_DIR", "LLM_CACHE_DIR", "LLM_LOG_DIR",
		"LLM_RAM_BUDGET", "LLM_KV_PLACEMENT", "LLM_KV_QUALITY", "LLM_ASSUME_YES",
		"LLM_BACKEND", "LLAMA_SERVER", "LLM_APP_HOME", "LLM_TUNE_ROUNDS",
		"LLM_VISION", "LLM_PARALLEL", "LLM_HOST", "LLM_SPEC",
	} {
		if v := os.Getenv(k); v != "" {
			m[k] = v
		}
	}
	return m
}

func applyEnvSnapshot(cfg *Config, snap map[string]string) {
	if v, ok := snap["LLM_PORT"]; ok {
		cfg.Port, _ = strconv.Atoi(v)
	}
	if v, ok := snap["LLM_CTX_SIZE"]; ok {
		applyCtxValue(cfg, v)
	}
	if v, ok := snap["LLM_MAX_RESTARTS"]; ok {
		cfg.MaxRestarts, _ = strconv.Atoi(v)
	}
	if v, ok := snap["LLM_KEEP_ALIVE"]; ok {
		cfg.KeepAlive, _ = strconv.Atoi(v)
	}
	if v, ok := snap["LLM_HEALTH_TIMEOUT"]; ok {
		cfg.HealthTimeout, _ = strconv.Atoi(v)
	}
	if v, ok := snap["LLM_MODEL_DIR"]; ok {
		cfg.ModelDir = v
	}
	if v, ok := snap["LLM_CACHE_DIR"]; ok {
		cfg.CacheDir = v
	}
	if v, ok := snap["LLM_LOG_DIR"]; ok {
		cfg.LogDir = v
	}
	if v, ok := snap["LLM_RAM_BUDGET"]; ok {
		cfg.RamBudget = v
	}
	if v, ok := snap["LLM_KV_PLACEMENT"]; ok {
		cfg.KVPlacement = v
	}
	if v, ok := snap["LLM_KV_QUALITY"]; ok {
		cfg.KVQuality = v
	}
	if v, ok := snap["LLM_ASSUME_YES"]; ok {
		cfg.AssumeYes = parseBool(v)
	}
	if v, ok := snap["LLM_BACKEND"]; ok {
		cfg.Backend = v
	}
	if v, ok := snap["LLAMA_SERVER"]; ok {
		cfg.LlamaServer = v
	}
	if v, ok := snap["LLM_APP_HOME"]; ok {
		cfg.AppHome = v
	}
	if v, ok := snap["LLM_TUNE_ROUNDS"]; ok {
		cfg.TuneRounds, _ = strconv.Atoi(v)
	}
	if v, ok := snap["LLM_VISION"]; ok {
		cfg.Vision = parseBool(v)
	}
	if v, ok := snap["LLM_PARALLEL"]; ok {
		cfg.Parallel, _ = strconv.Atoi(v)
	}
	if v, ok := snap["LLM_HOST"]; ok {
		cfg.Host = v
	}
	if v, ok := snap["LLM_SPEC"]; ok {
		cfg.Spec = v
	}
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

		switch strings.TrimPrefix(key, "LLM_") {
		case "PORT":
			cfg.Port, _ = strconv.Atoi(val)
		case "CTX_SIZE":
			applyCtxValue(cfg, val)
		case "MAX_RESTARTS":
			cfg.MaxRestarts, _ = strconv.Atoi(val)
		case "KEEP_ALIVE":
			cfg.KeepAlive, _ = strconv.Atoi(val)
		case "HEALTH_TIMEOUT":
			cfg.HealthTimeout, _ = strconv.Atoi(val)
		case "MODEL_DIR":
			cfg.ModelDir = val
		case "CACHE_DIR":
			cfg.CacheDir = val
		case "LOG_DIR":
			cfg.LogDir = val
		case "RAM_BUDGET":
			cfg.RamBudget = val
		case "KV_PLACEMENT":
			cfg.KVPlacement = val
		case "KV_QUALITY":
			cfg.KVQuality = val
		case "ASSUME_YES":
			cfg.AssumeYes = parseBool(val)
		case "BACKEND":
			cfg.Backend = val
		case "LLAMA_SERVER":
			cfg.LlamaServer = val
		case "APP_HOME":
			cfg.AppHome = val
		case "TUNE_ROUNDS":
			cfg.TuneRounds, _ = strconv.Atoi(val)
		case "VISION":
			cfg.Vision = parseBool(val)
		case "PARALLEL":
			cfg.Parallel, _ = strconv.Atoi(val)
		case "HOST":
			cfg.Host = val
		case "SPEC":
			cfg.Spec = val
		}
	}
	return scanner.Err()
}

func applyCtxValue(cfg *Config, val string) {
	val = strings.TrimSpace(strings.Trim(val, `"'`))
	switch strings.ToLower(val) {
	case "", "fit", "auto":
		cfg.CtxMode = "fit"
		cfg.CtxSize = 0
	case "max", "native":
		cfg.CtxMode = "max"
		cfg.CtxSize = 0
	default:
		if n, err := strconv.Atoi(val); err == nil && n > 0 {
			cfg.CtxMode = "manual"
			cfg.CtxSize = n
		}
	}
}

func (c *Config) CtxValue() string {
	switch c.CtxMode {
	case "fit", "auto":
		return "fit"
	case "max", "native":
		return "max"
	}
	if c.CtxSize > 0 {
		return strconv.Itoa(c.CtxSize)
	}
	return "fit"
}

func parseBool(v string) bool {
	v = strings.ToLower(strings.TrimSpace(v))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

// Save writes the config to the canonical config file.
func (c *Config) Save() error {
	path := Path()
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	fmt.Fprintf(f, "# llm-server configuration\n")
	fmt.Fprintf(f, "# Precedence: CLI flag > env var > this file > built-in default\n")
	fmt.Fprintf(f, "LLM_PORT=%d\n", c.Port)
	fmt.Fprintf(f, "LLM_CTX_SIZE=%q\n", c.CtxValue())
	fmt.Fprintf(f, "LLM_MAX_RESTARTS=%d\n", c.MaxRestarts)
	fmt.Fprintf(f, "LLM_KEEP_ALIVE=%d\n", c.KeepAlive)
	fmt.Fprintf(f, "LLM_HEALTH_TIMEOUT=%d\n", c.HealthTimeout)
	fmt.Fprintf(f, "LLM_MODEL_DIR=%q\n", c.ModelDir)
	fmt.Fprintf(f, "LLM_CACHE_DIR=%q\n", c.CacheDir)
	fmt.Fprintf(f, "LLM_LOG_DIR=%q\n", c.LogDir)
	fmt.Fprintf(f, "LLM_RAM_BUDGET=%q\n", c.RamBudget)
	fmt.Fprintf(f, "LLM_KV_PLACEMENT=%q\n", c.KVPlacement)
	fmt.Fprintf(f, "LLM_KV_QUALITY=%q\n", c.KVQuality)
	fmt.Fprintf(f, "LLM_ASSUME_YES=%v\n", c.AssumeYes)
	fmt.Fprintf(f, "LLM_BACKEND=%q\n", c.Backend)
	fmt.Fprintf(f, "LLAMA_SERVER=%q\n", c.LlamaServer)
	fmt.Fprintf(f, "LLM_APP_HOME=%q\n", c.AppHome)
	fmt.Fprintf(f, "LLM_TUNE_ROUNDS=%d\n", c.TuneRounds)
	fmt.Fprintf(f, "LLM_VISION=%v\n", c.Vision)
	fmt.Fprintf(f, "LLM_PARALLEL=%d\n", c.Parallel)
	fmt.Fprintf(f, "LLM_HOST=%q\n", c.Host)
	fmt.Fprintf(f, "LLM_SPEC=%q\n", c.Spec)
	return nil
}

// Show prints the current config with source attribution.
func (c *Config) Show() string {
	var b strings.Builder
	b.WriteString("llm-server configuration\n")
	b.WriteString("═══════════════════════\n\n")
	for _, k := range DefaultKeys {
		var val, source string
		switch k {
		case "PORT":
			val = strconv.Itoa(c.Port)
		case "CTX_SIZE":
			val = c.CtxValue()
		case "MAX_RESTARTS":
			val = strconv.Itoa(c.MaxRestarts)
		case "KEEP_ALIVE":
			val = strconv.Itoa(c.KeepAlive)
		case "HEALTH_TIMEOUT":
			val = strconv.Itoa(c.HealthTimeout)
		case "MODEL_DIR":
			val = c.ModelDir
		case "CACHE_DIR":
			val = c.CacheDir
		case "LOG_DIR":
			val = c.LogDir
		case "RAM_BUDGET":
			val = c.RamBudget
		case "KV_PLACEMENT":
			val = c.KVPlacement
		case "KV_QUALITY":
			val = c.KVQuality
		case "ASSUME_YES":
			val = strconv.FormatBool(c.AssumeYes)
		case "BACKEND":
			val = c.Backend
		case "LLAMA_SERVER":
			val = c.LlamaServer
		case "APP_HOME":
			val = c.AppHome
		case "TUNE_ROUNDS":
			val = strconv.Itoa(c.TuneRounds)
		case "VISION":
			val = strconv.FormatBool(c.Vision)
		case "PARALLEL":
			val = strconv.Itoa(c.Parallel)
		case "HOST":
			val = c.Host
		case "SPEC":
			val = c.Spec
		}
		if val == "" {
			val = "(empty)"
		}
		if os.Getenv("LLM_"+k) != "" || os.Getenv(k) != "" {
			source = "env"
		} else if fileExists(Path()) {
			source = "file"
		} else {
			source = "default"
		}
		b.WriteString(fmt.Sprintf("  %-18s %-20s (%s)\n", k+":", val, source))
	}
	return b.String()
}

// Edit opens the config file in the user's preferred editor.
func Edit() error {
	path := Path()
	if !fileExists(path) {
		cfg := Defaults()
		if err := cfg.Save(); err != nil {
			return err
		}
	}
	editor := os.Getenv("EDITOR")
	if editor == "" {
		editor = os.Getenv("VISUAL")
	}
	if editor == "" {
		if _, err := exec.LookPath("nano"); err == nil {
			editor = "nano"
		} else {
			editor = "vi"
		}
	}
	cmd := exec.Command(editor, path)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// Reset removes the config file (with backup).
func Reset() error {
	path := Path()
	if !fileExists(path) {
		return fmt.Errorf("no config file to reset")
	}
	backup := path + ".bak." + strconv.FormatInt(timeNow().Unix(), 10)
	if err := os.Rename(path, backup); err != nil {
		return err
	}
	return nil
}

// migrateLegacyConfig collapses legacy config.sh into the canonical config file.
func migrateLegacyConfig(canonical string) error {
	if strings.HasSuffix(canonical, ".sh") {
		return nil // already pointing at legacy
	}
	legacy := canonical + ".sh"
	if !fileExists(legacy) {
		return nil
	}
	if !fileExists(canonical) {
		return os.Rename(legacy, canonical)
	}
	// Both exist: merge (legacy values historically won)
	cfg := Defaults()
	loadFile(canonical, cfg)
	loadFile(legacy, cfg) // legacy wins
	if err := cfg.Save(); err != nil {
		return err
	}
	backup := legacy + ".bak." + strconv.FormatInt(timeNow().Unix(), 10)
	os.Rename(legacy, backup)
	return nil
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

var timeNow = func() time.Time { return time.Now() }
