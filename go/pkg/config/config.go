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

	"github.com/raketenkater/ggrun/pkg/backends"
)

// Config holds ggrun settings with precedence:
// CLI flag > env var > config file > built-in default.
// This is the single source of truth for all user-tunable settings.
type Config struct {
	Port            int    `json:"port"`
	Ctx             string `json:"ctx_size"` // fit, max, or a positive token count
	MaxRestarts     int    `json:"max_restarts"`
	KeepAlive       int    `json:"keep_alive"`
	HealthTimeout   int    `json:"health_timeout"`
	ModelDir        string `json:"model_dir"`
	CacheDir        string `json:"cache_dir"`
	LogDir          string `json:"log_dir"`
	RamBudget       string `json:"ram_budget"`
	RAMLimitPercent int    `json:"ram_limit_percent"` // maximum whole-host RAM use, 1-100
	VRAMHeadroom    string `json:"vram_headroom"`     // VRAM to hold back, e.g. "2G"
	RAMHeadroom     string `json:"ram_headroom"`      // system RAM to hold back, e.g. "8G"
	KVPlacement     string `json:"kv_placement"`
	KVQuality       string `json:"kv_quality"`
	AssumeYes       bool   `json:"assume_yes"`
	Backend         string `json:"backend"`
	LlamaServer     string `json:"llama_server"`
	AppHome         string `json:"app_home"`
	TuneRounds      int    `json:"tune_rounds"`
	Vision          bool   `json:"vision"`
	Parallel        int    `json:"parallel"`
	Host            string `json:"host"`
	Spec            string `json:"spec"` // off, auto, draft, eagle3, ngram, ngram-mod, ngram-k4v, mtp

	// sources is populated by Load and intentionally not serialized. Keeping
	// provenance next to the merged value lets `config show` report the source
	// of each individual setting rather than guessing from file/env existence.
	sources map[string]string
}

// DefaultKeys is the stable display order for config show / template generation.
var DefaultKeys = []string{
	"PORT", "CTX_SIZE", "MAX_RESTARTS", "KEEP_ALIVE", "HEALTH_TIMEOUT",
	"MODEL_DIR", "CACHE_DIR", "LOG_DIR",
	"RAM_BUDGET", "RAM_LIMIT_PERCENT", "VRAM_HEADROOM", "RAM_HEADROOM", "KV_PLACEMENT", "KV_QUALITY",
	"ASSUME_YES",
	"BACKEND", "LLAMA_SERVER", "APP_HOME",
	"TUNE_ROUNDS", "VISION", "PARALLEL", "HOST", "SPEC",
}

// Defaults returns the built-in defaults.
func Defaults() *Config {
	home, _ := os.UserHomeDir()
	return &Config{
		Port:            8081,
		Ctx:             "fit",
		MaxRestarts:     5,
		KeepAlive:       0,
		HealthTimeout:   0, // auto
		ModelDir:        filepath.Join(home, "ai_models"),
		CacheDir:        filepath.Join(home, ".cache", "ggrun"),
		LogDir:          "",
		RamBudget:       "",
		RAMLimitPercent: 95,
		VRAMHeadroom:    "",
		RAMHeadroom:     "",
		KVPlacement:     "auto",
		KVQuality:       "auto", // model-aware default: generic models use q8_0; stricter architectures may require f16
		AssumeYes:       false,
		Backend:         "",
		LlamaServer:     "",
		AppHome:         "",
		TuneRounds:      8,
		Vision:          false,
		Parallel:        1,
		Host:            "127.0.0.1",
		Spec:            "off",
		sources:         defaultSources(),
	}
}

// ParseBudgetMB parses a memory budget like "2G", "2048M", or "2048" into MB.
// Returns 0 for empty or unparseable input.
func ParseBudgetMB(s string) int {
	mb, err := ParseBudgetMBStrict(s)
	if err != nil {
		return 0
	}
	return mb
}

// ParseBudgetMBStrict parses a non-negative memory budget like "2G", "2048M",
// or "2048" into MB. Empty means no reservation. It rejects malformed and
// negative values so launch-time safety reservations cannot disappear silently.
func ParseBudgetMBStrict(s string) (int, error) {
	s = strings.TrimSpace(strings.ToUpper(s))
	if s == "" {
		return 0, nil
	}
	mult := 1
	if strings.HasSuffix(s, "G") || strings.HasSuffix(s, "GB") {
		mult = 1024
		s = strings.TrimSuffix(strings.TrimSuffix(s, "GB"), "G")
	} else if strings.HasSuffix(s, "M") || strings.HasSuffix(s, "MB") {
		s = strings.TrimSuffix(strings.TrimSuffix(s, "MB"), "M")
	}
	n, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil || n < 0 {
		return 0, fmt.Errorf("must be a non-negative memory amount such as 2G or 2048M")
	}
	if n > int(^uint(0)>>1)/mult {
		return 0, fmt.Errorf("memory amount is too large")
	}
	return n * mult, nil
}

// ParsePort validates a TCP port used by the local serving/control endpoints.
func ParsePort(s string) (int, error) {
	n, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil || n < 1 || n > 65535 {
		return 0, fmt.Errorf("must be an integer from 1 to 65535")
	}
	return n, nil
}

// ParseRAMLimitPercent validates the maximum whole-host RAM utilisation.
func ParseRAMLimitPercent(s string) (int, error) {
	n, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil || n < 1 || n > 100 {
		return 0, fmt.Errorf("must be an integer from 1 to 100")
	}
	return n, nil
}

func defaultSources() map[string]string {
	sources := make(map[string]string, len(DefaultKeys))
	for _, key := range DefaultKeys {
		sources[key] = "default"
	}
	return sources
}

// Path returns the canonical config file path. It resolves the app home via
// backends.AppHome() so the config file lives alongside backends.json
// (AppHome()/.config/) instead of diverging to a separate $HOME/.config/ggrun
// location.
func Path() string {
	if p := os.Getenv("LLM_CONFIG"); p != "" {
		return p
	}
	home := backends.AppHome()
	isGenericHome := home == os.Getenv("HOME")
	if !isGenericHome {
		if f := filepath.Join(home, ".config", "config"); fileExists(f) {
			return f
		}
		if f := filepath.Join(home, "config", "config"); fileExists(f) {
			return f
		}
	}
	if f := filepath.Join(home, ".config", "ggrun", "config"); fileExists(f) {
		return f
	}
	return filepath.Join(home, ".config", "ggrun", "config")
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
		return nil, fmt.Errorf("migrate config: %w", err)
	}

	if fileExists(cfgPath) {
		if err := loadFile(cfgPath, cfg); err != nil {
			return nil, fmt.Errorf("load config: %w", err)
		}
	}

	// Re-apply env snapshot so env wins over file
	if err := applyEnvSnapshot(cfg, envSnapshot); err != nil {
		return nil, fmt.Errorf("read environment: %w", err)
	}

	// Fill remaining unset keys from defaults (already in cfg)
	return cfg, nil
}

func snapshotEnv() map[string]string {
	m := make(map[string]string)
	for _, k := range []string{
		"LLM_PORT", "LLM_CTX_SIZE", "LLM_MAX_RESTARTS", "LLM_KEEP_ALIVE",
		"LLM_HEALTH_TIMEOUT", "LLM_MODEL_DIR", "LLM_CACHE_DIR", "LLM_LOG_DIR",
		"LLM_RAM_BUDGET", "LLM_RAM_LIMIT_PERCENT", "LLM_VRAM_HEADROOM", "LLM_RAM_HEADROOM", "LLM_KV_PLACEMENT", "LLM_KV_QUALITY", "LLM_ASSUME_YES",
		"LLM_BACKEND", "LLAMA_SERVER", "LLM_APP_HOME", "LLM_TUNE_ROUNDS",
		"LLM_VISION", "LLM_PARALLEL", "LLM_HOST", "LLM_SPEC",
	} {
		if v := os.Getenv(k); v != "" {
			m[k] = v
		}
	}
	return m
}

func applyEnvSnapshot(cfg *Config, snap map[string]string) error {
	for key, val := range snap {
		if err := setConfigValue(cfg, key, val, "env"); err != nil {
			return fmt.Errorf("%s: %w", key, err)
		}
	}
	return nil
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
		if err := setConfigValue(cfg, key, parts[1], "file"); err != nil {
			return fmt.Errorf("%s: %w", key, err)
		}
	}
	return scanner.Err()
}

func setConfigValue(cfg *Config, key, raw, source string) error {
	key = strings.TrimPrefix(strings.TrimSpace(key), "LLM_")
	val := strings.Trim(strings.TrimSpace(raw), `"'`)
	if cfg.sources == nil {
		cfg.sources = defaultSources()
	}
	mark := func() { cfg.sources[key] = source }
	switch key {
	case "PORT":
		n, err := ParsePort(val)
		if err != nil {
			return err
		}
		cfg.Port = n
	case "CTX_SIZE":
		if err := cfg.SetCtxValue(val); err != nil {
			return err
		}
	case "MAX_RESTARTS":
		n, err := parseNonNegativeInt(val)
		if err != nil {
			return err
		}
		cfg.MaxRestarts = n
	case "KEEP_ALIVE":
		n, err := parseNonNegativeInt(val)
		if err != nil {
			return err
		}
		cfg.KeepAlive = n
	case "HEALTH_TIMEOUT":
		n, err := parseNonNegativeInt(val)
		if err != nil {
			return err
		}
		cfg.HealthTimeout = n
	case "MODEL_DIR":
		cfg.ModelDir = val
	case "CACHE_DIR":
		cfg.CacheDir = val
	case "LOG_DIR":
		cfg.LogDir = val
	case "RAM_BUDGET":
		if _, err := ParseBudgetMBStrict(val); err != nil {
			return err
		}
		cfg.RamBudget = val
	case "RAM_LIMIT_PERCENT":
		n, err := ParseRAMLimitPercent(val)
		if err != nil {
			return err
		}
		cfg.RAMLimitPercent = n
	case "VRAM_HEADROOM":
		if _, err := ParseBudgetMBStrict(val); err != nil {
			return err
		}
		cfg.VRAMHeadroom = val
	case "RAM_HEADROOM":
		if _, err := ParseBudgetMBStrict(val); err != nil {
			return err
		}
		cfg.RAMHeadroom = val
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
		n, err := parseNonNegativeInt(val)
		if err != nil {
			return err
		}
		cfg.TuneRounds = n
	case "VISION":
		cfg.Vision = parseBool(val)
	case "PARALLEL":
		n, err := parseNonNegativeInt(val)
		if err != nil {
			return err
		}
		cfg.Parallel = n
	case "HOST":
		cfg.Host = val
	case "SPEC":
		cfg.Spec = val
	default:
		return nil
	}
	mark()
	return nil
}

func parseNonNegativeInt(val string) (int, error) {
	n, err := strconv.Atoi(strings.TrimSpace(val))
	if err != nil || n < 0 {
		return 0, fmt.Errorf("must be a non-negative integer")
	}
	return n, nil
}

// SetCtxValue updates the one canonical context representation.
func (c *Config) SetCtxValue(val string) error {
	normalized, err := normalizeCtxValue(val)
	if err != nil {
		return err
	}
	c.Ctx = normalized
	return nil
}

func (c *Config) CtxValue() string {
	if c == nil {
		return "fit"
	}
	value, err := normalizeCtxValue(c.Ctx)
	if err != nil {
		return "fit"
	}
	return value
}

// CtxMode derives the UI mode from the canonical context value.
func (c *Config) CtxMode() string {
	switch c.CtxValue() {
	case "fit":
		return "fit"
	case "max":
		return "max"
	default:
		return "manual"
	}
}

func normalizeCtxValue(val string) (string, error) {
	val = strings.TrimSpace(strings.Trim(val, `"'`))
	switch strings.ToLower(val) {
	case "", "fit", "auto":
		return "fit", nil
	case "max", "native":
		return "max", nil
	default:
		n, err := strconv.Atoi(val)
		if err != nil || n <= 0 {
			return "", fmt.Errorf("context must be fit, max, or a positive token count")
		}
		return strconv.Itoa(n), nil
	}
}

func parseBool(v string) bool {
	v = strings.ToLower(strings.TrimSpace(v))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

// Save writes the config to the canonical config file.
func (c *Config) Save() error {
	if _, err := ParsePort(strconv.Itoa(c.Port)); err != nil {
		return fmt.Errorf("PORT: %w", err)
	}
	ctx, err := normalizeCtxValue(c.Ctx)
	if err != nil {
		return fmt.Errorf("CTX_SIZE: %w", err)
	}
	c.Ctx = ctx
	for _, budget := range []struct {
		key, value string
	}{
		{"RAM_BUDGET", c.RamBudget},
		{"VRAM_HEADROOM", c.VRAMHeadroom},
		{"RAM_HEADROOM", c.RAMHeadroom},
	} {
		if _, err := ParseBudgetMBStrict(budget.value); err != nil {
			return fmt.Errorf("%s: %w", budget.key, err)
		}
	}
	for _, numeric := range []struct {
		key string
		val int
	}{
		{"MAX_RESTARTS", c.MaxRestarts}, {"KEEP_ALIVE", c.KeepAlive},
		{"HEALTH_TIMEOUT", c.HealthTimeout}, {"TUNE_ROUNDS", c.TuneRounds},
		{"PARALLEL", c.Parallel},
	} {
		if numeric.val < 0 {
			return fmt.Errorf("%s: must be a non-negative integer", numeric.key)
		}
	}
	if _, err := ParseRAMLimitPercent(strconv.Itoa(c.RAMLimitPercent)); err != nil {
		return fmt.Errorf("RAM_LIMIT_PERCENT: %w", err)
	}
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

	fmt.Fprintf(f, "# ggrun configuration\n")
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
	fmt.Fprintf(f, "LLM_RAM_LIMIT_PERCENT=%d\n", c.RAMLimitPercent)
	fmt.Fprintf(f, "LLM_VRAM_HEADROOM=%q\n", c.VRAMHeadroom)
	fmt.Fprintf(f, "LLM_RAM_HEADROOM=%q\n", c.RAMHeadroom)
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
	b.WriteString("ggrun configuration\n")
	b.WriteString("═══════════════════════\n\n")
	for _, k := range DefaultKeys {
		var val string
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
		case "RAM_LIMIT_PERCENT":
			val = strconv.Itoa(c.RAMLimitPercent)
		case "VRAM_HEADROOM":
			val = c.VRAMHeadroom
		case "RAM_HEADROOM":
			val = c.RAMHeadroom
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
		source := "default"
		if c.sources != nil && c.sources[k] != "" {
			source = c.sources[k]
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
	if err := cmd.Run(); err != nil {
		return err
	}
	if _, err := Load(); err != nil {
		return fmt.Errorf("configuration saved but is invalid: %w", err)
	}
	return nil
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
	if err := loadFile(canonical, cfg); err != nil {
		return fmt.Errorf("read canonical config: %w", err)
	}
	if err := loadFile(legacy, cfg); err != nil {
		return fmt.Errorf("read legacy config: %w", err)
	}
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
