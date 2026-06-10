package tune

import (
	"crypto/md5"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Entry holds one tuning attempt result.
type Entry struct {
	Timestamp     int64                  `json:"timestamp"`
	ModelPath     string                 `json:"model_path"`
	ModelName     string                 `json:"model_name"`
	HardwareHash  string                 `json:"hardware_hash"`
	Vision        bool                   `json:"vision"`
	Backend       string                 `json:"backend"`
	Round         int                    `json:"round"`
	Name          string                 `json:"name,omitempty"`
	Flags         map[string]string      `json:"flags"`
	OverrideFlags map[string]interface{} `json:"override_flags,omitempty"`
	Status        string                 `json:"status,omitempty"`
	Result        BenchmarkResult        `json:"result"`
	Best          bool                   `json:"best"`
}

// BenchmarkResult mirrors benchmark.Result.
type BenchmarkResult struct {
	PromptTokens    int     `json:"prompt_tokens"`
	PromptTPS       float64 `json:"prompt_tps"`
	GenTokens       int     `json:"gen_tokens"`
	GenTPS          float64 `json:"gen_tps"`
	DraftTokens     int     `json:"draft_tokens,omitempty"`
	DraftAccepted   int     `json:"draft_accepted,omitempty"`
	DraftAcceptRate float64 `json:"draft_accept_rate,omitempty"`
	PeakVRAMMB      int     `json:"peak_vram_mb"`
}

// Cache provides tune result persistence.
type Cache struct {
	path string
}

// NewCache opens the tune cache file.
func NewCache(dir string) *Cache {
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".cache", "llm-server")
	}
	return &Cache{path: filepath.Join(dir, "cache.json")}
}

// Load reads all cached entries.
func (c *Cache) Load() ([]Entry, error) {
	data, err := os.ReadFile(c.path)
	if err != nil {
		if os.IsNotExist(err) {
			return []Entry{}, nil
		}
		return nil, err
	}
	var entries []Entry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, err
	}
	return entries, nil
}

// Save writes entries to disk.
func (c *Cache) Save(entries []Entry) error {
	if err := os.MkdirAll(filepath.Dir(c.path), 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(c.path, data, 0644)
}

// FindBest returns the best entry for given model + hardware hash.
func (c *Cache) FindBest(modelPath, hwHash string) (*Entry, error) {
	entries, err := c.Load()
	if err != nil {
		return nil, err
	}
	var best *Entry
	for i := range entries {
		e := &entries[i]
		if e.ModelPath == modelPath && e.HardwareHash == hwHash && e.Best {
			if best == nil || e.Result.GenTPS > best.Result.GenTPS {
				best = e
			}
		}
	}
	return best, nil
}

// Add appends a new entry and marks previous best as non-best if this is better.
func (c *Cache) Add(entry Entry) error {
	entries, err := c.Load()
	if err != nil {
		return err
	}

	// Mark previous best as non-best if this is better in the same model,
	// hardware, backend, and vision scope. CUDA/IK and Vulkan tunes are not
	// interchangeable even when they run on the same GPUs.
	for i := range entries {
		if sameTuneScope(entries[i], entry) && entries[i].Best {
			if entry.Result.GenTPS > entries[i].Result.GenTPS {
				entries[i].Best = false
			} else {
				entry.Best = false
			}
		}
	}

	entries = append(entries, entry)
	return c.Save(entries)
}

func sameTuneScope(a, b Entry) bool {
	return a.ModelPath == b.ModelPath &&
		a.HardwareHash == b.HardwareHash &&
		a.Backend == b.Backend &&
		a.Vision == b.Vision
}

// TuneFileSummary is the loadable public tune artifact used by the CLI and GUI.
type TuneFileSummary struct {
	Name           string
	GenTPS         float64
	PPTPS          float64
	BaselineGenTPS float64
	BaselineWins   bool
	Flags          map[string]interface{}
}

// SaveTuneFile writes the Bash-compatible tune_<model>_<size>_hw<hash>_<backend>.json file.
func (c *Cache) SaveTuneFile(modelPath string, baseline, best *Entry, rounds int, backend string, vision bool, minImprovementPct float64, gpuNames []string, entries []Entry, complete bool) (string, error) {
	if c == nil {
		return "", nil
	}
	if baseline == nil {
		return "", fmt.Errorf("missing baseline entry")
	}
	if best == nil {
		best = baseline
	}
	path := TuneCachePath(filepath.Dir(c.path), modelPath, gpuNames, vision, backend)
	if path == "" {
		return "", fmt.Errorf("could not build tune cache path")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return "", err
	}

	if minImprovementPct <= 0 {
		minImprovementPct = 1.0
	}
	baselineWins := best.Round == baseline.Round || !meaningfulImprovement(best.Result.GenTPS, baseline.Result.GenTPS, minImprovementPct) || len(best.OverrideFlags) == 0
	bestName := best.Name
	if bestName == "" {
		bestName = "baseline"
	}
	bestFlags := map[string]interface{}{}
	if !baselineWins {
		for k, v := range best.OverrideFlags {
			bestFlags[k] = v
		}
	}
	allResults := make([]map[string]interface{}, 0, len(entries))
	for _, e := range entries {
		flags := e.OverrideFlags
		if flags == nil {
			flags = map[string]interface{}{}
		}
		status := e.Status
		if status == "" {
			status = "ok"
		}
		name := e.Name
		if name == "" && e.Round == 0 {
			name = "baseline"
		}
		allResults = append(allResults, map[string]interface{}{
			"round":   e.Round,
			"name":    name,
			"gen_tps": e.Result.GenTPS,
			"pp_tps":  e.Result.PromptTPS,
			"flags":   flags,
			"status":  status,
		})
	}

	completedRounds := 0
	for _, e := range entries {
		if e.Round > completedRounds {
			completedRounds = e.Round
		}
	}
	if complete && completedRounds < rounds {
		completedRounds = rounds
	}

	doc := map[string]interface{}{
		"model":               filepath.Base(modelPath),
		"tuned_at":            time.Now().UTC().Format(time.RFC3339),
		"provider":            "v3-go",
		"backend":             backendTagForFile(backend),
		"baseline_gen_tps":    baseline.Result.GenTPS,
		"baseline_pp_tps":     baseline.Result.PromptTPS,
		"baseline_wins":       baselineWins,
		"min_improvement_pct": minImprovementPct,
		"completed_rounds":    completedRounds,
		"complete":            complete,
		"best_config": map[string]interface{}{
			"name":    bestName,
			"flags":   bestFlags,
			"gen_tps": best.Result.GenTPS,
			"pp_tps":  best.Result.PromptTPS,
		},
		"rounds":      rounds,
		"all_results": allResults,
	}
	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return "", err
	}
	return path, os.WriteFile(path, data, 0644)
}

// TuneCachePath builds the public tune artifact path.
func TuneCachePath(dir, modelPath string, gpuNames []string, vision bool, backend string) string {
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".cache", "llm-server")
	}
	size, err := modelCacheSize(modelPath)
	if err != nil {
		return ""
	}
	visionSuffix := ""
	if vision {
		visionSuffix = "_v"
	}
	return filepath.Join(dir, fmt.Sprintf("tune_%s_%d_hw%s%s_%s.json",
		filepath.Base(modelPath), size, BashHardwareHash(gpuNames), visionSuffix, backendTagForFile(backend)))
}

func modelCacheSize(modelPath string) (int64, error) {
	st, err := os.Stat(modelPath)
	if err != nil {
		return 0, err
	}
	size := st.Size()
	base := filepath.Base(modelPath)
	stem := strings.TrimSuffix(base, ".gguf")
	marker := "-00001-of-"
	idx := strings.LastIndex(stem, marker)
	if idx < 0 {
		return size, nil
	}
	n, err := strconv.Atoi(stem[idx+len(marker):])
	if err != nil || n <= 1 {
		return size, nil
	}
	prefix := stem[:idx]
	dir := filepath.Dir(modelPath)
	var total int64
	for i := 1; i <= n; i++ {
		shard := filepath.Join(dir, fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, i, n))
		info, err := os.Stat(shard)
		if err != nil {
			return size, nil
		}
		total += info.Size()
	}
	if total > 0 {
		return total, nil
	}
	return size, nil
}

// LoadTuneFile loads the public tune artifact and validates the target model name.
func LoadTuneFile(path, modelName string) (*TuneFileSummary, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Model          string  `json:"model"`
		BaselineGenTPS float64 `json:"baseline_gen_tps"`
		BaselineWins   bool    `json:"baseline_wins"`
		BestConfig     struct {
			Name   string                 `json:"name"`
			Flags  map[string]interface{} `json:"flags"`
			GenTPS float64                `json:"gen_tps"`
			PPTPS  float64                `json:"pp_tps"`
		} `json:"best_config"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, err
	}
	if modelName != "" && doc.Model != "" && doc.Model != modelName {
		return nil, fmt.Errorf("tune cache is for %s, not %s", doc.Model, modelName)
	}
	flags := doc.BestConfig.Flags
	if flags == nil {
		flags = map[string]interface{}{}
	}
	return &TuneFileSummary{
		Name:           doc.BestConfig.Name,
		GenTPS:         doc.BestConfig.GenTPS,
		PPTPS:          doc.BestConfig.PPTPS,
		BaselineGenTPS: doc.BaselineGenTPS,
		BaselineWins:   doc.BaselineWins,
		Flags:          flags,
	}, nil
}

func backendTagForFile(backend string) string {
	b := strings.ToLower(strings.TrimSpace(backend))
	switch {
	case strings.Contains(b, "vulkan"):
		return "vulkan"
	case strings.Contains(b, "ik"):
		return "ik"
	case b == "":
		return "llama"
	default:
		return b
	}
}

// Key returns a cache key string.
func Key(modelPath, modelSize, hwHash, visionSuffix, backend string) string {
	return fmt.Sprintf("%s|%s|%s|%s|%s", modelPath, modelSize, hwHash, visionSuffix, backend)
}

// HardwareHash creates a simple hash from GPU names and total VRAM.
func HardwareHash(gpus []string, totalVRAM int) string {
	return fmt.Sprintf("%x-%d", hashStrings(gpus), totalVRAM)
}

// BashHardwareHash creates the filename hash used by the Bash tune cache files.
func BashHardwareHash(gpus []string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%d_", len(gpus))
	for _, name := range gpus {
		b.WriteString(name)
		b.WriteByte('_')
	}
	b.WriteByte('\n')
	sum := md5.Sum([]byte(b.String()))
	return fmt.Sprintf("%x", sum)[:8]
}

func hashStrings(ss []string) uint32 {
	var h uint32 = 5381
	for _, s := range ss {
		for i := 0; i < len(s); i++ {
			h = ((h << 5) + h) + uint32(s[i])
		}
	}
	return h
}

// Now returns the current Unix timestamp.
func Now() int64 { return time.Now().Unix() }

// LoadBashCache reads bash-format .conf files from the cache directory.
// Converts CACHED_GPU_ASSIGNMENTS, CACHED_MMAP, etc. into Go Entry format
// for backward compatibility with existing bash-tuned configs.
func (c *Cache) LoadBashCache() ([]Entry, error) {
	dir := filepath.Dir(c.path)
	pattern := filepath.Join(dir, "*.conf")
	files, err := filepath.Glob(pattern)
	if err != nil || len(files) == 0 {
		return nil, fmt.Errorf("no bash config files in %s", dir)
	}

	var entries []Entry
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}

		e := parseConfFile(string(data))
		if e != nil {
			entries = append(entries, *e)
		}
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("no valid bash tunes found")
	}
	return entries, nil
}

func parseConfFile(content string) *Entry {
	e := &Entry{
		Flags: make(map[string]string),
		Best:  true,
		Round: 1,
	}
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# Generated:") {
			parts := strings.Split(line, ": ")
			if len(parts) >= 2 {
				if t, err := time.Parse("Mon Jan  2 03:04:05 PM MST 2006", parts[1]); err == nil {
					e.Timestamp = t.Unix()
				}
			}
		}
		if !strings.HasPrefix(line, "CACHED_") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.Trim(strings.TrimSpace(parts[1]), "\"")
		switch key {
		case "CACHED_GPU_ASSIGNMENTS":
			e.Flags["gpu_assignments"] = val
		case "CACHED_MMAP":
			e.Flags["mmap"] = val
		case "CACHED_BATCH":
			e.Flags["batch"] = val
		case "CACHED_UBATCH":
			e.Flags["ubatch"] = val
		case "CACHED_PARALLEL":
			e.Flags["parallel"] = val
		case "CACHED_NCPUMOE":
			e.Flags["n_cpu_moe"] = val
		case "CACHED_KVUNIFIED":
			e.Flags["kv_unified"] = val
		case "CACHED_NO_PINNED":
			e.Flags["no_pinned"] = val
		default:
			e.Flags[strings.TrimPrefix(key, "CACHED_")] = val
		}
	}
	if len(e.Flags) == 0 {
		return nil
	}
	return e
}
