package tune

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Entry holds one tuning attempt result.
type Entry struct {
	Timestamp   int64             `json:"timestamp"`
	ModelPath   string            `json:"model_path"`
	ModelName   string            `json:"model_name"`
	HardwareHash string           `json:"hardware_hash"`
	Vision      bool              `json:"vision"`
	Backend     string            `json:"backend"`
	Round       int               `json:"round"`
	Flags       map[string]string `json:"flags"`
	Result      BenchmarkResult   `json:"result"`
	Best        bool              `json:"best"`
}

// BenchmarkResult mirrors benchmark.Result.
type BenchmarkResult struct {
	PromptTokens int     `json:"prompt_tokens"`
	PromptTPS    float64 `json:"prompt_tps"`
	GenTokens    int     `json:"gen_tokens"`
	GenTPS       float64 `json:"gen_tps"`
	PeakVRAMMB   int     `json:"peak_vram_mb"`
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

	// Mark previous best as non-best if this is better
	for i := range entries {
		if entries[i].ModelPath == entry.ModelPath &&
			entries[i].HardwareHash == entry.HardwareHash &&
			entries[i].Best {
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

// Key returns a cache key string.
func Key(modelPath, modelSize, hwHash, visionSuffix, backend string) string {
	return fmt.Sprintf("%s|%s|%s|%s|%s", modelPath, modelSize, hwHash, visionSuffix, backend)
}

// HardwareHash creates a simple hash from GPU names and total VRAM.
func HardwareHash(gpus []string, totalVRAM int) string {
	return fmt.Sprintf("%x-%d", hashStrings(gpus), totalVRAM)
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
		if err != nil { continue }

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
		Flags:   make(map[string]string),
		Best:    true,
		Round:   1,
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
		if !strings.HasPrefix(line, "CACHED_") { continue }
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 { continue }
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
	if len(e.Flags) == 0 { return nil }
	return e
}
