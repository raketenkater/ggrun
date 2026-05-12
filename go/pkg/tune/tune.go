package tune

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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
