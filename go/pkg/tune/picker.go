package tune

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ConfigEntry represents a cached tuned config for display/picking.
type ConfigEntry struct {
	Path   string
	Label  string
	GenTPS float64
}

// ListTunedConfigs returns tuned configs matching the model, backend, and vision flag,
// sorted by generation tok/s descending.
func ListTunedConfigs(cacheDir, modelName, backendTag string, wantVision bool) []ConfigEntry {
	pattern := filepath.Join(cacheDir, fmt.Sprintf("tune_%s_*.json", modelName))
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil
	}

	var entries []ConfigEntry
	for _, path := range matches {
		name := filepath.Base(path)
		if cacheBackend(name) != backendTag {
			continue
		}
		if isVisionCache(name) != wantVision {
			continue
		}
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var doc struct {
			Model          string                 `json:"model"`
			BaselineGenTPS float64                `json:"baseline_gen_tps"`
			BestConfig     struct {
				GenTPS float64                `json:"gen_tps"`
				Flags  map[string]interface{} `json:"flags"`
			} `json:"best_config"`
			Rounds   int    `json:"rounds"`
			TunedAt  string `json:"tuned_at"`
		}
		if err := json.Unmarshal(data, &doc); err != nil {
			continue
		}
		if doc.Model != modelName {
			continue
		}
		best := doc.BestConfig
		if best.GenTPS <= 0 {
			continue
		}
		gen := best.GenTPS
		base := doc.BaselineGenTPS
		var gain float64
		if base > 0 {
			gain = (gen - base) / base * 100.0
		}
		flags := best.Flags
		mode := modeFor(name, flags)
		kv := "base-kv"
		if v, ok := flags["--cache-type-k"].(string); ok && v != "" {
			kv = v
		} else if v, ok := flags["--cache-type-v"].(string); ok && v != "" {
			kv = v
		}
		age := ageLabel(doc.TunedAt)
		label := fmt.Sprintf("%.2f tok/s (%+.1f%%) | %s | %s | %s | %d rounds | %s",
			gen, gain, mode, backendTag, kv, doc.Rounds, age)
		entries = append(entries, ConfigEntry{
			Path:   path,
			Label:  label,
			GenTPS: gen,
		})
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].GenTPS > entries[j].GenTPS
	})
	return entries
}

// CountTunedConfigs returns the number of tuned configs for a model.
func CountTunedConfigs(cacheDir, modelName, backendTag string) int {
	return len(ListTunedConfigs(cacheDir, modelName, backendTag, false)) +
		len(ListTunedConfigs(cacheDir, modelName, backendTag, true))
}

func cacheBackend(name string) string {
	stem := name
	if strings.HasSuffix(name, ".json") {
		stem = name[:len(name)-5]
	}
	if strings.Contains(stem, "_vulkan") {
		return "vulkan"
	}
	if strings.Contains(stem, "_ik") {
		return "ik"
	}
	if strings.Contains(stem, "_llama") {
		return "llama"
	}
	return "llama"
}

func isVisionCache(name string) bool {
	return strings.Contains(name, "_v_")
}

func modeFor(name string, flags map[string]interface{}) string {
	placementKeys := []string{"--tensor-split", "--split-mode", "--device", "--main-gpu",
		"--mg", "-mg", "--ngl", "-ngl", "--n-gpu-layers"}
	if strings.Contains(name, "_unlimited") {
		return "perf+placement"
	}
	for _, k := range placementKeys {
		if _, ok := flags[k]; ok {
			return "perf+placement"
		}
	}
	return "perf"
}

func ageLabel(ts string) string {
	t, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		return "unknown"
	}
	return t.UTC().Format("2006-01-02")
}
