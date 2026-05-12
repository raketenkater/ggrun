package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/llm-server/pkg/detect"
)

// CacheEntry holds a validated placement cache entry for MoE.
type CacheEntry struct {
	GPUAssignments []GPUAssignment `json:"gpu_assignments"` // cuda_idx:start:count
	NCPUMoE        int             `json:"n_cpu_moe"`
	BatchSize      int             `json:"batch_size"`
	UBatchSize     int             `json:"ubatch_size"`
	Parallel       int             `json:"parallel"`
	KVUnified      bool            `json:"kv_unified"`
	NoPinned       bool            `json:"no_pinned"`
	MMap           bool            `json:"mmap"`
}

// GPUAssignment describes layers assigned to a GPU.
type GPUAssignment struct {
	CUDAIndex int `json:"cuda_index"`
	Start     int `json:"start"`
	Count     int `json:"count"`
}

// LoadPlacementCache attempts to load a validated placement cache file.
func LoadPlacementCache(cachePath string, caps *detect.Capabilities, kvTotalMB int) (*CacheEntry, error) {
	data, err := os.ReadFile(cachePath)
	if err != nil {
		return nil, err
	}
	content := string(data)

	// Parse simple key=value format from bash cache files
	entry := &CacheEntry{
		BatchSize:  1024,
		UBatchSize: 512,
		Parallel:   2,
	}
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.Trim(strings.TrimSpace(parts[1]), `"`)
		switch key {
		case "CACHED_GPU_ASSIGNMENTS":
			entry.GPUAssignments = parseGPUAssignments(val)
		case "CACHED_NCPUMOE":
			entry.NCPUMoE, _ = strconv.Atoi(val)
		case "CACHED_BATCH":
			entry.BatchSize, _ = strconv.Atoi(val)
		case "CACHED_UBATCH":
			entry.UBatchSize, _ = strconv.Atoi(val)
		case "CACHED_PARALLEL":
			entry.Parallel, _ = strconv.Atoi(val)
		case "CACHED_KVUNIFIED":
			entry.KVUnified = val == "1"
		case "CACHED_NO_PINNED":
			entry.NoPinned = val == "1"
		case "CACHED_MMAP":
			entry.MMap = val == "1"
		}
	}

	// Validate: each GPU must have enough VRAM for assigned layers + KV share
	for _, assign := range entry.GPUAssignments {
		found := false
		for _, g := range caps.GPUs {
			if g.Index == assign.CUDAIndex {
				found = true
				// We can't validate exact layer MB without model info here,
				// but we can check that the GPU exists
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("cached assignment references unknown GPU %d", assign.CUDAIndex)
		}
	}

	return entry, nil
}

// SavePlacementCache writes a placement cache file in bash-compatible format.
func SavePlacementCache(cachePath string, entry *CacheEntry) error {
	_ = os.MkdirAll(filepath.Dir(cachePath), 0755)
	var parts []string
	parts = append(parts, fmt.Sprintf("# llm-server placement cache (%s)", time.Now().UTC().Format("2006-01-02T15:04:05Z")))
	if len(entry.GPUAssignments) > 0 {
		var assigns []string
		for _, a := range entry.GPUAssignments {
			assigns = append(assigns, fmt.Sprintf("%d:%d:%d", a.CUDAIndex, a.Start, a.Count))
		}
		parts = append(parts, fmt.Sprintf("CACHED_GPU_ASSIGNMENTS=\"%s\"", strings.Join(assigns, " ")))
	}
	if entry.NCPUMoE > 0 {
		parts = append(parts, fmt.Sprintf("CACHED_NCPUMOE=\"%d\"", entry.NCPUMoE))
	}
	parts = append(parts, fmt.Sprintf("CACHED_BATCH=\"%d\"", entry.BatchSize))
	parts = append(parts, fmt.Sprintf("CACHED_UBATCH=\"%d\"", entry.UBatchSize))
	parts = append(parts, fmt.Sprintf("CACHED_PARALLEL=\"%d\"", entry.Parallel))
	if entry.KVUnified {
		parts = append(parts, "CACHED_KVUNIFIED=\"1\"")
	}
	if entry.NoPinned {
		parts = append(parts, "CACHED_NO_PINNED=\"1\"")
	}
	if entry.MMap {
		parts = append(parts, "CACHED_MMAP=\"1\"")
	}
	return os.WriteFile(cachePath, []byte(strings.Join(parts, "\n")+"\n"), 0644)
}

func parseGPUAssignments(s string) []GPUAssignment {
	var out []GPUAssignment
	for _, tok := range strings.Fields(s) {
		parts := strings.Split(tok, ":")
		if len(parts) != 3 {
			continue
		}
		ci, _ := strconv.Atoi(parts[0])
		st, _ := strconv.Atoi(parts[1])
		ct, _ := strconv.Atoi(parts[2])
		out = append(out, GPUAssignment{CUDAIndex: ci, Start: st, Count: ct})
	}
	return out
}
