package probe

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/gguf"
)

// ContextEstimate holds max and fit context sizes for a model.
type ContextEstimate struct {
	MaxCtx int // ContextLength from GGUF metadata
	FitCtx int // Largest empirically proven ctx from probe cache
}

// DetectMaxCtx returns the model's trained max context from GGUF metadata.
func DetectMaxCtx(modelPath string) int {
	info, err := gguf.Parse(modelPath)
	if err != nil {
		return 0
	}
	return info.ContextLength
}

// EstimateFitCtx returns the largest context size empirically proven to work
// for this model on this hardware, based on probe cache entries.
// It only returns values from successful loads — no extrapolation.
func EstimateFitCtx(modelPath, cacheDir string) int {
	info, err := gguf.Parse(modelPath)
	if err != nil {
		return 0
	}
	return EstimateFitCtxForInfo(modelPath, cacheDir, info, totalSystemMemoryMB())
}

// EstimateFitCtxForInfo is the metadata-aware form used by model browsers.
// Callers that already parsed a GGUF and detected hardware can avoid launching
// the parser and hardware probes again for every model in the directory.
func EstimateFitCtxForInfo(modelPath, cacheDir string, info *gguf.Info, totalSysMemMB int) int {
	if info == nil {
		return 0
	}
	layerCount := info.BlockCount
	if layerCount <= 0 {
		return 0
	}

	probeDir := filepath.Join(cacheDir, "probes")
	entries, err := os.ReadDir(probeDir)
	if err != nil {
		return 0
	}

	modelName := filepath.Base(modelPath)
	// Handle multi-part: strip shard suffix
	if idx := strings.Index(modelName, "-00001-of-"); idx > 0 {
		modelName = modelName[:idx] + ".gguf"
	}

	maxValidCtx := 0
	ctxRe := regexp.MustCompile(`ctx=(\d+)`)
	kvRe := regexp.MustCompile(`PROBED_KV_PER_LAYER_MB=(\d+)`)

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".probe") {
			continue
		}
		path := filepath.Join(probeDir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		content := string(data)
		// Only consider probes that mention this model
		if !strings.Contains(content, modelName) {
			continue
		}

		ctxMatch := ctxRe.FindStringSubmatch(content)
		kvMatch := kvRe.FindStringSubmatch(content)
		if len(ctxMatch) < 2 || len(kvMatch) < 2 {
			continue
		}
		probeCtx, _ := strconv.Atoi(ctxMatch[1])
		probeKVLayer, _ := strconv.Atoi(kvMatch[1])
		if probeCtx <= 0 || probeKVLayer <= 0 {
			continue
		}

		// Reject corrupted probes: implied total KV must fit in system memory
		if totalSysMemMB > 0 {
			impliedKVTotal := probeKVLayer * layerCount
			if impliedKVTotal > totalSysMemMB {
				continue
			}
		}

		if probeCtx > maxValidCtx {
			maxValidCtx = probeCtx
		}
	}

	// Cap at trained max context
	if info.ContextLength > 0 && maxValidCtx > info.ContextLength {
		maxValidCtx = info.ContextLength
	}

	return maxValidCtx
}

func totalSystemMemoryMB() int {
	caps, err := detect.Detect()
	if err != nil || caps == nil {
		return 0
	}
	return caps.TotalVRAM() + caps.RAM.TotalMB
}
