package tune

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// DefaultCommunityBaseURL serves community-shared tune files, named exactly
// like the local cache files (tune_<model>_<size>_hw<hash>_<backend>.json),
// so a lookup is a single GET keyed by model+hardware+backend.
const DefaultCommunityBaseURL = "https://raw.githubusercontent.com/raketenkater/ggrun-tunes/main/tunes"

// CommunityRepoURL is where users share their own tuned configs.
const CommunityRepoURL = "https://github.com/raketenkater/ggrun-tunes"

const (
	communityHitTTL  = 7 * 24 * time.Hour
	communityMissTTL = 24 * time.Hour
	communityMaxSize = 1 << 20 // 1 MiB — tune files are a few KiB
)

// FetchCommunityTune looks up a community-shared tuned config for this exact
// model + GPU set + backend. It returns a local path to a sanitized copy, or
// "" when none is available (offline, disabled, 404, invalid). Results and
// misses are cached on disk so launches stay fast and offline-safe.
func FetchCommunityTune(cacheDir, modelPath string, gpuNames []string, vision bool, backend string) string {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("LLM_COMMUNITY_TUNES"))) {
	case "off", "0", "false", "no":
		return ""
	}
	if cacheDir == "" || modelPath == "" {
		return ""
	}
	localPath := TuneCachePath(cacheDir, modelPath, gpuNames, vision, backend)
	if localPath == "" {
		return ""
	}
	name := filepath.Base(localPath)
	dst := filepath.Join(cacheDir, "community_"+name)
	if fi, err := os.Stat(dst); err == nil && time.Since(fi.ModTime()) < communityHitTTL {
		return dst
	}
	missMarker := dst + ".miss"
	if fi, err := os.Stat(missMarker); err == nil && time.Since(fi.ModTime()) < communityMissTTL {
		return ""
	}

	base := strings.TrimRight(os.Getenv("LLM_COMMUNITY_TUNES_URL"), "/")
	if base == "" {
		base = DefaultCommunityBaseURL
	}
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(base + "/" + name)
	if err != nil {
		return "" // offline: no marker, retry next launch
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		_ = os.MkdirAll(cacheDir, 0755)
		_ = os.WriteFile(missMarker, []byte("1\n"), 0644)
		return ""
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, communityMaxSize))
	if err != nil {
		return ""
	}
	sanitized, err := sanitizeCommunityTune(data, filepath.Base(modelPath))
	if err != nil {
		_ = os.MkdirAll(cacheDir, 0755)
		_ = os.WriteFile(missMarker, []byte("invalid: "+err.Error()+"\n"), 0644)
		return ""
	}
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return ""
	}
	tmp := dst + ".tmp"
	if err := os.WriteFile(tmp, sanitized, 0644); err != nil {
		return ""
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return ""
	}
	_ = os.Remove(missMarker)
	return dst
}

// sanitizeCommunityTune validates a downloaded tune document and strips every
// flag that is not on the launcher's allow-list. Community configs may only
// carry the same benign performance flags the AI-tune engine itself is allowed
// to propose — placement, model, and network flags can never be injected.
func sanitizeCommunityTune(data []byte, modelName string) ([]byte, error) {
	var doc map[string]interface{}
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("parse: %w", err)
	}
	if m, _ := doc["model"].(string); m != modelName {
		return nil, fmt.Errorf("model mismatch: %q", doc["model"])
	}
	if complete, _ := doc["complete"].(bool); !complete {
		return nil, fmt.Errorf("incomplete tune run")
	}
	best, _ := doc["best_config"].(map[string]interface{})
	if best == nil {
		return nil, fmt.Errorf("missing best_config")
	}
	if tps, _ := best["gen_tps"].(float64); tps <= 0 {
		return nil, fmt.Errorf("missing gen_tps")
	}
	flags, _ := best["flags"].(map[string]interface{})
	best["flags"] = sanitizeFlagValues(flags, DefaultProtectedFlags())
	doc["provenance"] = "community"
	// Drop per-round details; only the winning config is needed locally.
	delete(doc, "all_results")
	return json.MarshalIndent(doc, "", "  ")
}

// ShareHint returns the message printed after a successful tune so users can
// contribute their result back to the shared pool.
func ShareHint(tunePath string) string {
	if tunePath == "" {
		return ""
	}
	return fmt.Sprintf("[tune] Share this config with owners of the same GPUs:\n"+
		"[tune]   file: %s\n"+
		"[tune]   open a PR or issue at %s (drop the file into tunes/)",
		tunePath, CommunityRepoURL)
}
