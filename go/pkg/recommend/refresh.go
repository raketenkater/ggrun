package recommend

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// The catalog is embedded at build time, but new Artificial-Analysis-scored
// models land in the repo daily (see .github/workflows/update-recommendations.yml).
// To get those to users without shipping a new binary, the recommender refreshes
// a locally cached copy from the published catalog on a TTL, and always falls
// back to the embedded copy offline. This is best-effort and honours the same
// opt-out as the startup update check.
const (
	catalogURL   = "https://raw.githubusercontent.com/raketenkater/llm-server/main/go/pkg/recommend/catalog.json"
	catalogTTL   = 24 * time.Hour
	fetchTimeout = 6 * time.Second
	maxCatalog   = 8 << 20 // 8 MiB sanity cap on the download
	minModels    = 10      // reject truncated/garbage catalogs
)

// cachedCatalogPath returns where the runtime-refreshed catalog is stored, using
// the same locations the rest of the app uses for its cache.
func cachedCatalogPath() string {
	if d := os.Getenv("LLM_CACHE_DIR"); d != "" {
		return filepath.Join(d, "catalog.json")
	}
	if home := os.Getenv("LLM_APP_HOME"); home != "" {
		return filepath.Join(home, ".cache", "llm-server", "catalog.json")
	}
	if d, err := os.UserCacheDir(); err == nil {
		return filepath.Join(d, "llm-server", "catalog.json")
	}
	return ""
}

// catalogBytes prefers a valid cached catalog (fresher than the embedded one and
// usable offline), falling back to the embedded catalog.
func catalogBytes() []byte {
	if p := cachedCatalogPath(); p != "" {
		if b, err := os.ReadFile(p); err == nil && validCatalog(b) {
			return b
		}
	}
	return catalogJSON
}

func validCatalog(b []byte) bool {
	var doc catalogDoc
	if err := json.Unmarshal(b, &doc); err != nil {
		return false
	}
	return len(doc.Candidates) >= minModels
}

// MaybeRefresh refreshes the cached catalog from the published URL when it is
// missing or older than the TTL. Best-effort: any failure leaves the existing
// cache/embedded catalog untouched. Skipped when the user opted out of network
// update checks (LLM_SERVER_NO_UPDATE_CHECK).
func MaybeRefresh() {
	if os.Getenv("LLM_SERVER_NO_UPDATE_CHECK") != "" {
		return
	}
	path := cachedCatalogPath()
	if path == "" {
		return
	}
	if fi, err := os.Stat(path); err == nil && time.Since(fi.ModTime()) < catalogTTL {
		return // still fresh
	}

	client := &http.Client{Timeout: fetchTimeout}
	resp, err := client.Get(catalogURL)
	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxCatalog))
	if err != nil || !validCatalog(body) || !atLeastAsNew(body) {
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, body, 0o644); err != nil {
		return
	}
	_ = os.Rename(tmp, path) // atomic publish
}

// atLeastAsNew guards against replacing the current catalog with an older one
// (e.g. a local dev build embedding a catalog newer than main).
func atLeastAsNew(fetched []byte) bool {
	var f, cur catalogDoc
	if json.Unmarshal(fetched, &f) != nil {
		return false
	}
	_ = json.Unmarshal(catalogBytes(), &cur)
	if f.Version != cur.Version {
		return f.Version >= cur.Version
	}
	return f.GeneratedAt >= cur.GeneratedAt
}
