package recommend

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func writeCatalogCache(t *testing.T, generatedAt string, models int) {
	t.Helper()
	var embedded catalogDoc
	if err := json.Unmarshal(catalogJSON, &embedded); err != nil {
		t.Fatal(err)
	}
	doc := catalogDoc{Version: embedded.Version, GeneratedAt: generatedAt, Candidates: embedded.Candidates[:models]}
	data, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(os.Getenv("LLM_CACHE_DIR"), "catalog.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func servedGeneratedAt(t *testing.T) string {
	var doc catalogDoc
	if err := json.Unmarshal(catalogBytes(), &doc); err != nil {
		t.Fatal(err)
	}
	return doc.GeneratedAt
}

func TestOlderCachedCatalogDoesNotOverrideTheEmbeddedOne(t *testing.T) {
	t.Setenv("LLM_CACHE_DIR", t.TempDir())
	var embedded catalogDoc
	if err := json.Unmarshal(catalogJSON, &embedded); err != nil {
		t.Fatal(err)
	}
	writeCatalogCache(t, "2020-01-01T00:00:00Z", minModels)
	if got := servedGeneratedAt(t); got != embedded.GeneratedAt {
		t.Fatalf("an older cache was served: %s (embedded %s)", got, embedded.GeneratedAt)
	}
	writeCatalogCache(t, "2999-01-01T00:00:00Z", minModels)
	if got := servedGeneratedAt(t); got != "2999-01-01T00:00:00Z" {
		t.Fatalf("a newer cache was not served: %s", got)
	}
}
