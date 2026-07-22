package tune

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func communityTestModel(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	model := filepath.Join(dir, "TestModel-Q4_K_M.gguf")
	if err := os.WriteFile(model, make([]byte, 4096), 0644); err != nil {
		t.Fatal(err)
	}
	return model
}

func communityTuneDoc(model string, flags map[string]interface{}) []byte {
	doc := map[string]interface{}{
		"model":            model,
		"complete":         true,
		"baseline_gen_tps": 100.0,
		"best_config": map[string]interface{}{
			"name":    "community-best",
			"gen_tps": 120.0,
			"flags":   flags,
		},
	}
	data, _ := json.Marshal(doc)
	return data
}

func TestFetchCommunityTuneHitSanitizesFlags(t *testing.T) {
	model := communityTestModel(t)
	cacheDir := t.TempDir()
	payload := communityTuneDoc(filepath.Base(model), map[string]interface{}{
		"-b":             "4096",       // allowed perf flag
		"-m":             "/evil.gguf", // protected: must be stripped
		"--host":         "0.0.0.0",    // protected: must be stripped
		"--made-up-fla":  "x",          // unknown: must be stripped
		"--cache-type-k": "q8_0",       // quality-protected: must be stripped
		"--parallel":     "8",          // quality-protected: must be stripped
	})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(payload)
	}))
	defer srv.Close()
	t.Setenv("LLM_COMMUNITY_TUNES_URL", srv.URL)
	t.Setenv("LLM_COMMUNITY_TUNES", "")

	got := FetchCommunityTune(cacheDir, model, []string{"RTX 4070", "RTX 3060"}, false, "llama")
	if got == "" {
		t.Fatal("expected a community tune path")
	}
	data, err := os.ReadFile(got)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Provenance string `json:"provenance"`
		BestConfig struct {
			Flags map[string]interface{} `json:"flags"`
		} `json:"best_config"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	if doc.Provenance != "community" {
		t.Fatalf("expected community provenance, got %q", doc.Provenance)
	}
	flags := doc.BestConfig.Flags
	if flags["-b"] != "4096" {
		t.Fatalf("allowed perf flag missing: %v", flags)
	}
	for _, banned := range []string{"-m", "--host", "--made-up-fla", "--parallel"} {
		if _, ok := flags[banned]; ok {
			t.Fatalf("flag %s must be stripped from community config: %v", banned, flags)
		}
	}
}

func TestFetchCommunityTuneMissCaches404(t *testing.T) {
	model := communityTestModel(t)
	cacheDir := t.TempDir()
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		http.NotFound(w, r)
	}))
	defer srv.Close()
	t.Setenv("LLM_COMMUNITY_TUNES_URL", srv.URL)
	t.Setenv("LLM_COMMUNITY_TUNES", "")

	if got := FetchCommunityTune(cacheDir, model, []string{"GPU"}, false, "llama"); got != "" {
		t.Fatalf("expected miss, got %q", got)
	}
	if got := FetchCommunityTune(cacheDir, model, []string{"GPU"}, false, "llama"); got != "" {
		t.Fatalf("expected cached miss, got %q", got)
	}
	if hits != 1 {
		t.Fatalf("404 must be cached, server hit %d times", hits)
	}
}

func TestFetchCommunityTuneDisabled(t *testing.T) {
	model := communityTestModel(t)
	t.Setenv("LLM_COMMUNITY_TUNES", "off")
	t.Setenv("LLM_COMMUNITY_TUNES_URL", "http://127.0.0.1:1") // would fail loudly if contacted
	if got := FetchCommunityTune(t.TempDir(), model, []string{"GPU"}, false, "llama"); got != "" {
		t.Fatalf("expected disabled, got %q", got)
	}
}

func TestFetchCommunityTuneRejectsWrongModel(t *testing.T) {
	model := communityTestModel(t)
	cacheDir := t.TempDir()
	payload := communityTuneDoc("OtherModel.gguf", map[string]interface{}{"-b": "4096"})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(payload)
	}))
	defer srv.Close()
	t.Setenv("LLM_COMMUNITY_TUNES_URL", srv.URL)
	t.Setenv("LLM_COMMUNITY_TUNES", "")
	if got := FetchCommunityTune(cacheDir, model, []string{"GPU"}, false, "llama"); got != "" {
		t.Fatalf("model-mismatched config must be rejected, got %q", got)
	}
}

func TestFetchCommunityTuneRejectsIncomplete(t *testing.T) {
	model := communityTestModel(t)
	cacheDir := t.TempDir()
	doc := map[string]interface{}{
		"model":    filepath.Base(model),
		"complete": false,
		"best_config": map[string]interface{}{
			"gen_tps": 120.0,
			"flags":   map[string]interface{}{"-b": "4096"},
		},
	}
	payload, _ := json.Marshal(doc)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(payload)
	}))
	defer srv.Close()
	t.Setenv("LLM_COMMUNITY_TUNES_URL", srv.URL)
	t.Setenv("LLM_COMMUNITY_TUNES", "")
	if got := FetchCommunityTune(cacheDir, model, []string{"GPU"}, false, "llama"); got != "" {
		t.Fatalf("incomplete tune must be rejected, got %q", got)
	}
}
