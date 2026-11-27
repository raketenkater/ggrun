package backends

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"
	"time"
)

const syntheticCatalog = `{
  "version": 1,
  "recipes": [
    {
      "name": "old-fork", "description": "d", "tag": "old-fork",
      "git_url": "https://example.invalid/x.git", "branch": "b",
      "commit": "1111111111111111111111111111111111111111",
      "route_arch": "oldarch",
      "kind": "fork",
      "superseded_by": "mainline llama.cpp b9000 (PR 12345)",
      "evidence": ["https://example.invalid/pr/12345"],
      "expires": "2026-01-31"
    },
    {
      "name": "plain", "description": "d", "tag": "plain",
      "git_url": "https://example.invalid/y.git", "branch": "b",
      "commit": "2222222222222222222222222222222222222222",
      "route_arch": "plainarch"
    }
  ],
  "name_rules": []
}`

func TestCatalogSupersededByReachesRecipe(t *testing.T) {
	c, err := loadCatalog([]byte(syntheticCatalog), fstest.MapFS{})
	if err != nil {
		t.Fatal(err)
	}
	r := c.recipes[0]
	if r.SupersededBy != "mainline llama.cpp b9000 (PR 12345)" || r.Kind != "fork" || len(r.Evidence) != 1 || r.Expires != "2026-01-31" {
		t.Fatalf("catalog metadata not carried to the recipe: %+v", r)
	}
	if c.recipes[1].SupersededBy != "" {
		t.Fatal("an entry without superseded_by must stay unmarked")
	}
}

func TestAssessBackend(t *testing.T) {
	c, err := loadCatalog([]byte(syntheticCatalog), fstest.MapFS{})
	if err != nil {
		t.Fatal(err)
	}
	old, plain := c.recipes[0], c.recipes[1]
	now := time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC)
	built := now.Add(-72 * time.Hour)

	s := AssessBackend(Backend{
		Tag: "old-fork", Commit: "1111111111111111111111111111111111111111",
		Build: &BuildRecord{Seconds: 2400, At: built.Format(time.RFC3339)},
	}, &old, "3333333333333333333333333333333333333333", time.Time{}, now)
	if s.SupersededBy == "" || s.Recipe != "old-fork" || s.Kind != "fork" {
		t.Fatalf("superseded entry not reported: %+v", s)
	}
	if s.BuiltOffPin || !s.CheckoutOffPin {
		t.Fatalf("build is at the pin and the checkout is not: %+v", s)
	}
	if !s.ReviewOverdue {
		t.Fatal("expiry 2026-01-31 has passed")
	}
	if s.Age != 72*time.Hour || s.BuiltAtSource != "recorded" {
		t.Fatalf("age %v from %q", s.Age, s.BuiltAtSource)
	}

	// No recorded build: fall back to the binary's file time.
	s = AssessBackend(Backend{Tag: "plain", Commit: "9999999"}, &plain, "", now.Add(-time.Hour), now)
	if s.SupersededBy != "" || !s.BuiltOffPin || s.CheckoutOffPin || s.CheckoutCommit != "" {
		t.Fatalf("unexpected: %+v", s)
	}
	if s.BuiltAtSource != "binary file time" || s.Age != time.Hour {
		t.Fatalf("fallback age: %+v", s)
	}

	// A hand-added backend has no recipe: nothing to compare.
	s = AssessBackend(Backend{Tag: "mine", Commit: "abc"}, nil, "def", time.Time{}, now)
	if s.Recipe != "" || s.SupersededBy != "" || s.BuiltOffPin || s.CheckoutOffPin || !s.BuiltAt.IsZero() {
		t.Fatalf("unexpected: %+v", s)
	}
}

func TestSameCommitAllowsAbbreviations(t *testing.T) {
	full := "f46c95ee90d8c8200b0147c646b883405020b482"
	for _, ok := range []string{"f46c95e", strings.ToUpper(full[:12]), full} {
		if !SameCommit(full, ok) {
			t.Errorf("%s should match", ok)
		}
	}
	if SameCommit(full, "") || SameCommit(full, "abc1234") {
		t.Fatal("empty or different ids must not match")
	}
}

func TestBackendManifestKeepsBuildRecordAndReadsOldManifests(t *testing.T) {
	home := t.TempDir()
	t.Setenv("LLM_APP_HOME", home)
	old := `[{"tag":"legacy","path":"/x/.src/fork-a/build-cuda/bin/llama-server","commit":"abc"}]`
	if err := writeFileForTest(ManifestPath(), old); err != nil {
		t.Fatal(err)
	}
	list := Load()
	if len(list) != 1 || list[0].Build != nil {
		t.Fatalf("legacy manifest must load with no build record: %+v", list)
	}
	list[0].Build = &BuildRecord{Seconds: 123, TreeBytes: 456, Accel: "cuda", At: "2026-11-27T00:00:00Z"}
	if err := Save(list); err != nil {
		t.Fatal(err)
	}
	got := Load()
	if got[0].Build == nil || *got[0].Build != *list[0].Build {
		t.Fatalf("build record lost: %+v", got[0].Build)
	}
}

func writeFileForTest(path, body string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(body), 0o644)
}
