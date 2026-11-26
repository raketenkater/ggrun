package recommend

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"testing"
)

// S-09: the Go list of named publishers equals the scripts' scoring table.
func TestKnownQuantizersMatchTheCatalogScoring(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "..", "..", "tools", "models", "update_recommendations.py"))
	if err != nil {
		t.Fatal(err)
	}
	block := regexp.MustCompile(`(?s)TRUSTED_GGUF_OWNERS = \{(.*?)\}`).FindSubmatch(src)
	if block == nil {
		t.Fatal("owner table not found")
	}
	var py []string
	for _, m := range regexp.MustCompile(`"([^"]+)":`).FindAllSubmatch(block[1], -1) {
		py = append(py, string(m[1]))
	}
	got := append([]string(nil), knownQuantizers...)
	sort.Strings(py)
	sort.Strings(got)
	if len(py) != len(got) {
		t.Fatalf("python %v, go %v", py, got)
	}
	for i := range py {
		if py[i] != got[i] {
			t.Fatalf("python %v, go %v", py, got)
		}
	}
}

func TestOwnerLabelMarksOtherPublishers(t *testing.T) {
	for repo, want := range map[string]string{
		"unsloth/Qwen-GGUF": "unsloth", "Bartowski/x": "Bartowski", "someone/Qwen-GGUF": "someone*", "noslash": "?",
	} {
		if got := OwnerLabel(repo); got != want {
			t.Errorf("%s: %q, want %q", repo, got, want)
		}
	}
}

// Every catalog row that carries a revision carries a full commit id.
func TestCatalogRevisionsAreCommitIDs(t *testing.T) {
	re := regexp.MustCompile(`^[0-9a-f]{40}$`)
	for _, c := range Shortlist() {
		if c.Revision != "" && !re.MatchString(c.Revision) {
			t.Errorf("%s: revision %q", c.Name, c.Revision)
		}
	}
}
