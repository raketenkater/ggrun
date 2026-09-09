package backends

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestHotExpertTelemetryOverlayOnlyExposesBoundedCounters(t *testing.T) {
	// Reconstruct the added upstream file from the immutable feature patch.
	upstream := strings.Split(string(hotExpertsFeaturePatch), "diff --git a/src/llama-moecache.cpp b/src/llama-moecache.cpp\n")
	if len(upstream) != 2 {
		t.Fatal("upstream cache source missing")
	}
	section := strings.SplitN(upstream[1], "\ndiff --git ", 2)[0]
	var src strings.Builder
	for _, line := range strings.Split(section, "\n") {
		if strings.HasPrefix(line, "+") && !strings.HasPrefix(line, "+++") {
			src.WriteString(line[1:])
			src.WriteByte('\n')
		}
	}
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "src"), 0700); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "src", "llama-moecache.cpp")
	if err := os.WriteFile(file, []byte(src.String()), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("git", "apply", "--unsafe-paths", "-")
	cmd.Dir = dir
	cmd.Stdin = strings.NewReader(string(hotExpertsTelemetryPatch))
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("apply telemetry overlay: %v: %s", err, out)
	}
	got, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	old := `LLAMA_LOG_DEBUG("moe-cache: steps=`
	if strings.Count(src.String(), old) != 1 {
		t.Fatal("expected one bounded counter emission")
	}
	want := strings.Replace(src.String(), old, `LLAMA_LOG_INFO("moe-cache: steps=`, 1)
	if string(got) != want {
		t.Fatal("telemetry overlay changed behavior beyond the aggregate log level")
	}
	if !strings.Contains(string(got), "mc->n_steps % 512 == 0") {
		t.Fatal("bounded counter cadence lost")
	}
	recipe := RecipeByName("hot-experts")
	if len(recipe.Patches) != 2 || recipe.Patches[0].Name != "features/hot-experts/telemetry-info-v1" || recipe.Patches[1].Name != "features/hot-experts/inflight-dedup-v1" {
		t.Fatal("standalone pinned backend lacks telemetry overlay identity")
	}
}
