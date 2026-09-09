package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRelatedGrowthRejectsForeignScope(t *testing.T) {
	for _, change := range []string{"backend build", "backend feature", "model geometry", "model artifact", "renamed probe", "missing backend", "malformed slots"} {
		t.Run(change, func(t *testing.T) {
			dir := t.TempDir()
			gpus, model := growthCarryFixture()
			model.Path = filepath.Join(dir, "model.gguf")
			if err := os.WriteFile(model.Path, []byte("original"), 0600); err != nil {
				t.Fatal(err)
			}
			const tag = "llama@build-a"
			if err := RecordRuntimeGraphGrowth(dir, model, 131072, 256, "high", "gpu", tag, gpus, 1, map[int]int{0: 334}); err != nil {
				t.Fatal(err)
			}
			path := probeCachePath(dir, model, 131072, 256, "high", "gpu", tag, gpus, 0)
			requestedTag := tag
			switch change {
			case "backend build":
				requestedTag = "llama@build-b"
			case "backend feature":
				requestedTag = ScopedBackendRuntimeFeatureTag(tag, true, 0, 2)
			case "model geometry":
				model.NumExperts++
			case "model artifact":
				if err := os.WriteFile(model.Path, []byte("replacement with different bytes"), 0600); err != nil {
					t.Fatal(err)
				}
			case "renamed probe":
				if err := os.Rename(path, filepath.Join(dir, "unscoped.probe")); err != nil {
					t.Fatal(err)
				}
			default:
				data, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				old, replacement := "backend="+tag+" ", ""
				if change == "malformed slots" {
					old, replacement = "parallel=0", "parallel=invalid"
				}
				if err := os.WriteFile(path, []byte(strings.ReplaceAll(string(data), old, replacement)), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if got := RelatedModelRuntimeGraphGrowth(dir, model, gpus, 1, requestedTag); len(got) != 0 {
				t.Fatalf("%s contaminated this launch's growth reserve: %v", change, got)
			}
		})
	}
}

func TestForeignServeCannotRetireThisBackendsOOMReserve(t *testing.T) {
	dir := t.TempDir()
	gpus, model := growthCarryFixture()
	if err := RecordRuntimeGraphGrowthFromOOM(dir, model, 131072, 128, "high", "gpu", "llama@current", gpus, 1, 0, 900, false); err != nil {
		t.Fatal(err)
	}
	if err := RecordRuntimeGraphGrowth(dir, model, 131072, 256, "high", "gpu", "llama@other", gpus, 1, map[int]int{0: 100}); err != nil {
		t.Fatal(err)
	}
	if got := RelatedModelRuntimeGraphGrowth(dir, model, gpus, 1, "llama@current")[0]; got != 900 {
		t.Fatalf("another backend's healthy serve retired current OOM evidence: got %d, want 900", got)
	}
}

// The compute cost model must obey the same boundary as runtime growth. A
// cheaper/different backend cannot calibrate this build's allocation estimate.
func TestComputeExcessRequiresMatchingScope(t *testing.T) {
	for _, mismatch := range []string{"none", "backend", "slots", "artifact"} {
		t.Run(mismatch, func(t *testing.T) {
			dir := t.TempDir()
			gpus, _ := growthCarryFixture()
			model := glmProfile()
			writeObservedComputeProbe(t, dir, model, gpus, 1048576, 64, map[int]int{0: 4572}, "live-allocated")
			tag, slots := "llama", 1
			switch mismatch {
			case "backend":
				tag = "llama@new-build"
			case "slots":
				slots = 4
			case "artifact":
				model.NumExperts++
			}
			got := MeasuredComputeExcess(dir, model, gpus, slots, tag)
			if mismatch == "none" && got.BytesPerTokenCtx <= 0 {
				t.Fatal("compatible measurement was discarded")
			}
			if mismatch != "none" && got.BytesPerTokenCtx != 0 {
				t.Fatalf("%s leaked into compute prediction: %+v", mismatch, got)
			}
		})
	}
}

func TestRelatedGrowthPreservesCompatibleLegacyEvidence(t *testing.T) {
	for _, schema := range []string{"6", "7", "8"} {
		t.Run(schema, func(t *testing.T) {
			dir := t.TempDir()
			gpus, model := growthCarryFixture()
			// Empty defaults have always been normalized by probeCachePath. Keep
			// these old, correctly keyed records usable without another model load.
			if err := RecordRuntimeGraphGrowth(dir, model, 131072, 256, "", "", "", gpus, 1, map[int]int{0: 334}); err != nil {
				t.Fatal(err)
			}
			path := probeCachePath(dir, model, 131072, 256, "", "", "", gpus, 0)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			data = []byte(strings.ReplaceAll(string(data), fmt.Sprintf("PROBE_CACHE_SCHEMA=%d", probeCacheSchema), "PROBE_CACHE_SCHEMA="+schema))
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			if got := RelatedModelRuntimeGraphGrowth(dir, model, gpus, 1, "llama")[0]; got != 334 {
				t.Fatalf("valid legacy evidence lost: %d", got)
			}
		})
	}
}
