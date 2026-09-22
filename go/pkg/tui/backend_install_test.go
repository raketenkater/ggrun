package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func isolatedModelDir(t *testing.T) (modelPath string) {
	t.Helper()
	home := t.TempDir()
	modelDir := filepath.Join(home, "models")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_CONFIG", filepath.Join(home, "missing-config"))
	t.Setenv("LLM_APP_HOME", home)
	t.Setenv("LLM_MODEL_DIR", modelDir)
	t.Setenv("LLM_CACHE_DIR", filepath.Join(home, "cache"))
	t.Setenv("LLM_BACKEND", "")
	modelPath = filepath.Join(modelDir, "tiny-Q4_K_M.gguf")
	if err := os.WriteFile(modelPath, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	return modelPath
}

// The recipe tag from a model-aware install must stay scoped to that model.
// RunAfterBackendInstall copied it into m.backend, the session default, so
// backing out and choosing an unrelated model launched it with
// --backend <recipe> as an explicit pin — contradicting its own doc comment.
func TestBackendInstallDoesNotLeakTheRecipeTagIntoTheSessionDefault(t *testing.T) {
	modelPath := isolatedModelDir(t)
	before := InitialModel().backend

	m := afterBackendInstallModel(&LaunchRequest{ModelPath: modelPath, Backend: "some-recipe"})
	if m.backend != before {
		t.Fatalf("session default backend became %q after an install (was %q): every other model "+
			"would now be pinned to the recipe", m.backend, before)
	}
}

// A failed install returns to the model with the reason instead of claiming
// success (and, before, instead of ggrun exiting).
func TestFailedBackendInstallReturnsToTheModelWithTheError(t *testing.T) {
	modelPath := isolatedModelDir(t)

	m := afterBackendInstallModel(&LaunchRequest{
		ModelPath: modelPath, Backend: "some-recipe",
		BackendInstallError: "build failed: exit status 2",
	})
	if !strings.Contains(m.message, "build failed: exit status 2") || m.messageType != "error" {
		t.Fatalf("install failure not reported: type=%q message=%q", m.messageType, m.message)
	}
	if strings.Contains(m.message, "installed and auto-selected") {
		t.Fatalf("a failed install was reported as a success: %q", m.message)
	}
	if m.selectedModel < 0 {
		t.Fatal("fixture model was not found; this test would not exercise the model-config path")
	}
	if m.screen != ScreenModelConfig {
		t.Fatalf("the failure must reopen the model's configuration, got screen %v", m.screen)
	}

	ok := afterBackendInstallModel(&LaunchRequest{ModelPath: modelPath, Backend: "some-recipe"})
	if ok.messageType == "error" {
		t.Fatalf("a successful install was reported as a failure: %q", ok.message)
	}
}
