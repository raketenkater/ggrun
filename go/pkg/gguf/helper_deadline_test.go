package gguf

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// W5/C-02: a helper that never returns must not hang a launch.
func TestHungParserHelperReturnsATimeout(t *testing.T) {
	dir := t.TempDir()
	helper := "META = {'tensor_accounting_schema': 3}\nimport time\ntime.sleep(30)\n"
	if err := os.WriteFile(filepath.Join(dir, "parse_gguf.py"), []byte(helper), 0o755); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(model, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_SCRIPT_DIR", dir)
	old := parseHelperTimeout
	parseHelperTimeout = 300 * time.Millisecond
	t.Cleanup(func() { parseHelperTimeout = old })
	start := time.Now()
	_, err := Parse(model)
	if elapsed := time.Since(start); elapsed > 8*time.Second {
		t.Fatalf("Parse took %s with a hung helper", elapsed)
	}
	if err == nil || !strings.Contains(err.Error(), "did not answer within 300ms") {
		t.Fatalf("want a bounded timeout error, got %v", err)
	}
}
