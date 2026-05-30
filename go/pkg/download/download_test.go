package download

import (
	"testing"
)

func TestFindScript(t *testing.T) {
	path := findScript("")
	// May or may not exist in test environment
	_ = path
}

func TestDownloader(t *testing.T) {
	d := New("/tmp/models", "/tmp/cache", "")
	if d.ModelDir != "/tmp/models" {
		t.Fatalf("unexpected model dir")
	}
}
