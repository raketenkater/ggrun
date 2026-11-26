package download

import (
	"testing"
)

func TestDownloader(t *testing.T) {
	redirectScriptCache(t)
	d := New("/tmp/models", "/tmp/cache", "")
	if d.ModelDir != "/tmp/models" {
		t.Fatalf("unexpected model dir")
	}
	if d.ScriptPath == "" {
		t.Fatal("no downloader resolved; a bare binary must use the embedded copy")
	}
}
