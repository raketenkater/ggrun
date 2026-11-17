package gguf

import (
	"os"
	"testing"

	"github.com/raketenkater/ggrun/pkg/embedscript"
)

// Tests never write into the user's script cache.
func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "ggrun-gguf-scripts-")
	if err != nil {
		panic(err)
	}
	embedscript.CacheDir = func() (string, error) { return dir, nil }
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}
