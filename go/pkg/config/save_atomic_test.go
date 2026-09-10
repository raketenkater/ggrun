package config

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestSavePreservesExistingModeAndInvalidSavePreservesBytes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config")
	t.Setenv("LLM_CONFIG", path)
	cfg := Defaults()
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}
	cfg.Port = 9091
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Port = -1
	if err := cfg.Save(); err == nil {
		t.Fatal("invalid port saved")
	}
	after, err := os.ReadFile(path)
	if err != nil || string(before) != string(after) {
		t.Fatal("invalid save changed disk state")
	}
	if runtime.GOOS != "windows" {
		info, err := os.Stat(path)
		if err != nil || info.Mode().Perm() != 0600 {
			t.Fatalf("config permissions: %v %v", info, err)
		}
	}
}
