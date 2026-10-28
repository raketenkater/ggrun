package models

import (
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
)

func TestScanRootsFindsSinglesAndFirstShardOnly(t *testing.T) {
	root := t.TempDir()
	writeModel(t, root, "single.gguf", 4)
	writeModel(t, root, "UPPER.GGUF", 4)
	writeModel(t, root, "nested/bundle-00001-of-00002.gguf", 4)
	writeModel(t, root, "nested/bundle-00002-of-00002.gguf", 4)
	writeModel(t, root, "nested/incomplete-00001-of-00003.gguf", 4)
	writeModel(t, root, "nested/incomplete-00002-of-00003.gguf", 4)
	writeModel(t, root, "nested/ggml-vocab-llama.gguf", 4)
	writeModel(t, root, "nested/mmproj-model-f16.gguf", 4)
	writeModel(t, root, "nested/notes.txt", 4)

	result := scanRoots([]string{root}, nil, 100, time.Minute)
	want := []string{
		filepath.Join(root, "nested", "bundle-00001-of-00002.gguf"),
		filepath.Join(root, "single.gguf"),
		filepath.Join(root, "UPPER.GGUF"),
	}
	sort.Slice(want, func(i, j int) bool {
		return strings.ToLower(want[i]) < strings.ToLower(want[j])
	})
	if !reflect.DeepEqual(result.Paths, want) {
		t.Fatalf("paths = %#v, want %#v", result.Paths, want)
	}
	if result.Truncated {
		t.Fatal("small scan unexpectedly hit a limit")
	}
}

func TestScanRootsSkipsMountBoundariesAndNoiseDirectories(t *testing.T) {
	root := t.TempDir()
	boundary := filepath.Join(root, "separate-mount")
	writeModel(t, root, "keep.gguf", 4)
	writeModel(t, root, ".git/hidden.gguf", 4)
	writeModel(t, boundary, "scanned-separately.gguf", 4)

	result := scanRoots([]string{root}, []string{boundary}, 100, time.Minute)
	want := []string{filepath.Join(root, "keep.gguf")}
	if !reflect.DeepEqual(result.Paths, want) {
		t.Fatalf("paths = %#v, want %#v", result.Paths, want)
	}
}

func TestScanRootsStopsAtEntryLimit(t *testing.T) {
	root := t.TempDir()
	writeModel(t, root, "one.gguf", 4)
	writeModel(t, root, "two.gguf", 4)

	result := scanRoots([]string{root}, nil, 1, time.Minute)
	if !result.Truncated {
		t.Fatal("scan should report that its entry limit was reached")
	}
	if result.Entries <= 1 {
		t.Fatalf("entries = %d, expected the stopping entry to be counted", result.Entries)
	}
}

func TestLocalFilesystemClassification(t *testing.T) {
	tests := []struct {
		fsType     string
		mountpoint string
		want       bool
	}{
		{fsType: "ext4", mountpoint: "/", want: true},
		{fsType: "fuseblk", mountpoint: "/mnt/data", want: true},
		{fsType: "overlay", mountpoint: "/", want: true},
		{fsType: "overlay", mountpoint: "/var/lib/docker/overlay", want: false},
		{fsType: "tmpfs", mountpoint: "/run", want: false},
		{fsType: "nfs4", mountpoint: "/net", want: false},
		{fsType: "fuse.sshfs", mountpoint: "/remote", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.fsType+tt.mountpoint, func(t *testing.T) {
			if got := isLocalModelFilesystem(tt.fsType, tt.mountpoint); got != tt.want {
				t.Fatalf("isLocalModelFilesystem(%q, %q) = %v, want %v", tt.fsType, tt.mountpoint, got, tt.want)
			}
		})
	}
}

func TestLinuxMountScanPlanPrioritizesDataMountAndSkipsNetwork(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mountinfo parsing is Linux-specific")
	}
	local := t.TempDir()
	network := t.TempDir()
	mountInfo := strings.Join([]string{
		"36 25 0:32 / / rw - ext4 /dev/root rw",
		"37 36 8:1 / " + local + " rw - fuseblk /dev/sdb1 rw",
		"38 36 0:55 / " + network + " rw - nfs4 server:/models rw",
	}, "\n")

	roots, boundaries := linuxMountScanPlan(mountInfo)
	if len(roots) != 2 || roots[0] != local || roots[1] != "/" {
		t.Fatalf("roots = %#v, want data mount first and / last", roots)
	}
	boundarySet := make(map[string]bool)
	for _, path := range boundaries {
		boundarySet[path] = true
	}
	for _, want := range []string{"/", local, network} {
		if !boundarySet[want] {
			t.Fatalf("mount boundary %q missing from %#v", want, boundaries)
		}
	}
}

func TestDiscoveryCacheRoundTripDropsStaleAndInvalidPaths(t *testing.T) {
	cacheDir := t.TempDir()
	modelDir := t.TempDir()
	live := filepath.Join(modelDir, "live.gguf")
	stale := filepath.Join(modelDir, "stale.gguf")
	invalid := filepath.Join(modelDir, "notes.txt")
	incomplete := filepath.Join(modelDir, "partial-00001-of-00003.gguf")
	incompleteSecond := filepath.Join(modelDir, "partial-00002-of-00003.gguf")
	for _, path := range []string{live, stale, invalid, incomplete, incompleteSecond} {
		if err := os.WriteFile(path, []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := SaveDiscoveredPaths(cacheDir, []string{live, live, stale, invalid, incomplete}); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(stale); err != nil {
		t.Fatal(err)
	}

	if got := LoadDiscoveredPaths(cacheDir); !reflect.DeepEqual(got, []string{live}) {
		t.Fatalf("cached paths = %#v, want only %q", got, live)
	}
	if got := LoadDiscoveredPaths(""); got != nil {
		t.Fatalf("empty cache directory should not read a relative index, got %#v", got)
	}
}

func TestRememberModelDirAddsDownloadsAndKeepsPriorPaths(t *testing.T) {
	cacheDir := t.TempDir()
	prior := filepath.Join(t.TempDir(), "prior.gguf")
	downloadDir := filepath.Join(t.TempDir(), "custom folder")
	downloaded := filepath.Join(downloadDir, "repo", "model-Q4_K_M.gguf")
	for _, path := range []string{prior, downloaded} {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := SaveDiscoveredPaths(cacheDir, []string{prior}); err != nil {
		t.Fatal(err)
	}
	if err := RememberModelDir(cacheDir, downloadDir); err != nil {
		t.Fatal(err)
	}
	got := LoadDiscoveredPaths(cacheDir)
	want := []string{prior, downloaded}
	sort.Strings(got)
	sort.Strings(want)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("discovered paths = %#v, want %#v", got, want)
	}
	if err := RememberModelDir(cacheDir, t.TempDir()); err != nil {
		t.Fatal(err)
	}
	if got := LoadDiscoveredPaths(cacheDir); len(got) != 2 {
		t.Fatalf("an empty folder must not drop prior paths: %#v", got)
	}
}

func TestTruncatedDiscoveryScanMergesPriorValidPaths(t *testing.T) {
	cacheDir := t.TempDir()
	modelDir := t.TempDir()
	prior := filepath.Join(modelDir, "prior.gguf")
	foundNow := filepath.Join(modelDir, "new.gguf")
	for _, path := range []string{prior, foundNow} {
		if err := os.WriteFile(path, []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := SaveDiscoveredPaths(cacheDir, []string{prior}); err != nil {
		t.Fatal(err)
	}
	if err := SaveDiscoveredScan(cacheDir, ScanResult{Paths: []string{foundNow}, Truncated: true}); err != nil {
		t.Fatal(err)
	}
	got := LoadDiscoveredPaths(cacheDir)
	want := []string{prior, foundNow}
	sort.Strings(got)
	sort.Strings(want)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("truncated scan replaced prior paths: got %#v, want %#v", got, want)
	}

	if err := SaveDiscoveredScan(cacheDir, ScanResult{Paths: []string{foundNow}}); err != nil {
		t.Fatal(err)
	}
	if got := LoadDiscoveredPaths(cacheDir); !reflect.DeepEqual(got, []string{foundNow}) {
		t.Fatalf("complete scan should replace index, got %#v", got)
	}
}
