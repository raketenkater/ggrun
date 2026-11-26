package models

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	discoveryIndexVersion  = 1
	discoveryIndexFile     = "discovered-models.json"
	defaultScanMaxEntries  = 5_000_000
	defaultScanMaxTime     = 10 * time.Minute
	rememberScanMaxEntries = 200_000
	rememberScanMaxTime    = 30 * time.Second
)

var errScanStopped = errors.New("model discovery limit reached")

// ScanResult describes one bounded scan of the computer's local filesystems.
// Paths contains runnable GGUF entrypoints: a single-file model or shard 00001.
// The TUI performs the more expensive header parsing only after this filename
// pass has completed.
type ScanResult struct {
	Paths         []string
	Roots         []string
	Entries       int
	SkippedErrors int
	Truncated     bool
	Duration      time.Duration
	// Visited lists the roots whose walk actually started. Skipped lists
	// directories inside them that were not entered (mount boundaries such as
	// an unmounted automount, unreadable directories). A complete scan proves
	// absence only for paths covered by Visited and not by Skipped.
	Visited []string
	Skipped []string
}

type mountRecord struct {
	path   string
	fsType string
}

// ScanComputer discovers GGUFs on local disks without walking pseudo,
// container, or network filesystems. It is intentionally bounded: an unusual
// machine with millions of directory entries returns the models found so far
// instead of leaving the TUI apparently frozen forever.
func ScanComputer(configuredRoot string) ScanResult {
	roots, boundaries := computerScanPlan(configuredRoot)
	return scanRoots(roots, boundaries, defaultScanMaxEntries, defaultScanMaxTime)
}

// scanRoots is the testable scanner core. boundaries are mountpoints that must
// not be crossed while walking another root; each local mount is scanned once
// as its own root by ScanComputer.
func scanRoots(roots, boundaries []string, maxEntries int, maxDuration time.Duration) ScanResult {
	started := time.Now()
	result := ScanResult{Roots: append([]string(nil), roots...)}
	seen := make(map[string]bool)
	visited := make(map[string]bool)
	skipped := make(map[string]bool)
	boundarySet := make(map[string]bool, len(boundaries))
	for _, path := range boundaries {
		if clean := cleanAbs(path); clean != "" {
			boundarySet[clean] = true
		}
	}
	if maxEntries <= 0 {
		maxEntries = defaultScanMaxEntries
	}
	if maxDuration <= 0 {
		maxDuration = defaultScanMaxTime
	}

	for _, rawRoot := range roots {
		root := cleanAbs(rawRoot)
		if root == "" {
			continue
		}
		walkErr := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
			result.Entries++
			if result.Entries > maxEntries || time.Since(started) > maxDuration {
				result.Truncated = true
				return errScanStopped
			}
			if walkErr != nil {
				result.SkippedErrors++
				if clean := cleanAbs(path); clean == root {
					skipped[root] = true
					return filepath.SkipDir
				} else if entry != nil && entry.IsDir() {
					skipped[clean] = true
					return filepath.SkipDir
				}
				return nil
			}
			clean := cleanAbs(path)
			if clean == root {
				visited[root] = true
			}
			if entry.IsDir() {
				if clean != root && boundarySet[clean] {
					skipped[clean] = true
					return filepath.SkipDir
				}
				if clean != root && skipDiscoveryDirectory(clean, entry.Name()) {
					return filepath.SkipDir
				}
				return nil
			}
			if !isGGUFEntrypoint(entry.Name()) {
				return nil
			}
			info, err := os.Stat(path) // follow file symlinks, never directory symlinks
			if err != nil || info.IsDir() {
				result.SkippedErrors++
				return nil
			}
			if _, _, err := ResolveGGUFShardFiles(path); err != nil {
				// An interrupted split-model download is not runnable and must not
				// enter the persistent discovery cache as a deceptively small model.
				return nil
			}
			// Test fixtures and interrupted writes otherwise show up in the TUI as models.
			if ok, err := hasGGUFHeader(path); err != nil {
				result.SkippedErrors++
				return nil
			} else if !ok {
				return nil
			}
			real := clean
			if resolved, err := filepath.EvalSymlinks(clean); err == nil {
				real = cleanAbs(resolved)
			}
			if real != "" && !seen[real] {
				seen[real] = true
				result.Paths = append(result.Paths, clean)
			}
			return nil
		})
		if errors.Is(walkErr, errScanStopped) {
			break
		}
		if walkErr != nil {
			result.SkippedErrors++
		}
	}
	sort.Slice(result.Paths, func(i, j int) bool {
		return strings.ToLower(result.Paths[i]) < strings.ToLower(result.Paths[j])
	})
	for path := range visited {
		result.Visited = append(result.Visited, path)
		delete(skipped, path) // a boundary that was also walked as its own root
	}
	for path := range skipped {
		result.Skipped = append(result.Skipped, path)
	}
	sort.Strings(result.Visited)
	sort.Strings(result.Skipped)
	result.Duration = time.Since(started)
	return result
}

// hasGGUFHeader reports whether path starts with a plausible GGUF header:
// magic, a version of at least 1, then the tensor and KV counts (24 bytes).
func hasGGUFHeader(path string) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	var hdr [24]byte
	if _, err := io.ReadFull(f, hdr[:]); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return false, nil // shorter than a header
		}
		return false, err
	}
	return bytes.Equal(hdr[:4], []byte("GGUF")) && binary.LittleEndian.Uint32(hdr[4:8]) >= 1, nil
}

func isGGUFEntrypoint(name string) bool {
	lower := strings.ToLower(name)
	if !strings.HasSuffix(lower, ".gguf") {
		return false
	}
	// llama.cpp repositories ship tokenizer-only GGUF fixtures under this
	// name, and multimodal projectors are companions rather than launchable
	// language models. A whole-computer scan crosses many source checkouts, so
	// keeping these would bury the user's real models in hundreds of false
	// positives.
	if strings.HasPrefix(lower, "ggml-vocab-") || strings.Contains(lower, "mmproj") {
		return false
	}
	match := shardName.FindStringSubmatch(name)
	if len(match) == 2 {
		marker := strings.LastIndex(lower, "-")
		if marker < 9 {
			return false
		}
		// shardName guarantees the suffix shape; the five bytes immediately
		// before "-of-" are therefore the shard number.
		of := strings.LastIndex(lower, "-of-")
		return of >= 5 && lower[of-5:of] == "00001"
	}
	return true
}

func skipDiscoveryDirectory(path, name string) bool {
	for _, prefix := range []string{
		"/proc", "/sys", "/dev", "/run", "/var/lib/docker",
		"/var/lib/containerd", "/var/lib/containers", "/var/lib/snapd",
	} {
		if path == prefix || strings.HasPrefix(path, prefix+string(filepath.Separator)) {
			return true
		}
	}
	switch strings.ToLower(name) {
	case ".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "venv", ".tox", "$recycle.bin", "system volume information", ".snapshots":
		return true
	default:
		return false
	}
}

func cleanAbs(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	if abs, err := filepath.Abs(path); err == nil {
		path = abs
	}
	return filepath.Clean(path)
}

// readMountInfo and triggerAutomounts are variables so tests can inject a fake
// mount table without real mounts.
var (
	readMountInfo = func() (string, error) {
		data, err := os.ReadFile("/proc/self/mountinfo")
		return string(data), err
	}
	triggerAutomounts = defaultTriggerAutomounts
)

const automountTriggerTimeout = 3 * time.Second

// defaultTriggerAutomounts touches each given autofs mountpoint so the kernel
// mounts it, waiting at most automountTriggerTimeout in total. A dead disk
// blocks only abandoned goroutines, never the scan.
func defaultTriggerAutomounts(paths []string) {
	if len(paths) == 0 {
		return
	}
	var wg sync.WaitGroup
	for _, path := range paths {
		wg.Add(1)
		go func(path string) {
			defer wg.Done()
			if f, err := os.Open(path); err == nil {
				_, _ = f.ReadDir(1) // opening a systemd automount dir mounts it
				_ = f.Close()
			}
		}(path)
	}
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(automountTriggerTimeout):
	}
}

func computerScanPlan(configuredRoot string) ([]string, []string) {
	if runtime.GOOS == "linux" {
		if data, err := readMountInfo(); err == nil {
			if pending := pendingAutofsMounts(data); len(pending) > 0 {
				triggerAutomounts(pending)
				if again, err := readMountInfo(); err == nil {
					data = again
				}
			}
			if roots, boundaries := linuxMountScanPlan(data); len(roots) > 0 {
				return roots, boundaries
			}
		}
	}

	var roots []string
	addExistingRoot := func(path string) {
		path = cleanAbs(path)
		if path == "" {
			return
		}
		if info, err := os.Stat(path); err == nil && info.IsDir() {
			roots = append(roots, path)
		}
	}
	addExistingRoot(configuredRoot)
	if home, err := os.UserHomeDir(); err == nil {
		addExistingRoot(home)
	}
	if runtime.GOOS == "windows" {
		for drive := 'C'; drive <= 'Z'; drive++ {
			addExistingRoot(string(drive) + `:\`)
		}
	} else {
		for _, path := range []string{"/Volumes", "/mnt", "/media", "/data", "/srv", "/opt"} {
			addExistingRoot(path)
		}
		addExistingRoot("/")
	}
	return uniquePaths(roots), nil
}

// pendingAutofsMounts returns autofs mountpoints with no real filesystem
// mounted over them yet.
func pendingAutofsMounts(mountInfo string) []string {
	state := map[string]bool{} // path -> still only autofs
	var order []string
	for _, rec := range parseMountInfo(mountInfo) {
		path := cleanAbs(rec.path)
		if path == "" {
			continue
		}
		auto := strings.EqualFold(rec.fsType, "autofs")
		if prev, ok := state[path]; !ok {
			order = append(order, path)
			state[path] = auto
		} else {
			state[path] = prev && auto
		}
	}
	var out []string
	for _, path := range order {
		if state[path] {
			out = append(out, path)
		}
	}
	return out
}

func parseMountInfo(mountInfo string) []mountRecord {
	var records []mountRecord
	for _, line := range strings.Split(mountInfo, "\n") {
		fields := strings.Fields(line)
		sep := -1
		for i, field := range fields {
			if field == "-" {
				sep = i
				break
			}
		}
		if sep < 0 || len(fields) <= sep+1 || len(fields) < 5 {
			continue
		}
		records = append(records, mountRecord{
			path:   decodeMountInfoPath(fields[4]),
			fsType: fields[sep+1],
		})
	}
	return records
}

func linuxMountScanPlan(mountInfo string) ([]string, []string) {
	records := parseMountInfo(mountInfo)

	local := make(map[string]bool)
	boundaries := make(map[string]bool)
	for _, record := range records {
		path := cleanAbs(record.path)
		if path == "" {
			continue
		}
		boundaries[path] = true
		if isLocalModelFilesystem(record.fsType, path) {
			local[path] = true
		}
	}
	roots := make([]string, 0, len(local))
	for path := range local {
		if info, err := os.Stat(path); err == nil && info.IsDir() {
			roots = append(roots, path)
		}
	}
	// Search mounted data disks before / so a bounded scan finds the paths the
	// user is most likely missing even when the root filesystem is enormous.
	sort.Slice(roots, func(i, j int) bool {
		if roots[i] == "/" {
			return false
		}
		if roots[j] == "/" {
			return true
		}
		return strings.ToLower(roots[i]) < strings.ToLower(roots[j])
	})
	allBoundaries := make([]string, 0, len(boundaries))
	for path := range boundaries {
		allBoundaries = append(allBoundaries, path)
	}
	return roots, allBoundaries
}

func decodeMountInfoPath(path string) string {
	replacer := strings.NewReplacer(
		`\040`, " ",
		`\011`, "\t",
		`\012`, "\n",
		`\134`, `\`,
	)
	return replacer.Replace(path)
}

func isLocalModelFilesystem(fsType, mountpoint string) bool {
	fsType = strings.ToLower(strings.TrimSpace(fsType))
	switch fsType {
	case "proc", "sysfs", "devtmpfs", "devpts", "tmpfs", "cgroup", "cgroup2",
		"securityfs", "pstore", "bpf", "autofs", "mqueue", "hugetlbfs",
		"debugfs", "tracefs", "fusectl", "configfs", "binfmt_misc", "nsfs",
		"squashfs", "ramfs":
		return false
	case "overlay":
		return mountpoint == "/"
	}
	for _, prefix := range []string{"nfs", "cifs", "smb", "ceph", "afs", "9p", "sshfs", "fuse.sshfs"} {
		if strings.HasPrefix(fsType, prefix) {
			return false
		}
	}
	return true
}

func uniquePaths(paths []string) []string {
	seen := make(map[string]bool)
	out := make([]string, 0, len(paths))
	for _, path := range paths {
		path = cleanAbs(path)
		if path != "" && !seen[path] {
			seen[path] = true
			out = append(out, path)
		}
	}
	return out
}

type discoveryIndex struct {
	Version   int      `json:"version"`
	ScannedAt string   `json:"scanned_at"`
	Paths     []string `json:"paths"`
}

// SaveDiscoveredScan persists a system scan. A truncated scan has not proved
// that paths outside the visited prefix disappeared, so it merges with the
// prior index. A complete scan replaces only entries it actually covered: a
// root that was not visited (an automount disk not mounted yet, an unreadable
// root) or a directory it skipped keeps its saved entries.
func SaveDiscoveredScan(cacheDir string, result ScanResult) error {
	paths := append([]string(nil), result.Paths...)
	if result.Truncated {
		paths = append(LoadDiscoveredPaths(cacheDir), paths...)
	} else {
		for _, prior := range LoadDiscoveredPaths(cacheDir) {
			if !scanCovers(result, prior) {
				paths = append(paths, prior)
			}
		}
	}
	return SaveDiscoveredPaths(cacheDir, paths)
}

// scanCovers reports whether the scan walked the place path lives in: the
// longest visited-root or skipped-directory prefix of path must be a root.
func scanCovers(result ScanResult, path string) bool {
	path = cleanAbs(path)
	best, bestVisited := -1, false
	check := func(dirs []string, visited bool) {
		for _, dir := range dirs {
			dir = cleanAbs(dir)
			if dir == "" || !pathWithin(dir, path) {
				continue
			}
			if len(dir) > best || (len(dir) == best && visited) {
				best, bestVisited = len(dir), visited
			}
		}
	}
	check(result.Skipped, false)
	check(result.Visited, true)
	return best >= 0 && bestVisited
}

func pathWithin(dir, path string) bool {
	if dir == path {
		return true
	}
	if !strings.HasSuffix(dir, string(filepath.Separator)) {
		dir += string(filepath.Separator)
	}
	return strings.HasPrefix(path, dir)
}

// RememberModelDir adds the runnable GGUFs under dir to the discovered index,
// keeping every prior entry, so a model downloaded outside the configured model
// directory appears in the TUI without a full computer scan. The walk is
// bounded because dir is user-chosen and may be a large tree.
func RememberModelDir(cacheDir, dir string) error {
	result := scanRoots([]string{dir}, nil, rememberScanMaxEntries, rememberScanMaxTime)
	if len(result.Paths) == 0 {
		return nil
	}
	return SaveDiscoveredScan(cacheDir, ScanResult{Paths: result.Paths, Truncated: true})
}

// SaveDiscoveredPaths makes a system scan available on later TUI starts.
func SaveDiscoveredPaths(cacheDir string, paths []string) error {
	cacheDir = strings.TrimSpace(cacheDir)
	if cacheDir == "" {
		return errors.New("model discovery cache directory is empty")
	}
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return err
	}
	index := discoveryIndex{
		Version:   discoveryIndexVersion,
		ScannedAt: time.Now().UTC().Format(time.RFC3339),
		Paths:     uniquePaths(paths),
	}
	tmp, err := os.CreateTemp(cacheDir, ".discovered-models-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	enc := json.NewEncoder(tmp)
	enc.SetIndent("", "  ")
	if err := enc.Encode(index); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, filepath.Join(cacheDir, discoveryIndexFile))
}

// LoadDiscoveredPaths returns only paths that still exist and remain valid
// GGUF entrypoints. A removed or unplugged disk therefore cannot leave a dead
// launch row in the TUI.
func LoadDiscoveredPaths(cacheDir string) []string {
	cacheDir = strings.TrimSpace(cacheDir)
	if cacheDir == "" {
		return nil
	}
	data, err := os.ReadFile(filepath.Join(cacheDir, discoveryIndexFile))
	if err != nil {
		return nil
	}
	var index discoveryIndex
	if json.Unmarshal(data, &index) != nil || index.Version != discoveryIndexVersion {
		return nil
	}
	paths := make([]string, 0, len(index.Paths))
	for _, path := range uniquePaths(index.Paths) {
		if !isGGUFEntrypoint(filepath.Base(path)) {
			continue
		}
		if _, _, err := ResolveGGUFShardFiles(path); err == nil {
			paths = append(paths, path)
		}
	}
	return paths
}
