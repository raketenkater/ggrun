package models

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
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
				if entry != nil && entry.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
			clean := cleanAbs(path)
			if entry.IsDir() {
				if clean != root && (boundarySet[clean] || skipDiscoveryDirectory(clean, entry.Name())) {
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
	result.Duration = time.Since(started)
	return result
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

func computerScanPlan(configuredRoot string) ([]string, []string) {
	if runtime.GOOS == "linux" {
		if data, err := os.ReadFile("/proc/self/mountinfo"); err == nil {
			if roots, boundaries := linuxMountScanPlan(string(data)); len(roots) > 0 {
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

func linuxMountScanPlan(mountInfo string) ([]string, []string) {
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

// SaveDiscoveredScan persists a completed system scan. A truncated scan has
// not proved that paths outside the visited prefix disappeared, so merge its
// findings with the still-valid prior index instead of replacing that index.
func SaveDiscoveredScan(cacheDir string, result ScanResult) error {
	paths := append([]string(nil), result.Paths...)
	if result.Truncated {
		paths = append(LoadDiscoveredPaths(cacheDir), paths...)
	}
	return SaveDiscoveredPaths(cacheDir, paths)
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
