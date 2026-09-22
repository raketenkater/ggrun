package tui

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const (
	latestLaunchVersion = 1
	latestLaunchFile    = "latest-tui-launch.json"
)

// ErrNoLatestLaunch means the user has not launched a model from the TUI yet.
var ErrNoLatestLaunch = errors.New("no saved TUI launch configuration")

type latestLaunchRecord struct {
	Version int           `json:"version"`
	SavedAt string        `json:"saved_at"`
	Request LaunchRequest `json:"request"`
}

// SaveLatestLaunch records the exact request returned by the TUI immediately
// before the CLI launches or tunes it. Failed launch attempts are deliberately
// retained: after correcting a backend or memory problem, "Run latest
// configuration" is specifically meant to retry that configuration.
func SaveLatestLaunch(cacheDir string, req *LaunchRequest) error {
	cacheDir = strings.TrimSpace(cacheDir)
	if cacheDir == "" {
		return errors.New("latest-launch cache directory is empty")
	}
	if req == nil || strings.TrimSpace(req.ModelPath) == "" {
		return errors.New("latest launch has no model path")
	}
	if err := os.MkdirAll(cacheDir, 0o700); err != nil {
		return err
	}

	clean := *req
	// Only runnable model state belongs in this record. These mutually
	// exclusive TUI actions are never expected alongside ModelPath, but clear
	// them defensively so a future caller cannot turn replay into an update,
	// backend installation, or download operation.
	clean.Update = false
	clean.BackendArgs = nil
	clean.BackendInstallError = ""
	clean.DownloadRepo = ""
	clean.DownloadQuant = ""
	record := latestLaunchRecord{
		Version: latestLaunchVersion,
		SavedAt: time.Now().UTC().Format(time.RFC3339),
		Request: clean,
	}

	tmp, err := os.CreateTemp(cacheDir, ".latest-tui-launch-*.tmp")
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
	if err := enc.Encode(record); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}

	destination := filepath.Join(cacheDir, latestLaunchFile)
	if runtime.GOOS == "windows" {
		// Windows does not replace an existing file with os.Rename. The record is
		// already private and recoverable, so remove only this exact cache file.
		_ = os.Remove(destination)
	}
	return os.Rename(tmpPath, destination)
}

// LoadLatestLaunch returns a defensive copy of the last TUI model request and
// the time it was recorded. It does not require the model to still be mounted;
// the TUI performs that live check and can explain the unavailable path.
func LoadLatestLaunch(cacheDir string) (*LaunchRequest, time.Time, error) {
	cacheDir = strings.TrimSpace(cacheDir)
	if cacheDir == "" {
		return nil, time.Time{}, ErrNoLatestLaunch
	}
	data, err := os.ReadFile(filepath.Join(cacheDir, latestLaunchFile))
	if errors.Is(err, os.ErrNotExist) {
		return nil, time.Time{}, ErrNoLatestLaunch
	}
	if err != nil {
		return nil, time.Time{}, err
	}
	var record latestLaunchRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return nil, time.Time{}, fmt.Errorf("read latest TUI launch: %w", err)
	}
	if record.Version != latestLaunchVersion {
		return nil, time.Time{}, fmt.Errorf("unsupported latest TUI launch version %d", record.Version)
	}
	if strings.TrimSpace(record.Request.ModelPath) == "" {
		return nil, time.Time{}, errors.New("saved TUI launch has no model path")
	}
	// Treat the cache as data, not authority. Even if it was manually edited,
	// replay may only produce a model launch—not a different state-changing TUI
	// action that happens to share LaunchRequest.
	record.Request.Update = false
	record.Request.BackendArgs = nil
	record.Request.BackendInstallError = ""
	record.Request.DownloadRepo = ""
	record.Request.DownloadQuant = ""
	savedAt, err := time.Parse(time.RFC3339, record.SavedAt)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("invalid latest TUI launch timestamp: %w", err)
	}
	req := record.Request
	return &req, savedAt, nil
}
