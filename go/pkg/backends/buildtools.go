package backends

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// FitParamsTool is the backend's no-allocation memory oracle. Launch uses it to
// price a complete configuration without reading weights; without it every
// admission falls back to a contained full model load.
const FitParamsTool = "llama-fit-params"

// OptionalBuildTargets lists the auxiliary targets a configured CMake build tree
// defines beyond llama-server. A fork without the target is not an error: its
// launches keep the bounded contained-measurement fallback.
func OptionalBuildTargets(buildDir string) []string {
	if BuildDefinesTarget(buildDir, FitParamsTool) {
		return []string{FitParamsTool}
	}
	return nil
}

// BuildDefinesTarget reports whether a configured build tree defines target.
// Every CMake generator creates <binary-dir>/CMakeFiles/<target>.dir during
// configure, so this needs neither the generator nor a build.
func BuildDefinesTarget(buildDir, target string) bool {
	if buildDir == "" || target == "" {
		return false
	}
	for _, pattern := range []string{
		filepath.Join(buildDir, "CMakeFiles", target+".dir"),
		filepath.Join(buildDir, "*", "CMakeFiles", target+".dir"),
		filepath.Join(buildDir, "*", "*", "CMakeFiles", target+".dir"),
	} {
		if matches, _ := filepath.Glob(pattern); len(matches) > 0 {
			return true
		}
	}
	return false
}

// CheckFitParamsTool proves that a built oracle starts: it must print its usage
// and name the --fit-print mode launch relies on. A binary that cannot load its
// shared libraries fails here instead of at the first launch.
func CheckFitParamsTool(binary string) error {
	info, err := os.Stat(binary)
	if err != nil || info.IsDir() {
		return fmt.Errorf("%s missing: %s", FitParamsTool, binary)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o111 == 0 {
		return fmt.Errorf("%s is not executable: %s", FitParamsTool, binary)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, runErr := exec.CommandContext(ctx, binary, "--help").CombinedOutput()
	if ctx.Err() != nil {
		return fmt.Errorf("%s --help timed out", FitParamsTool)
	}
	if runErr != nil {
		return fmt.Errorf("%s --help failed: %w: %s", FitParamsTool, runErr, lastLine(string(out)))
	}
	if !strings.Contains(string(out), "--fit-print") {
		return fmt.Errorf("%s --help does not offer --fit-print", FitParamsTool)
	}
	return nil
}

// MissingBuildTools reports auxiliary targets the build tree defines but whose
// binaries are absent or do not start. A non-empty result means the build is
// incomplete even though its llama-server may pass conformance. A target this
// build already tried and failed is not reported again: rebuilding the same
// source would fail the same way, so only a new build retries it.
func MissingBuildTools(buildDir string) []string {
	var missing []string
	for _, target := range OptionalBuildTargets(buildDir) {
		if _, err := os.Stat(unavailableMarker(buildDir, target)); err == nil {
			continue
		}
		if CheckFitParamsTool(filepath.Join(buildDir, "bin", target)) != nil {
			missing = append(missing, target)
		}
	}
	return missing
}

// DiscardBrokenBuildTool removes an auxiliary binary that did not build or
// failed its check, so launch never selects it, and records why in the build
// directory. A missing oracle only costs the contained fallback; a broken one
// fails a launch.
func DiscardBrokenBuildTool(buildDir, target string, reason error) {
	_ = os.Remove(filepath.Join(buildDir, "bin", target))
	_ = os.MkdirAll(filepath.Join(buildDir, "bin"), 0o755)
	_ = os.WriteFile(unavailableMarker(buildDir, target), []byte(fmt.Sprintf("%v\n", reason)), 0o644)
}

func unavailableMarker(buildDir, target string) string {
	return filepath.Join(buildDir, "bin", ".ggrun-unavailable-"+target)
}

func lastLine(value string) string {
	lines := strings.Split(strings.TrimSpace(value), "\n")
	return strings.TrimSpace(lines[len(lines)-1])
}
