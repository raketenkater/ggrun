package backends

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"
)

// The launch preflight asks the backend itself whether a computed placement
// fits, by running `llama-fit-params --fit-print on`: it loads with
// no_alloc=true, builds the exact startup graphs, and prints per-device
// model/context/compute MiB without committing a byte of VRAM, in about a
// second. When that binary is absent, findFitParamsBin returns "" and the whole
// preflight silently does nothing -- so every candidate placement costs a real
// model load instead.
//
// Both build paths (`ggrun backend update` and fresh fork registration) built
// only the llama-server target. The oracle was therefore never installed on a
// machine that got its backends from ggrun, and the effect was invisible:
// preflight is designed to skip quietly when it is unavailable, so a rig simply
// paid ~5 minutes per candidate forever. Measured on GLM 5.3 Flash 2026-09-03:
// first tensor at ~38 s, warmup at ~304 s, per candidate.
//
// BuildFitParamsOracle closes that gap. It is deliberately best-effort: not
// every backend has the target (ik_llama.cpp has no fit-params tool at all), and
// a backend that serves correctly must never fail to install because an optional
// accelerator for the planner could not be compiled.

// FitParamsBinaryName is the oracle's binary name in a build's bin directory.
const FitParamsBinaryName = "llama-fit-params"

// fitParamsBuildTarget is the CMake target that produces it.
const fitParamsBuildTarget = "llama-fit-params"

// fitParamsBuildTimeout bounds the optional build. The target links against
// libraries the server build already produced, so it is normally seconds; the
// bound exists so a pathological configuration cannot hang a backend install.
const fitParamsBuildTimeout = 20 * time.Minute

// FitParamsPath returns where the oracle lives for a given build directory.
func FitParamsPath(buildDir string) string {
	return filepath.Join(buildDir, "bin", FitParamsBinaryName)
}

// HasFitParams reports whether the oracle is present in a build directory.
func HasFitParams(buildDir string) bool {
	fi, err := os.Stat(FitParamsPath(buildDir))
	return err == nil && !fi.IsDir()
}

// BuildFitParamsOracle compiles llama-fit-params into an already-configured
// build directory.
//
// Returns the binary path on success. Every failure is returned as an error for
// the caller to report, never to propagate: callers must treat a failure as
// "this backend has no oracle" and continue, exactly as ggrun behaved before the
// oracle existed. A backend that cannot build it still serves; it just pays a
// real load for each candidate placement instead of a one-second dry run.
func BuildFitParamsOracle(srcDir, buildDir string, jobs int) (string, error) {
	if buildDir == "" {
		return "", fmt.Errorf("no build directory")
	}
	// Always ask CMake to update the target: an existing binary can belong to
	// the previous server revision. CMake handles up-to-date targets cheaply.
	if jobs <= 0 {
		jobs = 1
	}
	ctx, cancel := context.WithTimeout(context.Background(), fitParamsBuildTimeout)
	defer cancel()
	// The build directory is already configured by the server build, so this
	// reuses that cache and only compiles the extra target.
	cmd := exec.CommandContext(ctx, "cmake", "--build", buildDir,
		"--config", "Release",
		"--parallel", strconv.Itoa(jobs),
		"--target", fitParamsBuildTarget)
	if srcDir != "" {
		cmd.Dir = srcDir
	}
	if _, err := cmd.CombinedOutput(); err != nil {
		// The caller may still activate the serving backend. Do not leave an
		// old or partially rebuilt oracle available beside that new server.
		if removeErr := os.Remove(FitParamsPath(buildDir)); removeErr != nil && !os.IsNotExist(removeErr) {
			return "", fmt.Errorf("building %s failed (%v); could not remove stale oracle: %w",
				fitParamsBuildTarget, err, removeErr)
		}
		if ctx.Err() != nil {
			return "", fmt.Errorf("building %s timed out after %s", fitParamsBuildTarget, fitParamsBuildTimeout)
		}
		return "", fmt.Errorf("target %s unavailable in this backend: %w", fitParamsBuildTarget, err)
	}
	path := FitParamsPath(buildDir)
	if !HasFitParams(buildDir) {
		return "", fmt.Errorf("build reported success but produced no %s", path)
	}
	return path, nil
}
