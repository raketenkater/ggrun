// Package devicelist runs a backend's own device listing, `<binary>
// --list-devices`, for every caller in one bounded and cached way.
//
// ggrun asked the backend for its devices from four places (the GPU
// capability check, the Vulkan device order, the Claude reviewer's device
// choice and the post-build smoke test), each with its own timeout, its own
// environment and no cache, so one launch could start the same listing
// process several times. Each caller still chooses its timeout, its child
// environment and how it reads the devices; this package only runs the
// process, grades the answer and remembers a clean one for the rest of the
// ggrun process.
//
// A listing has one of three grades: full (`--list-devices` answered), names
// (a build without `--list-devices`, such as ik_llama, named its devices in
// the buffer-type list of a deliberately invalid --override-tensor, without
// memory), or unprobed (no answer within the bound, or an inconclusive
// fallback). Unprobed means the caller keeps its behaviour from before any
// probe existed.
//
// The cache is in memory only. Its key is the binary (path, size,
// modification time and a digest of its first and last 64 KiB), the exact
// child environment, the bound and whether the fallback is allowed. Only a
// listing that exited 0, or a conclusive fallback, is cached: a timeout, a
// crash or a non-zero exit (a transient driver failure) is asked again next
// time. Cached output is for device names and totals only: the "MiB free" a
// listing prints is stale the moment it is read.
package devicelist

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// Probe is one listing request.
type Probe struct {
	Binary string
	// Env is the child's complete environment; nil inherits this process's
	// environment at the time of the call.
	Env []string
	// Timeout bounds the listing and its fallback together.
	Timeout time.Duration
	// BufferFallback allows the names-grade fallback for a build that
	// answers "unknown argument: --list-devices".
	BufferFallback bool
}

// Grade says how much a listing knows.
type Grade string

const (
	GradeFull     Grade = "full"     // --list-devices answered (Err may still be a non-zero exit)
	GradeNames    Grade = "names"    // the buffer-type fallback named the devices
	GradeUnprobed Grade = "unprobed" // nothing usable: keep the caller's old behaviour
)

// Result is what the backend printed and how it exited.
type Result struct {
	Grade Grade
	// Output is the text the grade was read from: the --list-devices output
	// (full, and unprobed after a timeout) or the fallback's output (names).
	Output string
	// Err is the --list-devices process error: nil on exit status 0, an
	// *exec.ExitError for a non-zero exit, or the start error.
	Err error
	// TimedOut is set when the bound expired before a usable answer.
	TimedOut bool
}

// DefaultTimeout bounds a caller that has no tighter bound of its own.
const DefaultTimeout = 60 * time.Second

var (
	mu    sync.Mutex
	cache = map[string]Result{}
	runs  int // listings actually started; read by tests
)

// Run returns the backend's listing, from the cache when this process already
// ran the identical listing with a clean answer.
func Run(p Probe) Result {
	timeout := p.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	env := p.Env
	if env == nil {
		env = os.Environ()
	}
	key, keyed := cacheKey(p, env, timeout)
	if keyed {
		mu.Lock()
		cached, ok := cache[key]
		mu.Unlock()
		if ok {
			return cached
		}
	}
	res := run(p, timeout)
	if keyed && ((res.Grade == GradeFull && res.Err == nil) || res.Grade == GradeNames) {
		mu.Lock()
		cache[key] = res
		mu.Unlock()
	}
	return res
}

// bufferProbeOverride is a tensor override no build accepts: argument parsing
// stops on it, after listing the registered buffer types, before any model
// or server starts.
const bufferProbeOverride = "ggrun_probe=__ggrun_invalid_buffer__"

func run(p Probe, timeout time.Duration) Result {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	out, err := command(ctx, p, "--list-devices").CombinedOutput()
	if ctx.Err() != nil {
		return Result{Grade: GradeUnprobed, Output: string(out), Err: err, TimedOut: true}
	}
	listing := Result{Grade: GradeFull, Output: string(out), Err: err}
	exitErr, exited := err.(*exec.ExitError)
	if !p.BufferFallback || !exited || exitErr.ExitCode() != 1 ||
		!strings.Contains(strings.ToLower(listing.Output), "unknown argument: --list-devices") {
		return listing
	}
	out, err = command(ctx, p, "--override-tensor", bufferProbeOverride).CombinedOutput()
	exitErr, exited = err.(*exec.ExitError)
	if ctx.Err() != nil || !exited || exitErr.ExitCode() != 1 ||
		!strings.Contains(strings.ToLower(string(out)), "invalid tensor buffer type override: "+bufferProbeOverride) {
		return Result{Grade: GradeUnprobed, Output: listing.Output, Err: listing.Err, TimedOut: ctx.Err() != nil}
	}
	return Result{Grade: GradeNames, Output: string(out), Err: listing.Err}
}

func command(ctx context.Context, p Probe, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, p.Binary, args...)
	if p.Env != nil {
		cmd.Env = p.Env
	}
	mu.Lock()
	runs++
	mu.Unlock()
	return cmd
}

// cacheKey identifies one listing; ok is false when the binary cannot be read,
// so a missing or unreadable binary is listed (and fails) every time.
func cacheKey(p Probe, env []string, timeout time.Duration) (string, bool) {
	fingerprint, ok := binaryFingerprint(p.Binary)
	if !ok {
		return "", false
	}
	h := sha256.New()
	fmt.Fprintf(h, "%s\x00%s\x00%d\x00%t\x00", p.Binary, fingerprint, timeout, p.BufferFallback)
	io.WriteString(h, strings.Join(env, "\x00"))
	return hex.EncodeToString(h.Sum(nil)), true
}

// binaryFingerprint changes whenever the file at path is replaced or
// rewritten, without reading a large binary in full.
func binaryFingerprint(path string) (string, bool) {
	f, err := os.Open(path)
	if err != nil {
		return "", false
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || info.IsDir() {
		return "", false
	}
	const edge = 64 << 10
	h := sha256.New()
	fmt.Fprintf(h, "%d\x00%d\x00", info.Size(), info.ModTime().UnixNano())
	if _, err := io.CopyN(h, f, edge); err != nil && err != io.EOF {
		return "", false
	}
	if info.Size() > 2*edge {
		if _, err := f.Seek(-edge, io.SeekEnd); err != nil {
			return "", false
		}
		if _, err := io.Copy(h, f); err != nil {
			return "", false
		}
	}
	return hex.EncodeToString(h.Sum(nil)), true
}

// StartedListings reports how many listing processes (listings and
// fallbacks) this ggrun process started. Tests use it to prove a listing was or was not repeated.
func StartedListings() int {
	mu.Lock()
	defer mu.Unlock()
	return runs
}
