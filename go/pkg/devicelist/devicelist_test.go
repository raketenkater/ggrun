package devicelist

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func fakeBackend(t *testing.T, body string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake backends are shell scripts")
	}
	path := filepath.Join(t.TempDir(), "llama-server")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+body), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// The same listing in one process runs once; its answer is returned again.
func TestRunCachesACompletedListing(t *testing.T) {
	bin := fakeBackend(t, "echo 'Available devices:'; echo \"  CUDA0: GPU $CUDA_VISIBLE_DEVICES\"\n")
	before := StartedListings()
	env := append(os.Environ(), "CUDA_VISIBLE_DEVICES=1")
	first := Run(Probe{Binary: bin, Env: env, Timeout: 10 * time.Second})
	second := Run(Probe{Binary: bin, Env: env, Timeout: 10 * time.Second})
	if first.Err != nil || first.Grade != GradeFull || !strings.Contains(first.Output, "CUDA0: GPU 1") {
		t.Fatalf("first listing = %+v", first)
	}
	if second != first {
		t.Fatalf("second listing %+v differs from the first %+v", second, first)
	}
	if got := StartedListings() - before; got != 1 {
		t.Fatalf("started %d listings, want 1", got)
	}
}

// NoCache lists again even beside a cached answer and leaves no entry, so a
// caller reading free memory never gets an old figure.
func TestRunNoCacheListsEveryTime(t *testing.T) {
	bin := fakeBackend(t, "echo 'Available devices:'; echo '  Vulkan0: GPU (100 MiB, 50 MiB free)'\n")
	env := os.Environ()
	before := StartedListings()
	Run(Probe{Binary: bin, Env: env, Timeout: 10 * time.Second, NoCache: true})
	Run(Probe{Binary: bin, Env: env, Timeout: 10 * time.Second, NoCache: true})
	Run(Probe{Binary: bin, Env: env, Timeout: 10 * time.Second})
	Run(Probe{Binary: bin, Env: env, Timeout: 10 * time.Second, NoCache: true})
	if got := StartedListings() - before; got != 4 {
		t.Fatalf("started %d listings, want 4", got)
	}
	Run(Probe{Binary: bin, Env: env, Timeout: 10 * time.Second})
	if got := StartedListings() - before; got != 4 {
		t.Fatalf("a cached listing was not reused (%d listings)", got)
	}
}

// A different child environment or timeout is a different listing.
func TestRunKeysOnEnvironmentAndTimeout(t *testing.T) {
	bin := fakeBackend(t, "echo \"  CUDA0: GPU $CUDA_VISIBLE_DEVICES\"\n")
	before := StartedListings()
	a := Run(Probe{Binary: bin, Env: []string{"CUDA_VISIBLE_DEVICES=0"}, Timeout: 10 * time.Second})
	b := Run(Probe{Binary: bin, Env: []string{"CUDA_VISIBLE_DEVICES=2"}, Timeout: 10 * time.Second})
	Run(Probe{Binary: bin, Env: []string{"CUDA_VISIBLE_DEVICES=2"}, Timeout: 11 * time.Second})
	if !strings.Contains(a.Output, "GPU 0") || !strings.Contains(b.Output, "GPU 2") {
		t.Fatalf("listings did not see their own environment: %q / %q", a.Output, b.Output)
	}
	if got := StartedListings() - before; got != 3 {
		t.Fatalf("started %d listings, want 3", got)
	}
}

// A rewritten binary at the same path is listed again.
func TestRunListsARewrittenBinaryAgain(t *testing.T) {
	bin := fakeBackend(t, "echo 'Available devices:'; echo '  CUDA0: old'\n")
	old := Run(Probe{Binary: bin, Timeout: 10 * time.Second})
	if err := os.WriteFile(bin, []byte("#!/bin/sh\necho 'Available devices:'; echo '  CUDA0: new'\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	fresh := Run(Probe{Binary: bin, Timeout: 10 * time.Second})
	if !strings.Contains(old.Output, "old") || !strings.Contains(fresh.Output, "new") {
		t.Fatalf("rewritten binary answered from the cache: %q then %q", old.Output, fresh.Output)
	}
}

// A listing that does not finish is bounded, reported and not cached.
func TestRunBoundsAHungListingAndDoesNotCacheIt(t *testing.T) {
	bin := fakeBackend(t, "exec sleep 30\n")
	before := StartedListings()
	start := time.Now()
	res := Run(Probe{Binary: bin, Timeout: 200 * time.Millisecond})
	if elapsed := time.Since(start); elapsed > 10*time.Second {
		t.Fatalf("hung listing took %s", elapsed)
	}
	if !res.TimedOut || res.Err == nil || res.Grade != GradeUnprobed {
		t.Fatalf("hung listing = %+v, want a timeout", res)
	}
	Run(Probe{Binary: bin, Timeout: 200 * time.Millisecond})
	if got := StartedListings() - before; got != 2 {
		t.Fatalf("started %d listings, want 2 (a timeout is never cached)", got)
	}
}

// A failing exit keeps its *exec.ExitError for the callers, and is listed
// again next time: a transient driver failure must not stick.
func TestRunKeepsTheExitErrorAndDoesNotCacheIt(t *testing.T) {
	bin := fakeBackend(t, "echo 'error: unknown argument: --list-devices'; exit 1\n")
	before := StartedListings()
	res := Run(Probe{Binary: bin, Timeout: 10 * time.Second})
	exitErr, ok := res.Err.(*exec.ExitError)
	if !ok || exitErr.ExitCode() != 1 || res.TimedOut || res.Grade != GradeFull || !strings.Contains(res.Output, "unknown argument") {
		t.Fatalf("result = %+v", res)
	}
	Run(Probe{Binary: bin, Timeout: 10 * time.Second})
	if got := StartedListings() - before; got != 2 {
		t.Fatalf("started %d listings, want 2 (no fallback was allowed, no failure is cached)", got)
	}
}

const fallbackBackend = "case \"$1\" in\n" +
	"--list-devices) echo 'error: unknown argument: --list-devices'; exit 1;;\n" +
	"--override-tensor) [ \"$2\" = 'ggrun_probe=__ggrun_invalid_buffer__' ] || exit 99\n" +
	"  %s\n" +
	"  echo 'error: invalid tensor buffer type override: ggrun_probe=__ggrun_invalid_buffer__'; exit 1;;\n" +
	"esac\nexit 99\n"

// A build without --list-devices names its devices through the buffer-type
// fallback: the names grade, cached like a clean listing.
func TestRunBufferFallbackGradesNames(t *testing.T) {
	bin := fakeBackend(t, strings.Replace(fallbackBackend, "%s", "echo 'Available buffer types:'; echo '  CPU'; echo '  CUDA0'", 1))
	before := StartedListings()
	res := Run(Probe{Binary: bin, Timeout: 10 * time.Second, BufferFallback: true})
	if res.Grade != GradeNames || !strings.Contains(res.Output, "CUDA0") || res.TimedOut {
		t.Fatalf("result = %+v", res)
	}
	if again := Run(Probe{Binary: bin, Timeout: 10 * time.Second, BufferFallback: true}); again != res {
		t.Fatalf("second answer %+v differs", again)
	}
	if got := StartedListings() - before; got != 2 {
		t.Fatalf("started %d processes, want 2 (one listing, one fallback)", got)
	}
}

// A fallback that does not answer the known way is unprobed, never a grade.
func TestRunInconclusiveFallbackIsUnprobed(t *testing.T) {
	bin := fakeBackend(t, "case \"$1\" in\n--list-devices) echo 'error: unknown argument: --list-devices'; exit 1;;\nesac\necho 'Available buffer types:'; echo '  CUDA0'; exit 139\n")
	before := StartedListings()
	res := Run(Probe{Binary: bin, Timeout: 10 * time.Second, BufferFallback: true})
	if res.Grade != GradeUnprobed || res.TimedOut || !strings.Contains(res.Output, "unknown argument") {
		t.Fatalf("result = %+v", res)
	}
	Run(Probe{Binary: bin, Timeout: 10 * time.Second, BufferFallback: true})
	if got := StartedListings() - before; got != 4 {
		t.Fatalf("started %d processes, want 4 (unprobed is never cached)", got)
	}
}

// A missing binary is an ordinary start error, tried again on every call.
func TestRunMissingBinary(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "absent")
	before := StartedListings()
	if res := Run(Probe{Binary: missing, Timeout: time.Second}); res.Err == nil || res.TimedOut {
		t.Fatalf("missing binary = %+v", res)
	}
	Run(Probe{Binary: missing, Timeout: time.Second})
	if got := StartedListings() - before; got != 2 {
		t.Fatalf("started %d listings, want 2", got)
	}
}

// A clean answer that names only the CPU (a driver that was not ready) is
// asked again; so is any answer older than the TTL.
func TestRunDoesNotKeepCPUOnlyOrOldAnswers(t *testing.T) {
	cpu := fakeBackend(t, "echo 'Available devices:'; echo '  (none)'\n")
	before := StartedListings()
	Run(Probe{Binary: cpu, Timeout: 10 * time.Second})
	Run(Probe{Binary: cpu, Timeout: 10 * time.Second})
	if got := StartedListings() - before; got != 2 {
		t.Fatalf("started %d listings for a CPU-only answer, want 2", got)
	}

	gpu := fakeBackend(t, "echo 'Available devices:'; echo '  Vulkan0: GPU (1000 MiB, 1 MiB free)'\n")
	clock := time.Now()
	now = func() time.Time { return clock }
	t.Cleanup(func() { now = time.Now })
	before = StartedListings()
	Run(Probe{Binary: gpu, Timeout: 10 * time.Second})
	clock = clock.Add(cacheTTL / 2)
	Run(Probe{Binary: gpu, Timeout: 10 * time.Second})
	clock = clock.Add(cacheTTL)
	Run(Probe{Binary: gpu, Timeout: 10 * time.Second})
	if got := StartedListings() - before; got != 2 {
		t.Fatalf("started %d listings, want 2 (one cached within the TTL, one after it)", got)
	}
}

// A rebuilt library beside an unchanged server binary is listed again.
func TestRunListsAgainAfterALibraryChanges(t *testing.T) {
	bin := fakeBackend(t, "echo 'Available devices:'; echo '  CUDA0: GPU'\n")
	lib := filepath.Join(filepath.Dir(bin), "libggml-cuda.so")
	if err := os.WriteFile(lib, []byte("one"), 0o644); err != nil {
		t.Fatal(err)
	}
	before := StartedListings()
	Run(Probe{Binary: bin, Timeout: 10 * time.Second})
	if err := os.WriteFile(lib, []byte("rebuilt"), 0o644); err != nil {
		t.Fatal(err)
	}
	Run(Probe{Binary: bin, Timeout: 10 * time.Second})
	if got := StartedListings() - before; got != 2 {
		t.Fatalf("started %d listings, want 2", got)
	}
}

// Zero sets no bound: a listing slower than any default still answers.
func TestRunZeroTimeoutIsUnbounded(t *testing.T) {
	bin := fakeBackend(t, "sleep 1; echo 'Available devices:'; echo '  CUDA0: slow'\n")
	res := Run(Probe{Binary: bin})
	if res.TimedOut || res.Err != nil || !strings.Contains(res.Output, "CUDA0: slow") {
		t.Fatalf("result = %+v", res)
	}
}
