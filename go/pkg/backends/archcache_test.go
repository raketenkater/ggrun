package backends

import (
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// isolateArchCache points the persisted cache at a fresh directory and starts
// from an empty in-process cache, as a new process would.
func isolateArchCache(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("LLM_CACHE_DIR", dir)
	resetArchCacheForTest()
	t.Cleanup(resetArchCacheForTest)
	return filepath.Join(dir, archCacheFileName)
}

func archScanCount() int {
	archCache.mu.Lock()
	defer archCache.mu.Unlock()
	return archCache.scans
}

// assertMatchesLiteralScan is the invariant: the token-set answer equals the
// literal scan's answer for every query.
func assertMatchesLiteralScan(t *testing.T, path string, queries []string) {
	t.Helper()
	for _, q := range queries {
		wantSup, wantProbed := backendSupportsArchByScan(path, strings.ToLower(strings.TrimSpace(q)))
		if len(strings.TrimSpace(q)) < 2 {
			wantSup, wantProbed = false, false
		}
		gotSup, gotProbed := BackendSupportsArch(path, q)
		if gotSup != wantSup || gotProbed != wantProbed {
			t.Errorf("%s: BackendSupportsArch(%q) = (%v,%v), literal scan = (%v,%v)",
				filepath.Base(path), q, gotSup, gotProbed, wantSup, wantProbed)
		}
	}
}

func TestArchTokenSetMatchesLiteralScan(t *testing.T) {
	isolateArchCache(t)
	long64 := strings.Repeat("a", 63) + "z"
	long65 := strings.Repeat("b", 64) + "z"
	body := strings.Join([]string{
		"startarch", // first run of the file: no NUL before it
		"/home/u/src/fork-llama.cpp-add-laguna/build/x.o",
		"qwen3moe",
		"llama",
		"LLAMA4",
		"hy_v3",
		"gpt-oss",
		"a b",
		long64,
		long65,
		"x",
		"",
		"mid\x01arch",
		"endarch",  // last NUL-terminated run
		"tailarch", // unterminated: no NUL after it
	}, "\x00")
	path := writeProbeFile(t, "strings.bin", []byte(body))

	queries := []string{
		"startarch", "laguna", "build", "x.o", "qwen3moe", "qwen3", "3moe", "wen3moe", "qwen3moe2",
		"llama", "llama4", "LLAMA4", "hy_v3", "HY_V3", "gpt-oss", "a b", long64, long65,
		strings.Repeat("b", 64), "x", "mid", "arch", "mid\x01arch", "endarch", "tailarch",
		"  llama  ", "l", "", "bogusarch",
	}
	assertMatchesLiteralScan(t, path, queries)

	// The cases the brief names, stated directly as well.
	for q, want := range map[string]bool{
		"laguna": false, "qwen3": false, "3moe": false, "startarch": false,
		"endarch": true, "tailarch": false, "qwen3moe": true, long64: true,
		// Too long for a token set: answered by the literal scan.
		long65: true, strings.Repeat("b", 64): false,
	} {
		if got, probed := BackendSupportsArch(path, q); got != want || !probed {
			t.Errorf("BackendSupportsArch(%q) = (%v,%v), want (%v,true)", q, got, probed, want)
		}
	}
}

func TestArchTokenSetMatchesLiteralScanOnRandomBytes(t *testing.T) {
	isolateArchCache(t)
	rng := rand.New(rand.NewSource(1))
	alphabet := []byte{0, 0, 'a', 'b', 'c', '-', 'X', '/'}
	letters := []byte{'a', 'b', 'c', '-', 'X', '/'}
	var queries []string
	var gen func(prefix string, depth int)
	gen = func(prefix string, depth int) {
		if len(prefix) >= 2 {
			queries = append(queries, prefix)
		}
		if depth == 0 {
			return
		}
		for _, c := range letters {
			gen(prefix+string(c), depth-1)
		}
	}
	gen("", 3)
	for i := 0; i < 4; i++ {
		body := make([]byte, 3000)
		for j := range body {
			body[j] = alphabet[rng.Intn(len(alphabet))]
		}
		path := writeProbeFile(t, "random.bin", body)
		assertMatchesLiteralScan(t, path, queries)
	}
}

func TestArchTokensAcrossChunkBoundary(t *testing.T) {
	isolateArchCache(t)
	for _, offset := range []int{-3, -1, 0, 1} {
		start := archProbeChunk + offset
		body := make([]byte, archProbeChunk*2)
		for i := range body {
			body[i] = 'x'
		}
		copy(body[start:], "\x00laguna\x00")
		path := writeProbeFile(t, "boundary.bin", body)
		tokens, err := scanArchTokens(path)
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := tokens["laguna"]; !ok {
			t.Errorf("token at chunk offset %d not found", offset)
		}
		assertMatchesLiteralScan(t, path, []string{"laguna", "lagun", "aguna"})
	}
}

func TestArchTokensRescanChangedFile(t *testing.T) {
	cachePath := isolateArchCache(t)
	path := writeProbeFile(t, "llama-server", []byte("\x00alpha\x00"))
	if sup, _ := BackendSupportsArch(path, "alpha"); !sup {
		t.Fatal("alpha not found")
	}
	scans := archScanCount()
	if sup, _ := BackendSupportsArch(path, "beta"); sup {
		t.Fatal("beta found before it was written")
	}
	if archScanCount() != scans {
		t.Fatal("an unchanged file was scanned again in the same process")
	}

	// Rewritten in place: a different size and an explicit mtime, so the test
	// does not depend on filesystem timestamp granularity.
	if err := os.WriteFile(path, []byte("\x00beta\x00\x00gamma\x00"), 0o600); err != nil {
		t.Fatal(err)
	}
	later := time.Now().Add(time.Hour)
	if err := os.Chtimes(path, later, later); err != nil {
		t.Fatal(err)
	}
	if sup, _ := BackendSupportsArch(path, "alpha"); sup {
		t.Error("stale token set served after the file changed")
	}
	if sup, _ := BackendSupportsArch(path, "beta"); !sup {
		t.Error("changed file was not rescanned")
	}

	// A new process must not trust the persisted entry for a changed file
	// either: same size, different mtime.
	resetArchCacheForTest()
	if err := os.WriteFile(path, []byte("\x00zeta\x00\x00gamma\x00"), 0o600); err != nil {
		t.Fatal(err)
	}
	evenLater := later.Add(time.Hour)
	if err := os.Chtimes(path, evenLater, evenLater); err != nil {
		t.Fatal(err)
	}
	if sup, _ := BackendSupportsArch(path, "beta"); sup {
		t.Error("persisted token set served after the file changed")
	}
	if sup, _ := BackendSupportsArch(path, "zeta"); !sup {
		t.Error("changed file was not rescanned after a restart")
	}
	if _, err := os.Stat(cachePath); err != nil {
		t.Fatalf("cache not written: %v", err)
	}
}

func TestArchTokenCachePersists(t *testing.T) {
	cachePath := isolateArchCache(t)
	kept := writeProbeFile(t, "llama-server", []byte("\x00qwen3moe\x00\x00llama\x00"))
	gone := writeProbeFile(t, "llama-server", []byte("\x00laguna\x00"))
	if sup, probed := BackendSupportsArch(kept, "qwen3moe"); !sup || !probed {
		t.Fatal("qwen3moe not found")
	}
	if sup, _ := BackendSupportsArch(gone, "laguna"); !sup {
		t.Fatal("laguna not found")
	}
	st, err := os.Stat(cachePath)
	if err != nil {
		t.Fatalf("cache not written: %v", err)
	}
	if runtime.GOOS != "windows" && st.Mode().Perm() != 0o600 {
		t.Errorf("cache mode %v, want 0600", st.Mode().Perm())
	}

	// A new process answers from the persisted sets without scanning.
	resetArchCacheForTest()
	if sup, probed := BackendSupportsArch(kept, "llama"); !sup || !probed {
		t.Error("persisted set lost a token")
	}
	if sup, probed := BackendSupportsArch(kept, "laguna"); sup || !probed {
		t.Error("persisted set gained a token")
	}
	if n := archScanCount(); n != 0 {
		t.Errorf("%d files scanned with a valid persisted cache, want 0", n)
	}

	// An entry whose file vanished is evicted on the next write.
	if err := os.Remove(gone); err != nil {
		t.Fatal(err)
	}
	resetArchCacheForTest()
	if sup, _ := BackendSupportsArch(kept, "qwen3moe"); !sup {
		t.Fatal("qwen3moe lost after eviction")
	}
	data, err := os.ReadFile(cachePath)
	if err != nil {
		t.Fatal(err)
	}
	var disk archCacheDisk
	if err := json.Unmarshal(data, &disk); err != nil {
		t.Fatalf("cache unreadable: %v", err)
	}
	if len(disk.Files) != 1 || disk.Files[0].Path != mustResolve(t, kept) {
		t.Errorf("cache after eviction holds %+v, want only %s", disk.Files, kept)
	}
	if entries, _ := os.ReadDir(filepath.Dir(cachePath)); len(entries) != 1 {
		t.Errorf("cache directory holds %d entries, want only the cache file", len(entries))
	}
}

func TestArchTokenCacheCorruptFallsBackToScan(t *testing.T) {
	cachePath := isolateArchCache(t)
	path := writeProbeFile(t, "llama-server", []byte("\x00qwen3moe\x00"))
	for _, garbage := range []string{"{not json", `{"version":999,"files":[]}`} {
		resetArchCacheForTest()
		if err := os.WriteFile(cachePath, []byte(garbage), 0o600); err != nil {
			t.Fatal(err)
		}
		if sup, probed := BackendSupportsArch(path, "qwen3moe"); !sup || !probed {
			t.Errorf("%q: corrupt cache changed the answer", garbage)
		}
		if archScanCount() == 0 {
			t.Errorf("%q: file was not scanned", garbage)
		}
		data, err := os.ReadFile(cachePath)
		if err != nil {
			t.Fatal(err)
		}
		var disk archCacheDisk
		if err := json.Unmarshal(data, &disk); err != nil || disk.Version != archCacheVersion || len(disk.Files) != 1 {
			t.Errorf("%q: cache not rebuilt: %s", garbage, data)
		}
	}
}

func mustResolve(t *testing.T, path string) string {
	t.Helper()
	r, err := filepath.EvalSymlinks(path)
	if err != nil {
		t.Fatal(err)
	}
	return r
}
