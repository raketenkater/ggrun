package backends

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// BackendSupportsArch is asked once per (backend, architecture) while
// recommendation rows are annotated. Re-running the literal scan for every
// question re-read the whole library closure each time: on a host with a dozen
// installed backends that was tens of thousands of opens of the same files and
// over a minute of TUI start-up.
//
// Each closure file is therefore tokenized once: every NUL-bracketed run of
// archTokenMin..archTokenMax bytes drawn from archTokenByte is collected into a
// set. The literal scan matches "\x00" + arch + "\x00"; because a query in that
// alphabet contains no NUL, it matches exactly when some run bounded by a NUL
// on both sides equals the query, which is set membership. A query outside the
// alphabet or length range falls back to the literal scan, so every answer is
// the one the scan alone would give.
//
// The sets are kept for the process and persisted in the cache directory,
// keyed by the file's resolved path, size and modification time; an entry
// whose key no longer matches the file is dropped and the file rescanned.

const (
	archTokenMin = 2
	archTokenMax = 64

	archCacheFileName = "arch-literals.json"
	archCacheVersion  = 1
	// archCacheMaxFiles and archCacheMaxBytes bound the persisted cache. A file
	// whose set alone exceeds archCacheMaxTokens is still served from memory
	// but not persisted: a set is never truncated to fit.
	archCacheMaxFiles  = 256
	archCacheMaxBytes  = 16 << 20
	archCacheMaxTokens = 200000
)

// archTokenByte is the alphabet of collected runs. Every architecture name
// llama.cpp registers is lower-case ASCII letters, digits, '_', '.' or '-';
// the query is lower-cased before matching, so upper-case runs could never be
// asked for through the set path.
func archTokenByte(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9') || b == '_' || b == '.' || b == '-'
}

// archTokenQuery reports whether a (lower-cased) query can be answered from a
// token set. Anything else goes through the literal scan.
func archTokenQuery(arch string) bool {
	if len(arch) < archTokenMin || len(arch) > archTokenMax {
		return false
	}
	for i := 0; i < len(arch); i++ {
		if !archTokenByte(arch[i]) {
			return false
		}
	}
	return true
}

type archFileKey struct {
	Path  string
	Size  int64
	MTime int64
}

type archTokenEntry struct {
	key    archFileKey
	tokens map[string]struct{}
	used   int64
}

type archProbeList struct {
	key   archFileKey
	files []string
}

var archCache struct {
	mu         sync.Mutex
	loadedFrom string
	loaded     bool
	dirty      bool
	files      map[string]*archTokenEntry // by resolved path
	probeLists map[string]archProbeList   // by binary path as given
	scans      int                        // files tokenized; observed by tests
}

// resetArchCacheForTest forgets every in-process cache, as a new process
// would start.
func resetArchCacheForTest() {
	archCache.mu.Lock()
	defer archCache.mu.Unlock()
	archCache.loadedFrom = ""
	archCache.loaded = false
	archCache.dirty = false
	archCache.files = nil
	archCache.probeLists = nil
	archCache.scans = 0
}

// archCachePath mirrors the catalog cache location.
func archCachePath() string {
	if d := os.Getenv("LLM_CACHE_DIR"); d != "" {
		return filepath.Join(d, archCacheFileName)
	}
	if home := os.Getenv("LLM_APP_HOME"); home != "" {
		return filepath.Join(home, ".cache", "ggrun", archCacheFileName)
	}
	if d, err := os.UserCacheDir(); err == nil {
		return filepath.Join(d, "ggrun", archCacheFileName)
	}
	return ""
}

// statKey is the cache key of a file as it is now.
func statKey(path string) (archFileKey, error) {
	st, err := os.Stat(path)
	if err != nil {
		return archFileKey{}, err
	}
	resolved := path
	if r, err := filepath.EvalSymlinks(path); err == nil && r != "" {
		resolved = r
	}
	return archFileKey{Path: resolved, Size: st.Size(), MTime: st.ModTime().UnixNano()}, nil
}

// cachedArchProbeFilesLocked memoizes archProbeFiles per binary. A binary
// rebuilt in place changes its size or mtime, and a listed file that vanished
// forces a fresh walk.
func cachedArchProbeFilesLocked(binaryPath string) []string {
	key, err := statKey(binaryPath)
	if err != nil {
		return archProbeFiles(binaryPath)
	}
	if archCache.probeLists == nil {
		archCache.probeLists = map[string]archProbeList{}
	}
	if l, ok := archCache.probeLists[binaryPath]; ok && l.key == key {
		intact := true
		for _, f := range l.files {
			if _, err := os.Stat(f); err != nil {
				intact = false
				break
			}
		}
		if intact {
			return l.files
		}
	}
	files := archProbeFiles(binaryPath)
	archCache.probeLists[binaryPath] = archProbeList{key: key, files: files}
	return files
}

// archTokensLocked returns the token set of one file, scanning it only when no
// entry matches its current key. On a read error the runs completed before the
// error are returned with it -- the literal scan also reported a match found
// before a failed read -- and nothing is cached.
func archTokensLocked(path string) (map[string]struct{}, error) {
	key, err := statKey(path)
	if err != nil {
		return nil, err
	}
	if archCache.files == nil {
		archCache.files = map[string]*archTokenEntry{}
	}
	now := time.Now().Unix()
	if e, ok := archCache.files[key.Path]; ok && e.key == key {
		e.used = now
		return e.tokens, nil
	}
	archCache.scans++
	tokens, err := scanArchTokens(path)
	if err != nil {
		return tokens, err
	}
	// A file being rewritten while it was read is answered from what was read,
	// as the literal scan would, but is not remembered.
	if after, err := statKey(path); err == nil && after == key {
		archCache.files[key.Path] = &archTokenEntry{key: key, tokens: tokens, used: now}
		archCache.dirty = true
	} else {
		delete(archCache.files, key.Path)
	}
	return tokens, nil
}

// scanArchTokens streams a file and collects every run of archTokenByte bytes,
// archTokenMin..archTokenMax long, with a NUL immediately before and after it.
// A file's first run has no NUL before it and an unterminated last run has none
// after it; neither is collected, exactly as the bracketed literal never
// matched there. A run that grows past archTokenMax or meets another byte is
// dead until the next NUL.
func scanArchTokens(path string) (map[string]struct{}, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	tokens := map[string]struct{}{}
	buf := make([]byte, archProbeChunk)
	run := make([]byte, 0, archTokenMax)
	open := false // a NUL was seen and the run since it is still viable
	for {
		n, readErr := f.Read(buf)
		data := buf[:n]
		for len(data) > 0 {
			if !open {
				i := bytes.IndexByte(data, 0)
				if i < 0 {
					break
				}
				data = data[i+1:]
				open = true
				run = run[:0]
				continue
			}
			b := data[0]
			data = data[1:]
			if b == 0 {
				if len(run) >= archTokenMin {
					if _, ok := tokens[string(run)]; !ok {
						tokens[string(run)] = struct{}{}
					}
				}
				run = run[:0]
				continue
			}
			if len(run) == archTokenMax || !archTokenByte(b) {
				open = false
				continue
			}
			run = append(run, b)
		}
		if readErr == io.EOF {
			return tokens, nil
		}
		if readErr != nil {
			return tokens, readErr
		}
	}
}

type archCacheDisk struct {
	Version int                 `json:"version"`
	Files   []archCacheDiskFile `json:"files"`
}

type archCacheDiskFile struct {
	Path   string   `json:"path"`
	Size   int64    `json:"size"`
	MTime  int64    `json:"mtime_ns"`
	Used   int64    `json:"used"`
	Tokens []string `json:"tokens"`
}

// loadArchCacheLocked merges the persisted sets into memory once per cache
// location. A missing, oversized or unparsable file is ignored and rebuilt by
// the next save; an entry whose file changed or vanished is dropped.
func loadArchCacheLocked() {
	path := archCachePath()
	if archCache.loaded && archCache.loadedFrom == path {
		return
	}
	archCache.loaded = true
	archCache.loadedFrom = path
	if archCache.files == nil {
		archCache.files = map[string]*archTokenEntry{}
	}
	if path == "" {
		return
	}
	st, err := os.Stat(path)
	if err != nil {
		return
	}
	if st.Size() > 2*archCacheMaxBytes {
		archCache.dirty = true
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var disk archCacheDisk
	if err := json.Unmarshal(data, &disk); err != nil || disk.Version != archCacheVersion {
		archCache.dirty = true
		return
	}
	for _, f := range disk.Files {
		key := archFileKey{Path: f.Path, Size: f.Size, MTime: f.MTime}
		if cur, err := statKey(f.Path); err != nil || cur != key || !validArchTokens(f.Tokens) {
			archCache.dirty = true
			continue
		}
		if _, ok := archCache.files[key.Path]; ok {
			continue
		}
		tokens := make(map[string]struct{}, len(f.Tokens))
		for _, t := range f.Tokens {
			tokens[t] = struct{}{}
		}
		archCache.files[key.Path] = &archTokenEntry{key: key, tokens: tokens, used: f.Used}
	}
}

func validArchTokens(tokens []string) bool {
	for _, t := range tokens {
		if !archTokenQuery(t) {
			return false
		}
	}
	return true
}

// saveArchCacheLocked writes the sets when something new was scanned, most
// recently used first within the bounds, privately and atomically. A failed
// write only costs a rescan in a later process.
func saveArchCacheLocked() {
	if !archCache.dirty {
		return
	}
	archCache.dirty = false
	path := archCachePath()
	if path == "" {
		return
	}
	entries := make([]*archTokenEntry, 0, len(archCache.files))
	for _, e := range archCache.files {
		if len(e.tokens) <= archCacheMaxTokens {
			entries = append(entries, e)
		}
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].used != entries[j].used {
			return entries[i].used > entries[j].used
		}
		return entries[i].key.Path < entries[j].key.Path
	})
	disk := archCacheDisk{Version: archCacheVersion}
	budget := archCacheMaxBytes
	for _, e := range entries {
		if len(disk.Files) >= archCacheMaxFiles {
			break
		}
		cost := len(e.key.Path) + 96
		tokens := make([]string, 0, len(e.tokens))
		for t := range e.tokens {
			tokens = append(tokens, t)
			cost += len(t) + 3
		}
		if cost > budget {
			continue
		}
		budget -= cost
		sort.Strings(tokens)
		disk.Files = append(disk.Files, archCacheDiskFile{
			Path: e.key.Path, Size: e.key.Size, MTime: e.key.MTime, Used: e.used, Tokens: tokens,
		})
	}
	data, err := json.Marshal(disk)
	if err != nil {
		return
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return
	}
	tmp, err := os.CreateTemp(dir, ".arch-literals-*.tmp")
	if err != nil {
		return
	}
	tmpPath := tmp.Name()
	_ = tmp.Chmod(0o600)
	_, werr := tmp.Write(data)
	cerr := tmp.Close()
	if werr != nil || cerr != nil {
		_ = os.Remove(tmpPath)
		return
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
	}
}
