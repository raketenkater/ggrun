package gguf

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	modelstore "github.com/raketenkater/ggrun/pkg/models"
)

// headerCacheFile lives in the ggrun cache directory.
const headerCacheFile = "gguf-headers.json"

// headerCacheVersion changes when the stored shape or its meaning changes.
const headerCacheVersion = 1

// headerCacheMaxEntries bounds the file; entries untouched in a session go
// first.
const headerCacheMaxEntries = 1024

// headerCacheMaxTemplate keeps an unusually large chat template out of the file
// (it is read again from the model each time instead).
const headerCacheMaxTemplate = 256 << 10

// parseModel is Parse; tests replace it to count calls.
var parseModel = Parse

// HeaderCache remembers what ggrun read from GGUF headers, so a model listing
// does not re-run the parser helper and re-walk every vocabulary on each start.
// An entry is valid only while the size and modification time of every file of
// the model, and the parser helper and the ggrun binary, are unchanged. A cache that cannot be
// read or written is simply empty: results are then computed as before.
type HeaderCache struct {
	dir         string
	fingerprint string

	mu      sync.Mutex
	entries map[string]*headerEntry
	sigs    map[string]string
	touched map[string]bool
	dirty   bool
}

type headerEntry struct {
	Sig             string                     `json:"sig"`
	Info            *Info                      `json:"info,omitempty"`
	ChatTemplate    string                     `json:"chat_template,omitempty"`
	HasChatTemplate bool                       `json:"has_chat_template,omitempty"`
	Values          map[string]json.RawMessage `json:"values,omitempty"`
}

type headerFile struct {
	Version int                     `json:"version"`
	Parser  string                  `json:"parser"`
	Entries map[string]*headerEntry `json:"entries"`
}

// OpenHeaderCache loads the cache stored in dir. It never returns nil. An empty
// dir gives a cache that is only kept in memory.
func OpenHeaderCache(dir string) *HeaderCache {
	c := &HeaderCache{
		dir:         strings.TrimSpace(dir),
		fingerprint: parserFingerprint(),
		entries:     map[string]*headerEntry{},
		sigs:        map[string]string{},
		touched:     map[string]bool{},
	}
	c.entries = c.readFile()
	return c
}

// executableIdentity names the running ggrun binary; tests replace it. Parse
// also post-processes the helper's output in Go, so a new binary must not
// reuse Info stored by an older one.
var executableIdentity = exeIdentity

// exeIdentity is the resolved path, size and modification time of the running
// executable (one stat, no hashing). Any failure gives a stable fallback.
func exeIdentity() string {
	exe, err := os.Executable()
	if err != nil {
		return "exe-unknown"
	}
	if resolved, err := filepath.EvalSymlinks(exe); err == nil {
		exe = resolved
	} else {
		return "exe-unknown"
	}
	st, err := os.Stat(exe)
	if err != nil {
		return "exe-unknown"
	}
	return fmt.Sprintf("%s:%d:%d", exe, st.Size(), st.ModTime().UnixNano())
}

// parserFingerprint identifies the parser helper Parse would run and the ggrun
// binary that post-processes its output, so a changed helper (new fields, fixed
// byte accounting) or a changed binary discards every stored Info.
func parserFingerprint() string {
	exe := sha256.Sum256([]byte(executableIdentity()))
	exeID := hex.EncodeToString(exe[:4])
	script := findParseScript()
	if script == "" {
		return "none-" + exeID
	}
	data, err := os.ReadFile(script)
	if err != nil {
		return "none-" + exeID
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:8]) + "-" + exeID
}

func (c *HeaderCache) readFile() map[string]*headerEntry {
	if c.dir == "" {
		return map[string]*headerEntry{}
	}
	data, err := os.ReadFile(filepath.Join(c.dir, headerCacheFile))
	if err != nil {
		return map[string]*headerEntry{}
	}
	var f headerFile
	if json.Unmarshal(data, &f) != nil || f.Version != headerCacheVersion || f.Parser != c.fingerprint || f.Entries == nil {
		return map[string]*headerEntry{}
	}
	return f.Entries
}

// signature is the identity of the model's files: the size and modification
// time of the file and, for a split model, of every shard. ok is false when the
// files cannot be resolved; the caller then computes without the cache.
func (c *HeaderCache) signature(path string) (string, bool) {
	c.mu.Lock()
	sig, seen := c.sigs[path]
	c.mu.Unlock()
	if seen {
		return sig, sig != ""
	}
	sig = computeSignature(path)
	c.mu.Lock()
	c.sigs[path] = sig
	c.mu.Unlock()
	return sig, sig != ""
}

func computeSignature(path string) string {
	files, _, err := modelstore.ResolveGGUFShardFiles(path)
	if err != nil || len(files) == 0 {
		return ""
	}
	var b strings.Builder
	for _, file := range files {
		st, err := os.Stat(file)
		if err != nil {
			return ""
		}
		fmt.Fprintf(&b, "%s:%d:%d;", filepath.Base(file), st.Size(), st.ModTime().UnixNano())
	}
	return b.String()
}

// entry returns the stored entry for path when its signature is current, or a
// fresh one. The caller holds c.mu.
func (c *HeaderCache) entry(path, sig string) *headerEntry {
	e := c.entries[path]
	if e == nil || e.Sig != sig {
		e = &headerEntry{Sig: sig}
		c.entries[path] = e
	}
	c.touched[path] = true
	return e
}

// Info is Parse, remembered. The returned value is shared: callers read it and
// do not modify it.
func (c *HeaderCache) Info(path string) (*Info, error) {
	sig, ok := c.signature(path)
	if !ok {
		return parseModel(path)
	}
	c.mu.Lock()
	if e := c.entries[path]; e != nil && e.Sig == sig && e.Info != nil {
		c.touched[path] = true
		info := e.Info
		c.mu.Unlock()
		return info, nil
	}
	c.mu.Unlock()
	info, err := parseModel(path)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	c.entry(path, sig).Info = info
	c.dirty = true
	c.mu.Unlock()
	return info, nil
}

// ChatTemplate is ChatTemplate, remembered. A model without an embedded
// template is remembered too.
func (c *HeaderCache) ChatTemplate(path string) string {
	sig, ok := c.signature(path)
	if !ok {
		return ChatTemplate(path)
	}
	c.mu.Lock()
	if e := c.entries[path]; e != nil && e.Sig == sig && e.HasChatTemplate {
		c.touched[path] = true
		tmpl := e.ChatTemplate
		c.mu.Unlock()
		return tmpl
	}
	c.mu.Unlock()
	tmpl := ChatTemplate(path)
	if len(tmpl) <= headerCacheMaxTemplate {
		c.mu.Lock()
		e := c.entry(path, sig)
		e.ChatTemplate, e.HasChatTemplate = tmpl, true
		c.dirty = true
		c.mu.Unlock()
	}
	return tmpl
}

// CachedValue remembers any other per-file header result under name, valid for
// exactly as long as that one file is unchanged. It works for any single file,
// including a non-first shard of a split model. compute reports ok=false for
// "no verdict" (an unreadable or unrecognized header); that is never stored, so
// it is retried on the next start. T must round-trip through JSON.
func CachedValue[T any](c *HeaderCache, path, name string, compute func() (T, bool)) (T, bool) {
	key := fileEntryPrefix + path
	sig, ok := c.fileSignature(key, path)
	if !ok {
		return compute()
	}
	c.mu.Lock()
	if e := c.entries[key]; e != nil && e.Sig == sig {
		if raw, found := e.Values[name]; found {
			var v T
			if json.Unmarshal(raw, &v) == nil {
				c.touched[key] = true
				c.mu.Unlock()
				return v, true
			}
		}
	}
	c.mu.Unlock()
	v, good := compute()
	if !good {
		return v, false
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return v, true
	}
	c.mu.Lock()
	e := c.entry(key, sig)
	if e.Values == nil {
		e.Values = map[string]json.RawMessage{}
	}
	e.Values[name] = raw
	c.dirty = true
	c.mu.Unlock()
	return v, true
}

// fileEntryPrefix keys per-file values apart from per-model entries, whose
// signature covers every shard.
const fileEntryPrefix = "file:"

func (c *HeaderCache) fileSignature(key, path string) (string, bool) {
	c.mu.Lock()
	sig, seen := c.sigs[key]
	c.mu.Unlock()
	if !seen {
		if st, err := os.Stat(path); err == nil && !st.IsDir() {
			sig = fmt.Sprintf("%d:%d", st.Size(), st.ModTime().UnixNano())
		}
		c.mu.Lock()
		c.sigs[key] = sig
		c.mu.Unlock()
	}
	return sig, sig != ""
}

// Save writes the cache when it changed. It merges with what other ggrun
// processes stored meanwhile, and a failed write is only reported: the next
// start recomputes.
func (c *HeaderCache) Save() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.dir == "" || !c.dirty {
		return nil
	}
	merged := c.readFile()
	for path, e := range c.entries {
		if c.touched[path] {
			merged[path] = e
		} else if _, present := merged[path]; !present {
			merged[path] = e
		}
	}
	for path := range merged {
		if len(merged) <= headerCacheMaxEntries {
			break
		}
		if !c.touched[path] {
			delete(merged, path)
		}
	}
	data, err := json.Marshal(headerFile{Version: headerCacheVersion, Parser: c.fingerprint, Entries: merged})
	if err != nil {
		return err
	}
	if err := os.MkdirAll(c.dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(c.dir, ".gguf-headers-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, filepath.Join(c.dir, headerCacheFile)); err != nil {
		return err
	}
	c.dirty = false
	return nil
}
