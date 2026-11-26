package gguf

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// stubParse replaces the parser helper with a counter.
func stubParse(t *testing.T) *int32 {
	t.Helper()
	var calls int32
	old := parseModel
	parseModel = func(path string) (*Info, error) {
		atomic.AddInt32(&calls, 1)
		return &Info{Architecture: "stub", BlockCount: 7, ExpertBytes: 11}, nil
	}
	t.Cleanup(func() { parseModel = old })
	return &calls
}

func writeCacheFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func setMtime(t *testing.T, path string, mtime time.Time) {
	t.Helper()
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		t.Fatal(err)
	}
}

func reopen(t *testing.T, c *HeaderCache, dir string) *HeaderCache {
	t.Helper()
	if err := c.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}
	return OpenHeaderCache(dir)
}

func TestHeaderCacheInfoHitSkipsParser(t *testing.T) {
	calls := stubParse(t)
	dir := t.TempDir()
	model := filepath.Join(t.TempDir(), "m.gguf")
	writeCacheFile(t, model, "GGUF-data")

	c := OpenHeaderCache(dir)
	if info, err := c.Info(model); err != nil || info.Architecture != "stub" {
		t.Fatalf("Info = %+v, %v", info, err)
	}
	if _, err := c.Info(model); err != nil || atomic.LoadInt32(calls) != 1 {
		t.Fatalf("second Info in one session parsed again: calls=%d err=%v", *calls, err)
	}

	c = reopen(t, c, dir)
	info, err := c.Info(model)
	if err != nil || atomic.LoadInt32(calls) != 1 {
		t.Fatalf("a new session re-ran the parser: calls=%d err=%v", *calls, err)
	}
	if info.Architecture != "stub" || info.BlockCount != 7 || info.ExpertBytes != 11 {
		t.Fatalf("cached Info lost fields: %+v", info)
	}
}

func TestHeaderCacheInvalidatesOnSizeOrMtimeChange(t *testing.T) {
	calls := stubParse(t)
	dir := t.TempDir()
	model := filepath.Join(t.TempDir(), "m.gguf")
	writeCacheFile(t, model, "GGUF-data")
	base := time.Now().Add(-time.Hour)
	setMtime(t, model, base)

	c := OpenHeaderCache(dir)
	if _, err := c.Info(model); err != nil {
		t.Fatal(err)
	}
	c = reopen(t, c, dir)
	if _, err := c.Info(model); err != nil || atomic.LoadInt32(calls) != 1 {
		t.Fatalf("unchanged file missed: calls=%d err=%v", *calls, err)
	}

	setMtime(t, model, base.Add(time.Minute))
	c = OpenHeaderCache(dir)
	if _, err := c.Info(model); err != nil || atomic.LoadInt32(calls) != 2 {
		t.Fatalf("changed mtime was served from the cache: calls=%d err=%v", *calls, err)
	}
	c = reopen(t, c, dir)

	writeCacheFile(t, model, "GGUF-data-longer")
	setMtime(t, model, base.Add(time.Minute))
	c = OpenHeaderCache(dir)
	if _, err := c.Info(model); err != nil || atomic.LoadInt32(calls) != 3 {
		t.Fatalf("changed size was served from the cache: calls=%d err=%v", *calls, err)
	}
}

func TestHeaderCacheSignatureCoversEveryShard(t *testing.T) {
	calls := stubParse(t)
	dir := t.TempDir()
	models := t.TempDir()
	first := filepath.Join(models, "m-00001-of-00002.gguf")
	second := filepath.Join(models, "m-00002-of-00002.gguf")
	writeCacheFile(t, first, "GGUF-one")
	writeCacheFile(t, second, "GGUF-two")

	c := OpenHeaderCache(dir)
	if _, err := c.Info(first); err != nil {
		t.Fatal(err)
	}
	c = reopen(t, c, dir)
	if _, err := c.Info(first); err != nil || atomic.LoadInt32(calls) != 1 {
		t.Fatalf("unchanged shards missed: calls=%d err=%v", *calls, err)
	}

	writeCacheFile(t, second, "GGUF-two-grown")
	c = OpenHeaderCache(dir)
	if _, err := c.Info(first); err != nil || atomic.LoadInt32(calls) != 2 {
		t.Fatalf("a changed second shard did not invalidate the model: calls=%d err=%v", *calls, err)
	}
}

func TestHeaderCacheDoesNotStoreParseErrors(t *testing.T) {
	var calls int32
	old := parseModel
	parseModel = func(path string) (*Info, error) {
		atomic.AddInt32(&calls, 1)
		return nil, errors.New("helper failed")
	}
	t.Cleanup(func() { parseModel = old })
	dir := t.TempDir()
	model := filepath.Join(t.TempDir(), "m.gguf")
	writeCacheFile(t, model, "GGUF")

	c := OpenHeaderCache(dir)
	if _, err := c.Info(model); err == nil {
		t.Fatal("want the parse error")
	}
	c = reopen(t, c, dir)
	if _, err := c.Info(model); err == nil || atomic.LoadInt32(&calls) != 2 {
		t.Fatalf("a failed parse was remembered: calls=%d err=%v", calls, err)
	}
}

func TestHeaderCacheMissingFileFallsThrough(t *testing.T) {
	calls := stubParse(t)
	c := OpenHeaderCache(t.TempDir())
	missing := filepath.Join(t.TempDir(), "gone.gguf")
	if _, err := c.Info(missing); err != nil {
		t.Fatalf("stubbed parser should be reached for an unresolvable path: %v", err)
	}
	if atomic.LoadInt32(calls) != 1 {
		t.Fatalf("calls = %d, want 1", *calls)
	}
	if got := c.ChatTemplate(missing); got != "" {
		t.Fatalf("ChatTemplate of a missing file = %q", got)
	}
}

func TestHeaderCacheChatTemplateHitAndInvalidation(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(t.TempDir(), "m.gguf")
	base := time.Now().Add(-time.Hour)
	writeChatTemplateFixture(t, model, "qwen35", "AAAA {{ messages }}")
	setMtime(t, model, base)

	c := OpenHeaderCache(dir)
	if got := c.ChatTemplate(model); got != "AAAA {{ messages }}" {
		t.Fatalf("ChatTemplate = %q", got)
	}
	c = reopen(t, c, dir)

	// Same size and mtime, different bytes: only a cache hit still reports the
	// old template, which proves the file was not read again.
	writeChatTemplateFixture(t, model, "qwen35", "BBBB {{ messages }}")
	setMtime(t, model, base)
	if got := c.ChatTemplate(model); got != "AAAA {{ messages }}" {
		t.Fatalf("an unchanged file was re-read: %q", got)
	}

	setMtime(t, model, base.Add(time.Minute))
	c = OpenHeaderCache(dir)
	if got := c.ChatTemplate(model); got != "BBBB {{ messages }}" {
		t.Fatalf("a changed mtime kept the stale template: %q", got)
	}
}

func TestHeaderCacheRemembersAbsentChatTemplate(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(t.TempDir(), "m.gguf")
	base := time.Now().Add(-time.Hour)
	writeChatTemplateFixture(t, model, "qwen35", "")
	setMtime(t, model, base)
	c := OpenHeaderCache(dir)
	if got := c.ChatTemplate(model); got != "" {
		t.Fatalf("ChatTemplate = %q", got)
	}
	c = reopen(t, c, dir)
	// An absent template is stored as a result, not as a miss.
	data, err := os.ReadFile(filepath.Join(dir, headerCacheFile))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "has_chat_template") {
		t.Fatalf("an absent template was not remembered: %s", data)
	}
	if got := c.ChatTemplate(model); got != "" {
		t.Fatalf("ChatTemplate after reopen = %q", got)
	}
}

func TestCachedValueHitInvalidationAndNoVerdict(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(t.TempDir(), "m.gguf")
	base := time.Now().Add(-time.Hour)
	writeCacheFile(t, model, "GGUF-data")
	setMtime(t, model, base)

	var computed int
	compute := func() (int64, bool) { computed++; return 4242, true }

	c := OpenHeaderCache(dir)
	if v, ok := CachedValue(c, model, "expected_size", compute); !ok || v != 4242 {
		t.Fatalf("CachedValue = %d, %v", v, ok)
	}
	c = reopen(t, c, dir)
	if v, ok := CachedValue(c, model, "expected_size", compute); !ok || v != 4242 || computed != 1 {
		t.Fatalf("hit recomputed: v=%d ok=%v computed=%d", v, ok, computed)
	}
	if _, ok := CachedValue(c, model, "other_name", compute); !ok || computed != 2 {
		t.Fatalf("a different name shared a slot: computed=%d", computed)
	}

	setMtime(t, model, base.Add(time.Minute))
	c = OpenHeaderCache(dir)
	if _, ok := CachedValue(c, model, "expected_size", compute); !ok || computed != 3 {
		t.Fatalf("changed mtime kept the stale value: computed=%d", computed)
	}

	var verdicts int
	noVerdict := func() (int64, bool) { verdicts++; return 0, false }
	c = OpenHeaderCache(dir)
	if _, ok := CachedValue(c, model, "unknown_type", noVerdict); ok {
		t.Fatal("no verdict reported as a verdict")
	}
	c = reopen(t, c, dir)
	if _, ok := CachedValue(c, model, "unknown_type", noVerdict); ok || verdicts != 2 {
		t.Fatalf("no-verdict was stored: verdicts=%d", verdicts)
	}
}

func TestHeaderCacheIgnoresOtherParserCorruptAndFutureFiles(t *testing.T) {
	calls := stubParse(t)
	dir := t.TempDir()
	model := filepath.Join(t.TempDir(), "m.gguf")
	writeCacheFile(t, model, "GGUF-data")

	c := OpenHeaderCache(dir)
	if _, err := c.Info(model); err != nil {
		t.Fatal(err)
	}
	if err := c.Save(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, headerCacheFile)
	good, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	var f headerFile
	if err := json.Unmarshal(good, &f); err != nil {
		t.Fatal(err)
	}
	cases := map[string][]byte{"corrupt": []byte("{not json")}
	f.Parser = "a-different-parser"
	other, _ := json.Marshal(f)
	cases["other parser"] = other
	f.Parser = c.fingerprint
	f.Version = headerCacheVersion + 1
	future, _ := json.Marshal(f)
	cases["other version"] = future

	for name, content := range cases {
		before := atomic.LoadInt32(calls)
		if err := os.WriteFile(path, content, 0o600); err != nil {
			t.Fatal(err)
		}
		c := OpenHeaderCache(dir)
		if _, err := c.Info(model); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if atomic.LoadInt32(calls) != before+1 {
			t.Fatalf("%s: stale cache was trusted", name)
		}
	}
}

func TestHeaderCacheConcurrentUseAndSaveMerge(t *testing.T) {
	stubParse(t)
	dir := t.TempDir()
	models := t.TempDir()
	var paths []string
	for _, name := range []string{"a", "b", "c", "d", "e", "f"} {
		p := filepath.Join(models, name+".gguf")
		writeCacheFile(t, p, "GGUF-"+name)
		paths = append(paths, p)
	}

	c := OpenHeaderCache(dir)
	var wg sync.WaitGroup
	for _, p := range paths {
		wg.Add(1)
		go func(p string) {
			defer wg.Done()
			if _, err := c.Info(p); err != nil {
				t.Error(err)
			}
			CachedValue(c, p, "v", func() (int, bool) { return 1, true })
		}(p)
	}
	wg.Wait()

	// A second process stored another model meanwhile; Save keeps both.
	other := OpenHeaderCache(dir)
	extra := filepath.Join(models, "extra.gguf")
	writeCacheFile(t, extra, "GGUF-extra")
	if _, err := other.Info(extra); err != nil {
		t.Fatal(err)
	}
	if err := other.Save(); err != nil {
		t.Fatal(err)
	}
	if err := c.Save(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, headerCacheFile))
	if err != nil {
		t.Fatal(err)
	}
	var f headerFile
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Entries) != len(paths)+1 {
		t.Fatalf("entries = %d, want %d (the other process's entry was lost)", len(f.Entries), len(paths)+1)
	}
}

func TestHeaderCacheWithoutDirectoryStaysInMemory(t *testing.T) {
	calls := stubParse(t)
	model := filepath.Join(t.TempDir(), "m.gguf")
	writeCacheFile(t, model, "GGUF")
	c := OpenHeaderCache("")
	for i := 0; i < 2; i++ {
		if _, err := c.Info(model); err != nil {
			t.Fatal(err)
		}
	}
	if err := c.Save(); err != nil || atomic.LoadInt32(calls) != 1 {
		t.Fatalf("Save err=%v calls=%d", err, *calls)
	}
}

// readCountingReader counts Read calls and hides any Discard method.
type readCountingReader struct {
	r     *bytes.Reader
	reads int
}

func (c *readCountingReader) Read(p []byte) (int, error) {
	c.reads++
	return c.r.Read(p)
}

func TestSkipFixedArrayIsNotPerElement(t *testing.T) {
	const n = 1 << 20
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, uint32(5)) // int32 elements
	binary.Write(buf, binary.LittleEndian, uint64(n))
	buf.Write(make([]byte, 4*n))
	buf.WriteString("tail")
	cr := &readCountingReader{r: bytes.NewReader(buf.Bytes())}
	if err := skipGGUFValue(cr, 9, 0); err != nil {
		t.Fatal(err)
	}
	if cr.reads > 4096 {
		t.Fatalf("skipping %d int32 elements took %d reads", n, cr.reads)
	}
	rest := make([]byte, 4)
	if _, err := cr.Read(rest); err != nil || string(rest) != "tail" {
		t.Fatalf("skip overshot: %q %v", rest, err)
	}
}

func TestSkipFixedArrayTruncatedIsAnError(t *testing.T) {
	for _, wrap := range []string{"plain", "buffered"} {
		buf := new(bytes.Buffer)
		binary.Write(buf, binary.LittleEndian, uint32(6)) // float32
		binary.Write(buf, binary.LittleEndian, uint64(1000))
		buf.Write(make([]byte, 100)) // far fewer than 4000 bytes
		var err error
		if wrap == "plain" {
			err = skipGGUFValue(bytes.NewReader(buf.Bytes()), 9, 0)
		} else {
			err = skipGGUFValue(bufio.NewReader(bytes.NewReader(buf.Bytes())), 9, 0)
		}
		if err == nil {
			t.Fatalf("%s: truncated array skipped without error", wrap)
		}
	}
	huge := new(bytes.Buffer)
	binary.Write(huge, binary.LittleEndian, uint32(10)) // 8-byte elements
	binary.Write(huge, binary.LittleEndian, ^uint64(0))
	if err := skipGGUFValue(bytes.NewReader(huge.Bytes()), 9, 0); err == nil {
		t.Fatal("an impossible element count was accepted")
	}
}

func TestChatTemplateAfterLargeVocabulary(t *testing.T) {
	const vocab = 200000
	tokens := fixtureKV{"tokenizer.ggml.tokens", 9, func(b *bytes.Buffer) {
		binary.Write(b, binary.LittleEndian, uint32(8))
		binary.Write(b, binary.LittleEndian, uint64(vocab))
		for i := 0; i < vocab; i++ {
			binary.Write(b, binary.LittleEndian, uint64(3))
			b.WriteString("tok")
		}
	}}
	scores := kvArray("tokenizer.ggml.scores", 6, make([]float64, vocab))
	types := kvArray("tokenizer.ggml.token_type", 5, make([]float64, vocab))
	path := filepath.Join(t.TempDir(), "m.gguf")
	writeGGUFMetadataFixture(t, path, []fixtureKV{
		kvString("general.architecture", "qwen35"), tokens, scores, types,
		kvString("tokenizer.chat_template", "{{ big vocab template }}"),
	})
	if got := ChatTemplate(path); got != "{{ big vocab template }}" {
		t.Fatalf("ChatTemplate = %q", got)
	}
}

func TestCachedValueWorksForNonFirstShard(t *testing.T) {
	dir := t.TempDir()
	models := t.TempDir()
	first := filepath.Join(models, "m-00001-of-00002.gguf")
	second := filepath.Join(models, "m-00002-of-00002.gguf")
	writeCacheFile(t, first, "GGUF-one")
	writeCacheFile(t, second, "GGUF-two")
	var computed int
	compute := func() (int64, bool) { computed++; return 7, true }

	c := OpenHeaderCache(dir)
	for _, p := range []string{first, second} {
		if _, ok := CachedValue(c, p, "expected_size", compute); !ok {
			t.Fatal("no value")
		}
	}
	c = reopen(t, c, dir)
	for _, p := range []string{first, second} {
		if v, ok := CachedValue(c, p, "expected_size", compute); !ok || v != 7 {
			t.Fatalf("value for %s = %d, %v", filepath.Base(p), v, ok)
		}
	}
	if computed != 2 {
		t.Fatalf("computed %d times, want once per shard", computed)
	}

	// Growing the second shard invalidates only that shard's value.
	writeCacheFile(t, second, "GGUF-two-grown")
	c = OpenHeaderCache(dir)
	CachedValue(c, first, "expected_size", compute)
	CachedValue(c, second, "expected_size", compute)
	if computed != 3 {
		t.Fatalf("computed %d times, want 3 (only the grown shard recomputed)", computed)
	}
}
