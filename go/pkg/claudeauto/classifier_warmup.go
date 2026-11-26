// Auto-mode check warm-up.
//
// About 99% of every Claude Code Auto-mode check is the same fixed classifier
// instructions (the request's system text, and tools if present); only the last
// few hundred tokens are session-specific. A backend that has not read those
// instructions since it started needs minutes to prefill them, while Claude
// Code allows a check 60 s, so the first checks after every server start were
// denied as unavailable. The router therefore keeps a local copy of the fixed
// part and replays it once in the background when serving starts, so the
// backend prompt cache already holds it before the first real check.
//
// Only system and tools are stored (mode 0600), never messages. Only a large
// payload is kept: warming a short prefix saves nothing, and ggrun-client's
// review-eval and canary prompts share ClassifierMarker but are short, so they
// must not replace the stored native instructions.
package claudeauto

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

const (
	warmupHeader    = "X-Ggrun-Warmup"
	warmupUserAgent = "ggrun-warmup/1"
	warmupText      = "ggrun classifier warm-up"
)

// Package variables so tests can shorten them.
var (
	warmupMinSystemBytes = 16 << 10
	warmupIdleFor        = 2 * time.Second
	warmupPoll           = 300 * time.Millisecond
	warmupAttemptTimeout = 20 * time.Minute
	warmupDeadline       = time.Hour
	warmupMaxAttempts    = 20
)

type warmupKey struct{}

func isWarmupRequest(req *http.Request) bool {
	if req == nil {
		return false
	}
	v, _ := req.Context().Value(warmupKey{}).(bool)
	return v
}

type classifierWarmup struct {
	path string

	mu     sync.Mutex
	system json.RawMessage
	tools  json.RawMessage
	digest [sha256.Size]byte

	writes atomic.Int64 // file writes, for tests

	running atomic.Bool
	wg      sync.WaitGroup
	ctx     context.Context
	cancel  context.CancelFunc

	pmu       sync.Mutex
	attempt   context.CancelFunc
	onMain    bool
	preempted bool
}

type warmupFile struct {
	System json.RawMessage `json:"system"`
	Tools  json.RawMessage `json:"tools,omitempty"`
	Saved  string          `json:"saved"`
}

// canonJSON puts a raw value in the form the file stores, so a value read back
// from disk hashes the same as the identical value in a new request.
func canonJSON(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	out, err := json.Marshal(raw)
	if err != nil {
		return nil
	}
	return out
}

func warmupDigest(system, tools json.RawMessage) [sha256.Size]byte {
	h := sha256.New()
	h.Write(system)
	h.Write([]byte{0})
	h.Write(tools)
	var d [sha256.Size]byte
	copy(d[:], h.Sum(nil))
	return d
}

func warmupAcceptable(system, tools json.RawMessage) bool {
	return systemHasClassifierMarker(system) && len(system)+len(tools) >= warmupMinSystemBytes
}

// EnableClassifierWarmup turns on capture and replay of the fixed Auto check
// instructions, kept at path. An empty path does nothing; a missing file is
// fine.
func (r *Router) EnableClassifierWarmup(path string) error {
	if r == nil || path == "" {
		return nil
	}
	w := &classifierWarmup{path: path}
	w.ctx, w.cancel = context.WithCancel(context.Background())
	r.warm.Store(w)
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	var f warmupFile
	if json.Unmarshal(data, &f) != nil {
		return nil // unreadable copy: the next real check replaces it
	}
	system, tools := canonJSON(f.System), canonJSON(f.Tools)
	if !warmupAcceptable(system, tools) {
		return nil
	}
	w.mu.Lock()
	w.system, w.tools, w.digest = system, tools, warmupDigest(system, tools)
	w.mu.Unlock()
	return nil
}

// captureClassifier remembers the fixed part of a real check. Never messages.
func (r *Router) captureClassifier(body []byte) {
	w := r.warm.Load()
	if w == nil {
		return
	}
	var req struct {
		System json.RawMessage `json:"system"`
		Tools  json.RawMessage `json:"tools"`
	}
	if json.Unmarshal(body, &req) != nil {
		return
	}
	system, tools := canonJSON(req.System), canonJSON(req.Tools)
	if !warmupAcceptable(system, tools) {
		return
	}
	digest := warmupDigest(system, tools)
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.system != nil && digest == w.digest {
		return
	}
	w.system, w.tools, w.digest = system, tools, digest
	if err := w.writeLocked(); err != nil {
		fmt.Fprintf(os.Stderr, "[claude-code] could not save Auto check instructions: %v\n", err)
	}
}

func (w *classifierWarmup) writeLocked() error {
	data, err := json.Marshal(warmupFile{System: w.system, Tools: w.tools, Saved: time.Now().UTC().Format(time.RFC3339)})
	if err != nil {
		return err
	}
	dir := filepath.Dir(w.path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".claude-classifier-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	_, werr := tmp.Write(data)
	cerr := tmp.Close()
	if err := errors.Join(werr, cerr, os.Chmod(name, 0o600)); err != nil {
		_ = os.Remove(name)
		return err
	}
	if err := os.Rename(name, w.path); err != nil {
		_ = os.Remove(name)
		return err
	}
	w.writes.Add(1)
	return nil
}

func (w *classifierWarmup) stored() (system, tools json.RawMessage) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.system, w.tools
}

func (w *classifierWarmup) setOnMain() {
	if w == nil {
		return
	}
	w.pmu.Lock()
	w.onMain = true
	w.pmu.Unlock()
}

// preempt cancels an in-flight main-route warm-up; the backend keeps the
// prompt it already read, so the retry resumes cheaply.
func (w *classifierWarmup) preempt() {
	if w == nil {
		return
	}
	w.pmu.Lock()
	defer w.pmu.Unlock()
	if w.onMain && w.attempt != nil {
		w.preempted = true
		w.attempt()
	}
}

func (w *classifierWarmup) stop() {
	if w == nil {
		return
	}
	w.cancel()
	w.wg.Wait()
}

// WarmClassifier replays the stored instructions once in the background. It
// reports whether a warm-up was started.
func (r *Router) WarmClassifier() bool {
	if r == nil {
		return false
	}
	w := r.warm.Load()
	if w == nil {
		return false
	}
	system, tools := w.stored()
	if system == nil || !w.running.CompareAndSwap(false, true) {
		return false
	}
	body := map[string]any{
		"model": MainAlias, "max_tokens": 1, "stream": false,
		"system":   system,
		"messages": []map[string]string{{"role": "user", "content": warmupText}},
	}
	if tools != nil {
		body["tools"] = tools
	}
	payload, err := json.Marshal(body)
	if err != nil {
		w.running.Store(false)
		return false
	}
	w.wg.Add(1)
	go func() {
		defer w.wg.Done()
		defer w.running.Store(false)
		r.runWarmup(w, payload)
	}()
	return true
}

func (r *Router) mainIdle() bool {
	return r.mainActive.Load() == 0 && r.mainQueued.Load() == 0 && r.mainInflight.Load() == 0
}

func (r *Router) waitIdle(ctx context.Context) bool {
	idleSince := time.Time{}
	for {
		if r.mainIdle() {
			if idleSince.IsZero() {
				idleSince = time.Now()
			} else if time.Since(idleSince) >= warmupIdleFor {
				return true
			}
		} else {
			idleSince = time.Time{}
		}
		select {
		case <-ctx.Done():
			return false
		case <-time.After(warmupPoll):
		}
	}
}

func (r *Router) runWarmup(w *classifierWarmup, payload []byte) {
	ctx, cancel := context.WithTimeout(w.ctx, warmupDeadline)
	defer cancel()
	start := time.Now()
	var last error
	for attempt := 1; attempt <= warmupMaxAttempts; attempt++ {
		if attempt > 1 && !r.waitIdle(ctx) {
			last = ctx.Err()
			break
		}
		err, preempted := r.warmupAttempt(ctx, w, payload)
		if err == nil {
			fmt.Fprintf(os.Stderr, "[claude-code] Auto check instructions warmed in %.0fs (first check after a restart stays fast)\n", time.Since(start).Seconds())
			return
		}
		last = err
		if !preempted || ctx.Err() != nil {
			break
		}
	}
	if w.ctx.Err() != nil {
		return // router closing
	}
	if last == nil {
		last = errors.New("too many attempts")
	}
	fmt.Fprintf(os.Stderr, "[claude-code] Auto check warm-up stopped: %v\n", last)
}

func (r *Router) warmupAttempt(ctx context.Context, w *classifierWarmup, payload []byte) (error, bool) {
	actx, cancel := context.WithTimeout(ctx, warmupAttemptTimeout)
	defer cancel()
	w.pmu.Lock()
	w.attempt, w.onMain, w.preempted = cancel, false, false
	w.pmu.Unlock()
	defer func() {
		w.pmu.Lock()
		w.attempt, w.onMain = nil, false
		w.pmu.Unlock()
	}()
	req, err := http.NewRequestWithContext(actx, http.MethodPost, r.URL()+"/v1/messages", bytes.NewReader(payload))
	if err != nil {
		return err, false
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(warmupHeader, "1")
	req.Header.Set("User-Agent", warmupUserAgent)
	resp, err := http.DefaultClient.Do(req)
	if err == nil {
		_, err = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if err == nil && resp.StatusCode != http.StatusOK {
			err = fmt.Errorf("HTTP %d", resp.StatusCode)
		}
	}
	w.pmu.Lock()
	preempted := w.preempted
	w.pmu.Unlock()
	return err, preempted
}
