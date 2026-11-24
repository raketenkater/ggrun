package claudeauto

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func shortWarmup(t *testing.T) {
	t.Helper()
	min, idle, poll := warmupMinSystemBytes, warmupIdleFor, warmupPoll
	warmupMinSystemBytes, warmupIdleFor, warmupPoll = 2048, 50*time.Millisecond, 10*time.Millisecond
	t.Cleanup(func() { warmupMinSystemBytes, warmupIdleFor, warmupPoll = min, idle, poll })
}

func largeSystem(tag string) string {
	return ClassifierMarker + " " + tag + " " + strings.Repeat("rule. ", 600)
}

func bigCheck(system, secret string) string {
	return fmt.Sprintf(`{"model":"local","system":[{"type":"text","text":%q}],"tools":[{"name":"t","description":"d"}],`+
		`"messages":[{"role":"user","content":%q}]}`, system, secret)
}

const okBody = `{"type":"message","role":"assistant","content":[{"type":"text","text":"<block>no</block>"}],"usage":{"input_tokens":1,"output_tokens":1}}`

func warmRouter(t *testing.T, backend string) (*Router, string, string) {
	t.Helper()
	shortWarmup(t)
	router, metrics := newTestRouter(t, backend, 1)
	file := filepath.Join(t.TempDir(), "sub", "claude-classifier-system.json")
	if err := router.EnableClassifierWarmup(file); err != nil {
		t.Fatal(err)
	}
	return router, metrics, file
}

func TestWarmupCaptureStoresOnlySystemAndTools(t *testing.T) {
	var hits atomic.Int64
	backend := countingBackend(t, &hits)
	router, _, file := warmRouter(t, backend.URL)
	const secret = "SECRET-TRANSCRIPT-9f3a"
	sys := largeSystem("a")
	post(t, router, bigCheck(sys, secret))

	data, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), ClassifierMarker) || !strings.Contains(string(data), "rule. rule.") {
		t.Fatal("saved file lacks system text")
	}
	if strings.Contains(string(data), secret) || strings.Contains(string(data), `"messages"`) {
		t.Fatal("saved file contains messages")
	}
	if st, _ := os.Stat(file); st.Mode().Perm() != 0o600 {
		t.Fatalf("mode = %v", st.Mode().Perm())
	}
	w := router.warm.Load()
	if got := w.writes.Load(); got != 1 {
		t.Fatalf("writes = %d", got)
	}
	post(t, router, bigCheck(sys, "other transcript"))
	if got := w.writes.Load(); got != 1 {
		t.Fatalf("identical check rewrote file: writes = %d", got)
	}
	post(t, router, bigCheck(largeSystem("b"), secret))
	if got := w.writes.Load(); got != 2 {
		t.Fatalf("changed system did not rewrite: writes = %d", got)
	}
	// A short classifier request (review-eval, canary) never replaces it.
	before, _ := os.ReadFile(file)
	post(t, router, classifierBody(16))
	after, _ := os.ReadFile(file)
	if string(before) != string(after) || w.writes.Load() != 2 {
		t.Fatal("small classifier request changed the stored copy")
	}
	// Reload from disk by a fresh router.
	r2, _ := newTestRouter(t, backend.URL, 1)
	if err := r2.EnableClassifierWarmup(file); err != nil {
		t.Fatal(err)
	}
	post(t, r2, bigCheck(largeSystem("b"), "x"))
	if got := r2.warm.Load().writes.Load(); got != 0 {
		t.Fatalf("identical-after-reload check rewrote file: %d", got)
	}
}

type recBackend struct {
	mu     sync.Mutex
	bodies []map[string]json.RawMessage
}

func (b *recBackend) add(body []byte) int {
	var m map[string]json.RawMessage
	_ = json.Unmarshal(body, &m)
	b.mu.Lock()
	defer b.mu.Unlock()
	b.bodies = append(b.bodies, m)
	return len(b.bodies)
}

func (b *recBackend) count() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.bodies)
}

func waitUntil(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestWarmClassifierReplaysSystemOnly(t *testing.T) {
	rec := &recBackend{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, 1<<20)
		n, _ := r.Body.Read(buf)
		rec.add(buf[:n])
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okBody))
	}))
	t.Cleanup(srv.Close)
	router, metrics, _ := warmRouter(t, srv.URL)
	post(t, router, bigCheck(largeSystem("a"), "SECRET"))
	if router.watch.classifier.Load() != 1 {
		t.Fatal("real check not counted")
	}
	if !router.WarmClassifier() {
		t.Fatal("WarmClassifier returned false")
	}
	waitUntil(t, "warm-up request", func() bool { return rec.count() == 2 })
	records := waitForRecords(t, metrics, 2)
	time.Sleep(100 * time.Millisecond)
	if rec.count() != 2 {
		t.Fatalf("backend saw %d requests", rec.count())
	}
	real, warm := rec.bodies[0], rec.bodies[1]
	if string(warm["system"]) != string(real["system"]) {
		t.Errorf("warm system differs from real system")
	}
	if !strings.Contains(string(warm["messages"]), warmupText) || strings.Contains(string(warm["messages"]), "SECRET") {
		t.Errorf("warm messages = %s", warm["messages"])
	}
	if string(warm["tools"]) != string(real["tools"]) {
		t.Errorf("tools differ")
	}
	if !strings.Contains(string(warm["chat_template_kwargs"]), `"enable_thinking":false`) {
		t.Errorf("thinking not off: %s", warm["chat_template_kwargs"])
	}
	kinds := map[string]int{}
	for _, r := range records {
		kinds[r.Kind]++
	}
	if kinds[kindWarmup] != 1 || kinds[kindClassifier] != 1 {
		t.Errorf("kinds = %v", kinds)
	}
	if router.watch.classifier.Load() != 1 {
		t.Error("warm-up counted as a real review")
	}
	if got := router.warm.Load().writes.Load(); got != 1 {
		t.Errorf("warm-up captured: writes = %d", got)
	}
}

func TestWarmClassifierWithoutSavedFile(t *testing.T) {
	var hits atomic.Int64
	backend := countingBackend(t, &hits)
	router, _, _ := warmRouter(t, backend.URL)
	if router.WarmClassifier() {
		t.Fatal("started without stored instructions")
	}
	time.Sleep(50 * time.Millisecond)
	if hits.Load() != 0 {
		t.Fatal("sent a request")
	}
}

func TestWarmClassifierPreemptedByMainWork(t *testing.T) {
	var warmSeen, warmCancelled atomic.Int64
	var normalDone atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, 1<<20)
		n, _ := r.Body.Read(buf)
		body := string(buf[:n])
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(body, warmupText) {
			if warmSeen.Add(1) == 1 {
				<-r.Context().Done()
				warmCancelled.Add(1)
				return
			}
			_, _ = w.Write([]byte(okBody))
			return
		}
		if strings.Contains(body, "NORMAL") {
			normalDone.Store(true)
		}
		_, _ = w.Write([]byte(okBody))
	}))
	t.Cleanup(srv.Close)
	router, _, _ := warmRouter(t, srv.URL)
	post(t, router, bigCheck(largeSystem("a"), "x"))
	if !router.WarmClassifier() {
		t.Fatal("no warm-up")
	}
	waitUntil(t, "first warm-up", func() bool { return warmSeen.Load() == 1 })
	waitUntil(t, "warm-up on main route", func() bool {
		w := router.warm.Load()
		w.pmu.Lock()
		defer w.pmu.Unlock()
		return w.onMain
	})
	post(t, router, `{"model":"local","system":"coding","messages":[{"role":"user","content":"NORMAL"}]}`)
	if !normalDone.Load() {
		t.Fatal("normal request did not complete")
	}
	waitUntil(t, "warm-up cancelled", func() bool { return warmCancelled.Load() == 1 })
	waitUntil(t, "warm-up retried", func() bool { return warmSeen.Load() == 2 })
	waitUntil(t, "warm-up finished", func() bool { return !router.warm.Load().running.Load() })
}

func TestWarmClassifierReviewerFailureDoesNotReachMain(t *testing.T) {
	var mainHits, revHits atomic.Int64
	mainSrv := countingBackend(t, &mainHits)
	revSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		revHits.Add(1)
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	t.Cleanup(revSrv.Close)
	shortWarmup(t)
	router, _ := newTestRouterPair(t, mainSrv.URL, revSrv.URL, 1)
	file := filepath.Join(t.TempDir(), "c.json")
	if err := router.EnableClassifierWarmup(file); err != nil {
		t.Fatal(err)
	}
	// Seed the stored copy directly: a real check would fall back to main.
	w := router.warm.Load()
	w.mu.Lock()
	w.system = canonJSON(json.RawMessage(fmt.Sprintf("%q", largeSystem("a"))))
	w.mu.Unlock()
	if !router.WarmClassifier() {
		t.Fatal("no warm-up")
	}
	waitUntil(t, "reviewer hit", func() bool { return revHits.Load() >= 1 })
	waitUntil(t, "warm-up end", func() bool { return !w.running.Load() })
	if mainHits.Load() != 0 {
		t.Fatalf("main backend hit %d times", mainHits.Load())
	}
}

func TestCloseStopsWarmup(t *testing.T) {
	var seen atomic.Int64
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen.Add(1)
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })
	router, _, _ := warmRouter(t, srv.URL)
	w := router.warm.Load()
	w.mu.Lock()
	w.system = canonJSON(json.RawMessage(fmt.Sprintf("%q", largeSystem("a"))))
	w.mu.Unlock()
	router.WarmClassifier()
	waitUntil(t, "warm-up in flight", func() bool { return seen.Load() == 1 })
	done := make(chan struct{})
	go func() { _ = router.Close(); close(done) }()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("Close hung on warm-up")
	}
	if w.running.Load() {
		t.Fatal("warm-up still running after Close")
	}
}
