package claudeauto

import (
	"context"
	"encoding/json"
	"errors"
	"io"
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

func TestQueueWaitReturnsRetryableOverloadWithoutTakingASlot(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	var calls atomic.Int64
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if calls.Add(1) == 1 {
			close(entered)
			select {
			case <-release:
			case <-req.Context().Done():
				return
			}
		}
		_, _ = io.WriteString(w, `{"usage":{"input_tokens":100,"output_tokens":10}}`)
	}))
	defer backend.Close()

	// Generous: the cancelled request below must be seen as cancelled by the
	// router well before this bound, even on a slow runner.
	bound := time.Second
	router, err := startRouter(backend.URL, backend.URL, false, 1, bound)
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		once.Do(func() { close(release) })
		if !closed {
			_ = router.Close()
		}
	}()
	metrics := filepath.Join(t.TempDir(), "metrics.jsonl")
	if err := router.EnableMetrics(metrics); err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Timeout: 3 * time.Second}
	defer client.CloseIdleConnections()
	post := func(ctx context.Context) (int, string, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, router.URL()+"/v1/messages", strings.NewReader(`{"model":"local"}`))
		if err != nil {
			return 0, "", err
		}
		res, err := client.Do(req)
		if err != nil {
			return 0, "", err
		}
		defer res.Body.Close()
		_, err = io.Copy(io.Discard, res.Body)
		return res.StatusCode, res.Header.Get("Retry-After"), err
	}
	first := make(chan int, 1)
	go func() {
		code, _, _ := post(context.Background())
		first <- code
	}()
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("initial turn failed to reach the backend")
	}

	// The queue expires while the first request continues service.
	code, retry, err := post(context.Background())
	if err != nil || code != http.StatusTooManyRequests || retry != "5" || calls.Load() != 1 {
		t.Fatalf("queued overload: status=%d retry=%q calls=%d err=%v", code, retry, calls.Load(), err)
	}
	// A client that gives up while queued. Cancel only once the router shows the
	// request queued, and release the backend only once the router has seen the
	// disconnect: otherwise the abandoned request could still be admitted when
	// the slot frees and be recorded as served instead of cancelled in queue.
	ctx, cancel := context.WithCancel(context.Background())
	cancelled := make(chan error, 1)
	go func() {
		_, _, err := post(ctx)
		cancelled <- err
	}()
	waitForQueued(t, router, 1)
	cancel()
	waitForQueueEmpty(t, router)
	if err := <-cancelled; !errors.Is(err, context.Canceled) {
		t.Fatalf("client cancellation: %v", err)
	}
	once.Do(func() { close(release) })
	if code := <-first; code != http.StatusOK {
		t.Fatalf("already-admitted request ended with HTTP %d", code)
	}
	if code, _, err := post(context.Background()); code != http.StatusOK || err != nil {
		t.Fatalf("next request could not take the released slot: %d %v", code, err)
	}
	err = router.Close()
	closed = true
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(metrics)
	if err != nil {
		t.Fatal(err)
	}
	var rejectCount, clientCancelCount int
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var rec RequestRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatal(err)
		}
		switch {
		case rec.Status == http.StatusTooManyRequests:
			rejectCount++
			if rec.Aborted || rec.Usage != (Usage{}) || rec.QueueMS < bound.Milliseconds() {
				t.Fatalf("overload must record a queue-only HTTP response: %+v", rec)
			}
		case rec.Aborted && rec.CancelPhase == "queue":
			clientCancelCount++
		}
	}
	if rejectCount != 1 || clientCancelCount != 1 {
		t.Fatalf("overloads=%d client cancellations=%d", rejectCount, clientCancelCount)
	}
}

// waitForQueueEmpty polls until the router has no queued main request.
func waitForQueueEmpty(t *testing.T, router *Router) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if router.mainQueued.Load() == 0 {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("router never noticed the client leaving the queue")
}

func TestQueueWaitFromEnv(t *testing.T) {
	for value, want := range map[string]time.Duration{"": 0, "1": time.Second, "120": 2 * time.Minute, "3600": time.Hour} {
		got, err := queueWaitFromEnv(value)
		if err != nil || got != want {
			t.Errorf("queueWaitFromEnv(%q) = %v, %v; want %v", value, got, err, want)
		}
	}
	for _, value := range []string{"0", "-5", "3601", "2m", "1.5", " 120"} {
		if _, err := queueWaitFromEnv(value); err == nil {
			t.Errorf("queueWaitFromEnv(%q) accepted an invalid bound", value)
		}
	}
}

func TestQueueWaitNeedsAdmission(t *testing.T) {
	t.Setenv(queueWaitEnv, "120")
	if router, err := StartRouter("http://127.0.0.1:1", "http://127.0.0.1:1", false, 0); err == nil {
		_ = router.Close()
		t.Fatal("a queue-wait bound without main-model admission was accepted")
	}
}
