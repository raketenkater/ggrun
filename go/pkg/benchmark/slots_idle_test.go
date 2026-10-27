package benchmark

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// busySlotServer simulates a llama.cpp backend still prefilling a request its
// client already cancelled. /slots reports a processing slot until busyFor has
// passed (mainline is_processing, or ik_llama.cpp state when ik is set); while
// blocking is set the /slots poll itself stalls, as the busy scheduler does.
// Each chat request records whether it arrived while the slot was busy.
type busySlotServer struct {
	*httptest.Server
	busyUntil time.Time
	mu        sync.Mutex
	chats     []bool // true = arrived while busy
	slotPolls int
}

func newBusySlotServer(t *testing.T, busyFor time.Duration, ik, blocking bool, chatDelay time.Duration) *busySlotServer {
	t.Helper()
	s := &busySlotServer{busyUntil: time.Now().Add(busyFor)}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		busy := time.Now().Before(s.busyUntil)
		switch {
		case strings.HasSuffix(r.URL.Path, "/slots"):
			s.mu.Lock()
			s.slotPolls++
			s.mu.Unlock()
			if busy && blocking {
				select {
				case <-time.After(time.Until(s.busyUntil)):
				case <-r.Context().Done():
					return
				}
				busy = false
			}
			slot := map[string]interface{}{"id": 0, "is_processing": busy}
			if ik {
				state := 0
				if busy {
					state = 1
				}
				slot = map[string]interface{}{"id": 0, "state": state}
			}
			_ = json.NewEncoder(w).Encode([]interface{}{slot})
		case strings.HasSuffix(r.URL.Path, "/health"):
			_, _ = w.Write([]byte(`{"status":"ok"}`))
		case strings.HasSuffix(r.URL.Path, "/props"):
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"default_generation_settings": map[string]interface{}{"n_ctx": 262144},
			})
		case strings.HasSuffix(r.URL.Path, "/tokenize"):
			var body struct {
				Content string `json:"content"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"tokens": make([]int, len(body.Content)/4)})
		default:
			s.mu.Lock()
			s.chats = append(s.chats, busy)
			s.mu.Unlock()
			if chatDelay > 0 {
				select {
				case <-time.After(chatDelay):
				case <-r.Context().Done():
					return
				}
			}
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"choices": []interface{}{map[string]interface{}{"message": map[string]string{"content": "GGRUN_OK"}}},
				"usage":   map[string]interface{}{"prompt_tokens": 900},
				"timings": map[string]interface{}{"cache_n": 800, "prompt_per_second": 5000.0},
			})
		}
	}))
	t.Cleanup(s.Server.Close)
	return s
}

func (s *busySlotServer) chatsWhileBusy() (total, busy int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, b := range s.chats {
		if b {
			busy++
		}
	}
	return len(s.chats), busy
}

func fastIdleRunner(url string, wait time.Duration) *Runner {
	return &Runner{
		BaseURL: url, Model: "m", Timeout: 2 * time.Second, ContextTokens: 262144,
		PrefillBudget: time.Hour, IdleSlotWait: wait,
		idlePollInterval: 10 * time.Millisecond, idlePollTimeout: 50 * time.Millisecond,
	}
}

// Invariant: a timed probe never starts while the backend still reports a
// processing slot, and the wait is not charged to the probe's own timeout:
// the backend stays busy longer than the per-request Timeout and the canary
// still passes.
func TestTimedProbesWaitForIdleSlots(t *testing.T) {
	for _, c := range []struct {
		name         string
		ik, blocking bool
	}{
		{"mainline is_processing", false, false},
		{"ik_llama state", true, false},
		{"scheduler too busy to answer /slots", false, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			srv := newBusySlotServer(t, 400*time.Millisecond, c.ik, c.blocking, 0)
			runner := fastIdleRunner(srv.URL, time.Minute)
			runner.Timeout = 150 * time.Millisecond
			result, err := runner.RunCacheCanary()
			if err != nil {
				t.Fatalf("canary after a busy slot: %v", err)
			}
			if !result.Functional || result.TooSlow {
				t.Fatalf("canary result %+v", result)
			}
			total, busy := srv.chatsWhileBusy()
			if total == 0 || busy != 0 {
				t.Fatalf("%d of %d timed requests started while a slot was processing", busy, total)
			}
		})
	}
	t.Run("agent prefill pilot", func(t *testing.T) {
		srv := newBusySlotServer(t, 300*time.Millisecond, false, false, 0)
		if _, err := fastIdleRunner(srv.URL, time.Minute).ProbeAgentPrefill(); err != nil {
			t.Fatal(err)
		}
		if total, busy := srv.chatsWhileBusy(); total != 1 || busy != 0 {
			t.Fatalf("pilot sent %d requests, %d while busy", total, busy)
		}
	})
}

// The wait is bounded by IdleSlotWait and by the runner's Deadline, and a
// server without /slots (or zero IdleSlotWait) is not waited on at all.
func TestIdleSlotWaitIsBounded(t *testing.T) {
	alwaysBusy := newBusySlotServer(t, time.Hour, false, false, 0)
	runner := fastIdleRunner(alwaysBusy.URL, 150*time.Millisecond)
	if waited := runner.waitForIdleSlots(); waited < 150*time.Millisecond || waited > 2*time.Second {
		t.Fatalf("IdleSlotWait bound: waited %s", waited)
	}
	runner = fastIdleRunner(alwaysBusy.URL, time.Hour)
	runner.Deadline = time.Now().Add(100 * time.Millisecond)
	if waited := runner.waitForIdleSlots(); waited > 2*time.Second {
		t.Fatalf("Deadline bound: waited %s", waited)
	}
	runner = fastIdleRunner(alwaysBusy.URL, 0)
	if waited := runner.waitForIdleSlots(); waited != 0 {
		t.Fatalf("disabled wait still waited %s", waited)
	}

	noSlots := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/slots") {
			http.Error(w, `{"error":{"message":"This server does not support slots endpoint."}}`, http.StatusNotImplemented)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"choices": []interface{}{}})
	}))
	defer noSlots.Close()
	started := time.Now()
	if waited := fastIdleRunner(noSlots.URL, time.Hour).waitForIdleSlots(); waited > time.Second || time.Since(started) > time.Second {
		t.Fatalf("server without /slots was waited on for %s", waited)
	}
	unknownFormat := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`[{"id":0}]`))
	}))
	defer unknownFormat.Close()
	if waited := fastIdleRunner(unknownFormat.URL, time.Hour).waitForIdleSlots(); waited > time.Second {
		t.Fatalf("unrecognized /slots format was waited on for %s", waited)
	}
}

// A caller that refuses to measure a busy backend learns that the bound passed
// with a slot still processing; idle and missing evidence stay true.
func TestWaitForIdleSlotsReportsAStillBusyBackend(t *testing.T) {
	alwaysBusy := newBusySlotServer(t, time.Hour, false, false, 0)
	runner := fastIdleRunner(alwaysBusy.URL, time.Hour)
	runner.Deadline = time.Now().Add(100 * time.Millisecond)
	started := time.Now()
	if runner.WaitForIdleSlots() {
		t.Fatal("a backend busy past the deadline was reported idle")
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("the wait outlived its deadline by %s", elapsed)
	}
	if !fastIdleRunner(newBusySlotServer(t, 100*time.Millisecond, true, false, 0).URL, time.Hour).WaitForIdleSlots() {
		t.Fatal("a backend that went idle was reported busy")
	}
	unknownFormat := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`[{"id":0}]`))
	}))
	defer unknownFormat.Close()
	if !fastIdleRunner(unknownFormat.URL, time.Hour).WaitForIdleSlots() {
		t.Fatal("unrecognized /slots evidence was reported busy")
	}
}

// Invariant: a speed probe that times out on a healthy backend measured
// nothing and must not reject the profile; on an unhealthy backend it stays an
// error. Health comes from the caller when supplied, else from /health.
func TestCanarySpeedProbeTimeoutOnHealthyBackendIsTooSlow(t *testing.T) {
	srv := newBusySlotServer(t, 0, false, false, 500*time.Millisecond)
	for _, c := range []struct {
		name    string
		healthy func() bool
		tooSlow bool
	}{
		{"caller reports healthy", func() bool { return true }, true},
		{"/health answers", nil, true},
		{"caller reports unhealthy", func() bool { return false }, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			runner := fastIdleRunner(srv.URL, time.Minute)
			runner.Timeout = 100 * time.Millisecond
			runner.BackendHealthy = c.healthy
			result, err := runner.RunCacheCanary()
			if c.tooSlow {
				if err != nil || result == nil || !result.TooSlow || result.Functional || result.Passed {
					t.Fatalf("healthy slow backend: result=%+v err=%v", result, err)
				}
				if !strings.Contains(result.Reason, "endpoint answered too slowly to measure") {
					t.Fatalf("reason %q", result.Reason)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), "prefill speed probe") {
				t.Fatalf("unhealthy backend was not an error: result=%+v err=%v", result, err)
			}
		})
	}
	down := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := down.URL
	down.Close()
	if (&Runner{BaseURL: url}).backendHealthy() {
		t.Fatal("a closed listener was judged healthy")
	}
}
