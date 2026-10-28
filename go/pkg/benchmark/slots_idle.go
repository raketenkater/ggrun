package benchmark

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"time"
)

// llama.cpp honors a cancelled request only between ubatches, so a probe whose
// client timed out can leave the backend prefilling for many minutes (GLM-5.3,
// 2026-10-26: a 2-minute calibration pilot kept the backend busy 22 more
// minutes and the canary's speed probe then missed its own 20-minute timeout).
// A timed probe therefore first waits, bounded and outside its own timeout,
// until no slot is processing.
const (
	idleSlotPollInterval = 5 * time.Second
	// /slots is answered by the backend's scheduler, so during a long ubatch the
	// poll itself waits. A poll that times out is a busy scheduler.
	idleSlotPollTimeout = 15 * time.Second
)

type slotActivity int

const (
	slotsUnknown slotActivity = iota
	slotsIdle
	slotsBusy
)

// waitForIdleSlots blocks until the backend reports no processing slot, the
// IdleSlotWait bound passes, or the runner's Deadline arrives. A server without
// /slots, or with a format it does not recognize, is not waited on: missing
// evidence keeps the old behavior instead of delaying every probe.
func (r *Runner) waitForIdleSlots() time.Duration {
	waited, idle := r.idleSlotWait()
	if !idle {
		fmt.Fprintf(os.Stderr, "[benchmark] backend still processing an earlier request after %s; starting the timed probe anyway\n", waited.Round(time.Second))
	}
	return waited
}

// WaitForIdleSlots is the same bounded wait for a caller that decides itself
// what a still-busy backend means. It reports false when /slots still showed a
// processing slot at the bound, when the deadline had already passed, or when
// a reply arrived at or after it; idle, unknown and a zero IdleSlotWait within
// the bound report true.
func (r *Runner) WaitForIdleSlots() bool {
	_, idle := r.idleSlotWait()
	return idle
}

func (r *Runner) idleSlotWait() (time.Duration, bool) {
	limit := r.IdleSlotWait
	if limit <= 0 {
		return 0, true
	}
	started := time.Now()
	// One absolute end bounds the polls too: a /slots request answered by a
	// busy scheduler must not carry the wait past the caller's deadline, and a
	// reply that arrives after it is not success.
	end := started.Add(limit)
	if !r.Deadline.IsZero() && r.Deadline.Before(end) {
		end = r.Deadline
	}
	if !started.Before(end) {
		return 0, false
	}
	interval := r.idlePollInterval
	if interval <= 0 {
		interval = idleSlotPollInterval
	}
	pollTimeout := r.idlePollTimeout
	if pollTimeout <= 0 {
		pollTimeout = idleSlotPollTimeout
	}
	announced := false
	for {
		activity := r.slotActivity(min(pollTimeout, time.Until(end)), end)
		if !time.Now().Before(end) {
			return time.Since(started), false
		}
		if activity != slotsBusy {
			if announced {
				fmt.Fprintf(os.Stderr, "[benchmark] backend idle after %s; starting the timed probe\n", time.Since(started).Round(time.Second))
			}
			return time.Since(started), true
		}
		if !announced {
			fmt.Fprintf(os.Stderr, "[benchmark] backend is still processing an earlier request; waiting up to %s before the timed probe\n", time.Until(end).Round(time.Second))
			announced = true
		}
		time.Sleep(min(interval, time.Until(end)))
	}
}

// slotActivity reads GET /slots. Mainline llama.cpp reports is_processing per
// slot; ik_llama.cpp reports state (0 idle, 1 processing).
func (r *Runner) slotActivity(timeout time.Duration, end time.Time) slotActivity {
	ctx, cancel := context.WithDeadline(context.Background(), end)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(r.BaseURL, "/")+"/slots", nil)
	if err != nil {
		return slotsUnknown
	}
	resp, err := (&http.Client{Timeout: timeout}).Do(req)
	if err != nil {
		if isTimeout(err) {
			return slotsBusy
		}
		return slotsUnknown
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return slotsUnknown
	}
	var slots []map[string]json.RawMessage
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&slots); err != nil {
		if isTimeout(err) {
			return slotsBusy
		}
		return slotsUnknown
	}
	activity := slotsUnknown
	for _, slot := range slots {
		if raw, ok := slot["is_processing"]; ok {
			var processing bool
			if json.Unmarshal(raw, &processing) != nil {
				return slotsUnknown
			}
			if processing {
				return slotsBusy
			}
			activity = slotsIdle
			continue
		}
		if raw, ok := slot["state"]; ok {
			var state int
			if json.Unmarshal(raw, &state) != nil {
				return slotsUnknown
			}
			if state != 0 {
				return slotsBusy
			}
			activity = slotsIdle
			continue
		}
		return slotsUnknown
	}
	return activity
}

// backendHealthy prefers the caller's process-level judgement. Without one, any
// /health answer counts, as in the launch watchdog; ik_llama.cpp answers /health
// from its scheduler, so callers that own the process should supply it.
func (r *Runner) backendHealthy() bool {
	if r.BackendHealthy != nil {
		return r.BackendHealthy()
	}
	client := &http.Client{Timeout: idleSlotPollTimeout}
	resp, err := client.Get(strings.TrimRight(r.BaseURL, "/") + "/health")
	if err != nil {
		return false
	}
	_ = resp.Body.Close()
	return true
}

func isTimeout(err error) bool {
	var netErr net.Error
	return errors.Is(err, os.ErrDeadlineExceeded) || (errors.As(err, &netErr) && netErr.Timeout())
}
