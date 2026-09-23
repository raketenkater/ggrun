package main

import (
	"fmt"
	"os"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// hostMemorySettle bounds how long a launch waits for host memory that is still
// being returned by a process that just exited.
type hostMemorySettle struct {
	poll, flatFor, limit time.Duration
}

var defaultHostMemorySettle = hostMemorySettle{poll: 2 * time.Second, flatFor: 10 * time.Second, limit: 90 * time.Second}

// validateHostMemoryContainmentSettled is validateHostMemoryContainment that
// tolerates memory still being released by a process that just stopped.
//
// The containment ceiling is the host's available memory at detection time. A
// clean relaunch of MiMo-V2.6 started right after the previous 96 GiB backend
// had stopped and read ~96 GiB less than the first launch had; the fail-closed
// gate then refused a plan that fits, with "exceeds the 102048 MiB whole-host
// ceiling". While available memory is still rising the reading is stale, so
// re-read it and re-check; once it has been flat for a while the refusal is
// genuine and stands.
func validateHostMemoryContainmentSettled(req *launchRequest, caps *detect.Capabilities, strategy *placement.Strategy,
	s hostMemorySettle, available func() int, sleep func(time.Duration)) error {
	err := validateHostMemoryContainment(req, caps, strategy)
	if err == nil || caps == nil || available == nil {
		return err
	}
	best := caps.RAM.FreeMB
	announced := false
	var flat time.Duration
	for waited := time.Duration(0); waited < s.limit && flat < s.flatFor; waited += s.poll {
		sleep(s.poll)
		now := available()
		if now <= best+256 {
			flat += s.poll
			continue
		}
		flat = 0
		if !announced {
			fmt.Fprintf(os.Stderr, "[launch] host memory is still being released (%d MiB available and rising); waiting before the containment check\n", now)
			announced = true
		}
		best = now
		caps.RAM.FreeMB = now
		if err = validateHostMemoryContainment(req, caps, strategy); err == nil {
			fmt.Fprintf(os.Stderr, "[launch] host memory settled at %d MiB available; containment check passes\n", now)
			return nil
		}
	}
	return err
}

func validateHostMemoryContainmentWaiting(req *launchRequest, caps *detect.Capabilities, strategy *placement.Strategy) error {
	return validateHostMemoryContainmentSettled(req, caps, strategy, defaultHostMemorySettle, currentAvailableRAMMB, time.Sleep)
}

// waitForShutdownRelease holds ggrun's exit until the stopped backend's RAM and
// VRAM are back at the pre-launch baseline, bounded by timeout.
//
// Stop returns once the process and scope are stopped, but tearing down a
// 96 GiB-RAM / 43 GiB-VRAM MiMo-V2.6 backend outlasts that. ggrun exited, the
// port was free, and an immediate relaunch planned against 1.9 GiB of free
// VRAM on the 4070 and a 130 GiB RAM ceiling, derated as far as it safely
// could and failed closed. A clean shutdown has to mean the memory is free.
func waitForShutdownRelease(baseline *detect.Capabilities, timeout time.Duration, atBaseline func(*detect.Capabilities) bool, sleep func(time.Duration)) bool {
	if baseline == nil || atBaseline == nil {
		return true
	}
	if atBaseline(baseline) {
		return true
	}
	fmt.Fprintln(os.Stderr, "[launch] waiting for the backend to release its memory...")
	for waited := time.Duration(0); waited < timeout; waited += 500 * time.Millisecond {
		sleep(500 * time.Millisecond)
		if atBaseline(baseline) {
			fmt.Fprintf(os.Stderr, "[launch] backend memory released after %s\n", (waited + 500*time.Millisecond).Round(time.Second))
			return true
		}
	}
	fmt.Fprintf(os.Stderr, "[launch] warning: RAM/VRAM had not returned to the pre-launch level within %s; a relaunch now may see less free memory\n", timeout)
	return false
}
