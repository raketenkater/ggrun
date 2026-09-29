package main

import (
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// The MiMo-V2.6 relaunch: the plan needs 92,776 MiB + a 10,240 MiB reserve,
// detection ran while the previous 96 GiB backend was still being torn down.
func relaunchFixture(freeMB int) (*launchRequest, *detect.Capabilities, *placement.Strategy) {
	req := &launchRequest{RAMHeadroomMB: 4096, CgroupHeadroomMB: 4096, NoMMap: true}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 212000, FreeMB: freeMB}}
	strategy := &placement.Strategy{Type: placement.MoEOffload, PlannedHostFootprintMB: 92776, CRAM: 10240}
	return req, caps, strategy
}

func fakeClock() (func(time.Duration), *time.Duration) {
	var waited time.Duration
	return func(d time.Duration) { waited += d }, &waited
}

func TestRelaunchWaitsForMemoryStillBeingReleased(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("host containment applies on Linux")
	}
	req, caps, strategy := relaunchFixture(106144) // ceiling 102048: refused
	if validateHostMemoryContainment(req, caps, strategy) == nil {
		t.Fatal("fixture must start below the ceiling")
	}
	readings := []int{130000, 170000, 201000}
	available := func() int {
		v := readings[0]
		if len(readings) > 1 {
			readings = readings[1:]
		}
		return v
	}
	sleep, waited := fakeClock()
	if err := validateHostMemoryContainmentSettled(req, caps, strategy, defaultHostMemorySettle, available, sleep); err != nil {
		t.Fatalf("memory returned by the previous backend was not waited for: %v", err)
	}
	if *waited > 20*time.Second {
		t.Errorf("waited %s; should pass as soon as the reading clears the gate", *waited)
	}
}

// A host that is simply too small must still be refused, and quickly.
func TestGenuinelyInsufficientMemoryStillRefusedPromptly(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("host containment applies on Linux")
	}
	req, caps, strategy := relaunchFixture(106144)
	sleep, waited := fakeClock()
	err := validateHostMemoryContainmentSettled(req, caps, strategy, defaultHostMemorySettle, func() int { return 106144 }, sleep)
	if err == nil {
		t.Fatal("a plan that does not fit was admitted")
	}
	if *waited > defaultHostMemorySettle.flatFor+defaultHostMemorySettle.poll {
		t.Errorf("waited %s on flat memory; must give up after %s", *waited, defaultHostMemorySettle.flatFor)
	}
}

// ggrun must not exit while the backend it stopped still holds memory.
func TestShutdownWaitsForBackendMemoryRelease(t *testing.T) {
	baseline := &detect.Capabilities{}
	checks := 0
	releasedAfter := 6
	sleep, waited := fakeClock()
	ok := waitForShutdownRelease(baseline, 2*time.Minute, func(*detect.Capabilities) bool {
		checks++
		return checks > releasedAfter
	}, sleep)
	if !ok || checks != releasedAfter+1 {
		t.Fatalf("ok=%v after %d checks; want release observed on check %d", ok, checks, releasedAfter+1)
	}
	if *waited != time.Duration(releasedAfter)*500*time.Millisecond {
		t.Errorf("waited %s", *waited)
	}

	sleep, waited = fakeClock()
	if waitForShutdownRelease(baseline, 5*time.Second, func(*detect.Capabilities) bool { return false }, sleep) {
		t.Fatal("reported release that never happened")
	}
	if *waited > 5*time.Second {
		t.Errorf("waited %s past its bound", *waited)
	}
}

// MiniMax-M3 (about 120 GiB resident) outlasted a 2-minute exit wait and the
// supervising harness force-killed the launcher. Exit waits briefly, and the
// memory still returning is waited for by the next launch instead.
func TestPendingReleaseMovesTheWaitToTheNextLaunch(t *testing.T) {
	if shutdownReleaseWait >= 30*time.Second {
		t.Fatalf("exit wait %s is not below a supervisor's typical 30 s grace", shutdownReleaseWait)
	}
	dir := t.TempDir()
	if releaseIsPending(dir) {
		t.Fatal("marker present before any shutdown")
	}
	calls := 0
	read := func() (int, int) { calls++; return 100000, 1000 }
	waitForPendingRelease(dir, time.Minute, read, func(time.Duration) {})
	if calls != 0 {
		t.Fatal("an ordinary launch waited for a release that was never pending")
	}
	markReleasePending(dir)
	if !releaseIsPending(dir) {
		t.Fatal("marker not written")
	}
	ram, vram := 60000, 40000
	read = func() (int, int) {
		calls++
		if ram < 180000 {
			ram += 40000
			vram -= 13000
		}
		return ram, vram
	}
	waitForPendingRelease(dir, time.Minute, read, func(time.Duration) {})
	if ram < 180000 {
		t.Fatalf("launch stopped waiting while memory was still returning (ram %d)", ram)
	}
	if releaseIsPending(dir) {
		t.Fatal("marker not cleared after waiting")
	}
}

// A plan the containment gate refuses is re-planned once with the gate's
// reserve held back from placement, and the re-plan must pass the same gate.
func TestContainHostMemoryReplansWithTheReserveHeldBack(t *testing.T) {
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 131072, FreeMB: 131072}}
	gate := func(req *launchRequest, _ *detect.Capabilities, s *placement.Strategy) error {
		if s.PlannedHostFootprintMB+max(req.CgroupHeadroomMB, s.CRAM) > req.RamBudgetMB {
			return fmt.Errorf("planned host footprint %d MiB exceeds the ceiling", s.PlannedHostFootprintMB)
		}
		return nil
	}
	first := &placement.Strategy{Type: placement.MoEOffload, ContextSize: 78848, PlannedHostFootprintMB: 26497, CRAM: 1024}
	contained := &placement.Strategy{Type: placement.MoEOffload, ContextSize: 58368, PlannedHostFootprintMB: 24434, CRAM: 1024}

	req := &launchRequest{RamBudgetMB: 28672, CgroupHeadroomMB: 4096}
	var reserveSeen int
	got, err := containHostMemory(req, caps, first, func(r *launchRequest) (*placement.Strategy, error) {
		reserveSeen = r.PlacementHostReserveMB
		return contained, nil
	}, gate)
	if err != nil || got != contained {
		t.Fatalf("refused plan was not re-planned into containment: %v %+v", err, got)
	}
	if reserveSeen != 4096 {
		t.Fatalf("re-plan held back %d MiB, want the gate's 4096 MiB reserve", reserveSeen)
	}

	// A re-plan that still does not fit is refused, not launched.
	req = &launchRequest{RamBudgetMB: 28672, CgroupHeadroomMB: 4096}
	if _, err := containHostMemory(req, caps, first, func(*launchRequest) (*placement.Strategy, error) { return first, nil }, gate); err == nil {
		t.Fatal("an uncontainable re-plan was accepted")
	}

	// A plan that already fits is returned unchanged with no re-plan.
	req = &launchRequest{RamBudgetMB: 28672, CgroupHeadroomMB: 4096}
	got, err = containHostMemory(req, caps, contained, func(*launchRequest) (*placement.Strategy, error) {
		t.Fatal("a contained plan was re-planned")
		return nil, nil
	}, gate)
	if err != nil || got != contained || req.PlacementHostReserveMB != 0 {
		t.Fatalf("contained plan changed: %v %+v reserve=%d", err, got, req.PlacementHostReserveMB)
	}
}
