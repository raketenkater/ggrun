package main

import (
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
