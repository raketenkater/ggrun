package main

import (
	"os"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// The growth probe's numbers: 11,685 MiB after load, 11,857 MiB at 28k tokens.
// The recorder must file the peak growth once, under the runtime index, and
// never a smaller value later.
func TestServingGrowthRecorderFilesThePeakUnderTheRuntimeIndex(t *testing.T) {
	samples := []map[int]int{{0: 11815}, {0: 11857}, {0: 11843}}
	i := 0
	var recorded []map[int]int
	// --gpus 2: runtime index 0 is physical GPU 2.
	r := newServingGrowthRecorder(map[int]int{2: 11685}, []detect.GPU{{Index: 0}}, map[int]int{0: 2},
		func() map[int]int { s := map[int]int{2: samples[i][0]}; i++; return s },
		func(g map[int]int) error { recorded = append(recorded, g); return nil })
	for range samples {
		r.observe()
	}
	r.flush()
	if len(recorded) != 1 || recorded[0][0] != 172 {
		t.Fatalf("want one record of 172 MiB on runtime CUDA0, got %v", recorded)
	}
	r.flush()
	if len(recorded) != 1 {
		t.Fatalf("an unchanged peak was recorded again: %v", recorded)
	}
}

func TestServingGrowthRecorderIgnoresNoiseAndOtherDevices(t *testing.T) {
	var recorded []map[int]int
	r := newServingGrowthRecorder(map[int]int{0: 10000}, []detect.GPU{{Index: 0}, {Index: 1}}, nil,
		func() map[int]int { return map[int]int{0: 10010, 1: 9000} },
		func(g map[int]int) error { recorded = append(recorded, g); return nil })
	r.observe()
	r.flush()
	if len(recorded) != 0 {
		t.Fatalf("10 MiB of pool noise or an unsnapshotted device was recorded: %v", recorded)
	}
}

func TestServingGrowthRecorderStopFlushesAndIsSafe(t *testing.T) {
	var recorded []map[int]int
	r := newServingGrowthRecorder(map[int]int{0: 1000}, []detect.GPU{{Index: 0}}, nil,
		func() map[int]int { return map[int]int{0: 1500} },
		func(g map[int]int) error { recorded = append(recorded, g); return nil })
	go r.run(1<<40, 1<<40) // only stop drives it
	r.stop()
	r.stop()
	if len(recorded) != 1 || recorded[0][0] != 500 {
		t.Fatalf("stop must take a last sample and flush: %v", recorded)
	}
	var none *servingGrowthRecorder
	none.stop()
}

func TestProcessTreeIncludesTheRoot(t *testing.T) {
	tree := processTreePIDs(os.Getpid())
	if tree == nil {
		t.Skip("no /proc on this platform")
	}
	if !tree[os.Getpid()] {
		t.Fatal("the root process is part of its own tree")
	}
}
