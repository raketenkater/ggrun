package placement

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func TestPackGateUpChunks_FillsRemainderByLargest(t *testing.T) {
	// 3 GPUs, remainders after whole-layer packing (MB). gate+up chunk ~2/3 of a
	// 3370MB layer = 2247MB. Only the two big fragments can hold a chunk; the
	// 1000MB fragment (the x1 3060) is too small.
	remainder := []int{2900, 1000, 2700}
	gpuOrder := []int{0, 2, 1} // bandwidth order: 3090Ti, 4070, 3060
	pins, moved := packGateUpChunks(remainder, gpuOrder, 2247, 8, 35)

	if len(pins) != 2 {
		t.Fatalf("want 2 pins, got %d: %+v", len(pins), pins)
	}
	// Biggest remainder first (GPU0=2900), then GPU2=2700.
	if pins[0].Layer != 8 || pins[0].GI != 0 {
		t.Errorf("pin0 = %+v, want {Layer:8 GI:0}", pins[0])
	}
	if pins[1].Layer != 9 || pins[1].GI != 2 {
		t.Errorf("pin1 = %+v, want {Layer:9 GI:2}", pins[1])
	}
	if moved != 2*2247 {
		t.Errorf("moved = %d, want %d", moved, 2*2247)
	}
	if remainder[0] != 2900 { // must not mutate caller's slice
		t.Errorf("packGateUpChunks mutated caller's remainder slice")
	}
}

func TestPackGateUpChunks_NoRoom(t *testing.T) {
	pins, moved := packGateUpChunks([]int{500, 400, 900}, []int{0, 1, 2}, 2247, 8, 35)
	if len(pins) != 0 || moved != 0 {
		t.Fatalf("want no pins, got %d pins moved=%d", len(pins), moved)
	}
}

func TestPackGateUpChunks_CappedByCPULayers(t *testing.T) {
	// Plenty of VRAM but only 1 CPU-bound layer to pack.
	pins, moved := packGateUpChunks([]int{100000}, []int{0}, 2247, 40, 1)
	if len(pins) != 1 || moved != 2247 {
		t.Fatalf("want 1 pin moved=2247, got %d pins moved=%d", len(pins), moved)
	}
	if pins[0].Layer != 40 {
		t.Errorf("layer = %d, want 40", pins[0].Layer)
	}
}

func TestBuildOTStringWithSubPins_EmptyMatchesFromStart(t *testing.T) {
	gpus := []detect.GPU{{Index: 0}, {Index: 1}, {Index: 2}}
	layers := []int{3, 3, 2}
	order := []int{0, 1, 2}
	got := buildOTStringWithSubPins(layers, nil, gpus, order, 0, "llama")
	want := buildOTStringFromStart(layers, gpus, order, 0, "llama")
	if got != want {
		t.Errorf("with no sub-pins:\n got=%s\nwant=%s", got, want)
	}
}

func TestBuildOTStringWithSubPins_EmitsGateUpBeforeCatchAll(t *testing.T) {
	gpus := []detect.GPU{{Index: 0}, {Index: 1}, {Index: 2}}
	layers := []int{3, 0, 0} // whole layers 0,1,2 on GPU0
	order := []int{0, 1, 2}
	pins := []subExpertPin{{Layer: 3, GI: 0}, {Layer: 4, GI: 2}}
	got := buildOTStringWithSubPins(layers, pins, gpus, order, 0, "llama")

	if !strings.Contains(got, `blk\.(3)\.ffn_(gate_up|up_gate|gate|up)_exps.*=CUDA0`) {
		t.Errorf("missing gate+up pin for layer 3 -> CUDA0 in %q", got)
	}
	if !strings.Contains(got, `blk\.(4)\.ffn_(gate_up|up_gate|gate|up)_exps.*=CUDA2`) {
		t.Errorf("missing gate+up pin for layer 4 -> CUDA2 in %q", got)
	}
	// gate+up pins must precede the exps=CPU catch-all (first-match-wins).
	if i := strings.Index(got, "gate_up|up_gate|gate|up)_exps"); i < 0 || i > strings.LastIndex(got, "exps=CPU") {
		t.Errorf("gate+up pin must come before exps=CPU catch-all: %q", got)
	}
	if !strings.HasSuffix(got, "exps=CPU") {
		t.Errorf("must end with exps=CPU catch-all: %q", got)
	}
	// down must NOT be pinned to a GPU (stays CPU): the gate+up pattern excludes down.
	if strings.Contains(got, `gate|up)_exps.*=CUDA`) && strings.Contains(got, "down_exps.*=CUDA0,blk") {
		t.Errorf("down_exps should not be pinned by a gate+up rule: %q", got)
	}
}
