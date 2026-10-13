package placement

import (
	"slices"
	"testing"
)

// A context shift silently drops half of an agent's conversation, and on
// ik_llama (shift on by default) Gemma 4 12B crashed with an illegal memory
// access right after one (matrix11). Every model gets --no-context-shift when
// the backend takes it; a backend that does not list it gets nothing new.
func TestContextShiftIsOffWhenTheBackendCanTurnItOff(t *testing.T) {
	ik := "  --no-context-shift       disable context-shift.\n  --context-shift (auto|on|off|0|1)"
	dense := &Strategy{Type: SingleGPU, ContextSize: 43008, KVType: "q4_0", Parallel: 1,
		BackendSupportsNoContextShift: backendHelpSupportsExactFlag(ik, "--no-context-shift")}
	if !slices.Contains(dense.Args("gemma.gguf", 8081), "--no-context-shift") {
		t.Fatal("a dense model kept the backend's context shift")
	}
	old := &Strategy{Type: SingleGPU, ContextSize: 4096, Parallel: 1,
		BackendSupportsNoContextShift: backendHelpSupportsExactFlag("  --no-context-shift-foo  x", "--no-context-shift")}
	if slices.Contains(old.Args("m.gguf", 8081), "--no-context-shift") {
		t.Fatal("emitted --no-context-shift to a backend that does not list it")
	}
	ssm := &Strategy{Type: SingleGPU, ContextSize: 4096, Parallel: 1, HasSSM: true}
	if args := ssm.Args("m.gguf", 8081); !slices.Contains(args, "--no-context-shift") {
		t.Fatal("a recurrent model lost --no-context-shift")
	}
}
