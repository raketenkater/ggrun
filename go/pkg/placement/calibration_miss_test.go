package placement

import "testing"

func TestExplainCalibrationScopeMissNamesChangedComponents(t *testing.T) {
	dir := t.TempDir()
	recorded := CalibrationScopeKey{ModelIdentity: "m", BackendIdentity: "b1", ContextSize: 151552, Parallel: 2}
	if _, err := SaveCalibrationDecision(dir, CalibrationDecision{ScopeKey: recorded.String(), ModelBasename: "q.gguf", Winner: "default", Scope: &recorded}); err != nil {
		t.Fatal(err)
	}
	current := recorded
	current.BackendIdentity = "b2"
	current.Threads = 8
	file, differing := ExplainCalibrationScopeMiss(dir, "q.gguf", current)
	if file == "" || len(differing) != 2 || differing[0] != "BackendIdentity" || differing[1] != "Threads" {
		t.Fatalf("file=%q differing=%v", file, differing)
	}
	if file, _ := ExplainCalibrationScopeMiss(dir, "other.gguf", current); file != "" {
		t.Fatalf("another model's decision was compared: %s", file)
	}
}
