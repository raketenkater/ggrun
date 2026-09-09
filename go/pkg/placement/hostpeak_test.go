package placement

import "testing"

// TestKillPeakRaisesTheScopeFloor is the regression for a loop that killed a
// live run. The scope is sized from a plan estimate; when the estimate is low
// the backend is SIGKILLed inside its own cgroup. runtimeCgroupOOM already read
// the peak at the kill, but nothing stored it, so the next launch derived the
// same estimate and could die at the same point. Measured 2026-09-06 on GLM 5.3
// Flash: killed at a 127742 MiB ceiling with the prompt cache still growing.
func TestKillPeakRaisesTheScopeFloor(t *testing.T) {
	dir := t.TempDir()
	model := "/models/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf"

	if got := HostPeakFloorMB(dir, model, 4096); got != 0 {
		t.Fatalf("an unobserved model must demand no floor, got %d", got)
	}
	if err := RecordMeasuredHostPeak(dir, model, HostPeak{
		PeakMB: 127742, FromKill: true, ContextSize: 942080, NCPUMoE: 43,
	}); err != nil {
		t.Fatalf("record: %v", err)
	}
	// A kill at P proves the scope must exceed P, so the floor clears it.
	if got := HostPeakFloorMB(dir, model, 4096); got <= 127742 {
		t.Fatalf("a kill must raise the floor above the peak it died at: got %d", got)
	}
	if got := MeasuredHostPeak(dir, model); !got.FromKill || got.ContextSize != 942080 {
		t.Fatalf("the kill and its layout must be recorded: %+v", got)
	}
}

// TestHostPeakNeverShrinks mirrors RecordCompanionVRAM: a ceiling must cover the
// worst case ever seen, and a quiet run that peaked lower is not evidence the
// busy one will not recur.
func TestHostPeakNeverShrinks(t *testing.T) {
	dir := t.TempDir()
	model := "/models/m.gguf"
	if err := RecordMeasuredHostPeak(dir, model, HostPeak{PeakMB: 120000, FromKill: true}); err != nil {
		t.Fatalf("record big: %v", err)
	}
	if err := RecordMeasuredHostPeak(dir, model, HostPeak{PeakMB: 40000}); err != nil {
		t.Fatalf("record small: %v", err)
	}
	if got := MeasuredHostPeak(dir, model).PeakMB; got != 120000 {
		t.Fatalf("a quieter run must not lower the peak: got %d, want 120000", got)
	}
}

// TestHealthyPeakNeedsNoExtraHeadroom keeps the two sources distinct. A kill
// proves the scope was too small and must be exceeded; a healthy peak only
// needs to be covered, and padding it would withhold host RAM for no observed
// reason.
func TestHealthyPeakNeedsNoExtraHeadroom(t *testing.T) {
	dir := t.TempDir()
	model := "/models/m.gguf"
	if err := RecordMeasuredHostPeak(dir, model, HostPeak{PeakMB: 90000, FromKill: false}); err != nil {
		t.Fatalf("record: %v", err)
	}
	if got := HostPeakFloorMB(dir, model, 8192); got != 90000 {
		t.Fatalf("a healthy peak needs covering, not padding: got %d, want 90000", got)
	}
}
