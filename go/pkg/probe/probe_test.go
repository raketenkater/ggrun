package probe

import (
	"testing"
)

func TestProbe(t *testing.T) {
	m, err := Probe()
	if err != nil {
		t.Fatalf("probe failed: %v", err)
	}
	if m.RAMFreeMB == 0 {
		t.Fatalf("expected non-zero RAM free")
	}
	_ = m.TotalFree()
	_ = m.String()
}
