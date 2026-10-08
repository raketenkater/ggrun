package main

import (
	"testing"
	"time"
)

// The load cap stops churn, not progress. A converging preflight earns at most
// maxConvergingAdmissionLoads extra starts; the window and cap still bound it.
func TestConvergingPreflightEarnsBoundedExtraStarts(t *testing.T) {
	w := newAdmissionWork("start", time.Hour, time.Minute, maxStartAdmissionLoads, "", nil)
	for i := 0; i < maxStartAdmissionLoads; i++ {
		if _, err := w.beginLoad(time.Minute); err != nil {
			t.Fatalf("start %d refused inside the base allowance: %v", i+1, err)
		}
	}
	if _, err := w.beginLoad(time.Minute); err == nil || !isAdmissionBudgetError(err) {
		t.Fatalf("a start beyond the allowance without progress must be refused, got %v", err)
	}
	for i := 0; i < maxConvergingAdmissionLoads; i++ {
		if !w.grantConvergingLoad() {
			t.Fatalf("converging grant %d refused", i+1)
		}
		if _, err := w.beginLoad(time.Minute); err != nil {
			t.Fatalf("granted start %d refused: %v", i+1, err)
		}
	}
	if w.grantConvergingLoad() {
		t.Fatal("converging grants must be bounded")
	}
	if _, err := w.beginLoad(time.Minute); err == nil {
		t.Fatal("a start beyond base plus converging grants must be refused")
	}
	var unbounded *admissionWork
	if unbounded.grantConvergingLoad() {
		t.Fatal("nil admission work has nothing to grant")
	}
}
