package placement

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"
)

// boundedDecisionCases covers every validation level and finalist outcome the
// retry predicate distinguishes.
func boundedDecisionCases() []CalibrationDecision {
	var out []CalibrationDecision
	levels := []string{CalibrationValidationScreened, CalibrationValidationWorkflow, CalibrationValidationAdmission, CalibrationValidationBaselineBounded}
	outcomes := []string{"unavailable", "baseline-won", "promoted", FinalistOutcomeUnmeasured}
	for _, level := range levels {
		for _, outcome := range outcomes {
			for _, failed := range []bool{false, true} {
				for _, attempts := range []int{1, MaxUnmeasuredFinalistAttempts} {
					d := CalibrationDecision{Winner: "default", ValidationLevel: level, Finalist: "f", FinalistOutcome: outcome, FinalistAttempts: attempts}
					if failed {
						d.FinalistFailureClass, d.FinalistFailureReason = "class", "reason"
					}
					out = append(out, d)
				}
			}
		}
	}
	return out
}

// When the measured finalist is the calculated one, recording it changes
// nothing: the new predicate equals the old one in every case.
func TestSuppressesAutomaticSearch_EqualsRetryPredicateWithoutSwap(t *testing.T) {
	for _, d := range boundedDecisionCases() {
		legacy := d
		recorded := d
		recorded.CalculatedFinalist, recorded.CalculatedFingerprint = d.Finalist, "fp"
		for _, asked := range []string{"f", "g", ""} {
			want := asked != "" && d.SuppressesAutomaticAdmissionRetry(asked)
			if got := legacy.SuppressesAutomaticSearch(asked, "fp"); got != want {
				t.Errorf("legacy %+v asked %q: got %v want %v", d, asked, got, want)
			}
			if got := recorded.SuppressesAutomaticSearch(asked, "fp"); got != want {
				t.Errorf("recorded %+v asked %q: got %v want %v", d, asked, got, want)
			}
		}
	}
}

// A recorded calculated finalist binds the decision to that plan only. It is
// never ORed with the measured name: a plan whose calculated finalist is the
// measured one, or any other candidate, is a different plan and searches.
func TestSuppressesAutomaticSearch_BindsToRecordedCalculatedFinalist(t *testing.T) {
	d := CalibrationDecision{Winner: "default", ValidationLevel: CalibrationValidationBaselineBounded,
		Finalist: "directed", CalculatedFinalist: "calculated", CalculatedFingerprint: "fp", FinalistOutcome: "baseline-won"}
	if !d.SuppressesAutomaticSearch("calculated", "fp") {
		t.Error("a bounded decision does not cover the plan it searched")
	}
	for _, other := range []string{"directed", "another", ""} {
		if d.SuppressesAutomaticSearch(other, "fp") {
			t.Errorf("a decision for calculated finalist %q suppressed the plan with %q", d.CalculatedFinalist, other)
		}
	}
	// The same name with another strategy is another experiment.
	for _, fp := range []string{"other", ""} {
		if d.SuppressesAutomaticSearch("calculated", fp) {
			t.Errorf("same-name finalist with strategy %q was suppressed", fp)
		}
	}
	nameOnly := d
	nameOnly.CalculatedFingerprint = ""
	if nameOnly.SuppressesAutomaticSearch("calculated", "") || nameOnly.SuppressesAutomaticSearch("calculated", "fp") {
		t.Error("a recorded name without a strategy fingerprint suppressed the search")
	}
	d.Winner = "directed"
	d.FinalistOutcome = "promoted"
	if d.SuppressesAutomaticSearch("calculated", "fp") {
		t.Error("a promoted winner went through the retry predicate")
	}
	var nilDecision *CalibrationDecision
	if nilDecision.SuppressesAutomaticSearch("calculated", "fp") {
		t.Error("a nil decision suppressed the search")
	}
}

// The field is additive: an empty value is not written, a record written
// without it (main before this field) loads with it empty, and a record with
// a key this binary does not know still loads, as it does for older binaries
// reading a record that has calculated_finalist.
func TestCalibrationDecisionCalculatedFinalistIsOptional(t *testing.T) {
	data, err := json.Marshal(CalibrationDecision{ScopeKey: "k", Finalist: "f"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "calculated_finalist") {
		t.Errorf("an empty calculated finalist was written: %s", data)
	}
	dir := t.TempDir()
	write := func(body string) {
		t.Helper()
		path := CalibrationPath(dir, "k")
		if err := os.MkdirAll(dir+"/calibration", 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	schema := strconv.Itoa(CalibrationSchemaVersion)
	write(`{"schema_version":` + schema + `,"scope_key":"k","winner":"default","finalist":"f","finalist_outcome":"baseline-won"}`)
	if d, err := LoadCalibrationDecision(dir, "k"); err != nil || d.CalculatedFinalist != "" || d.Finalist != "f" {
		t.Errorf("a record without the field did not load unchanged: %+v err=%v", d, err)
	}
	write(`{"schema_version":` + schema + `,"scope_key":"k","winner":"default","finalist":"f","calculated_finalist":"c","future_field":1}`)
	if d, err := LoadCalibrationDecision(dir, "k"); err != nil || d.CalculatedFinalist != "c" {
		t.Errorf("a record with the field and an unknown key did not load: %+v err=%v", d, err)
	}
	if _, err := SaveCalibrationDecision(dir, CalibrationDecision{ScopeKey: "k", Finalist: "f", CalculatedFinalist: "c"}); err != nil {
		t.Fatal(err)
	}
	if d, err := LoadCalibrationDecision(dir, "k"); err != nil || d.CalculatedFinalist != "c" {
		t.Errorf("the field did not round-trip: %+v err=%v", d, err)
	}
}
