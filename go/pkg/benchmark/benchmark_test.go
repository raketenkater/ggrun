package benchmark

import (
	"encoding/json"
	"testing"
)

func TestEstimateTokens(t *testing.T) {
	if got := estimateTokens("hello world"); got != 2 {
		t.Fatalf("expected ~2 tokens for 11 chars, got %d", got)
	}
}

func TestResultJSON(t *testing.T) {
	r := &Result{
		Model:     "test",
		GenTPS:    15.5,
		Timestamp: 1234567890,
	}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if len(data) == 0 {
		t.Fatalf("empty json")
	}
}
