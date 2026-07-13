package main

import "testing"

func TestSpecTestComparisonMath(t *testing.T) {
	if got := percentGain(12, 10); got < 19.99 || got > 20.01 {
		t.Fatalf("percentGain = %f", got)
	}
	if got := inversePercentGain(8, 10); got < 24.99 || got > 25.01 {
		t.Fatalf("inversePercentGain = %f", got)
	}
	if got := percentRegression(95, 100); got < 4.99 || got > 5.01 {
		t.Fatalf("percentRegression = %f", got)
	}
	if got := percentRegression(105, 100); got != 0 {
		t.Fatalf("faster prompt processing reported regression: %f", got)
	}
	if got := absolutePercentDelta(90, 100); got < 9.99 || got > 10.01 {
		t.Fatalf("absolutePercentDelta = %f", got)
	}
}

func TestSpecTestIsKnownCommand(t *testing.T) {
	if !knownCommand("spec-test") {
		t.Fatal("spec-test must bypass compatibility launch dispatch")
	}
}

func TestSpecLaunchIdentityIgnoresNetworkAddressButTracksRuntimeFlags(t *testing.T) {
	a := specLaunchIdentity([]string{"server-a", "--model", "m.gguf", "--port", "8081", "--host", "127.0.0.1", "-ub", "256"})
	b := specLaunchIdentity([]string{"server-b", "--model", "m.gguf", "--port", "9090", "--host", "0.0.0.0", "-ub", "256"})
	if a != b {
		t.Fatal("network-only changes invalidated launch identity")
	}
	c := specLaunchIdentity([]string{"server-a", "--model", "m.gguf", "--port", "8081", "-ub", "128"})
	if c == a {
		t.Fatal("performance flag change did not invalidate launch identity")
	}
}
