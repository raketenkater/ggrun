package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// The plain form is parsed by installers and update checks; it must not change.
func TestVersionPlainFormatUnchanged(t *testing.T) {
	read := captureStdout(t)
	cmdVersion(nil)
	if got := read(); got != "ggrun "+version+"\n" {
		t.Fatalf("plain version output changed: %q", got)
	}
}

func TestVersionJSONCarriesIdentity(t *testing.T) {
	read := captureStdout(t)
	cmdVersion([]string{"--json"})
	got := read()
	var id buildIdentity
	if err := json.Unmarshal([]byte(strings.TrimSpace(got)), &id); err != nil {
		t.Fatalf("version --json is not JSON: %q: %v", got, err)
	}
	if id.Version != version || id.GoVersion == "" || id.Platform == "" {
		t.Fatalf("identity incomplete: %+v", id)
	}
}
