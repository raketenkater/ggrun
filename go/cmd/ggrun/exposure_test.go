package main

import (
	"strings"
	"testing"
)

// S-03: a non-loopback bind without an API key is announced; nothing else is.
func TestExposedHostWarning(t *testing.T) {
	none := func(string) string { return "" }
	for _, host := range []string{"", "127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"} {
		if w := exposedHostWarning(host, 8081, nil, none); w != "" {
			t.Errorf("%q warned: %s", host, w)
		}
	}
	for _, host := range []string{"0.0.0.0", "::", "192.168.1.5"} {
		w := exposedHostWarning(host, 8081, nil, none)
		if !strings.Contains(w, "no API key") || !strings.Contains(w, "8081") {
			t.Errorf("%q: %q", host, w)
		}
	}
	if w := exposedHostWarning("0.0.0.0", 8081, []string{"--api-key", "k"}, none); w != "" {
		t.Errorf("--api-key still warned")
	}
	if w := exposedHostWarning("0.0.0.0", 8081, []string{"--api-key-file=/k"}, none); w != "" {
		t.Errorf("--api-key-file still warned")
	}
	if w := exposedHostWarning("0.0.0.0", 8081, nil, func(k string) string {
		if k == "LLAMA_API_KEY" {
			return "k"
		}
		return ""
	}); w != "" {
		t.Errorf("LLAMA_API_KEY still warned")
	}
}
