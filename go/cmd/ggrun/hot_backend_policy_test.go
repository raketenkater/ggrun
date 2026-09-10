package main

import "testing"

func TestRequiredHotExpertBackendPolicy(t *testing.T) {
	for _, policy := range []string{"", "off", "auto", "on", "32"} {
		for _, help := range []string{"--parallel N", "--moe-expert-cache N", "--moe-expert-cache N --moe-expert-cache-inserts N"} {
			req := &launchRequest{HotExperts: policy}
			err := validateRequiredHotExpertBackend(req, &backendInfo{Tag: "merged", Help: help})
			wantErr := (policy == "on" || policy == "32") && help != "--moe-expert-cache N --moe-expert-cache-inserts N"
			if (err != nil) != wantErr {
				t.Fatalf("policy=%q help=%q: err=%v, want error=%v", policy, help, err, wantErr)
			}
		}
	}
}
