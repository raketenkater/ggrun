package main

import (
	"testing"

	"github.com/raketenkater/ggrun/clientkit"
)

func TestRemoteUtilityCapacityIncludesReviewOnlyAndFallback(t *testing.T) {
	p := clientkit.Profile{ContextPerSlot: 65536, Slots: 2}
	for _, tc := range []struct {
		name           string
		runtime        *claudeAutoRuntime
		context, slots int
		separate       bool
	}{
		{"main only", nil, 65536, 2, false},
		{"unseated companion", &claudeAutoRuntime{companion: &claudeCompanionProfile{ContextTokens: 8192}}, 65536, 2, false},
		{"review only", &claudeAutoRuntime{reviewerPort: 1, companion: &claudeCompanionProfile{ContextTokens: 8192, ServesWorkers: false}}, 8192, 1, true},
		{"fallback smaller", &claudeAutoRuntime{reviewerPort: 1, companion: &claudeCompanionProfile{ContextTokens: 262144, ServesWorkers: true}}, 65536, 1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			u := remoteUtilityCapacity(p, tc.runtime)
			if u.ContextPerSlot != tc.context || u.Slots != tc.slots || u.SeparateBackend != tc.separate || u.FallbackToMain != tc.separate {
				t.Fatalf("wrong utility capacity: %+v", u)
			}
		})
	}
}
