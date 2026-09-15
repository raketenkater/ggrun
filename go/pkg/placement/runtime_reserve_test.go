package placement

import "testing"

// Graph growth is model-graph state, so RelatedModelRuntimeGraphGrowth relaxes
// the backend match deliberately. A backend FEATURE allocation is not model-graph
// state: an expert cache is memory one build reserves on its own account.
//
// Measured on GLM-5.3-Flash 2026-09-15: a probe recorded under
// backend=glm-5-3-flash-hot-experts|hot-experts=14 carried 6,406 MiB on CUDA1
// into stock-backend launches, against 76 MiB of growth actually observed.
func TestBackendExpertCacheDeclarationIsRead(t *testing.T) {
	for backend, want := range map[string]bool{
		"glm-5-3-flash-hot-experts@llama-server-4d57452|hot-experts=14,inserts=2": true,
		"glm-5-3-flash-hot-experts@srv|hot-experts=1":                             true,
		"llama-server-abc":               false,
		"glm-5-3-flash@llama-server-abc": false,
		// A declared cache of zero is no cache.
		"fork@srv|hot-experts=0": false,
		"fork@srv|hot-experts=":  false,
		"fork@srv|hot-experts=x": false,
		"":                       false,
		"hot-experts=3":          true,
	} {
		if got := backendDeclaresExpertCache(backend); got != want {
			t.Errorf("backendDeclaresExpertCache(%q) = %v, want %v", backend, got, want)
		}
	}
}

// The guard must be symmetric: a cache-running backend must not inherit a
// no-cache measurement either, since that would under-reserve and risk the OOM
// the carry exists to prevent.
func TestExpertCacheGuardIsSymmetric(t *testing.T) {
	cache := "fork@srv|hot-experts=14"
	plain := "llama-server-abc"
	if backendDeclaresExpertCache(cache) == backendDeclaresExpertCache(plain) {
		t.Fatal("a cache-declaring backend and a plain one compared equal")
	}
	// Same class compares equal, so ordinary carrying is unaffected.
	if backendDeclaresExpertCache(plain) != backendDeclaresExpertCache("llama-server-different-build") {
		t.Error("two plain backends were treated as different classes")
	}
	if backendDeclaresExpertCache(cache) != backendDeclaresExpertCache("other@srv|hot-experts=2") {
		t.Error("two cache-declaring backends were treated as different classes")
	}
}
