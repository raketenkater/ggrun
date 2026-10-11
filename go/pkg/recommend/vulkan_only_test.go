package recommend

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Full rig, matrix15: MiMo-V2.6-Flash Q2_K was Best overall at a predicted
// 10.5 tok/s as "CUDA / ik_llama", but only the Vulkan build loads mimo2 and it
// decoded 2.9 tok/s. Such a row is marked, listed after rows that run as
// estimated where the ranking uses speed, and keeps its capability rank in
// Smartest.
func TestVulkanOnlyArchIsMarkedAndListedAfterLoadableRows(t *testing.T) {
	t.Cleanup(resetBackendProbes)
	SetInstalledArchSupport(func(*detect.Capabilities, string) (bool, bool) { return true, true })
	SetVulkanOnCUDA(func(_ *detect.Capabilities, arch string, _ bool, _ int) bool { return arch == "mimo2" })
	r := Recommendation{Candidate: Candidate{Name: "MiMo", Arch: "mimo2"}, BackendHint: "CUDA / ik_llama"}
	markBackendStatus(nil, &r)
	if r.BackendStatus != BackendNeedsSetup || r.BackendHint != "Vulkan" {
		t.Fatalf("Vulkan-only row not marked: %+v", r)
	}
	q := Recommendation{Candidate: Candidate{Name: "Qwen", Arch: "qwen35"}, BackendHint: "CUDA / ik_llama"}
	markBackendStatus(nil, &q)
	if q.BackendStatus != BackendReady {
		t.Fatalf("a CUDA-loadable row was marked %q", q.BackendStatus)
	}
	rows := []Recommendation{r, q}
	preferServable(rows, true)
	if rows[0].Name != "Qwen" {
		t.Fatalf("the Vulkan-only row stayed first: %v", []string{rows[0].Name, rows[1].Name})
	}
	rows = []Recommendation{r, q}
	preferServable(rows, false)
	if rows[0].Name != "MiMo" {
		t.Fatalf("Smartest moved the Vulkan-only row: %v", []string{rows[0].Name, rows[1].Name})
	}
}

// Without a build route the Vulkan build is what serves the row: it is ready,
// not waiting on setup, and still ranks after rows that run as estimated where
// the ranking uses speed.
func TestVulkanOnlyArchWithoutBuildRouteIsReady(t *testing.T) {
	t.Cleanup(resetBackendProbes)
	SetInstalledArchSupport(func(*detect.Capabilities, string) (bool, bool) { return true, true })
	SetVulkanOnCUDA(func(_ *detect.Capabilities, arch string, _ bool, _ int) bool { return arch == "mimo2" })
	SetBackendRoute(func(*detect.Capabilities, string, bool) bool { return false })
	r := Recommendation{Candidate: Candidate{Name: "MiMo", Arch: "mimo2"}, BackendHint: "CUDA / ik_llama"}
	markBackendStatus(nil, &r)
	if r.BackendStatus != BackendReady || r.BackendHint != "Vulkan" || strings.Contains(r.Reason, "offers") {
		t.Fatalf("Vulkan-only row without a build route: %+v", r)
	}
	q := Recommendation{Candidate: Candidate{Name: "Qwen", Arch: "qwen35"}, BackendStatus: BackendReady}
	rows := []Recommendation{r, q}
	preferServable(rows, true)
	if rows[0].Name != "Qwen" {
		t.Fatalf("the Vulkan-only row stayed first: %v", []string{rows[0].Name, rows[1].Name})
	}
}
