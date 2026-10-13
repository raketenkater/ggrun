package recommend

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// A row the installed Vulkan build loads is ready now. When the first launch
// offers the CUDA build its estimate assumes, that is a separate fact
// (CUDABuildOffered). Smartest keeps its capability rank; Best overall and
// Fastest list it after rows that run as estimated now, as they do a row
// whose backend the launch builds: MiMo-V2.6-Flash Q2_K, served on Vulkan,
// took 46 minutes to ready and 19-53 minutes per task (matrix15).
func TestVulkanRowWithCUDABuildOfferedIsReady(t *testing.T) {
	t.Cleanup(resetBackendProbes)
	SetInstalledArchSupport(func(*detect.Capabilities, string) (bool, bool) { return true, true })
	SetVulkanOnCUDA(func(_ *detect.Capabilities, arch string, _ bool, _ int) bool { return arch == "mimo2" })
	r := Recommendation{Candidate: Candidate{Name: "MiMo", Arch: "mimo2"}, BackendHint: "CUDA / ik_llama"}
	markBackendStatus(nil, &r)
	if r.BackendStatus != BackendReady || !r.CUDABuildOffered || r.BackendHint != "Vulkan" || !strings.Contains(r.Reason, "offers a CUDA build") {
		t.Fatalf("Vulkan row with a CUDA build offered: %+v", r)
	}
	q := Recommendation{Candidate: Candidate{Name: "Qwen", Arch: "qwen35"}, BackendHint: "CUDA / ik_llama"}
	markBackendStatus(nil, &q)
	if q.BackendStatus != BackendReady || q.CUDABuildOffered {
		t.Fatalf("a CUDA-loadable row was marked %q offered=%v", q.BackendStatus, q.CUDABuildOffered)
	}
	for speedRanked, first := range map[bool]string{true: "Qwen", false: "MiMo"} {
		rows := []Recommendation{r, q}
		preferServable(rows, speedRanked)
		if rows[0].Name != first {
			t.Fatalf("speedRanked=%v: %v, want %s first", speedRanked, []string{rows[0].Name, rows[1].Name}, first)
		}
	}
}

// Full rig, matrix15: MiMo-V2.6-Flash Q2_K was Best overall at a predicted
// 10.5 tok/s as "CUDA / ik_llama", but only the Vulkan build loads mimo2 and it
// decoded 2.9 tok/s. Without a CUDA build route the Vulkan build is what
// serves it: ready, listed after rows that run as estimated where the ranking
// uses speed, and kept at its capability rank in Smartest.
func TestVulkanOnlyArchWithoutBuildRouteIsReady(t *testing.T) {
	t.Cleanup(resetBackendProbes)
	SetInstalledArchSupport(func(*detect.Capabilities, string) (bool, bool) { return true, true })
	SetVulkanOnCUDA(func(_ *detect.Capabilities, arch string, _ bool, _ int) bool { return arch == "mimo2" })
	SetBackendRoute(func(*detect.Capabilities, string, bool) bool { return false })
	r := Recommendation{Candidate: Candidate{Name: "MiMo", Arch: "mimo2"}, BackendHint: "CUDA / ik_llama"}
	markBackendStatus(nil, &r)
	if r.BackendStatus != BackendReady || r.CUDABuildOffered || r.BackendHint != "Vulkan" || strings.Contains(r.Reason, "offers") {
		t.Fatalf("Vulkan-only row without a build route: %+v", r)
	}
	q := Recommendation{Candidate: Candidate{Name: "Qwen", Arch: "qwen35"}, BackendStatus: BackendReady}
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
