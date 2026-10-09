package recommend

import (
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Full rig, matrix15: MiMo-V2.6-Flash Q2_K was Best overall at a predicted
// 10.5 tok/s as "CUDA / ik_llama", but only the Vulkan build loads mimo2 and it
// decoded 2.9 tok/s. Such a row is marked and listed after rows that run now.
func TestVulkanOnlyArchIsMarkedAndListedAfterLoadableRows(t *testing.T) {
	t.Cleanup(func() { SetInstalledArchSupport(nil); SetVulkanOnCUDA(nil) })
	SetInstalledArchSupport(func(*detect.Capabilities, string) (bool, bool) { return true, true })
	SetVulkanOnCUDA(func(_ *detect.Capabilities, arch string, _ bool, _ int) bool { return arch == "mimo2" })
	r := Recommendation{Candidate: Candidate{Name: "MiMo", Arch: "mimo2"}, BackendHint: "CUDA / ik_llama"}
	markBackendBuild(nil, &r)
	if !r.NeedsBackendBuild || r.BackendHint != "Vulkan" {
		t.Fatalf("Vulkan-only row not marked: %+v", r)
	}
	q := Recommendation{Candidate: Candidate{Name: "Qwen", Arch: "qwen35"}, BackendHint: "CUDA / ik_llama"}
	markBackendBuild(nil, &q)
	if q.NeedsBackendBuild {
		t.Fatal("a CUDA-loadable row was marked")
	}
	rows := []Recommendation{r, q}
	preferLoadable(rows)
	if rows[0].Name != "Qwen" {
		t.Fatalf("the Vulkan-only row stayed first: %v", []string{rows[0].Name, rows[1].Name})
	}
}
