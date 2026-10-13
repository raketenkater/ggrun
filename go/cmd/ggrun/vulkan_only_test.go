package main

import "testing"

// Full rig, matrix15: both the bundled ik CUDA build and the Vulkan build
// carry the mimo2 literal, and the large-MoE file-backed tie-break served
// MiMo-V2.6-Flash Q2_K (117.5 GiB) on Vulkan at 2.9 tok/s. The recommendation
// must see the same choice the launch makes.
func TestRecommendationSeesTheLaunchsVulkanChoice(t *testing.T) {
	ik := &backendInfo{Path: "/app/.bin/backends/ik_llama-server-cuda/llama-server", Tag: "ik_llama", Dialect: "ik_llama", IsIK: true}
	vk := &backendInfo{Path: "/app/.bin/backends/llama-server-vulkan/llama-server", Tag: "vulkan", Dialect: "vulkan"}
	candidates := []autoBackendCandidate{{info: ik, canonical: true}, {info: vk, canonical: true}}
	both := func(path, arch string) (bool, bool) { return true, true }
	if !defaultLaunchUsesVulkan(candidates, "mimo2", true, 120371, both) {
		t.Fatal("a large MoE both builds load was not seen as served on Vulkan")
	}
	if defaultLaunchUsesVulkan(candidates, "qwen35", false, 10679, both) {
		t.Fatal("a small dense model was seen as served on Vulkan")
	}
	ikOnly := func(path, arch string) (bool, bool) { return path == ik.Path, true }
	if defaultLaunchUsesVulkan(candidates, "glm-dsa", true, 227000, ikOnly) {
		t.Fatal("an arch only ik loads was seen as served on Vulkan")
	}
}
