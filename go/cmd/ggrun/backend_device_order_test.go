package main

import (
	"os"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// The reference rig as ggrun detects it (NVIDIA PCI order) and as the Vulkan
// build lists it.
func rigGPUs() *detect.Capabilities {
	return &detect.Capabilities{GPUs: []detect.GPU{
		{Index: 0, Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12282, PCIBusID: "00000000:17:00.0"},
		{Index: 1, Name: "NVIDIA GeForce RTX 3090 Ti", VRAMTotalMB: 24564, PCIBusID: "00000000:65:00.0"},
		{Index: 2, Name: "NVIDIA GeForce RTX 3060", VRAMTotalMB: 12288, PCIBusID: "00000000:B3:00.0"},
	}}
}

const rigVulkanList = `Available devices:
  Vulkan0: NVIDIA GeForce RTX 3090 Ti (24810 MiB, 24321 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4070 (12528 MiB, 12075 MiB free)
  Vulkan2: NVIDIA GeForce RTX 3060 (12534 MiB, 12125 MiB free)
`

func listing(out string) func(string) ([]listedDevice, error) {
	return func(string) ([]listedDevice, error) { return parseVulkanDeviceList(out), nil }
}

func vulkanBackend() *backendInfo {
	return &backendInfo{Path: "/x/llama-server-vulkan", Dialect: "vulkan"}
}

// MiMo-V2.6 failed to allocate because the split was in ggrun's order and the
// backend's Vulkan1 was the 4070, not the 3090 Ti the 58% share was meant for.
func TestVulkanDevicesAreListedInDetectedOrder(t *testing.T) {
	t.Setenv("GGML_VK_VISIBLE_DEVICES", "")
	env, err := vulkanDeviceOrderEnv(&launchRequest{}, vulkanBackend(), rigGPUs(), listing(rigVulkanList))
	if err != nil {
		t.Fatal(err)
	}
	if env != "GGML_VK_VISIBLE_DEVICES=1,0,2" || os.Getenv("GGML_VK_VISIBLE_DEVICES") != "1,0,2" {
		t.Fatalf("want the backend's 4070,3090 Ti,3060 = 1,0,2, got %q (env %q)", env, os.Getenv("GGML_VK_VISIBLE_DEVICES"))
	}
}

// --gpus 1 means the 3090 Ti in ggrun's numbering; the generic path wrote
// GGML_VK_VISIBLE_DEVICES=1, which is the 4070 to the Vulkan backend.
func TestVulkanGPUSelectionTargetsThePhysicalCard(t *testing.T) {
	t.Setenv("GGML_VK_VISIBLE_DEVICES", "1")
	env, err := vulkanDeviceOrderEnv(&launchRequest{GPUsFlag: "1"}, vulkanBackend(), rigGPUs(), listing(rigVulkanList))
	if err != nil || env != "GGML_VK_VISIBLE_DEVICES=0" {
		t.Fatalf("--gpus 1 (3090 Ti) must select Vulkan device 0, got %q err=%v", env, err)
	}
	env, _ = vulkanDeviceOrderEnv(&launchRequest{GPUsFlag: "0,2"}, vulkanBackend(), rigGPUs(), listing(rigVulkanList))
	if env != "GGML_VK_VISIBLE_DEVICES=1,2" {
		t.Fatalf("--gpus 0,2 (4070, 3060) must be 1,2, got %q", env)
	}
}

func TestVulkanDeviceOrderLeavesMatchingOrderAndOtherBackendsAlone(t *testing.T) {
	t.Setenv("GGML_VK_VISIBLE_DEVICES", "")
	os.Unsetenv("GGML_VK_VISIBLE_DEVICES")
	same := "  Vulkan0: NVIDIA GeForce RTX 4070 (12528 MiB, 1 MiB free)\n  Vulkan1: NVIDIA GeForce RTX 3090 Ti (24810 MiB, 1 MiB free)\n  Vulkan2: NVIDIA GeForce RTX 3060 (12534 MiB, 1 MiB free)\n"
	if env, err := vulkanDeviceOrderEnv(&launchRequest{}, vulkanBackend(), rigGPUs(), listing(same)); env != "" || err != nil {
		t.Fatalf("an order that already matches must set nothing, got %q err=%v", env, err)
	}
	if _, set := os.LookupEnv("GGML_VK_VISIBLE_DEVICES"); set {
		t.Fatal("GGML_VK_VISIBLE_DEVICES was set; an empty value would hide every device")
	}
	cuda := &backendInfo{Path: "/x/llama-server", Dialect: "ik_llama"}
	if env, err := vulkanDeviceOrderEnv(&launchRequest{}, cuda, rigGPUs(), listing(rigVulkanList)); env != "" || err != nil {
		t.Fatalf("a CUDA backend must not be touched, got %q err=%v", env, err)
	}
}

func TestVulkanDeviceOrderRefusesAnUnmappableList(t *testing.T) {
	other := "  Vulkan0: AMD Radeon RX 7800 XT (16368 MiB, 1 MiB free)\n  Vulkan1: NVIDIA GeForce RTX 4070 (12528 MiB, 1 MiB free)\n  Vulkan2: NVIDIA GeForce RTX 3060 (12534 MiB, 1 MiB free)\n"
	if _, err := vulkanDeviceOrderEnv(&launchRequest{}, vulkanBackend(), rigGPUs(), listing(other)); err == nil ||
		!strings.Contains(err.Error(), "3090 Ti") {
		t.Fatalf("a list without the 3090 Ti must be refused by name, got %v", err)
	}
}
