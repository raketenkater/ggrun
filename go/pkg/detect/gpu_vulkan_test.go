package detect

import "testing"

func TestParseVulkanHeapVRAMUsesDeviceLocalHeap(t *testing.T) {
	// Representative full `vulkaninfo` excerpt: a discrete GPU with a 11.99 GiB
	// DEVICE_LOCAL heap and a larger non-local system-RAM heap that must be
	// ignored, plus an integrated GPU whose DEVICE_LOCAL heap (shared RAM) must
	// be skipped so it falls back to the name heuristic.
	full := `
GPU0:
VkPhysicalDeviceProperties:
	deviceName = AMD Radeon RX 7900 XTX
	deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
VkPhysicalDeviceMemoryProperties:
	memoryHeaps: count = 2
		memoryHeaps[0]:
			size   = 12878610432 (0x2ff800000) (11.99 GiB)
			flags: count = 1
				MEMORY_HEAP_DEVICE_LOCAL_BIT
		memoryHeaps[1]:
			size   = 67219730432 (0xfa6000000) (62.60 GiB)
			flags: count = 0
				None
GPU1:
VkPhysicalDeviceProperties:
	deviceName = AMD Radeon Graphics
	deviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
VkPhysicalDeviceMemoryProperties:
	memoryHeaps: count = 1
		memoryHeaps[0]:
			size   = 33609865216 (0x7d3000000) (31.30 GiB)
			flags: count = 1
				MEMORY_HEAP_DEVICE_LOCAL_BIT
`
	heaps := parseVulkanHeapVRAM(full)
	if got := heaps["AMD Radeon RX 7900 XTX"]; got != 12282 {
		t.Fatalf("discrete GPU: expected 12282 MB from the DEVICE_LOCAL heap, got %d", got)
	}
	if _, ok := heaps["AMD Radeon Graphics"]; ok {
		t.Fatalf("integrated GPU must be skipped (its DEVICE_LOCAL heap is shared RAM), got %d", heaps["AMD Radeon Graphics"])
	}
}
