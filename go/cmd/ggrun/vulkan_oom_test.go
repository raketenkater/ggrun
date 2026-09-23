package main

import "testing"

// Lines from the MiMo-V2.6 Vulkan probe that failed closed: the measured
// compute-buffer shortfall must reach the ubatch/expert-layer recovery.
func TestVulkanAllocationFailuresReachStartupOOMRecovery(t *testing.T) {
	compute := "ggml_vulkan: Device memory allocation of size 1140850688 failed.\n" +
		"ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory\n" +
		"12.47.431.169 E ggml_gallocr_reserve_n_impl: failed to allocate Vulkan0 buffer of size 1962934272\n" +
		"12.47.431.174 E graph_reserve: failed to allocate compute buffers\n"
	device, mb, isCompute, ok := startupLogCUDAOOMDetailed(compute)
	if !ok || device != 0 || mb != 1872 || !isCompute {
		t.Fatalf("compute-buffer OOM: device=%d MiB=%d compute=%v ok=%v; want 0, 1872, true, true", device, mb, isCompute, ok)
	}

	weights := "0.01.013.527 E alloc_tensor_range: failed to allocate Vulkan1 buffer of size 822083584\n" +
		"0.01.324.983 E llama_model_load: error loading model: unable to allocate Vulkan1 buffer\n"
	device, mb, isCompute, ok = startupLogCUDAOOMDetailed(weights)
	if !ok || device != 1 || mb != 784 || isCompute {
		t.Fatalf("weights OOM: device=%d MiB=%d compute=%v ok=%v; want 1, 784, false, true", device, mb, isCompute, ok)
	}
}

// CUDA's specific line keeps priority over the generic one it is printed with.
func TestCUDAOOMLineKeepsPriority(t *testing.T) {
	log := "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2048.00 MiB on device 1: cudaMalloc failed: out of memory\n" +
		"ggml_gallocr_reserve_n_impl: failed to allocate CUDA1 buffer of size 2147483648\n"
	device, mb, isCompute, ok := startupLogCUDAOOMDetailed(log)
	if !ok || device != 1 || mb != 2048 || !isCompute {
		t.Fatalf("got device=%d MiB=%d compute=%v ok=%v", device, mb, isCompute, ok)
	}
}
