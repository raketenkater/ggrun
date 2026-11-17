package recovery

import (
	"testing"
)

func TestParseCUDAOOMDetails(t *testing.T) {
	device, allocMB, ok := ParseCUDAOOM("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 11875.43 MiB on device 1: cudaMalloc failed: out of memory")
	if !ok || device != 1 || allocMB != 11876 {
		t.Fatalf("unexpected CUDA OOM parse: device=%d alloc=%d ok=%v", device, allocMB, ok)
	}
}

func TestParseCUDADeviceFromVMMOOM(t *testing.T) {
	device, ok := ParseCUDADevice("120.54 E   current device: 0, in function alloc at ggml-cuda.cu:529")
	if !ok || device != 0 {
		t.Fatalf("unexpected CUDA device parse: device=%d ok=%v", device, ok)
	}
	if _, ok := ParseCUDADevice("CUDA error: out of memory"); ok {
		t.Fatal("OOM marker without a current-device diagnostic must not invent a device")
	}
}
