package detect

import (
	"testing"
)

func TestDetect(t *testing.T) {
	caps, err := Detect()
	if err != nil {
		t.Fatalf("detect failed: %v", err)
	}
	if caps.OS == "" {
		t.Fatalf("OS should not be empty")
	}
	if caps.Arch == "" {
		t.Fatalf("Arch should not be empty")
	}
	if caps.CPU.Cores == 0 {
		t.Fatalf("CPU cores should not be zero")
	}
}

func TestTotalVRAM(t *testing.T) {
	caps := &Capabilities{
		GPUs: []GPU{
			{Index: 0, VRAMTotalMB: 12288},
			{Index: 1, VRAMTotalMB: 12288},
		},
	}
	if got := caps.TotalVRAM(); got != 24576 {
		t.Fatalf("expected 24576 MB, got %d", got)
	}
}

func TestJSON(t *testing.T) {
	caps := &Capabilities{
		OS:   "linux",
		Arch: "amd64",
		GPUs: []GPU{{Index: 0, Name: "RTX 4070", VRAMTotalMB: 12288}},
		RAM:  RAMInfo{TotalMB: 65536, FreeMB: 32768},
		CPU:  CPUInfo{Model: "AMD Ryzen", Cores: 16},
	}
	data, err := caps.JSON()
	if err != nil {
		t.Fatalf("json failed: %v", err)
	}
	if len(data) == 0 {
		t.Fatalf("empty json")
	}
}

func TestDetectBackends(t *testing.T) {
	b := detectBackends()
	// At least llama-server might exist on this system
	_ = b
}

func TestDetectCPU(t *testing.T) {
	cpu := detectCPU()
	if cpu.Cores == 0 {
		t.Fatalf("cpu cores should not be zero")
	}
}

func TestDetectRAMLinux(t *testing.T) {
	ram := detectRAMLinux()
	if ram.TotalMB == 0 {
		t.Fatalf("expected non-zero RAM total")
	}
}

func TestDetectNVIDIA(t *testing.T) {
	gpus := detectNVIDIA()
	// May or may not have nvidia-smi
	_ = gpus
}

func TestDetectROCm(t *testing.T) {
	gpus := detectROCm()
	// May or may not have rocm-smi
	_ = gpus
}

func TestParseVulkanGPUsKeepsMetadataWithDeviceBlock(t *testing.T) {
	summary := `GPU0:
	apiVersion         = 1.3.277
	driverVersion      = 550.54.14
	deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName         = NVIDIA GeForce RTX 4070
GPU1:
	apiVersion         = 1.3.274
	driverVersion      = 24.0.0
	deviceType         = PHYSICAL_DEVICE_TYPE_CPU
	deviceName         = llvmpipe (LLVM 17.0.6, 256 bits)
`

	gpus := parseVulkanGPUs(summary)
	if len(gpus) != 1 {
		t.Fatalf("expected one non-software GPU, got %d: %#v", len(gpus), gpus)
	}
	if gpus[0].Name != "NVIDIA GeForce RTX 4070" {
		t.Fatalf("unexpected GPU name: %q", gpus[0].Name)
	}
	if gpus[0].Driver != "550.54.14" {
		t.Fatalf("expected discrete GPU driver, got %q", gpus[0].Driver)
	}
	if gpus[0].ComputeCap != "1.3.277" {
		t.Fatalf("expected discrete GPU apiVersion, got %q", gpus[0].ComputeCap)
	}
	if gpus[0].VRAMTotalMB != 12288 {
		t.Fatalf("expected RTX 4070 VRAM estimate, got %d", gpus[0].VRAMTotalMB)
	}
}

func TestParseVulkanGPUsUsesConservativeIntegratedBudget(t *testing.T) {
	summary := `GPU0:
	apiVersion         = 1.3.250
	driverVersion      = Mesa 24.0.0
	deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
	deviceName         = Intel(R) Iris(R) Xe Graphics
`

	gpus := parseVulkanGPUs(summary)
	if len(gpus) != 1 {
		t.Fatalf("expected one integrated GPU, got %d: %#v", len(gpus), gpus)
	}
	if gpus[0].VRAMTotalMB != 2048 {
		t.Fatalf("expected conservative integrated budget, got %d", gpus[0].VRAMTotalMB)
	}
}

func TestEstimateVRAMFromNameUsesConservativeUnknownDefault(t *testing.T) {
	if got := estimateVRAMFromName("Unknown Vulkan Device"); got != 4096 {
		t.Fatalf("expected conservative unknown default, got %d", got)
	}
}
