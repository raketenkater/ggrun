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

func TestNVIDIAPCIeLinkUsesObservedWidthWithMaxGen(t *testing.T) {
	links := parseNVIDIAPCIeLinks("1, 1, 3, 16\n1, 4, 3, 16\n")
	if len(links) != 2 {
		t.Fatalf("expected two parsed links, got %d", len(links))
	}
	gpu := GPU{}
	applyNVIDIAPCIeLink(&gpu, links[0])
	if gpu.PCIGen != 3 || gpu.PCILanes != 1 || gpu.BandwidthMBps != pcieBandwidth(3, 1) || gpu.BandwidthSource != "observed_width" {
		t.Fatalf("unexpected observed-width link: %+v", gpu)
	}
}

func TestNVIDIAPCIeLinkKeepsMaxWhenWidthMatches(t *testing.T) {
	links := parseNVIDIAPCIeLinks("1, 16, 3, 16\n")
	if len(links) != 1 {
		t.Fatalf("expected one parsed link, got %d", len(links))
	}
	gpu := GPU{}
	applyNVIDIAPCIeLink(&gpu, links[0])
	if gpu.PCIGen != 3 || gpu.PCILanes != 16 || gpu.BandwidthMBps != pcieBandwidth(3, 16) || gpu.BandwidthSource != "max" {
		t.Fatalf("unexpected max link: %+v", gpu)
	}
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

func TestAppleSiliconGPUSizing(t *testing.T) {
	gpu, ok := appleSiliconGPU(32*1024*1024*1024, "Apple M2 Pro")
	if !ok {
		t.Fatal("expected a GPU for 32GB unified memory")
	}
	// Metal's default working-set limit is ~75% of unified memory.
	if gpu.VRAMTotalMB != 24576 {
		t.Fatalf("expected 24576 MB (75%% of 32GB), got %d", gpu.VRAMTotalMB)
	}
	if gpu.Index != 0 || gpu.Name != "Apple M2 Pro" {
		t.Fatalf("unexpected GPU entry: %+v", gpu)
	}
	if _, ok := appleSiliconGPU(0, "x"); ok {
		t.Fatal("zero memsize must not produce a GPU")
	}
}

func TestApplyVRAMHeadroom(t *testing.T) {
	caps := &Capabilities{GPUs: []GPU{
		{Index: 0, VRAMTotalMB: 24000},
		{Index: 1, VRAMTotalMB: 12000},
		{Index: 2, VRAMTotalMB: 12000},
	}}
	// Reserve 4800 MB total across 48000 MB => 10% off each GPU.
	out := ApplyVRAMHeadroom(caps, 4800)
	if got := out.TotalVRAM(); got != 48000-4800 {
		t.Fatalf("expected total %d, got %d", 48000-4800, got)
	}
	if out.GPUs[0].VRAMTotalMB != 21600 || out.GPUs[1].VRAMTotalMB != 10800 {
		t.Fatalf("expected proportional split, got %d / %d", out.GPUs[0].VRAMTotalMB, out.GPUs[1].VRAMTotalMB)
	}
	// Original caps must be untouched (returns a copy).
	if caps.GPUs[0].VRAMTotalMB != 24000 {
		t.Fatalf("ApplyVRAMHeadroom mutated the input caps")
	}
	// Zero/negative headroom is a no-op returning the same pointer.
	if ApplyVRAMHeadroom(caps, 0) != caps {
		t.Fatalf("zero headroom should be a no-op")
	}
}

func TestParseBudgetMBViaHeadroomCases(t *testing.T) {
	// Sanity for the shared budget parser used by --vram-headroom and config.
	caps := &Capabilities{GPUs: []GPU{{VRAMTotalMB: 10000}}}
	if ApplyVRAMHeadroom(caps, 100000).GPUs[0].VRAMTotalMB != 0 {
		t.Fatalf("headroom larger than VRAM should floor at 0")
	}
}

func TestApplyRAMHeadroom(t *testing.T) {
	caps := &Capabilities{RAM: RAMInfo{TotalMB: 128000, FreeMB: 100000}}
	out := ApplyRAMHeadroom(caps, 8000)
	if out.RAM.TotalMB != 120000 || out.RAM.FreeMB != 92000 {
		t.Fatalf("expected 120000/92000, got %d/%d", out.RAM.TotalMB, out.RAM.FreeMB)
	}
	if caps.RAM.TotalMB != 128000 {
		t.Fatalf("ApplyRAMHeadroom mutated the input caps")
	}
	if ApplyRAMHeadroom(caps, 0) != caps {
		t.Fatalf("zero headroom should be a no-op")
	}
	if ApplyRAMHeadroom(caps, 999999).RAM.FreeMB != 0 {
		t.Fatalf("headroom larger than RAM should floor at 0")
	}
}
