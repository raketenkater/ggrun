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
