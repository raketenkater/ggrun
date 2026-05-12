package detect

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// Capabilities represents the full hardware and environment profile.
type Capabilities struct {
	OS       string    `json:"os"`
	Arch     string    `json:"arch"`
	GPUs     []GPU     `json:"gpus"`
	RAM      RAMInfo   `json:"ram"`
	CPU      CPUInfo   `json:"cpu"`
	Backends []Backend `json:"backends"`
}

// GPU represents a single GPU device.
type GPU struct {
	Index      int    `json:"index"`
	Name       string `json:"name"`
	VRAMTotalMB int   `json:"vram_total_mb"`
	VRAMUsedMB  int   `json:"vram_used_mb,omitempty"`
	Driver     string `json:"driver,omitempty"`
}

// RAMInfo represents system memory.
type RAMInfo struct {
	TotalMB int `json:"total_mb"`
	FreeMB  int `json:"free_mb"`
}

// CPUInfo represents CPU details.
type CPUInfo struct {
	Model      string `json:"model"`
	Cores      int    `json:"cores"`
	Threads    int    `json:"threads"`
	Flags      string `json:"flags,omitempty"`
}

// Backend represents a discovered inference backend binary.
type Backend struct {
	Name    string `json:"name"`
	Path    string `json:"path"`
	Version string `json:"version,omitempty"`
}

// Detect probes the system and returns capabilities.
func Detect() (*Capabilities, error) {
	gpus := detectNVIDIA()
	if len(gpus) == 0 {
		gpus = detectROCm()
	}

	ram := detectRAM()
	cpu := detectCPU()
	backends := detectBackends()

	return &Capabilities{
		OS:       runtime.GOOS,
		Arch:     runtime.GOARCH,
		GPUs:     gpus,
		RAM:      ram,
		CPU:      cpu,
		Backends: backends,
	}, nil
}

func detectNVIDIA() []GPU {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,name,memory.total,memory.used,driver_version",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}
	var gpus []GPU
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		parts := strings.Split(line, ", ")
		if len(parts) < 4 {
			continue
		}
		idx, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
		vramTotal, _ := strconv.Atoi(strings.TrimSpace(parts[2]))
		vramUsed, _ := strconv.Atoi(strings.TrimSpace(parts[3]))
		driver := ""
		if len(parts) >= 5 {
			driver = strings.TrimSpace(parts[4])
		}
		gpus = append(gpus, GPU{
			Index:       idx,
			Name:        strings.TrimSpace(parts[1]),
			VRAMTotalMB: vramTotal,
			VRAMUsedMB:  vramUsed,
			Driver:      driver,
		})
	}
	return gpus
}

func detectROCm() []GPU {
	if _, err := exec.LookPath("rocm-smi"); err != nil {
		return nil
	}
	out, err := exec.Command("rocm-smi", "--showproductname", "--showmeminfo", "vram").Output()
	if err != nil {
		return nil
	}
	// rocm-smi output parsing is less structured; minimal support
	var gpus []GPU
	for i, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if strings.Contains(line, "GPU") && strings.Contains(line, ":") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				gpus = append(gpus, GPU{
					Index:       i,
					Name:        strings.TrimSpace(parts[1]),
					VRAMTotalMB: 0, // would need more parsing
				})
			}
		}
	}
	return gpus
}

func detectRAM() RAMInfo {
	switch runtime.GOOS {
	case "linux":
		return detectRAMLinux()
	case "darwin":
		return detectRAMDarwin()
	default:
		return RAMInfo{}
	}
}

func detectRAMLinux() RAMInfo {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return RAMInfo{}
	}
	var totalKB, freeKB int
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "MemTotal:") {
			fmt.Sscanf(line, "MemTotal: %d kB", &totalKB)
		} else if strings.HasPrefix(line, "MemAvailable:") {
			fmt.Sscanf(line, "MemAvailable: %d kB", &freeKB)
		}
	}
	return RAMInfo{
		TotalMB: totalKB / 1024,
		FreeMB:  freeKB / 1024,
	}
}

func detectRAMDarwin() RAMInfo {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return RAMInfo{}
	}
	bytes, _ := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	return RAMInfo{TotalMB: int(bytes / 1024 / 1024)}
}

func detectCPU() CPUInfo {
	cores := runtime.NumCPU()
	model := "unknown"
	flags := ""

	if runtime.GOOS == "linux" {
		data, _ := os.ReadFile("/proc/cpuinfo")
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "model name") {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					model = strings.TrimSpace(parts[1])
					break
				}
			}
		}
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "flags") {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					flags = strings.TrimSpace(parts[1])
				}
				break
			}
		}
	} else if runtime.GOOS == "darwin" {
		out, _ := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
		model = strings.TrimSpace(string(out))
	}

	return CPUInfo{
		Model:   model,
		Cores:   cores,
		Threads: cores,
		Flags:   flags,
	}
}

func detectBackends() []Backend {
	var backends []Backend
	for _, name := range []string{"llama-server", "ik_llama", "ik_llama-server"} {
		path, err := exec.LookPath(name)
		if err != nil {
			continue
		}
		backends = append(backends, Backend{Name: name, Path: path})
	}
	return backends
}

// VRAMFreeMB returns free VRAM for this GPU.
func (g GPU) VRAMFreeMB() int {
	free := g.VRAMTotalMB - g.VRAMUsedMB
	if free < 0 {
		return 0
	}
	return free
}

// TotalVRAM returns the sum of total VRAM across all detected GPUs.
func (c *Capabilities) TotalVRAM() int {
	total := 0
	for _, g := range c.GPUs {
		total += g.VRAMTotalMB
	}
	return total
}

// TotalVRAMFree returns the sum of free VRAM across all detected GPUs.
func (c *Capabilities) TotalVRAMFree() int {
	total := 0
	for _, g := range c.GPUs {
		total += g.VRAMFreeMB()
	}
	return total
}

// JSON returns a pretty-printed JSON representation.
func (c *Capabilities) JSON() ([]byte, error) {
	return json.MarshalIndent(c, "", "  ")
}
