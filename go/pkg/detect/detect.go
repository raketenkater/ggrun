package detect

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"sort"
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
	Index         int    `json:"index"`
	Name          string `json:"name"`
	VRAMTotalMB   int    `json:"vram_total_mb"`
	VRAMUsedMB    int    `json:"vram_used_mb,omitempty"`
	Driver        string `json:"driver,omitempty"`
	PCIGen        int    `json:"pci_gen,omitempty"`
	PCILanes      int    `json:"pci_lanes,omitempty"`
	BandwidthMBps int    `json:"bandwidth_mbps,omitempty"`
	PCIBusID      string `json:"pci_bus_id,omitempty"`
	ComputeCap    string `json:"compute_cap,omitempty"`
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
		"--query-gpu=index,pci.bus_id,name,memory.total,memory.used,driver_version,compute_cap",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}
	// Query PCIe bandwidth separately
	pcieOut, _ := exec.Command("nvidia-smi",
		"--query-gpu=pcie.link.gen.gpucurrent,pcie.link.width.current",
		"--format=csv,noheader,nounits").Output()
	pcieLines := strings.Split(strings.TrimSpace(string(pcieOut)), "\n")

	var gpus []GPU
	for i, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		parts := strings.Split(line, ", ")
		if len(parts) < 6 {
			continue
		}
		idx, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
		pciBusID := strings.TrimSpace(parts[1])
		vramTotal, _ := strconv.Atoi(strings.TrimSpace(parts[3]))
		vramUsed, _ := strconv.Atoi(strings.TrimSpace(parts[4]))
		driver := ""
		if len(parts) >= 6 {
			driver = strings.TrimSpace(parts[5])
		}
		computeCap := ""
		if len(parts) >= 7 {
			computeCap = strings.TrimSpace(parts[6])
		}
		gpu := GPU{
			Index:       idx,
			Name:        strings.TrimSpace(parts[2]),
			VRAMTotalMB: vramTotal,
			VRAMUsedMB:  vramUsed,
			Driver:      driver,
			PCIBusID:    pciBusID,
			ComputeCap:  computeCap,
		}
		// Parse PCIe bandwidth
		if i < len(pcieLines) {
			pcieParts := strings.Split(pcieLines[i], ",")
			if len(pcieParts) >= 2 {
				gen, _ := strconv.Atoi(strings.TrimSpace(pcieParts[0]))
				lanes, _ := strconv.Atoi(strings.TrimSpace(pcieParts[1]))
				gpu.PCIGen = gen
				gpu.PCILanes = lanes
				gpu.BandwidthMBps = pcieBandwidth(gen, lanes)
			// Fallback: if nvidia-smi returned 0 (GPU under load), try sysfs
			if gpu.BandwidthMBps <= 0 && gpu.PCIBusID != "" {
				gpu.BandwidthMBps = pcieBandwidthFromSysfs(gpu.PCIBusID)
			}
			}
		}
		gpus = append(gpus, gpu)
	}

	// Sort GPUs by PCI bus ID ascending to match CUDA_DEVICE_ORDER=PCI_BUS_ID.
	// The Go server sets this env var when launching llama-server, so CUDA
	// assigns device 0 to the lowest PCI bus ID. Re-index 0..N-1.
	sort.Slice(gpus, func(i, j int) bool {
		return gpus[i].PCIBusID < gpus[j].PCIBusID
	})
	for i := range gpus {
		gpus[i].Index = i
	}

	return gpus
}

// parseComputeCap parses "8.9" → 809, "8.6" → 806 for comparison.
func parseComputeCap(cc string) int {
	parts := strings.SplitN(cc, ".", 2)
	if len(parts) != 2 {
		return 0
	}
	major, _ := strconv.Atoi(parts[0])
	minor, _ := strconv.Atoi(parts[1])
	return major*100 + minor
}

// pcieBandwidth computes PCIe bandwidth in MB/s from generation and lane count.
func pcieBandwidth(gen, lanes int) int {
	// Per-lane bandwidth in MB/s (unidirectional)
	perLane := map[int]int{
		1: 250,
		2: 500,
		3: 985,  // ~984.6 MB/s
		4: 1969, // ~1969.0 MB/s
		5: 3938, // ~3938.0 MB/s
	}
	bw, ok := perLane[gen]
	if !ok {
		bw = 985 // default to gen3
	}
	return bw * lanes
}

// pcieBandwidthFromSysfs tries to read max PCIe link from sysfs.
// Used as fallback when nvidia-smi returns 0 (GPU under load).
func pcieBandwidthFromSysfs(busID string) int {
	if busID == "" {
		return 0
	}
	// busID is like "00000000:01:00.0"
	// sysfs path: /sys/bus/pci/devices/0000:01:00.0/
	dev := strings.TrimPrefix(busID, "0000")
	if dev == busID {
		dev = busID
	}
	sysPath := "/sys/bus/pci/devices/0000" + dev

	// Read max link speed (1=2.5GT/s, 2=5GT/s, 3=8GT/s, 4=16GT/s)
	speedBytes, err := os.ReadFile(sysPath + "/max_link_speed")
	if err != nil {
		return 0
	}
	speedStr := strings.TrimSpace(string(speedBytes))
	// Format: "8.0 GT/s PCIe" or just "8"
	speedStr = strings.TrimSuffix(speedStr, " GT/s")
	speedStr = strings.TrimSuffix(speedStr, " GT/s PCIe")
	speedStr = strings.TrimSpace(speedStr)
	speed, _ := strconv.ParseFloat(speedStr, 64)
	gen := int(speed / 2.5) // 2.5GT/s = Gen1, 5=Gen2, 8=Gen3, 16=Gen4

	// Read max link width
	widthBytes, err := os.ReadFile(sysPath + "/max_link_width")
	if err != nil {
		return 0
	}
	widthStr := strings.TrimSpace(string(widthBytes))
	lanes, _ := strconv.Atoi(widthStr)

	return pcieBandwidth(gen, lanes)
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
	freeMB := detectRAMFreeMB()
	totalMB := freeMB
	// Try to get total from /proc/meminfo on Linux
	if runtime.GOOS == "linux" {
		data, _ := os.ReadFile("/proc/meminfo")
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "MemTotal:") {
				var kb int
				fmt.Sscanf(line, "MemTotal: %d kB", &kb)
				totalMB = kb / 1024
				break
			}
		}
	}
	return RAMInfo{TotalMB: totalMB, FreeMB: freeMB}
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
	threads := runtime.NumCPU()
	cores := detectPhysicalCores()
	model := "unknown"
	flags := ""

	if runtime.GOOS == "linux" {
		data, _ := os.ReadFile("/proc/cpuinfo")
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "model name") {
				if parts := strings.SplitN(line, ":", 2); len(parts) == 2 {
					model = strings.TrimSpace(parts[1])
				}
			}
			if strings.HasPrefix(line, "flags") {
				if parts := strings.SplitN(line, ":", 2); len(parts) == 2 {
					flags = strings.TrimSpace(parts[1])
				}
			}
		}
	} else if runtime.GOOS == "darwin" {
		out, _ := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
		model = strings.TrimSpace(string(out))
	}

	return CPUInfo{
		Model:   model,
		Cores:   cores,
		Threads: threads,
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
