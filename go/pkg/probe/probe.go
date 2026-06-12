package probe

import (
	"fmt"
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/raketenkater/llm-server/pkg/detect"
)

// Memory holds GPU and system memory state.
type Memory struct {
	GPUFreeMB []int `json:"gpu_free_mb"`
	RAMFreeMB int   `json:"ram_free_mb"`
}

// Probe checks current free memory on GPUs and system.
func Probe() (*Memory, error) {
	gpuFree := probeGPUFree()
	ramFree := probeRAMFree()
	return &Memory{
		GPUFreeMB: gpuFree,
		RAMFreeMB: ramFree,
	}, nil
}

func probeGPUFree() []int {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,memory.free",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}
	var free []int
	re := regexp.MustCompile(`(\d+),\s*(\d+)`)
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		m := re.FindStringSubmatch(line)
		if len(m) >= 3 {
			mb, _ := strconv.Atoi(strings.TrimSpace(m[2]))
			free = append(free, mb)
		}
	}
	return free
}

func probeRAMFree() int {
	if caps, err := detect.Detect(); err == nil && caps.RAM.FreeMB > 0 {
		return caps.RAM.FreeMB
	}
	if runtime.GOOS != "linux" {
		return 0
	}
	data, err := exec.Command("awk", "/MemAvailable:/ {print int($2/1024)}", "/proc/meminfo").Output()
	if err != nil {
		return 0
	}
	mb, _ := strconv.Atoi(strings.TrimSpace(string(data)))
	return mb
}

// TotalFree returns the sum of free GPU memory.
func (m *Memory) TotalFree() int {
	total := 0
	for _, v := range m.GPUFreeMB {
		total += v
	}
	return total
}

// String returns a human-readable summary.
func (m *Memory) String() string {
	parts := []string{fmt.Sprintf("RAM: %d MB free", m.RAMFreeMB)}
	for i, v := range m.GPUFreeMB {
		parts = append(parts, fmt.Sprintf("GPU%d: %d MB free", i, v))
	}
	return strings.Join(parts, ", ")
}
