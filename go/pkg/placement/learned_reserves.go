package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// LearnedReserve is one stored overhead the planner charges on every plan
// for this GPU set until the probe cache is reset (review P-14). It exists to
// be shown: the planner's behaviour does not depend on this type.
type LearnedReserve struct {
	Device string // "CUDA0".., or "host"
	MB     int
	Date   string // when it was measured, "" when the record does not say
	Source string // what measured it
	Path   string // the record
}

// LearnedReserves lists the stored overheads loadSystemProbe would apply to
// gpus, in device order then host. Empty when none is stored.
func LearnedReserves(cacheDir string, gpus []detect.GPU) []LearnedReserve {
	sp := loadSystemProbe(cacheDir, gpus)
	if sp == nil {
		return nil
	}
	path := systemProbePath(cacheDir, gpus)
	date, source := systemProbeProvenance(path)
	var out []LearnedReserve
	idx := make([]int, 0, len(sp.CUDAOverheadByGPU))
	for i, mb := range sp.CUDAOverheadByGPU {
		if mb > 0 {
			idx = append(idx, i)
		}
	}
	sort.Ints(idx)
	for _, i := range idx {
		out = append(out, LearnedReserve{Device: "CUDA" + strconv.Itoa(i), MB: sp.CUDAOverheadByGPU[i], Date: date, Source: source, Path: path})
	}
	if len(idx) == 0 && sp.CUDAOverheadMB > 0 {
		out = append(out, LearnedReserve{Device: "each GPU", MB: sp.CUDAOverheadMB, Date: date, Source: source, Path: path})
	}
	if sp.HostOverheadMB > 0 {
		out = append(out, LearnedReserve{Device: "host", MB: sp.HostOverheadMB, Date: date, Source: source, Path: path})
	}
	return out
}

func systemProbePath(cacheDir string, gpus []detect.GPU) string {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	return filepath.Join(cacheDir, fmt.Sprintf("system_%s.cache", gpuIdentityHash(gpus)))
}

// systemProbeProvenance reads the record's "# Generated:" date and its first
// comment line, which names what wrote it.
func systemProbeProvenance(path string) (date, source string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", "system probe (migrated record)"
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "#") {
			continue
		}
		text := strings.TrimSpace(strings.TrimPrefix(line, "#"))
		if d, ok := strings.CutPrefix(text, "Generated:"); ok {
			date = strings.TrimSpace(d)
		} else if source == "" {
			source = text
		}
	}
	if source == "" {
		source = "system probe"
	}
	return date, source
}

// LearnedReserveLines formats reserves for the plan summary and dry run.
func LearnedReserveLines(label string, reserves []LearnedReserve) []string {
	lines := make([]string, 0, len(reserves))
	for _, r := range reserves {
		date := r.Date
		if date == "" {
			date = "date unknown"
		}
		lines = append(lines, fmt.Sprintf("[%s] learned reserve %s %d MiB (measured %s; %s; %s; `ggrun probe reset` clears it)",
			label, r.Device, r.MB, date, r.Source, r.Path))
	}
	return lines
}
