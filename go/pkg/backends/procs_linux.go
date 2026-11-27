package backends

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func platformProcessLister() func() ([]ProcInfo, error) { return listProcProcesses }

// listProcProcesses reads /proc/<pid>/exe and /proc/<pid>/cwd. A process that
// exits mid-scan, or belongs to a user whose links cannot be read, is skipped.
func listProcProcesses() ([]ProcInfo, error) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil, err
	}
	var out []ProcInfo
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		dir := filepath.Join("/proc", e.Name())
		exe, _ := os.Readlink(filepath.Join(dir, "exe"))
		cwd, _ := os.Readlink(filepath.Join(dir, "cwd"))
		// A binary replaced on disk reads back with this suffix.
		exe = strings.TrimSuffix(exe, " (deleted)")
		if exe == "" && cwd == "" {
			continue
		}
		out = append(out, ProcInfo{PID: pid, Exe: exe, Cwd: cwd})
	}
	return out, nil
}
