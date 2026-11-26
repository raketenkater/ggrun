//go:build !windows

package detect

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestParseLinuxCPUList(t *testing.T) {
	got := parseLinuxCPUList("0-2,8,10-11")
	for _, id := range []int{0, 1, 2, 8, 10, 11} {
		if !got[id] {
			t.Fatalf("allowed CPU %d missing from %#v", id, got)
		}
	}
	for _, id := range []int{3, 9, 12} {
		if got[id] {
			t.Fatalf("CPU %d incorrectly allowed in %#v", id, got)
		}
	}
}

func TestParseLinuxCPUListRejectsMalformedRanges(t *testing.T) {
	got := parseLinuxCPUList("4-2,-1,nope,7")
	if len(got) != 1 || !got[7] {
		t.Fatalf("malformed list was not rejected conservatively: %#v", got)
	}
}

func withCPUFixtures(t *testing.T, root string) {
	t.Helper()
	a, b, c, d := sysCPUDir, procSelfCgroup, cgroupFSRoot, procSelfStatus
	sysCPUDir = filepath.Join(root, "sys/devices/system/cpu")
	procSelfCgroup = filepath.Join(root, "proc/self/cgroup")
	procSelfStatus = filepath.Join(root, "proc/self/status")
	cgroupFSRoot = filepath.Join(root, "sys/fs/cgroup")
	t.Cleanup(func() { sysCPUDir, procSelfCgroup, cgroupFSRoot, procSelfStatus = a, b, c, d })
}

func writeFixture(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// C-11: a container quota caps the worker count below the visible cores.
func TestCPUQuotaFromCgroupV2(t *testing.T) {
	root := t.TempDir()
	withCPUFixtures(t, root)
	writeFixture(t, procSelfCgroup, "0::/kubepods/pod1/ctr\n")
	writeFixture(t, filepath.Join(cgroupFSRoot, "kubepods/pod1/ctr/cpu.max"), "max 100000\n")
	writeFixture(t, filepath.Join(cgroupFSRoot, "kubepods/pod1/cpu.max"), "350000 100000\n")
	writeFixture(t, filepath.Join(cgroupFSRoot, "kubepods/cpu.max"), "800000 100000\n")
	if got := cpuQuotaCores(); got != 4 {
		t.Fatalf("quota cores %d, want 4 (3.5 rounded up, smallest on the path)", got)
	}
	writeFixture(t, filepath.Join(cgroupFSRoot, "kubepods/pod1/cpu.max"), "max 100000\n")
	writeFixture(t, filepath.Join(cgroupFSRoot, "kubepods/cpu.max"), "max 100000\n")
	if got := cpuQuotaCores(); got != 0 {
		t.Fatalf("unlimited reads as %d", got)
	}
}

// A hybrid CPU: SMT performance cores and single-thread efficiency cores.
// Each physical core counts once, whatever its thread count, within the
// process affinity mask.
func TestHybridTopologyCountsEachCoreOnceWithinAffinity(t *testing.T) {
	root := t.TempDir()
	withCPUFixtures(t, root)
	// cpu0..7: 4 P-cores x 2 threads; cpu8..11: 4 E-cores.
	for id := 0; id < 12; id++ {
		core := id / 2
		if id >= 8 {
			core = id - 4
		}
		base := filepath.Join(sysCPUDir, "cpu"+strconv.Itoa(id), "topology")
		writeFixture(t, filepath.Join(base, "physical_package_id"), "0\n")
		writeFixture(t, filepath.Join(base, "core_id"), strconv.Itoa(core)+"\n")
	}
	writeFixture(t, procSelfStatus, "Name:\tggrun\nCpus_allowed_list:\t0-11\n")
	if ids := detectPhysicalCPUList(); len(ids) != 8 {
		t.Fatalf("hybrid cores %v, want 8", ids)
	}
	writeFixture(t, procSelfStatus, "Cpus_allowed_list:\t0-3,8-9\n")
	if ids := detectPhysicalCPUList(); len(ids) != 4 {
		t.Fatalf("cores within affinity %v, want 4 (2 P + 2 E)", ids)
	}
}

// ARM cores have no SMT and /proc/cpuinfo has no "physical id": all cores.
func TestFallbackCoresOnARMUsesEveryCore(t *testing.T) {
	if got := fallbackPhysicalCores("arm64", 8); got != 8 {
		t.Fatalf("arm64 fallback %d, want 8", got)
	}
	if got := fallbackPhysicalCores("amd64", 8); got != 4 {
		t.Fatalf("amd64 fallback %d, want 4", got)
	}
	if got := fallbackPhysicalCores("amd64", 2); got != 2 {
		t.Fatalf("small amd64 fallback %d", got)
	}
}
