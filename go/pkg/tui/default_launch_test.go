package tui

import (
	"github.com/raketenkater/ggrun/pkg/detect"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// TUIDefaultLaunchArgs is the argv a fresh TUI emits for a model with every
// setting left at its default. cmd/ggrun proves it resolves to exactly the same
// launch as plain `ggrun <model>`; keep the two tests in step.
var TUIDefaultLaunchArgsTail = []string{"--port", "8081", "--ctx-size", "fit", "--kv-placement", "auto", "--support-expert", "auto"}

func TestFreshTUIDefaultLaunchArgs(t *testing.T) {
	modelPath := isolatedModelDir(t)
	m := InitialModel()
	m.models = []ModelItem{{Name: "tiny", Path: modelPath}}
	m.selectedModel = 0
	req := m.buildLaunchRequest()
	if req == nil {
		t.Fatal("no launch request")
	}
	want := append([]string{modelPath}, TUIDefaultLaunchArgsTail...)
	if got := req.LaunchArgs(); !reflect.DeepEqual(got, want) {
		t.Fatalf("fresh TUI default argv changed:\n got  %q\n want %q\nupdate the CLI equivalence test in cmd/ggrun as well", got, want)
	}
}

// A TUI opened with a hardware restriction carries it into every launch and
// into its recommendations; without one the default argv is unchanged.
func TestHardwareRestrictionReachesLaunchAndRecommendations(t *testing.T) {
	t.Cleanup(func() { SetHardwareRestriction("", false) })
	modelPath := isolatedModelDir(t)
	m := InitialModel()
	m.models = []ModelItem{{Name: "tiny", Path: modelPath}}
	m.selectedModel = 0

	SetHardwareRestriction("2", false)
	if got := m.buildLaunchRequest().LaunchArgs(); !reflect.DeepEqual(got[:3], []string{modelPath, "--gpus", "2"}) {
		t.Fatalf("GPU restriction missing from TUI launch: %q", got)
	}
	caps := &detect.Capabilities{GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 12282}, {Index: 1, VRAMTotalMB: 24564}, {Index: 2, VRAMTotalMB: 12288}}}
	if got := restrictedCapabilities(caps); len(got.GPUs) != 1 || got.GPUs[0].Index != 2 || len(caps.GPUs) != 3 {
		t.Fatalf("recommendations not restricted to GPU 2: %+v", got.GPUs)
	}

	SetHardwareRestriction("", true)
	if got := m.buildLaunchRequest().LaunchArgs(); !reflect.DeepEqual(got[:2], []string{modelPath, "--cpu"}) {
		t.Fatalf("CPU-only missing from TUI launch: %q", got)
	}
	if got := restrictedCapabilities(caps); len(got.GPUs) != 0 {
		t.Fatalf("CPU-only recommendations still see GPUs: %+v", got.GPUs)
	}

	SetHardwareRestriction("", false)
	want := append([]string{modelPath}, TUIDefaultLaunchArgsTail...)
	if got := m.buildLaunchRequest().LaunchArgs(); !reflect.DeepEqual(got, want) {
		t.Fatalf("unrestricted TUI argv changed: %q", got)
	}
}

// The Recommended header names the inventory its recommendations were planned
// against, not every card on the machine.
func TestRecommendedHeaderShowsTheRestrictedInventory(t *testing.T) {
	t.Cleanup(func() { SetHardwareRestriction("", false) })
	t.Setenv("LLM_CACHE_DIR", t.TempDir())
	SetHardwareRestriction("2", false)
	m := Model{caps: &detect.Capabilities{CPU: detect.CPUInfo{Cores: 8}, RAM: detect.RAMInfo{TotalMB: 212 * 1024, FreeMB: 200 * 1024},
		GPUs: []detect.GPU{{Index: 0, Name: "NVIDIA GeForce RTX 4070", VRAMTotalMB: 12282}, {Index: 2, Name: "NVIDIA GeForce RTX 3060", VRAMTotalMB: 12288}}},
		ramBudgetMB: 28 * 1024, ramLimitPercent: 90, width: 120}
	m.refreshRecommendations()
	view := m.viewRecommended()
	if !strings.Contains(view, "RTX 3060 12G · 28GB RAM") || strings.Contains(view, "RTX 4070") {
		t.Fatalf("recommended header does not show the restricted inventory:\n%s", strings.SplitN(view, "\n", 3)[1])
	}
}

// A model linked into the model directory is listed with its target's size,
// not the few bytes of the link.
func TestSymlinkedModelListsTheTargetSize(t *testing.T) {
	target := isolatedModelDir(t)
	if err := os.Truncate(target, 3<<30); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(t.TempDir(), "linked.gguf")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	info, err := os.Lstat(link)
	if err != nil {
		t.Fatal(err)
	}
	item, _, ok := modelItemFromPath(link, info, false)
	if !ok || item.SizeGB < 2.9 {
		t.Fatalf("symlinked model listed at %.1f GB, want its 3 GB target", item.SizeGB)
	}
}
