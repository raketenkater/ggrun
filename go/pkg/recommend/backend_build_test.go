package recommend

import (
	"reflect"
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
)

func cpuSixteenCaps() *detect.Capabilities {
	return &detect.Capabilities{OS: "linux", RAM: detect.RAMInfo{TotalMB: 16384, FreeMB: 15000}}
}

func repos(rows []Recommendation) []string {
	out := make([]string, len(rows))
	for i, r := range rows {
		out[i] = r.Repo
	}
	return out
}

// A model no installed backend loads is still listed, but never ahead of one
// the install can serve now: the release backend lacked k2-horizon while the
// catalog (stamped from upstream master) called it runnable, and a fresh
// install's top CPU pick failed to load.
func TestUnloadableArchitecturesRankAfterLoadableOnes(t *testing.T) {
	defer SetInstalledArchSupport(nil)
	SetInstalledArchSupport(nil)
	base := TopCategories(cpuSixteenCaps(), 6)

	blocked := ""
	for _, r := range base.Balanced {
		if r.Arch != "" {
			blocked = strings.ToLower(r.Arch)
			break
		}
	}
	if blocked == "" {
		t.Skip("catalog rows carry no architecture")
	}
	SetInstalledArchSupport(func(_ *detect.Capabilities, arch string) (bool, bool) {
		return strings.ToLower(arch) != blocked, true
	})
	got := TopCategories(cpuSixteenCaps(), 6)
	for name, group := range map[string][]Recommendation{"balanced": got.Balanced, "smartest": got.Smartest, "fastest": got.Fastest} {
		seenBuild := false
		for _, r := range group {
			want := strings.ToLower(r.Arch) == blocked
			if r.NeedsBackendBuild != want {
				t.Fatalf("%s: %s NeedsBackendBuild=%v, arch %q", name, r.Repo, r.NeedsBackendBuild, r.Arch)
			}
			if want && !strings.Contains(r.Reason, "no installed backend loads "+r.Arch) {
				t.Fatalf("%s: flagged row does not say why: %q", name, r.Reason)
			}
			if seenBuild && !r.NeedsBackendBuild {
				t.Fatalf("%s: loadable %s ranked after a row that needs a backend build", name, r.Repo)
			}
			seenBuild = seenBuild || r.NeedsBackendBuild
		}
	}
	if got.Balanced[0].NeedsBackendBuild {
		t.Fatalf("top pick needs a backend build: %s", got.Balanced[0].Repo)
	}

	// Loadable rows keep their relative order.
	var keptBase, keptGot []string
	for _, r := range base.Balanced {
		if strings.ToLower(r.Arch) != blocked {
			keptBase = append(keptBase, r.Repo)
		}
	}
	for _, r := range got.Balanced {
		if !r.NeedsBackendBuild && len(keptGot) < len(keptBase) {
			keptGot = append(keptGot, r.Repo)
		}
	}
	if !reflect.DeepEqual(keptBase[:len(keptGot)], keptGot) {
		t.Fatalf("loadable order changed:\nbefore %v\nafter  %v", keptBase, keptGot)
	}
}

// An unanswerable probe must not reorder or annotate anything.
func TestUnknownBackendSupportLeavesRankingAlone(t *testing.T) {
	defer SetInstalledArchSupport(nil)
	SetInstalledArchSupport(nil)
	base := TopCategories(cpuSixteenCaps(), 6)
	SetInstalledArchSupport(func(*detect.Capabilities, string) (bool, bool) { return false, false })
	got := TopCategories(cpuSixteenCaps(), 6)
	if !reflect.DeepEqual(repos(base.Balanced), repos(got.Balanced)) || !reflect.DeepEqual(repos(base.Fastest), repos(got.Fastest)) {
		t.Fatalf("unknown support changed the ranking:\n%v\n%v", repos(base.Balanced), repos(got.Balanced))
	}
	for _, r := range got.Balanced {
		if r.NeedsBackendBuild {
			t.Fatalf("%s flagged without a probe answer", r.Repo)
		}
	}
}
