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

func categoryGroups(c Categories) map[string][]Recommendation {
	return map[string][]Recommendation{"balanced": c.Balanced, "smartest": c.Smartest, "fastest": c.Fastest}
}

func resetBackendProbes() {
	SetInstalledArchSupport(nil)
	SetBackendRoute(nil)
	SetVulkanOnCUDA(nil)
}

// blockedUpstreamArch picks the architecture of the first Best overall row the
// catalog stamps as upstream-registered, so blocking it exercises a build route.
func blockedUpstreamArch(t *testing.T, c Categories) string {
	t.Helper()
	for _, r := range c.Balanced {
		if r.Arch != "" && !r.NeedsForkBackend() {
			return strings.ToLower(r.Arch)
		}
	}
	t.Skip("catalog rows carry no upstream architecture")
	return ""
}

// K2 Horizon: the release backend lacked k2-horizon while the catalog (stamped
// from upstream master) called it runnable, nothing offered a build, and a
// fresh install's top CPU pick failed to load. A row no supported route can
// serve is still listed, but after every usable row in every category.
func TestNoBackendRouteRanksAfterUsableRows(t *testing.T) {
	t.Cleanup(resetBackendProbes)
	resetBackendProbes()
	base := TopCategories(cpuSixteenCaps(), 6)
	blocked := blockedUpstreamArch(t, base)
	SetInstalledArchSupport(func(_ *detect.Capabilities, arch string) (bool, bool) {
		return strings.ToLower(arch) != blocked, true
	})
	SetBackendRoute(func(*detect.Capabilities, string, bool) bool { return false })
	got := TopCategories(cpuSixteenCaps(), 6)
	for name, group := range categoryGroups(got) {
		seenUnsupported := false
		for _, r := range group {
			want := BackendReady
			if strings.ToLower(r.Arch) == blocked {
				want = BackendUnsupported
				if !strings.Contains(r.Reason, "no supported route builds one") {
					t.Fatalf("%s: unsupported row does not say why: %q", name, r.Reason)
				}
			}
			if r.Arch != "" && r.BackendStatus != want {
				t.Fatalf("%s: %s BackendStatus=%q, want %q", name, r.Repo, r.BackendStatus, want)
			}
			if seenUnsupported && r.BackendStatus != BackendUnsupported {
				t.Fatalf("%s: usable %s ranked after a row no backend route serves", name, r.Repo)
			}
			seenUnsupported = seenUnsupported || r.BackendStatus == BackendUnsupported
		}
	}
	if got.Balanced[0].BackendStatus == BackendUnsupported {
		t.Fatalf("top pick has no backend route: %s", got.Balanced[0].Repo)
	}

	// Usable rows keep their relative order.
	var keptBase, keptGot []string
	for _, r := range base.Balanced {
		if strings.ToLower(r.Arch) != blocked {
			keptBase = append(keptBase, r.Repo)
		}
	}
	for _, r := range got.Balanced {
		if r.BackendStatus != BackendUnsupported && len(keptGot) < len(keptBase) {
			keptGot = append(keptGot, r.Repo)
		}
	}
	if !reflect.DeepEqual(keptBase[:len(keptGot)], keptGot) {
		t.Fatalf("usable order changed:\nbefore %v\nafter  %v", keptBase, keptGot)
	}
}

// Installation state is not capability: a row whose backend the first launch
// builds through a supported route keeps its rank in every category, and is
// marked so a caller that needs a model ready now can select one.
func TestBackendSetupRouteKeepsRankInEveryCategory(t *testing.T) {
	t.Cleanup(resetBackendProbes)
	resetBackendProbes()
	base := TopCategories(cpuSixteenCaps(), 6)
	blocked := blockedUpstreamArch(t, base)
	SetInstalledArchSupport(func(_ *detect.Capabilities, arch string) (bool, bool) {
		return strings.ToLower(arch) != blocked, true
	})
	got := TopCategories(cpuSixteenCaps(), 6)
	baseGroups := categoryGroups(base)
	for name, group := range categoryGroups(got) {
		if !reflect.DeepEqual(repos(baseGroups[name]), repos(group)) {
			t.Fatalf("%s: a backend build route changed the ranking:\nbefore %v\nafter  %v", name, repos(baseGroups[name]), repos(group))
		}
		for _, r := range group {
			if strings.ToLower(r.Arch) != blocked {
				continue
			}
			if r.BackendStatus != BackendNeedsSetup || !strings.Contains(r.Reason, "the first launch offers to build one") {
				t.Fatalf("%s: %s status %q reason %q", name, r.Repo, r.BackendStatus, r.Reason)
			}
		}
	}
}

// An unanswerable probe must not reorder or annotate anything.
func TestUnknownBackendSupportLeavesRankingAlone(t *testing.T) {
	t.Cleanup(resetBackendProbes)
	resetBackendProbes()
	base := TopCategories(cpuSixteenCaps(), 6)
	SetInstalledArchSupport(func(*detect.Capabilities, string) (bool, bool) { return false, false })
	got := TopCategories(cpuSixteenCaps(), 6)
	baseGroups := categoryGroups(base)
	for name, group := range categoryGroups(got) {
		if !reflect.DeepEqual(repos(baseGroups[name]), repos(group)) {
			t.Fatalf("%s: unknown support changed the ranking:\n%v\n%v", name, repos(baseGroups[name]), repos(group))
		}
		for _, r := range group {
			if r.BackendStatus != BackendUnknown {
				t.Fatalf("%s: %s marked %q without a probe answer", name, r.Repo, r.BackendStatus)
			}
		}
	}
}

func TestSeniorSmartestPreservesIntelligenceAcrossBackendSetup(t *testing.T) {
	t.Cleanup(resetBackendProbes)
	resetBackendProbes()
	yes := true
	strong := Candidate{Name: "strong", Repo: "fixture/strong", Arch: "qwen2", Runnable: &yes, AAIntelligence: 55, TotalParamsB: 4, SizeGB: 2.6, Quants: []QuantOption{{Name: "Q4_K_M", SizeGB: 2.6}}}
	weak := Candidate{Name: "weak", Repo: "fixture/weak", Arch: "llama", Runnable: &yes, AAIntelligence: 20, TotalParamsB: 4, SizeGB: 2.6, Quants: []QuantOption{{Name: "Q4_K_M", SizeGB: 2.6}}}
	intelligenceTestCatalog(t, strong, weak)
	SetInstalledArchSupport(func(_ *detect.Capabilities, arch string) (bool, bool) { return arch == "llama", true })
	rows := TopCategories(intelligenceTestHardware(), 1).Smartest
	if len(rows) != 1 || rows[0].Repo != strong.Repo {
		t.Fatalf("Smartest chose %s (intelligence %.1f) over buildable strong model (55): backend installation state overrode intelligence", rows[0].Repo, rows[0].AAIntelligence)
	}
}

// Per category: a smarter, faster model that needs backend setup outranks a
// ready one when a supported route builds its backend. When none does it stays
// listed with its status, after every usable row.
func TestBackendRouteDecidesRankPerCategory(t *testing.T) {
	t.Cleanup(resetBackendProbes)
	yes := true
	strong := Candidate{Name: "strong", Repo: "fixture/strong", Arch: "qwen2", Runnable: &yes, AAIntelligence: 55, TotalParamsB: 4, SizeGB: 2.6, Quants: []QuantOption{{Name: "Q4_K_M", SizeGB: 2.6}}}
	weak := Candidate{Name: "weak", Repo: "fixture/weak", Arch: "llama", Runnable: &yes, AAIntelligence: 30, TotalParamsB: 8, SizeGB: 5.2, Quants: []QuantOption{{Name: "Q4_K_M", SizeGB: 5.2}}}
	for _, c := range []struct {
		route bool
		first string
		want  BackendStatus
	}{
		{route: true, first: strong.Repo, want: BackendNeedsSetup},
		{route: false, first: weak.Repo, want: BackendUnsupported},
	} {
		resetBackendProbes()
		intelligenceTestCatalog(t, strong, weak)
		SetInstalledArchSupport(func(_ *detect.Capabilities, arch string) (bool, bool) { return arch == "llama", true })
		route := c.route
		SetBackendRoute(func(*detect.Capabilities, string, bool) bool { return route })
		for name, group := range categoryGroups(TopCategories(intelligenceTestHardware(), 50)) {
			if len(group) < 2 || group[0].Repo != c.first {
				t.Fatalf("route=%v %s: got %v, want %s first", c.route, name, repos(group), c.first)
			}
			last := group[len(group)-1]
			if !c.route && (last.Repo != strong.Repo || last.BackendStatus != c.want) {
				t.Fatalf("route=%v %s: unsupported row not listed last: %v", c.route, name, repos(group))
			}
			for _, r := range group {
				if r.Repo == strong.Repo && r.BackendStatus != c.want {
					t.Fatalf("route=%v %s: status %q, want %q", c.route, name, r.BackendStatus, c.want)
				}
				if r.Repo == weak.Repo && r.BackendStatus != BackendReady {
					t.Fatalf("route=%v %s: ready row marked %q", c.route, name, r.BackendStatus)
				}
			}
		}
	}
}

// Without the launcher's answer a route is assumed exactly when upstream
// registers the architecture; a fork-only one has none.
func TestCatalogOnlyRouteFollowsUpstreamStamp(t *testing.T) {
	t.Cleanup(resetBackendProbes)
	resetBackendProbes()
	SetInstalledArchSupport(func(*detect.Capabilities, string) (bool, bool) { return false, true })
	yes, no := true, false
	upstream := Recommendation{Candidate: Candidate{Name: "up", Arch: "k2-horizon", Runnable: &yes}}
	fork := Recommendation{Candidate: Candidate{Name: "fork", Arch: "inkling", Runnable: &no}}
	markBackendStatus(nil, &upstream)
	markBackendStatus(nil, &fork)
	if upstream.BackendStatus != BackendNeedsSetup || fork.BackendStatus != BackendUnsupported {
		t.Fatalf("upstream %q, fork-only %q", upstream.BackendStatus, fork.BackendStatus)
	}
}
