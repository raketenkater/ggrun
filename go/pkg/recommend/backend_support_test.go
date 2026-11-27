package recommend

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/detect"
)

// With the shared resolver installed, a row's status and wording come from its
// state: the status ranking uses, BackendNote verbatim, and the note in Reason
// for every row that cannot load now.
func TestBackendSupportDecidesTheRow(t *testing.T) {
	t.Cleanup(func() { resetBackendProbes(); SetBackendSupport(nil) })
	SetInstalledArchSupport(func(*detect.Capabilities, string) (bool, bool) {
		t.Fatal("the old arch hook must not decide a row when the resolver is installed")
		return false, false
	})
	var answer backends.Support
	SetBackendSupport(func(_ *detect.Capabilities, r Recommendation) backends.Support { return answer })
	cases := []struct {
		state      backends.SupportState
		status     BackendStatus
		inReason   bool
		requiresIt bool
	}{
		{backends.SupportReady, BackendReady, false, false},
		{backends.SupportInstall, BackendNeedsSetup, true, false},
		{backends.SupportUnsupported, BackendUnsupported, true, false},
		{backends.SupportNoEncoding, BackendUnsupported, true, true},
		{backends.SupportUnknownArch, BackendUnsupported, true, false},
		{backends.SupportUnknown, BackendUnknown, false, false},
		{backends.SupportUnknown, BackendUnknown, true, true},
	}
	for _, c := range cases {
		answer = backends.Support{State: c.state, Reason: "note for " + string(c.state)}
		if c.requiresIt {
			answer.Requirement = backends.Requirement{Unchecked: []string{"entry"}}
		}
		r := Recommendation{Candidate: Candidate{Name: "M", Arch: "somearch"}, Reason: "base"}
		markBackendStatus(nil, &r)
		if r.BackendStatus != c.status || r.BackendSupport != c.state || r.BackendNote != answer.Note() {
			t.Fatalf("%s: got %+v", c.state, r)
		}
		if got := strings.Contains(r.Reason, answer.Note()); got != c.inReason {
			t.Fatalf("%s (requirement %v): reason %q", c.state, c.requiresIt, r.Reason)
		}
		if StatusForSupport(c.state) != c.status {
			t.Fatalf("%s: StatusForSupport disagrees", c.state)
		}
	}
}

// The Vulkan-on-CUDA probe still runs first, and a row with no architecture is
// still unknown without asking the resolver.
func TestBackendSupportKeepsTheVulkanAndNoArchPaths(t *testing.T) {
	t.Cleanup(func() { resetBackendProbes(); SetBackendSupport(nil) })
	SetInstalledArchSupport(func(*detect.Capabilities, string) (bool, bool) { return true, true })
	SetVulkanOnCUDA(func(_ *detect.Capabilities, arch string, _ bool, _ int) bool { return arch == "vk" })
	SetBackendSupport(func(_ *detect.Capabilities, r Recommendation) backends.Support {
		if r.Arch == "vk" || r.Arch == "" {
			t.Fatalf("resolver asked about %q", r.Arch)
		}
		return backends.Support{State: backends.SupportReady}
	})
	r := Recommendation{Candidate: Candidate{Name: "V", Arch: "vk"}}
	markBackendStatus(nil, &r)
	if r.BackendStatus != BackendReady || !r.ServedOnVulkan() || r.BackendSupport != "" {
		t.Fatalf("Vulkan row: %+v", r)
	}
	n := Recommendation{Candidate: Candidate{Name: "N"}}
	markBackendStatus(nil, &n)
	if n.BackendStatus != BackendUnknown {
		t.Fatalf("no-arch row: %+v", n)
	}
}
