package backends

import (
	"strings"
	"testing"

	"github.com/raketenkater/ggrun/pkg/gguf"
)

func TestResolveSupportTable(t *testing.T) {
	c := mustCatalog(t,
		encRecipe("needs-stq", "hy_v4", stqLike)+","+
			encRecipe("fork-arch", "forky", "")+","+
			`{"name":"helper-src","description":"d","tag":"helper-src","git_url":"u","branch":"b","commit":"`+testGoodCommit+`","route_arch":"helped","helper_only":true}`,
		`{"match":"arch_prefix","value":"minimax-m","requires":"ik_llama"}`)

	stq := hist(gguf.TypeUsage{Type: 44, Tensors: 58, Names: []string{"blk.*.ffn_up_exps.weight"}})
	plainTypes := hist(gguf.TypeUsage{Type: 12, Tensors: 300})
	loads := InstalledBackend{Name: "main", Path: "/b/main", LoadsArch: true, Probed: true}
	lacks := InstalledBackend{Name: "main", Path: "/b/main", Probed: true}
	unread := InstalledBackend{Name: "odd", Path: "/b/odd"}
	withProbe := func(b InstalledBackend, found bool) InstalledBackend {
		p := &EncodingProbe{TableSeen: true, Complete: true, requested: map[string]bool{"stq1_0": true}}
		if found {
			p.Found = map[string]string{"STQ1_0": "libggml.so"}
		}
		b.Encodings = p
		return b
	}
	stqTagged := InstalledBackend{Name: "needs-stq", Path: "/b/stq", LoadsArch: true, Probed: true}
	const blocked = "source builds are not available on Windows"

	cases := []struct {
		name    string
		in      SupportInput
		state   SupportState
		backend string
		reason  string
	}{
		{"no architecture", SupportInput{Installed: []InstalledBackend{loads}}, SupportUnknown, "", "architecture is not known"},
		{"no installed backend", SupportInput{Arch: "llama"}, SupportUnknown, "", "no installed backend was found"},
		{"unreadable backend", SupportInput{Arch: "llama", Installed: []InstalledBackend{lacks, unread}}, SupportUnknown, "", "could not be read"},
		{"arch unknown to catalog but loaded", SupportInput{Arch: "llama", Installed: []InstalledBackend{lacks, loads}}, SupportReady, "main", "loads with installed backend main"},
		{"unreadable does not hide a loader", SupportInput{Arch: "llama", Installed: []InstalledBackend{unread, loads}}, SupportReady, "main", ""},
		{"mainline registers: build", SupportInput{Arch: "llama", Installed: []InstalledBackend{lacks}, MainlineRegisters: true, MainlineKnown: true}, SupportInstall, "", "build current llama.cpp"},
		{"mainline registers, build blocked", SupportInput{Arch: "llama", Installed: []InstalledBackend{lacks}, MainlineRegisters: true, MainlineKnown: true, BuildBlocked: blocked}, SupportUnsupported, "", blocked},
		{"recipe route names the command", SupportInput{Arch: "forky", Installed: []InstalledBackend{lacks}}, SupportInstall, "", "ggrun backend install fork-arch"},
		{"recipe route blocked", SupportInput{Arch: "forky", Installed: []InstalledBackend{lacks}, BuildBlocked: blocked}, SupportUnsupported, "", "fork-arch recipe needs a source build, but " + blocked},
		{"helper recipe: accelerated build, no install command", SupportInput{Arch: "helped", Installed: []InstalledBackend{lacks}}, SupportInstall, "", "accelerated backend from the reviewed helper-src"},
		{"family rule blocks a mainline build", SupportInput{Arch: "minimax-m9", Installed: []InstalledBackend{lacks}, MainlineRegisters: true, MainlineKnown: true}, SupportUnsupported, "", "ik_llama backend family"},
		{"stamped not-mainline arch, no route", SupportInput{Arch: "inkling", Installed: []InstalledBackend{lacks}, MainlineKnown: true}, SupportUnsupported, "", "no supported route builds one here"},
		{"arch unknown everywhere", SupportInput{Arch: "neverheard", Installed: []InstalledBackend{lacks}}, SupportUnknownArch, "", "neither the backend catalog nor the model catalog knows"},

		// Tensor encodings.
		{"encoding entry, types unknown, loader: cannot tell", SupportInput{Arch: "hy_v4", Installed: []InstalledBackend{loads}}, SupportUnknown, "", "needs-stq need tensor encodings"},
		{"types known, encoding not used: ready", SupportInput{Arch: "hy_v4", Types: plainTypes, Installed: []InstalledBackend{loads}}, SupportReady, "main", ""},
		{"encoding used, no probe cached: cannot tell", SupportInput{Arch: "hy_v4", Types: stq, Installed: []InstalledBackend{loads}}, SupportUnknown, "", "ggrun backend install needs-stq"},
		{"encoding used, cached probe finds it", SupportInput{Arch: "hy_v4", Types: stq, Installed: []InstalledBackend{withProbe(loads, true)}}, SupportReady, "main", "carries the tensor encodings STQ1_0"},
		{"encoding used, cached probe lacks it: build", SupportInput{Arch: "hy_v4", Types: stq, Installed: []InstalledBackend{withProbe(loads, false)}}, SupportInstall, "", "no installed backend has; build the reviewed needs-stq"},
		{"encoding used, lacks it, build blocked", SupportInput{Arch: "hy_v4", Types: stq, Installed: []InstalledBackend{withProbe(loads, false)}, BuildBlocked: blocked}, SupportNoEncoding, "", blocked},
		{"encoding used, the entry's own backend is installed", SupportInput{Arch: "hy_v4", Types: stq, Installed: []InstalledBackend{loads, stqTagged}}, SupportReady, "needs-stq", "the catalog entry for"},
		{"encoding used, nothing loads the arch", SupportInput{Arch: "hy_v4", Types: stq, Installed: []InstalledBackend{lacks}}, SupportInstall, "", "needs-stq"},
		{"encoding used, nothing installed", SupportInput{Arch: "hy_v4", Types: stq}, SupportUnknown, "", "not known yet"},
	}
	for _, tc := range cases {
		got := c.resolveSupport(tc.in)
		if got.State != tc.state || got.Backend != tc.backend || !strings.Contains(got.Reason, tc.reason) {
			t.Errorf("%s: got state=%s backend=%q reason=%q", tc.name, got.State, got.Backend, got.Reason)
		}
		if got.Note() != got.Reason {
			t.Errorf("%s: Note is not the reason", tc.name)
		}
		if (got.State == SupportReady || got.State == SupportUnknown) != (got.Short() == "") {
			t.Errorf("%s: Short %q does not match state %s", tc.name, got.Short(), got.State)
		}
	}
}

// The embedded catalog answers through the same code; a nil catalog never
// panics and still answers from the installed backends.
func TestResolveSupportEmbeddedAndNilCatalog(t *testing.T) {
	in := SupportInput{Arch: "llama", Installed: []InstalledBackend{{Name: "main", LoadsArch: true, Probed: true}}}
	if got := ResolveSupport(in); got.State != SupportReady {
		t.Fatalf("embedded: %+v", got)
	}
	var nilCat *parsedCatalog
	if got := nilCat.resolveSupport(in); got.State != SupportReady {
		t.Fatalf("nil catalog: %+v", got)
	}
	in.Installed[0].LoadsArch = false
	if got := nilCat.resolveSupport(in); got.State != SupportUnknownArch {
		t.Fatalf("nil catalog, nothing loads: %+v", got)
	}
}
