package download

import (
	"regexp"
	"slices"
	"strconv"
	"testing"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/recommend"
)

func argValue(args []string, flag string) string {
	if i := slices.Index(args, flag); i >= 0 && i+1 < len(args) {
		return args[i+1]
	}
	return ""
}

// --yes used to be added whenever stdin was not a terminal, so an unattended
// run silently took whatever the default was; an explicit --yes from the user
// never reached the script. Now --yes means the user said so, and a run with
// nobody to ask is told so and fails unless the quant is explicit.
func TestScriptArgsSeparateYesFromNoTerminal(t *testing.T) {
	d := &Downloader{ScriptPath: "dl.py", ModelDir: "/m", CacheDir: "/c"}
	caps := &detect.Capabilities{RAM: detect.RAMInfo{TotalMB: 65536, FreeMB: 65536}}

	args := d.scriptArgs("org/repo", "", caps, false)
	if slices.Contains(args, "--yes") || !slices.Contains(args, "--non-interactive") {
		t.Fatalf("no terminal without --yes: %v", args)
	}
	args = d.scriptArgs("org/repo", "", caps, true)
	if slices.Contains(args, "--yes") || slices.Contains(args, "--non-interactive") {
		t.Fatalf("interactive run must keep its prompts: %v", args)
	}
	d.AssumeYes = true
	if args = d.scriptArgs("org/repo", "", caps, true); !slices.Contains(args, "--yes") {
		t.Fatalf("--yes was dropped: %v", args)
	}
	if got := argValue(args, "--ram"); got != "65536" {
		t.Fatalf("--ram = %q, want the planning ceiling", got)
	}
}

// A catalog MoE whose repo name carries no A##B was sized as dense.
func TestScriptArgsPassCatalogActiveParams(t *testing.T) {
	named := regexp.MustCompile(`(?i)A\d+(\.\d+)?B([-_]|$)`)
	for _, c := range recommend.Shortlist() {
		if !c.MoE || c.ActiveParamsB <= 0 || named.MatchString(c.Repo) {
			continue
		}
		args := (&Downloader{}).scriptArgs(c.Repo, "", nil, true)
		if got := argValue(args, "--active-params-b"); got != strconv.FormatFloat(c.ActiveParamsB, 'f', -1, 64) {
			t.Fatalf("%s: --active-params-b = %q, want %v", c.Repo, got, c.ActiveParamsB)
		}
		if args := (&Downloader{}).scriptArgs("org/unknown-dense-GGUF", "", nil, true); slices.Contains(args, "--active-params-b") {
			t.Fatalf("non-catalog repo got active params: %v", args)
		}
		return
	}
	t.Skip("catalog has no MoE with inferred active params and an unnamed repo")
}
