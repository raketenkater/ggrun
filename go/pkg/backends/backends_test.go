package backends

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
)

func TestHY3RecipeIsPinnedAndRouted(t *testing.T) {
	recipe := RecipeByName("hy3")
	if recipe == nil {
		t.Fatal("HY3 recipe missing")
	}
	if recipe.RouteArch != "hy_v3" {
		t.Fatalf("HY3 route arch = %q, want hy_v3", recipe.RouteArch)
	}
	if recipe.Branch != "hy3-support" || len(recipe.Commit) != 40 {
		t.Fatalf("HY3 source is not reproducibly pinned: %#v", recipe)
	}
	if recipe.GitURL != "https://github.com/noonr48/ik_llama-hy3.git" {
		t.Fatalf("unexpected HY3 fork: %s", recipe.GitURL)
	}
	if got := recipe.PatchNames(); len(got) != 1 || got[0] != "hy3/0001-fix-router-tensor-name" {
		t.Fatalf("HY3 recipe patches = %#v", got)
	}

	// Callers receive a copy and cannot mutate the built-in catalog.
	recipe.Commit = "changed"
	again := RecipeByName("HY3")
	if again == nil || again.Commit == "changed" {
		t.Fatal("recipe lookup leaked mutable catalog state")
	}
}

func TestHotExpertsRecipeIsPinnedCapabilityOnlyAndNotArchitectureRouted(t *testing.T) {
	recipe := RecipeByName("hot-experts")
	if recipe == nil {
		t.Fatal("hot-experts recipe missing")
	}
	if recipe.Tag != "hot-experts" || recipe.RouteArch != "" || recipe.HelperOnly {
		t.Fatalf("performance backend must remain explicit and architecture-neutral: %#v", recipe)
	}
	if recipe.GitURL != "https://github.com/csantiago78/llama.cpp.git" ||
		recipe.Branch != "moe-expert-cache" || recipe.Commit != "bccbacdb8945680f1cfc7e6bffd1e59014705750" {
		t.Fatalf("hot-experts source is not the reviewed immutable revision: %#v", recipe)
	}
	if recipe.Accel != "cuda" {
		t.Fatalf("hot-experts acceleration=%q, want cuda", recipe.Accel)
	}
	if len(recipe.RequiredFlags) != 2 || recipe.RequiredFlags[0] != "--moe-expert-cache" ||
		recipe.RequiredFlags[1] != "--moe-expert-cache-inserts" {
		t.Fatalf("hot-experts capability contract=%v", recipe.RequiredFlags)
	}
	for _, arch := range []string{"qwen3moe", "glm4moe", "deepseek2"} {
		if got := ForArch(arch); got != nil && got.Tag == recipe.Tag {
			t.Fatalf("hot-experts recipe hijacked %s routing: %#v", arch, got)
		}
	}
}

func TestHotExpertsFeatureIsEmbeddedAndCapabilityGated(t *testing.T) {
	feature := FeatureByName("HOT-EXPERTS")
	if feature == nil {
		t.Fatal("hot-experts source feature missing")
	}
	if feature.Accel != "cuda" || len(feature.Patches) != 3 {
		t.Fatalf("unexpected hot-experts feature: %#v", feature)
	}
	if len(feature.RequiredFlags) != 2 || feature.RequiredFlags[0] != "--moe-expert-cache" ||
		feature.RequiredFlags[1] != "--moe-expert-cache-inserts" {
		t.Fatalf("hot-experts feature capability contract=%v", feature.RequiredFlags)
	}
	patch := string(feature.Patches[0].contents)
	if !strings.Contains(patch, "From bccbacdb8945680f1cfc7e6bffd1e59014705750") ||
		!strings.Contains(patch, "diff --git a/src/llama-moecache.cpp") {
		t.Fatal("embedded hot-experts artifact is not the reviewed upstream commit")
	}

	// Catalog lookups return copies rather than mutable global state.
	feature.RequiredFlags[0] = "changed"
	again := FeatureByName("hot-experts")
	if again == nil || again.RequiredFlags[0] == "changed" {
		t.Fatal("feature lookup leaked mutable catalog state")
	}
}

func TestComposeRecipeKeepsBaseForkAndAddsFeature(t *testing.T) {
	base := Backend{
		Tag: "glm5next", GitURL: "https://github.com/eauchs/llama.cpp.git",
		Branch: "glm5next/add-glm-5.3-flash", Commit: strings.Repeat("a", 40),
		RouteArch: "glm5next",
	}
	recipe, err := ComposeRecipe(base, "hot-experts")
	if err != nil {
		t.Fatal(err)
	}
	if recipe.Tag != "glm5next-hot-experts" || recipe.BaseTag != "glm5next" ||
		recipe.GitURL != base.GitURL || recipe.Branch != base.Branch || recipe.Commit != base.Commit ||
		recipe.RouteArch != base.RouteArch || !recipe.HelperOnly || recipe.Accel != "cuda" {
		t.Fatalf("composed recipe lost base identity or isolation: %#v", recipe)
	}
	if len(recipe.Features) != 1 || recipe.Features[0] != "hot-experts" ||
		len(recipe.PatchNames()) != 3 || !strings.HasPrefix(recipe.PatchNames()[0], "features/hot-experts/") {
		t.Fatalf("composed recipe lost feature provenance: %#v", recipe)
	}
}

func TestComposeRecipePreservesReviewedBasePatchesAndRejectsUnknownOnes(t *testing.T) {
	hy3 := RecipeByName("hy3")
	if hy3 == nil {
		t.Fatal("hy3 recipe missing")
	}
	base := Backend{Tag: hy3.Tag, GitURL: hy3.GitURL, Branch: hy3.Branch, Commit: hy3.Commit, RouteArch: hy3.RouteArch}
	composed, err := ComposeRecipe(base, "hot-experts")
	if err != nil {
		t.Fatal(err)
	}
	names := composed.PatchNames()
	if len(names) != 4 || names[0] != "hy3/0001-fix-router-tensor-name" || !strings.HasPrefix(names[1], "features/hot-experts/") {
		t.Fatalf("base/feature patch order=%v", names)
	}

	custom := Backend{
		Tag: "custom", GitURL: "https://example.test/llama.cpp.git", Commit: strings.Repeat("b", 40),
		AppliedPatches: []string{"private/unavailable"},
	}
	if got, err := ComposeRecipe(custom, "hot-experts"); err == nil || got != nil {
		t.Fatalf("unreproducible custom patch stack was accepted: recipe=%#v err=%v", got, err)
	}
}

func TestFeatureBackendAndRecipeReconstructionAreScopedToExactBase(t *testing.T) {
	appHome := t.TempDir()
	t.Setenv("LLM_APP_HOME", appHome)
	baseBin := filepath.Join(appHome, "base-server")
	featureBin := filepath.Join(appHome, "feature-server")
	for _, path := range []string{baseBin, featureBin} {
		if err := os.WriteFile(path, []byte("binary"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	base := Backend{
		Tag: "glm5next", Path: baseBin, GitURL: "https://github.com/eauchs/llama.cpp.git",
		Branch: "glm5next/add-glm-5.3-flash", Commit: strings.Repeat("c", 40), RouteArch: "glm5next",
	}
	recipe, err := ComposeRecipe(base, "hot-experts")
	if err != nil {
		t.Fatal(err)
	}
	composite := Backend{
		Tag: recipe.Tag, Path: featureBin, RouteArch: recipe.RouteArch, HelperOnly: true,
		GitURL: recipe.GitURL, Branch: recipe.Branch, Commit: recipe.Commit,
		BaseTag: recipe.BaseTag, Features: append([]string(nil), recipe.Features...), AppliedPatches: recipe.PatchNames(),
	}
	if err := Save([]Backend{base, composite}); err != nil {
		t.Fatal(err)
	}
	got := FeatureBackend("glm5next", "hot-experts")
	if got == nil || got.Tag != composite.Tag {
		t.Fatalf("feature backend lookup=%#v", got)
	}
	if wrong := FeatureBackend("another-base", "hot-experts"); wrong != nil {
		t.Fatalf("feature backend leaked across base identity: %#v", wrong)
	}
	rebuilt, err := RecipeForBackend(composite)
	if err != nil {
		t.Fatal(err)
	}
	if rebuilt.Tag != composite.Tag || strings.Join(rebuilt.PatchNames(), ",") != strings.Join(composite.AppliedPatches, ",") {
		t.Fatalf("reconstructed recipe=%#v patches=%v", rebuilt, rebuilt.PatchNames())
	}
}

func TestMiniMaxM3RecipeIsPinnedAndRouted(t *testing.T) {
	recipe := RecipeByName("minimax-m3")
	if recipe == nil {
		t.Fatal("MiniMax-M3 recipe missing")
	}
	if recipe.RouteArch != "minimax-m3" {
		t.Fatalf("MiniMax-M3 route arch = %q, want minimax-m3", recipe.RouteArch)
	}
	if recipe.Branch != "minimax-m3" || len(recipe.Commit) != 40 {
		t.Fatalf("MiniMax-M3 source is not reproducibly pinned: %#v", recipe)
	}
	if recipe.GitURL != "https://github.com/danielhanchen/llama.cpp.git" {
		t.Fatalf("unexpected MiniMax-M3 fork: %s", recipe.GitURL)
	}
	if got := recipe.PatchNames(); len(got) != 0 {
		t.Fatalf("MiniMax-M3 recipe unexpectedly carries patches: %#v", got)
	}

	byTag := RecipeByName("MiniMax-M3")
	if byTag == nil || byTag.Commit != recipe.Commit {
		t.Fatal("MiniMax-M3 tag lookup did not resolve the pinned recipe")
	}
}

func TestNanoBeigeSupportRecipeIsHelperOnly(t *testing.T) {
	recipe := RecipeByName("nanbeige42")
	if recipe == nil || !recipe.HelperOnly || recipe.RouteArch != "nanbeige" {
		t.Fatalf("NanoBeige support recipe must retain conformance arch without main routing: %#v", recipe)
	}
	if got := recipe.PatchNames(); len(got) != 1 || got[0] != "0001-accept-loop-count-gguf-schema.patch" {
		t.Fatalf("Nanbeige4.2 recipe lost its reviewed GGUF compatibility patch: %v", got)
	}

	appHome := t.TempDir()
	t.Setenv("LLM_APP_HOME", appHome)
	binary := filepath.Join(appHome, "llama-server-cpu")
	if err := os.WriteFile(binary, []byte("binary"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := Save([]Backend{{Tag: "nanbeige42", Path: binary, RouteArch: "nanbeige", HelperOnly: true}}); err != nil {
		t.Fatal(err)
	}
	if got := ForArch("nanbeige"); got != nil {
		t.Fatalf("helper-only CPU backend hijacked main-model routing: %#v", got)
	}
}

func TestLegacyNanoBeigeManifestCannotRouteMainModel(t *testing.T) {
	appHome := t.TempDir()
	t.Setenv("LLM_APP_HOME", appHome)
	binary := filepath.Join(appHome, "llama-server")
	if err := os.WriteFile(binary, []byte("binary"), 0o755); err != nil {
		t.Fatal(err)
	}
	// Deliberately omit HelperOnly, as manifests written before that field was
	// introduced do.
	if err := Save([]Backend{{Tag: "nanbeige42", Path: binary, RouteArch: "nanbeige"}}); err != nil {
		t.Fatal(err)
	}
	loaded := ByTag("nanbeige42")
	if loaded == nil || !loaded.HelperOnly {
		t.Fatalf("builtin helper-only policy was not applied to legacy record: %#v", loaded)
	}
	if got := ForArch("nanbeige"); got != nil {
		t.Fatalf("legacy helper backend hijacked main routing: %#v", got)
	}
	if got := RegisteredForArch("nanbeige"); len(got) != 0 {
		t.Fatalf("legacy helper backend leaked into TUI route candidates: %#v", got)
	}
}

// The default Laguna recipe must be poolside's own fork, not the upstream pull
// request. The PR implements the target architecture only: its loader builds 69
// of the 76 tensors the published DFlash drafter contains, so every speculative
// launch dies during model load. On a RAM-resident MoE, where a forward pass
// costs nearly the same carrying one token or a micro-batch, that forfeits the
// largest measured lever available.
func TestLagunaRecipeUsesThePoolsideForkForDFlash(t *testing.T) {
	recipe := RecipeByName("laguna")
	if recipe == nil {
		t.Fatal("Laguna recipe missing")
	}
	if recipe.RouteArch != "laguna" {
		t.Errorf("Laguna route arch = %q, want laguna", recipe.RouteArch)
	}
	if recipe.GitURL != "https://github.com/poolsideai/llama.cpp.git" {
		t.Errorf("Laguna source = %q, want poolside's fork (the upstream PR cannot load the DFlash drafter)", recipe.GitURL)
	}
	if recipe.Branch != "laguna" {
		t.Errorf("Laguna branch = %q, want laguna", recipe.Branch)
	}
	if len(recipe.Commit) != 40 {
		t.Errorf("Laguna source is not reproducibly pinned: %#v", recipe)
	}
}

// The upstream PR stays selectable: it is what heads for mainline, so it is the
// build to test against when the PR merges, and it is a working target-only
// fallback if the poolside fork regresses.
func TestLagunaUpstreamPRRecipeRemainsAvailable(t *testing.T) {
	recipe := RecipeByName("laguna-upstream-pr")
	if recipe == nil {
		t.Fatal("upstream-PR Laguna recipe missing")
	}
	if recipe.GitURL != "https://github.com/joerowell/llama.cpp.git" || recipe.Branch != "add-laguna" {
		t.Errorf("unexpected upstream-PR source: %#v", recipe)
	}
	if recipe.RouteArch != "laguna" || len(recipe.Commit) != 40 {
		t.Errorf("upstream-PR recipe is not routed or pinned: %#v", recipe)
	}
	// The two recipes must be distinguishable, or installing one would silently
	// register over the other.
	if def := RecipeByName("laguna"); def != nil && def.Tag == recipe.Tag {
		t.Errorf("both Laguna recipes share tag %q", recipe.Tag)
	}
}

func TestHY3RecipePatchAppliesAndRevertsCleanly(t *testing.T) {
	recipe := RecipeByName("hy3")
	if recipe == nil {
		t.Fatal("HY3 recipe missing")
	}
	dir := t.TempDir()
	target := filepath.Join(dir, "src", "llama-model.cpp")
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	original := `static const std::map<llm_arch, std::map<llm_tensor, std::string>> LLM_TENSOR_NA
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
`
	if err := os.WriteFile(target, []byte(original), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := recipe.ApplyPatches(dir); err != nil {
		t.Fatalf("apply HY3 patch: %v", err)
	}
	patched, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(patched), `LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp"`) {
		t.Fatalf("router tensor name was not patched:\n%s", patched)
	}
	if err := recipe.ApplyPatches(dir); err != nil {
		t.Fatalf("reapplying HY3 patch must be idempotent: %v", err)
	}
	if err := recipe.RevertPatches(dir); err != nil {
		t.Fatalf("revert HY3 patch: %v", err)
	}
	restored, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(restored) != original {
		t.Fatalf("patch revert changed source:\n%s", restored)
	}
}

// The installer's default target is ~/.local/bin, whose parent holds no ggrun
// state. Claiming it as an app home made a second copy of ggrun report no
// models and no backends while the real install sat untouched.
func TestAppHomeFromExeIgnoresAStatelessBinParent(t *testing.T) {
	root := t.TempDir()
	stateless := filepath.Join(root, ".local")
	if err := os.MkdirAll(filepath.Join(stateless, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := AppHomeFromExe(filepath.Join(stateless, "bin", "ggrun")); got != "" {
		t.Errorf("AppHomeFromExe claimed a stateless bin parent: %q", got)
	}
}

// A real self-contained install must still be recognised from its own binary,
// with no environment variable set.
func TestAppHomeFromExeAcceptsARealInstall(t *testing.T) {
	install := t.TempDir()
	if err := os.MkdirAll(filepath.Join(install, ".config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(install, ".config", "backends.json"), []byte("[]"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, binDir := range []string{".bin", "bin"} {
		exe := filepath.Join(install, binDir, "ggrun")
		if got := AppHomeFromExe(exe); got != install {
			t.Errorf("AppHomeFromExe(%s) = %q, want %q", binDir, got, install)
		}
	}
}

// A self-contained production tree may keep the binary directly in its app
// home instead of under .bin. It must identify itself without relying on a
// stale global pointer left by another install.
func TestAppHomeFromExeAcceptsDirectInstallBinary(t *testing.T) {
	install := t.TempDir()
	if err := os.MkdirAll(filepath.Join(install, ".config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(install, ".config", "config"), []byte("# test\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := AppHomeFromExe(filepath.Join(install, "ggrun")); got != install {
		t.Errorf("AppHomeFromExe = %q, want direct install %q", got, install)
	}
}

func TestAppHomeFromExeIgnoresUnrelatedLocations(t *testing.T) {
	stateless := t.TempDir()
	if got := AppHomeFromExe(filepath.Join(stateless, "ggrun")); got != "" {
		t.Errorf("AppHomeFromExe = %q, want empty for stateless directory", got)
	}
	if got := AppHomeFromExe(""); got != "" {
		t.Errorf("AppHomeFromExe(\"\") = %q, want empty", got)
	}
}

func TestHasStateRecognisesRealInstallsOnly(t *testing.T) {
	dir := t.TempDir()
	if HasState(dir) {
		t.Error("empty directory reported as an app home")
	}
	if err := os.MkdirAll(filepath.Join(dir, ".config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if HasState(dir) {
		t.Error("bare .config directory reported as an app home")
	}
	if err := os.WriteFile(filepath.Join(dir, ".config", "backends.json"), []byte("[]"), 0o600); err != nil {
		t.Fatal(err)
	}
	if !HasState(dir) {
		t.Error("a directory holding backends.json is an app home")
	}
}

func TestRequiredBackendForArch(t *testing.T) {
	for _, arch := range []string{"minimax-m2", "MiniMax-M3", " minimax-m4 "} {
		if got := RequiredBackendForArch(arch); got != "ik_llama" {
			t.Errorf("RequiredBackendForArch(%q) = %q, want ik_llama", arch, got)
		}
	}
	if got := RequiredBackendForArch("laguna"); got != "" {
		t.Errorf("RequiredBackendForArch(laguna) = %q, want no generic requirement", got)
	}
}

// Discovery is what saves a user whose shell no longer exports LLM_APP_HOME.
func TestDiscoverAppHomeFindsANestedInstall(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	install := filepath.Join(home, "my-project", "ggrun")
	if err := os.MkdirAll(filepath.Join(install, ".config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(install, ".config", "backends.json"), []byte("[]"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := DiscoverAppHome(); got != install {
		t.Errorf("DiscoverAppHome() = %q, want %q", got, install)
	}
}

func TestDiscoverAppHomePrefersTheConventionalLayout(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	for _, dir := range []string{
		filepath.Join(home, "ggrun"),
		filepath.Join(home, "elsewhere", "ggrun"),
	} {
		if err := os.MkdirAll(filepath.Join(dir, ".config"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, ".config", "backends.json"), []byte("[]"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if got := DiscoverAppHome(); got != filepath.Join(home, "ggrun") {
		t.Errorf("DiscoverAppHome() = %q, want the conventional ~/ggrun", got)
	}
}

func TestDiscoverAppHomeReturnsEmptyWhenNoInstallExists(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := os.MkdirAll(filepath.Join(home, "unrelated", "stuff"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := DiscoverAppHome(); got != "" {
		t.Errorf("DiscoverAppHome() = %q, want empty", got)
	}
}

// A recorded pointer must survive the install being deleted without sending
// every later lookup to a dead path.
func TestPointerIsIgnoredWhenTheRecordedInstallIsGone(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("LLM_APP_HOME", "")
	install := filepath.Join(home, "gone")
	if err := os.MkdirAll(filepath.Join(install, ".config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(install, ".config", "backends.json"), []byte("[]"), 0o600); err != nil {
		t.Fatal(err)
	}
	RecordAppHome(install)
	if got := AppHome(); got != install {
		t.Fatalf("AppHome() = %q, want the recorded %q", got, install)
	}
	if err := os.RemoveAll(install); err != nil {
		t.Fatal(err)
	}
	if got := AppHome(); got == install {
		t.Error("AppHome() still points at a deleted install")
	}
}

func TestScanIsBounded(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	for i := 0; i < 60; i++ {
		if err := os.MkdirAll(filepath.Join(home, "d"+strconv.Itoa(i), "sub"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// No install anywhere: must terminate and report nothing rather than walk
	// the tree indefinitely.
	if got := DiscoverAppHome(); got != "" {
		t.Errorf("DiscoverAppHome() = %q, want empty", got)
	}
}

func TestConcurrentUpsertPreservesEveryBackend(t *testing.T) {
	appHome := t.TempDir()
	t.Setenv("LLM_APP_HOME", appHome)
	const count = 24
	errCh := make(chan error, count)
	var wg sync.WaitGroup
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errCh <- Upsert(Backend{Tag: "fork-" + strconv.Itoa(i), Path: "/tmp/fork-" + strconv.Itoa(i)})
		}(i)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			t.Fatalf("concurrent upsert: %v", err)
		}
	}
	if got := len(Load()); got != count {
		t.Fatalf("manifest lost a concurrent update: got %d entries, want %d", got, count)
	}
}

func TestUpsertRefusesToOverwriteCorruptManifest(t *testing.T) {
	appHome := t.TempDir()
	t.Setenv("LLM_APP_HOME", appHome)
	path := ManifestPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	original := []byte("{not-json\n")
	if err := os.WriteFile(path, original, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := Upsert(Backend{Tag: "must-not-replace"}); err == nil {
		t.Fatal("corrupt registry was silently treated as empty")
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(original) {
		t.Fatalf("failed mutation changed corrupt registry: %q", got)
	}
}
