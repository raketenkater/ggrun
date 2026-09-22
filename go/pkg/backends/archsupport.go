package backends

import (
	"bytes"
	"debug/elf"
	"debug/pe"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// A backend that cannot serve a model's architecture fails at model load with a
// message from deep inside the loader, and nothing tells the user that a
// reviewed fork exists which can. On this project a Laguna launch silently
// routed to mainline, whose loader has no laguna architecture at all, and the
// only signal was a tensor-count error.
//
// The check here is a real capability probe rather than a guess: llama.cpp
// registers every architecture as a C string literal in the loader
// (`{ LLM_ARCH_DFLASH, "dflash" }`), so the name is present exactly when the
// build knows the architecture.
//
// Two details decide whether the probe is right or useless:
//
// The literal usually is not in the executable. Mainline-style builds link
// llama-server against libllama, so the executable carries no architecture at
// all while the shared object carries all of them; ik_llama places libllama in
// a sibling directory, and the poolside build reaches libllama only through
// libllama-server-impl. Scanning just the given path reports "unsupported" for
// every architecture on every dynamically linked backend, so the probe follows
// the library graph.
//
// A plain substring is not evidence. Build directories end up embedded in the
// binaries, so a fork built in .../fork-llama.cpp-add-laguna/ contains dozens
// of copies of "laguna" inside path strings -- libggml-cuda alone had 69, none
// of them an architecture. Matching the name NUL-bracketed keeps only genuine
// standalone C literals, which drops every one of those paths while still
// finding each real architecture.
//
// One inaccuracy survives, and it is the safe one. Architecture names are not
// the only NUL-terminated literals in a loader: the ik_llama forks list
// "laguna" among their tokenizer pre-types, so a probe of those backends
// reports the architecture as known when it is not. Nothing distinguishes the
// two tables by content, so the probe over-reports rather than under-reports.
// That direction is deliberate -- an over-report costs the user the suggestion
// they would not have had anyway, while an under-report would talk them out of
// a backend that works. Mainline llama.cpp carries no such literal, so the case
// this feature exists for, a fork-only architecture launched on mainline, still
// probes correctly.

// archProbeChunk is the read size for scanning a file. Backend libraries reach
// hundreds of megabytes; streaming keeps memory bounded regardless.
const archProbeChunk = 4 << 20

// archProbeMaxFiles bounds the dependency walk. Real backends pull in a handful
// of llama objects; the cap only stops a pathological or cyclic graph.
const archProbeMaxFiles = 24

// BackendSupportsArch reports whether a built backend knows an architecture.
//
// The second return value is false when the question could not be answered --
// an unreadable binary, or an architecture name too short to match safely. A
// caller must not treat "could not probe" as "not supported": refusing a launch
// on a failed probe would be worse than the cryptic loader error it replaces.
func BackendSupportsArch(binaryPath, arch string) (supported, probed bool) {
	arch = strings.ToLower(strings.TrimSpace(arch))
	if binaryPath == "" || len(arch) < 2 {
		return false, false
	}
	// Genuine architecture literals are NUL-terminated and follow the previous
	// string's terminator, so requiring both rejects substrings of paths and of
	// longer architecture names alike.
	needle := make([]byte, 0, len(arch)+2)
	needle = append(needle, 0)
	needle = append(needle, arch...)
	needle = append(needle, 0)

	files := archProbeFiles(binaryPath)
	scannedAny := false
	for _, f := range files {
		hit, err := fileHasLiteral(f, needle)
		if err != nil {
			continue
		}
		scannedAny = true
		if hit {
			return true, true
		}
	}
	if !scannedAny {
		return false, false
	}
	return false, true
}

// archProbeFiles returns the binary plus the llama libraries it loads, directly
// or transitively. Traversal stays on objects whose name carries "llama": the
// architecture table lives there, and following everything else would mean
// scanning libc and the CUDA runtime to no purpose.
func archProbeFiles(binaryPath string) []string {
	// Resolve launcher symlinks before walking DT_NEEDED. The canonical app home
	// deliberately links into separately stored build trees; using the link's
	// directory as $ORIGIN finds no libllama and falsely reports every
	// architecture as unsupported.
	if resolved, err := filepath.EvalSymlinks(binaryPath); err == nil && resolved != "" {
		binaryPath = resolved
	}
	out := []string{binaryPath}
	seen := map[string]bool{binaryPath: true}

	for i := 0; i < len(out) && len(out) < archProbeMaxFiles; i++ {
		deps := elfLlamaDeps(out[i])
		if len(deps) == 0 {
			deps = peLlamaDeps(out[i])
		}
		for _, dep := range deps {
			if seen[dep] {
				continue
			}
			seen[dep] = true
			out = append(out, dep)
			if len(out) >= archProbeMaxFiles {
				break
			}
		}
	}
	return out
}

// elfLlamaDeps resolves the llama-named DT_NEEDED entries of one object against
// its DT_RUNPATH/DT_RPATH. Only $ORIGIN-relative and absolute entries are
// resolved -- backends built from source always locate their own libraries that
// way, and guessing at system search paths would invite the wrong libllama.
// peLlamaDeps is elfLlamaDeps for Windows builds. llama-server.exe carries no
// architecture literal; they live in llama.dll, reached through
// llama-server-impl.dll. Scanning only the .exe reported every architecture,
// even "llama", as proven unsupported, so automatic selection on Windows would
// refuse every model (the installer's pinned LLM_BACKEND="llama" only hid it).
// Windows resolves a DLL from the executable's own directory first, and that is
// where the bundle ships them.
func peLlamaDeps(path string) []string {
	f, err := pe.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	// debug/pe's ImportedLibraries is an unimplemented stub that always returns
	// nil; ImportedSymbols does parse the import table, as "symbol:library".
	symbols, err := f.ImportedSymbols()
	if err != nil {
		return nil
	}
	seen := map[string]bool{}
	var imports []string
	for _, sym := range symbols {
		i := strings.LastIndexByte(sym, ':')
		if i < 0 || seen[strings.ToLower(sym[i+1:])] {
			continue
		}
		seen[strings.ToLower(sym[i+1:])] = true
		imports = append(imports, sym[i+1:])
	}
	return resolveSiblingLlamaLibs(filepath.Dir(path), imports)
}

// resolveSiblingLlamaLibs finds the llama libraries among imports in dir,
// ignoring case the way Windows does.
func resolveSiblingLlamaLibs(dir string, imports []string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	byLower := make(map[string]string, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			byLower[strings.ToLower(e.Name())] = e.Name()
		}
	}
	var out []string
	for _, name := range imports {
		lower := strings.ToLower(filepath.Base(name))
		if !strings.Contains(lower, "llama") {
			continue
		}
		if actual, ok := byLower[lower]; ok {
			out = append(out, filepath.Join(dir, actual))
		}
	}
	return out
}

func elfLlamaDeps(path string) []string {
	f, err := elf.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	needed, err := f.ImportedLibraries()
	if err != nil || len(needed) == 0 {
		return nil
	}
	origin := filepath.Dir(path)

	var dirs []string
	for _, tag := range []elf.DynTag{elf.DT_RUNPATH, elf.DT_RPATH} {
		vals, err := f.DynString(tag)
		if err != nil {
			continue
		}
		for _, v := range vals {
			for _, entry := range strings.Split(v, ":") {
				if entry = strings.TrimSpace(entry); entry == "" {
					continue
				}
				entry = strings.ReplaceAll(entry, "${ORIGIN}", origin)
				entry = strings.ReplaceAll(entry, "$ORIGIN", origin)
				if filepath.IsAbs(entry) {
					dirs = append(dirs, entry)
				}
			}
		}
	}
	// The object's own directory is where a build most often keeps its
	// libraries, and it stays valid when RUNPATH was stripped.
	dirs = append(dirs, origin)

	var out []string
	for _, name := range needed {
		if !strings.Contains(strings.ToLower(name), "llama") {
			continue
		}
		found := ""
		for _, dir := range dirs {
			cand := filepath.Join(dir, name)
			if st, err := os.Stat(cand); err == nil && !st.IsDir() {
				found = cand
				break
			}
		}
		if found == "" {
			// Some builds ship no RUNPATH at all and rely on the launcher's
			// LD_LIBRARY_PATH -- the ik_llama fork keeps llama-server in
			// build/bin while its libllama.so stays in build/src. Searching the
			// build tree finds those without encoding any one fork's layout.
			found = findLibInBuildTree(filepath.Dir(origin), name)
		}
		if found != "" {
			out = append(out, found)
		}
	}
	return out
}

// archProbeSearchDirs bounds the fallback search so an unexpected directory
// tree cannot turn a probe into a full-disk walk.
const archProbeSearchDirs = 512

// findLibInBuildTree looks for a library by exact name under a build root,
// shallowly. Build layouts keep their objects a couple of levels down, so a
// small depth limit finds them while keeping the walk cheap.
func findLibInBuildTree(root, name string) string {
	if root == "" || name == "" {
		return ""
	}
	rootDepth := strings.Count(filepath.Clean(root), string(os.PathSeparator))
	found := ""
	visited := 0

	filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || found != "" {
			if found != "" {
				return filepath.SkipAll
			}
			return nil
		}
		if d.IsDir() {
			visited++
			if visited > archProbeSearchDirs {
				return filepath.SkipAll
			}
			if strings.Count(filepath.Clean(path), string(os.PathSeparator))-rootDepth >= 3 {
				return filepath.SkipDir
			}
			return nil
		}
		if d.Name() == name {
			found = path
			return filepath.SkipAll
		}
		return nil
	})
	return found
}

// fileHasLiteral streams a file looking for an exact byte sequence, overlapping
// successive reads so a match straddling a chunk boundary is still found.
func fileHasLiteral(path string, needle []byte) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()

	buf := make([]byte, archProbeChunk)
	overlap := len(needle) - 1
	carry := make([]byte, 0, overlap)
	for {
		n, readErr := f.Read(buf)
		if n > 0 {
			window := append(append(make([]byte, 0, len(carry)+n), carry...), buf[:n]...)
			if bytes.Contains(window, needle) {
				return true, nil
			}
			if len(window) > overlap {
				carry = append(carry[:0], window[len(window)-overlap:]...)
			} else {
				carry = append(carry[:0], window...)
			}
		}
		if readErr == io.EOF {
			return false, nil
		}
		if readErr != nil {
			return false, readErr
		}
	}
}

// RecipesForArch returns reviewed recipes that route an architecture, so a
// launch that cannot proceed can name the exact fix instead of describing the
// problem. Generic by construction: it consults the catalog rather than
// special-casing any model.
func RecipesForArch(arch string) []Recipe {
	arch = strings.ToLower(strings.TrimSpace(arch))
	if arch == "" {
		return nil
	}
	var out []Recipe
	for _, r := range Recipes() {
		if !r.HelperOnly && strings.EqualFold(strings.TrimSpace(r.RouteArch), arch) {
			out = append(out, r)
		}
	}
	return out
}

// RegisteredForArch returns registered backends routing an architecture.
func RegisteredForArch(arch string) []Backend {
	arch = strings.ToLower(strings.TrimSpace(arch))
	if arch == "" {
		return nil
	}
	var out []Backend
	for _, b := range Load() {
		if !IsHelperOnly(b) && strings.EqualFold(strings.TrimSpace(b.RouteArch), arch) {
			out = append(out, b)
		}
	}
	return out
}

// SoleHelperForArch returns a helper-only backend when it is the ONLY
// registered backend routing an architecture, and nil otherwise.
//
// ForArch and RegisteredForArch deliberately skip helper-only forks so they
// never globally route main-model launches. But a helper-only fork can be the
// only build that knows a fork-only architecture (Nanbeige's "nanbeige" arch is
// the one case on this project — mainline llama.cpp has no such loader). A model
// of that arch then has no canonical backend that can load it, so the helper
// fork is the only path. This returns it exactly in that situation: sole
// registered backend for the arch, and the arch has no non-helper candidate at
// all. It stays nil when a real (non-helper) backend also serves the arch, so
// the helper fork never shadows a canonical route.
func SoleHelperForArch(arch string) *Backend {
	arch = strings.ToLower(strings.TrimSpace(arch))
	if arch == "" {
		return nil
	}
	var helper *Backend
	haveNonHelper := false
	for _, b := range Load() {
		if !strings.EqualFold(strings.TrimSpace(b.RouteArch), arch) {
			continue
		}
		if _, err := os.Stat(b.Path); err != nil {
			continue
		}
		if IsHelperOnly(b) {
			if helper == nil {
				b := b
				helper = &b
			}
			continue
		}
		haveNonHelper = true
	}
	if haveNonHelper || helper == nil {
		return nil
	}
	return helper
}
