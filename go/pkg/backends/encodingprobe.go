package backends

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// A tensor type id does not say which encoding a file uses, and a backend that
// lacks the encoding fails at load with "invalid ggml type N". The backend's own
// type table does say: every encoding it can read is a C string literal in
// ggml's type table. These functions look for the names the catalog lists, the
// way archsupport.go looks for architecture names, and report what they found
// as data. Nothing here executes a backend; the help text is passed in by the
// caller that already runs --help.
//
// Matching mirrors the architecture probe where it matters:
//   - in a binary a name counts only as a whole NUL-bracketed literal, either
//     as written or in lower case (ggml spells type_name in lower case, "q8_0"):
//     "q1_0" is a substring of "stq1_0" and a prefix of "q1_0_g128", and build
//     paths embedded in a binary contain names too;
//   - in help text it counts as a whole word (letters, digits and underscore
//     delimit), ignoring case, since help text is prose.
//
// A catalog entry's encoding name is therefore the type_name literal of the
// backend's own type table, usually lower case.
//
// What a hit means: the literal is in the type table, so the backend knows the
// name; it does not prove the GPU backends have kernels for it, and a help hit
// may only be the KV-cache type list. What absence means: see EncodingProbe.Complete.

const (
	encodingScanChunk    = 4 << 20
	encodingMaxFileBytes = 256 << 20 // read per file
	encodingMaxTotal     = 1 << 30   // read per backend
	encodingMaxFiles     = 48

	// encodingCanary is a literal every ggml type table carries (F16's
	// type_name). Finding it shows the table was among the bytes read.
	encodingCanary = "f16"
)

// EncodingProbe is what a backend build is known to carry.
type EncodingProbe struct {
	// Found maps each requested name (as the caller spelled it) that was found
	// to the source that carries it: "help" or a file's base name.
	Found map[string]string
	// Complete is true when every requested name was found, or when every file
	// of the build was read in full and the type table was among them: a file
	// whose name contains "ggml" was read in full, or the canary literal "f16"
	// (in every type table) was found. Only then does a name missing from Found
	// mean the backend lacks it. When false it proves nothing: a file was
	// unreadable or over the read bounds, no build was given, or the libraries
	// holding the type table were not located (a system install without RUNPATH
	// gives the binary alone).
	Complete bool
}

// EncodingsInHelp returns the names that occur as whole words in help,
// ignoring case, sorted.
func EncodingsInHelp(help string, names []string) []string {
	lower := strings.ToLower(help)
	var out []string
	for _, name := range uniqueEncodingNames(names) {
		n := strings.ToLower(name)
		for from := 0; from < len(lower); {
			i := strings.Index(lower[from:], n)
			if i < 0 {
				break
			}
			i += from
			end := i + len(n)
			if (i == 0 || !encodingWordByte(lower[i-1])) && (end == len(lower) || !encodingWordByte(lower[end])) {
				out = append(out, name)
				break
			}
			from = i + 1
		}
	}
	return out
}

func encodingWordByte(b byte) bool {
	return b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9' || b == '_'
}

func uniqueEncodingNames(names []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, n := range names {
		if n != "" && !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	sort.Strings(out)
	return out
}

// scanEncodingLiterals streams r for NUL-bracketed literals of names (exact
// case), reading at most limit bytes (0 = no limit) in reads of chunk bytes.
// Successive reads overlap, so a literal across a chunk boundary is found. It
// returns the names found, the bytes read, and whether the source was read to
// its end (or every name was found first).
func scanEncodingLiterals(r io.Reader, names []string, chunk int, limit int64) (found []string, read int64, complete bool, err error) {
	remaining := map[string][][]byte{}
	overlap := 0
	for _, name := range uniqueEncodingNames(names) {
		variants := [][]byte{encodingNeedle(name)}
		if lower := strings.ToLower(name); lower != name {
			variants = append(variants, encodingNeedle(lower))
		}
		remaining[name] = variants
		for _, v := range variants {
			if len(v)-1 > overlap {
				overlap = len(v) - 1
			}
		}
	}
	if chunk < 1 {
		chunk = encodingScanChunk
	}
	buf := make([]byte, chunk)
	carry := make([]byte, 0, overlap)
	idle := 0
	for len(remaining) > 0 {
		want := len(buf)
		if limit > 0 {
			if read >= limit {
				return found, read, false, nil
			}
			if int64(want) > limit-read {
				want = int(limit - read)
			}
		}
		n, readErr := r.Read(buf[:want])
		read += int64(n)
		if n > 0 {
			idle = 0
			window := append(append(make([]byte, 0, len(carry)+n), carry...), buf[:n]...)
			for name, variants := range remaining {
				for _, needle := range variants {
					if bytes.Contains(window, needle) {
						found = append(found, name)
						delete(remaining, name)
						break
					}
				}
			}
			if len(window) > overlap {
				carry = append(carry[:0], window[len(window)-overlap:]...)
			} else {
				carry = append(carry[:0], window...)
			}
		} else if readErr == nil {
			if idle++; idle > 100 {
				return found, read, false, io.ErrNoProgress
			}
		}
		if readErr == io.EOF {
			sort.Strings(found)
			return found, read, true, nil
		}
		if readErr != nil {
			sort.Strings(found)
			return found, read, false, readErr
		}
	}
	sort.Strings(found)
	return found, read, true, nil
}

// encodingNeedle is a name as a whole NUL-bracketed C literal.
func encodingNeedle(name string) []byte {
	return append(append([]byte{0}, name...), 0)
}

// encodingProbeFiles is the binary, the llama libraries it loads (the
// architecture probe's closure) and the ggml libraries beside any of them: the
// type table lives in a ggml library, which the llama closure does not follow.
func encodingProbeFiles(binaryPath string) []string {
	files := archProbeFiles(binaryPath)
	seen := map[string]bool{}
	for _, f := range files {
		seen[f] = true
	}
	var dirs []string
	seenDir := map[string]bool{}
	for _, f := range files {
		if d := filepath.Dir(f); !seenDir[d] {
			seenDir[d] = true
			dirs = append(dirs, d)
		}
	}
	for _, d := range dirs {
		entries, err := os.ReadDir(d)
		if err != nil {
			continue
		}
		for _, e := range entries {
			lower := strings.ToLower(e.Name())
			if e.IsDir() || !strings.Contains(lower, "ggml") {
				continue
			}
			if !(strings.HasSuffix(lower, ".dll") || strings.HasSuffix(lower, ".dylib") || strings.Contains(lower, ".so")) {
				continue
			}
			if p := filepath.Join(d, e.Name()); !seen[p] {
				seen[p] = true
				files = append(files, p)
			}
		}
	}
	return files
}

// ProbeBackendEncodings reports which of names a backend build carries. help is
// the build's --help output (may be empty); binaryPath is its server binary,
// whose library closure is read within fixed bounds (encodingMax*). It is a
// pure function of those inputs: it executes nothing and caches nothing.
func ProbeBackendEncodings(binaryPath, help string, names []string) EncodingProbe {
	return probeBackendEncodings(binaryPath, help, names, encodingMaxFileBytes, encodingMaxTotal, encodingMaxFiles)
}

// probeBackendEncodings is ProbeBackendEncodings with its read bounds as
// parameters, so tests can use small ones.
func probeBackendEncodings(binaryPath, help string, names []string, maxFileBytes, maxTotal int64, maxFiles int) EncodingProbe {
	probe := EncodingProbe{Found: map[string]string{}}
	names = uniqueEncodingNames(names)
	if len(names) == 0 {
		probe.Complete = true
		return probe
	}
	for _, n := range EncodingsInHelp(help, names) {
		probe.Found[n] = "help"
	}
	var remaining []string
	for _, n := range names {
		if _, ok := probe.Found[n]; !ok {
			remaining = append(remaining, n)
		}
	}
	if len(remaining) == 0 {
		probe.Complete = true
		return probe
	}
	if binaryPath == "" {
		return probe
	}
	files := encodingProbeFiles(binaryPath)
	allRead := len(files) <= maxFiles
	if !allRead {
		files = files[:maxFiles]
	}
	budget := maxTotal
	tableSeen := false // a ggml library was read in full, or the canary was found
	for _, f := range files {
		if len(remaining) == 0 {
			break
		}
		if budget <= 0 {
			allRead = false
			break
		}
		limit := maxFileBytes
		if budget < limit {
			limit = budget
		}
		scanNames := append([]string{}, remaining...)
		if !containsString(scanNames, encodingCanary) {
			scanNames = append(scanNames, encodingCanary)
		}
		found, read, ok, err := scanEncodingFile(f, scanNames, limit)
		budget -= read
		if err != nil || !ok {
			allRead = false
		} else if strings.Contains(strings.ToLower(filepath.Base(f)), "ggml") {
			tableSeen = true
		}
		if containsString(found, encodingCanary) {
			tableSeen = true
		}
		var left []string
		for _, n := range remaining {
			if containsString(found, n) {
				probe.Found[n] = filepath.Base(f)
			} else {
				left = append(left, n)
			}
		}
		remaining = left
	}
	probe.Complete = len(remaining) == 0 || (allRead && tableSeen)
	return probe
}

func containsString(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

func scanEncodingFile(path string, names []string, limit int64) ([]string, int64, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, false, err
	}
	defer f.Close()
	return scanEncodingLiterals(f, names, encodingScanChunk, limit)
}
