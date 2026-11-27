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
//   - in a binary a name counts only as a whole NUL-bracketed literal, case as
//     written: "Q1_0" is a substring of "STQ1_0" and a prefix of "Q1_0_G128",
//     and build paths embedded in a binary contain names too;
//   - in help text it counts as a whole word (letters, digits and underscore
//     delimit), ignoring case, since help text is prose.

const (
	encodingScanChunk    = 4 << 20
	encodingMaxFileBytes = 256 << 20 // read per file
	encodingMaxTotal     = 1 << 30   // read per backend
	encodingMaxFiles     = 48
)

// EncodingProbe is what a backend build is known to carry.
type EncodingProbe struct {
	// Found maps each requested name (as the caller spelled it) that was found
	// to the source that carries it: "help" or a file's base name.
	Found map[string]string
	// Complete is true when every requested name was found or every file of the
	// build was read in full. When false, a name missing from Found proves
	// nothing: a file was unreadable, over the read bounds, or no build was given.
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
	remaining := map[string][]byte{}
	overlap := 0
	for _, name := range uniqueEncodingNames(names) {
		needle := append(append([]byte{0}, name...), 0)
		remaining[name] = needle
		if len(needle)-1 > overlap {
			overlap = len(needle) - 1
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
			for name, needle := range remaining {
				if bytes.Contains(window, needle) {
					found = append(found, name)
					delete(remaining, name)
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
	complete := len(files) <= encodingMaxFiles
	if !complete {
		files = files[:encodingMaxFiles]
	}
	budget := int64(encodingMaxTotal)
	for _, f := range files {
		if len(remaining) == 0 {
			break
		}
		limit := int64(encodingMaxFileBytes)
		if budget < limit {
			limit = budget
		}
		found, read, ok, err := scanEncodingFile(f, remaining, limit)
		budget -= read
		if err != nil || !ok {
			complete = false
		}
		var left []string
		for _, n := range remaining {
			hit := false
			for _, g := range found {
				if g == n {
					hit = true
					probe.Found[n] = filepath.Base(f)
					break
				}
			}
			if !hit {
				left = append(left, n)
			}
		}
		remaining = left
	}
	probe.Complete = complete || len(remaining) == 0
	return probe
}

func scanEncodingFile(path string, names []string, limit int64) ([]string, int64, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, false, err
	}
	defer f.Close()
	return scanEncodingLiterals(f, names, encodingScanChunk, limit)
}
