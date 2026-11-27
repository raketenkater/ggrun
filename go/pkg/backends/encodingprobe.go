package backends

import (
	"bytes"
	"debug/elf"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// A tensor type id does not say which encoding a file uses, and a backend that
// lacks the encoding fails at load with "invalid ggml type N". The backend's own
// type table does say: ggml keeps the name of every type it knows in one table
// of C string literals. These functions read that table out of a backend build
// and report what it holds as data. Nothing here executes a backend; the help
// text is passed in by the caller that already runs --help.
//
// Recognising the table. A linker stores identical string tails once, and the
// table is not labelled, so it is found by its content: the names of ggml's
// legacy quants sit as consecutive NUL-terminated strings, "q4_0", "q4_1", ...
// "q8_0", "q8_1" (checked on mainline llama.cpp and ik_llama.cpp builds; a
// "DEPRECATED" placeholder may sit between them). The anchor is
// "\0q4_0\0q4_1\0" followed within 48 bytes by "\0q8_0\0q8_1\0"; one pair
// alone also occurs outside the table. The table is the NUL-separated strings
// from 64 bytes before the anchor to 4096 bytes after it, without the two cut at
// the edges. The first file that holds an anchor decides; a build whose table
// does not contain the anchor is reported as not read (Complete false), never
// as lacking names.
//
// Matching a name. Names compare case-insensitively and as a whole table string
// (ggml spells them as "q4_K", "iq4_xs", "bf16"), so "Q4_K" matches "q4_K". A
// name that is no string of the table but is a proper suffix of one ("f16" in
// "bf16", "q2_0" in "tq2_0") is undetermined: the linker merges such tails, so
// the bytes cannot show whether the shorter name is a type of its own.
// Undetermined is neither found nor absent. Only strings in the table count, so
// an unrelated table elsewhere in the build cannot report a name found.
//
// Names outside the table. The linker keeps one copy of an identical short
// literal ("f32", "i64"), and that copy may sit in another string table, so the
// table region can lack a name the backend has. The scan of the file that holds
// the table therefore also records every requested name that occurs anywhere in
// it as "name" followed by NUL (case-insensitive, any left boundary, so also as
// the tail of a longer string). Such a name that is not a table string is
// Undetermined, never absent. This costs no extra pass: the file is read in
// full once, and when a limit cuts it short Complete is false.
//
// What a hit means: the backend's type table has the name; it does not prove
// the GPU backends have kernels for it, and a help hit may only be the
// KV-cache type list.

const (
	encodingScanChunk    = 4 << 20
	encodingMaxFileBytes = 512 << 20 // read per file; a whole library (an ik build's is 346 MB)
	encodingMaxTotal     = 1 << 30   // read per backend
	encodingMaxFiles     = 48

	tableRegionBefore = 64   // bytes kept before the anchor
	tableRegionAfter  = 4096 // bytes kept after the anchor
	tableAnchorGap    = 48   // the second pair starts within this many bytes
	maxTableToken     = 32
	maxDirEntries     = 4096
	maxRunpathDirs    = 16
)

var (
	tableAnchorA = []byte("\x00q4_0\x00q4_1\x00")
	tableAnchorB = []byte("\x00q8_0\x00q8_1\x00")
)

// EncodingProbe is what a backend build is known to carry.
type EncodingProbe struct {
	// Found maps each requested name (as the caller spelled it) that is in the
	// build's type table to its source: "help", or the base name of the file
	// holding the table (a symlink resolved).
	Found map[string]string
	// Undetermined maps each requested name that is no string of the type table
	// but may still be a type of the backend to the evidence: the longer table
	// string it ends (name -> "bf16"), or "outside the type table" when the name
	// occurs elsewhere in the file as a NUL-terminated string. It is neither
	// found nor absent: use Absent, never "not in Found", to ask whether a name
	// is missing.
	Undetermined map[string]string
	// TableSeen is true when a file with a recognisable type table was read.
	TableSeen bool
	// Complete is true when every requested name was found, or the file holding
	// the type table was read in full: TableSeen and not cut short by a read
	// bound. Only then does a name that is in neither Found nor
	// Undetermined mean the backend lacks it. When false nothing is known about
	// the names not in Found: no build was given, no file held the table, a file
	// was unreadable or over the read bounds, or the libraries holding the table
	// were not located (a system install without RUNPATH, or a platform whose
	// libraries are not looked up beyond the binary's own directory).
	Complete bool
}

// Absent reports that the build is known to lack name: the table was read and
// it is neither found nor undetermined.
func (p EncodingProbe) Absent(name string) bool {
	_, found := p.Found[name]
	_, undetermined := p.Undetermined[name]
	return p.Complete && !found && !undetermined
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

// scanTypeTable streams r once, reading at most limit bytes (0 = no limit) in
// reads of chunk bytes. It returns the region around the type table anchor (nil
// when the stream holds none) and which of names occur anywhere in the stream as
// "name" followed by NUL, ignoring case. complete is true only when the stream
// was read to its end: a limit or an error leaves region and occurs as far as
// they got, with complete false.
func scanTypeTable(r io.Reader, chunk int, limit int64, names []string) (region []byte, occurs map[string]bool, read int64, complete bool, err error) {
	if chunk < 1 {
		chunk = encodingScanChunk
	}
	buf := make([]byte, chunk)
	tr := newNeedleSet(names)
	var win []byte
	from := 0 // win[from:] has not been searched for an anchor start
	eof := false
	idle := 0
	for {
		for region == nil {
			i := bytes.Index(win[from:], tableAnchorA)
			if i < 0 {
				if n := len(win) - len(tableAnchorA) + 1; n > from {
					from = n
				}
				break
			}
			i += from
			if len(win) < i+tableRegionAfter && !eof {
				from = i // wait for the rest of the region
				break
			}
			if tableAnchorHolds(win[i:]) {
				lo := i - tableRegionBefore
				if lo < 0 {
					lo = 0
				}
				hi := i + tableRegionAfter
				if hi > len(win) {
					hi = len(win)
				}
				region = append([]byte(nil), win[lo:hi]...)
				win = nil
				break
			}
			from = i + 1
		}
		if eof {
			return region, tr.result(), read, true, nil
		}
		if region == nil {
			if drop := from - tableRegionBefore; drop > 0 {
				win = append([]byte(nil), win[drop:]...)
				from -= drop
			}
		}
		want := len(buf)
		if limit > 0 {
			if read >= limit {
				return region, tr.result(), read, false, nil
			}
			if int64(want) > limit-read {
				want = int(limit - read)
			}
		}
		n, readErr := r.Read(buf[:want])
		read += int64(n)
		tr.feed(buf[:n])
		if region == nil {
			win = append(win, buf[:n]...)
		}
		if n == 0 && readErr == nil {
			if idle++; idle > 100 {
				return region, tr.result(), read, false, io.ErrNoProgress
			}
		} else {
			idle = 0
		}
		if readErr == io.EOF {
			eof = true
		} else if readErr != nil {
			return region, tr.result(), read, false, readErr
		}
	}
}

// needleSet records which names occur in a stream as lower-case "name\0".
type needleSet struct {
	names   []string
	needles [][]byte
	seen    []bool
	left    int
	overlap int
	carry   []byte
}

func newNeedleSet(names []string) *needleSet {
	s := &needleSet{}
	for _, n := range uniqueEncodingNames(names) {
		needle := append([]byte(strings.ToLower(n)), 0)
		s.names = append(s.names, n)
		s.needles = append(s.needles, needle)
		if len(needle)-1 > s.overlap {
			s.overlap = len(needle) - 1
		}
	}
	s.seen = make([]bool, len(s.names))
	s.left = len(s.names)
	return s
}

// feed adds the next bytes of the stream.
func (s *needleSet) feed(data []byte) {
	if s.left == 0 || len(data) == 0 {
		return
	}
	window := make([]byte, 0, len(s.carry)+len(data))
	window = append(append(window, s.carry...), data...)
	if len(window) > s.overlap {
		s.carry = append(s.carry[:0], window[len(window)-s.overlap:]...)
	} else {
		s.carry = append(s.carry[:0], window...)
	}
	for i, c := range window { // ASCII lower-casing, in place
		if c >= 'A' && c <= 'Z' {
			window[i] = c + 0x20
		}
	}
	for i, needle := range s.needles {
		if !s.seen[i] && bytes.Contains(window, needle) {
			s.seen[i] = true
			s.left--
		}
	}
}

// result lists the names seen.
func (s *needleSet) result() map[string]bool {
	out := map[string]bool{}
	for i, n := range s.names {
		if s.seen[i] {
			out[n] = true
		}
	}
	return out
}

// tableAnchorHolds checks the second pair after a first pair at w[0:].
func tableAnchorHolds(w []byte) bool {
	start := len(tableAnchorA) - 1 // the NUL ending the first pair may start the second
	end := start + tableAnchorGap + len(tableAnchorB)
	if end > len(w) {
		end = len(w)
	}
	return start < end && bytes.Contains(w[start:end], tableAnchorB)
}

// typeTableTokens are the strings of a table region: whole NUL-separated runs
// of letters, digits and underscore (the two cut at the region's edges are
// dropped).
func typeTableTokens(region []byte) []string {
	parts := bytes.Split(region, []byte{0})
	if len(parts) < 3 {
		return nil
	}
	var out []string
	for _, p := range parts[1 : len(parts)-1] {
		if len(p) == 0 || len(p) > maxTableToken {
			continue
		}
		ok := true
		for _, c := range p {
			if !encodingWordByte(c) {
				ok = false
				break
			}
		}
		if ok {
			out = append(out, string(p))
		}
	}
	return out
}

// outsideTable is the Undetermined evidence for a name that is no table string
// but occurs elsewhere in the file as a NUL-terminated string.
const outsideTable = "outside the type table"

// classifyTypeNames splits names into those that are strings of the table
// (found) and those that are only a suffix of one (undetermined: name -> string),
// or occur elsewhere in the file (undetermined: outsideTable).
func classifyTypeNames(names, tokens []string, occurs map[string]bool) (found []string, undetermined map[string]string) {
	undetermined = map[string]string{}
	lower := make([]string, len(tokens))
	for i, t := range tokens {
		lower[i] = strings.ToLower(t)
	}
	for _, name := range names {
		ln := strings.ToLower(name)
		hit := false
		for _, t := range lower {
			if t == ln {
				hit = true
				break
			}
		}
		if hit {
			found = append(found, name)
			continue
		}
		for i, t := range lower {
			if len(t) > len(ln) && strings.HasSuffix(t, ln) {
				undetermined[name] = tokens[i]
				break
			}
		}
		if _, ok := undetermined[name]; !ok && occurs[name] {
			undetermined[name] = outsideTable
		}
	}
	return found, undetermined
}

// elfRunpathEntries returns the raw DT_RUNPATH and DT_RPATH strings of an ELF
// file (nil for anything else). It reads the dynamic section only.
func elfRunpathEntries(path string) []string {
	f, err := elf.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	var out []string
	for _, tag := range []elf.DynTag{elf.DT_RUNPATH, elf.DT_RPATH} {
		vals, err := f.DynString(tag)
		if err == nil {
			out = append(out, vals...)
		}
	}
	return out
}

// runpathEntries is a seam for tests.
var runpathEntries = elfRunpathEntries

// expandRunpath resolves $ORIGIN and ${ORIGIN} in colon-separated runpath
// strings against origin and keeps the absolute results, cleaned.
func expandRunpath(origin string, raw []string) []string {
	var out []string
	for _, r := range raw {
		for _, e := range strings.Split(r, ":") {
			e = strings.TrimSpace(e)
			if e == "" {
				continue
			}
			e = strings.ReplaceAll(strings.ReplaceAll(e, "${ORIGIN}", origin), "$ORIGIN", origin)
			if filepath.IsAbs(e) {
				out = append(out, filepath.Clean(e))
			}
		}
	}
	return out
}

// within reports whether path is root or below it.
func within(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}

// encodingProbeFiles lists the files to read for a binary: the binary, the llama
// libraries it loads (the architecture probe's closure), then the ggml
// libraries (names containing "ggml": .so*, .dll, .dylib) in the directories a
// build keeps its libraries: the binary's own, and the DT_RUNPATH/DT_RPATH
// directories of the binary and of those libraries. The build tree is the
// parent of the binary's directory; nothing outside it is listed and a symlink
// whose target is outside it is skipped. Each file appears once, as the
// resolved path.
func encodingProbeFiles(binaryPath string) []string {
	closure := archProbeFiles(binaryPath)
	if len(closure) == 0 {
		return nil
	}
	binDir := filepath.Dir(closure[0])
	root := filepath.Dir(binDir)
	if r, err := filepath.EvalSymlinks(root); err == nil {
		root = r
	}
	var files []string
	seen := map[string]bool{}
	add := func(p string) bool {
		r, err := filepath.EvalSymlinks(p)
		if err != nil {
			r = p
		}
		if seen[r] {
			return false
		}
		seen[r] = true
		files = append(files, r)
		return true
	}
	for _, f := range closure {
		add(f)
	}
	dirs := []string{binDir}
	seenDir := map[string]bool{binDir: true}
	for _, f := range closure {
		origin := filepath.Dir(f)
		for _, d := range expandRunpath(origin, runpathEntries(f)) {
			if len(dirs) >= maxRunpathDirs {
				break
			}
			real, err := filepath.EvalSymlinks(d)
			if err != nil || !within(root, real) || seenDir[real] {
				continue
			}
			seenDir[real] = true
			dirs = append(dirs, real)
		}
		if d := origin; !seenDir[d] && within(root, d) && len(dirs) < maxRunpathDirs {
			seenDir[d] = true
			dirs = append(dirs, d)
		}
	}
	for _, d := range dirs {
		entries, err := os.ReadDir(d)
		if err != nil {
			continue
		}
		for i, e := range entries {
			if i >= maxDirEntries {
				break
			}
			lower := strings.ToLower(e.Name())
			if e.IsDir() || !strings.Contains(lower, "ggml") {
				continue
			}
			if !(strings.HasSuffix(lower, ".dll") || strings.HasSuffix(lower, ".dylib") || strings.Contains(lower, ".so")) {
				continue
			}
			p := filepath.Join(d, e.Name())
			if real, err := filepath.EvalSymlinks(p); err != nil || !within(root, real) {
				continue
			}
			add(p)
		}
	}
	return files
}

// ProbeBackendEncodings reports which of names a backend build carries. help is
// the build's --help output (may be empty); binaryPath is its server binary,
// whose libraries are located and read within fixed bounds (encodingMax*). It
// is a pure function of those inputs: it executes nothing and caches nothing.
func ProbeBackendEncodings(binaryPath, help string, names []string) EncodingProbe {
	return probeBackendEncodings(binaryPath, help, names, encodingMaxFileBytes, encodingMaxTotal, encodingMaxFiles)
}

// probeBackendEncodings is ProbeBackendEncodings with its read bounds as
// parameters, so tests can use small ones.
func probeBackendEncodings(binaryPath, help string, names []string, maxFileBytes, maxTotal int64, maxFiles int) EncodingProbe {
	probe := EncodingProbe{Found: map[string]string{}, Undetermined: map[string]string{}}
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
	if len(files) > maxFiles {
		files = files[:maxFiles]
	}
	budget := maxTotal
	fullyRead := false // the file holding the table was read to its end
	for _, f := range files {
		if budget <= 0 {
			break
		}
		limit := maxFileBytes
		if budget < limit {
			limit = budget
		}
		region, occurs, read, complete, _ := scanEncodingFile(f, limit, remaining)
		budget -= read
		if region == nil {
			continue
		}
		probe.TableSeen = true
		fullyRead = complete
		found, undetermined := classifyTypeNames(remaining, typeTableTokens(region), occurs)
		for _, n := range found {
			probe.Found[n] = filepath.Base(f)
		}
		for n, t := range undetermined {
			probe.Undetermined[n] = t
		}
		break
	}
	probe.Complete = probe.TableSeen && fullyRead
	return probe
}

func scanEncodingFile(path string, limit int64, names []string) ([]byte, map[string]bool, int64, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, false, err
	}
	defer f.Close()
	return scanTypeTable(f, encodingScanChunk, limit, names)
}
