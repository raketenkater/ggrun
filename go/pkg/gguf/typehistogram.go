package gguf

import (
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	modelstore "github.com/raketenkater/ggrun/pkg/models"
)

// TypeUsage is what a GGUF's tensor table says about one tensor type id: how
// many table entries use it, their element count and the bytes they occupy on
// disk (offset spans, as in ReadQuantProfile). The id is reported as the number
// in the file and never named here: ids from 40 up mean different encodings in
// different backends (upstream Q2_0 is another fork's F8_E4M3_B128), so which
// encoding an id is can only be answered for one backend, by the catalog.
//
// Names are the distinct tensor-name patterns that use the id, with every
// all-digit dot segment written "*" ("blk.12.ffn_down.weight" is
// "blk.*.ffn_down.weight"), sorted; they let a catalog entry tell two
// encodings that share an id apart. Only printable ASCII names are listed.
type TypeUsage struct {
	Type     uint32   `json:"type"`
	Tensors  int64    `json:"tensors"`  // every table entry, including empty tensors
	Elements int64    `json:"elements"` // saturates at MaxInt64
	Bytes    int64    `json:"bytes"`    // spans of the non-empty tensors
	Names    []string `json:"names,omitempty"`
	// NamesTruncated is true when more distinct patterns exist than Names lists.
	NamesTruncated bool `json:"names_truncated,omitempty"`
}

// ErrTooManyTensorTypes: a header uses more distinct type ids than any real
// model does, so no histogram is reported.
var ErrTooManyTensorTypes = errors.New("tensor table uses too many distinct types")

// Bounds on what one header can make the histogram hold.
const (
	maxTypeIDs  = 256  // distinct type ids
	maxNameSet  = 1024 // distinct name patterns tracked per id while reading
	maxNamesOut = 128  // name patterns listed per id
)

// typeAcc accumulates a histogram while the tensor table is walked.
type typeAcc struct {
	by      map[uint32]*typeAccEntry
	tooMany bool
}

type typeAccEntry struct {
	use      TypeUsage
	names    map[string]struct{}
	namesCut bool
}

func newTypeAcc() *typeAcc { return &typeAcc{by: map[uint32]*typeAccEntry{}} }

// entry counts one table entry.
func (a *typeAcc) entry(ttype uint32, name string, elems uint64) {
	e := a.by[ttype]
	if e == nil {
		if len(a.by) >= maxTypeIDs {
			a.tooMany = true
			return
		}
		e = &typeAccEntry{use: TypeUsage{Type: ttype}, names: map[string]struct{}{}}
		a.by[ttype] = e
	}
	e.use.Tensors++
	e.use.Elements = satAddInt64(e.use.Elements, elems)
	if pat, ok := tensorNamePattern(name); ok {
		if _, dup := e.names[pat]; !dup {
			if len(e.names) < maxNameSet {
				e.names[pat] = struct{}{}
			} else {
				e.namesCut = true
			}
		}
	}
}

// addBytes adds one non-empty tensor's span.
func (a *typeAcc) addBytes(ttype uint32, n uint64) {
	if e := a.by[ttype]; e != nil {
		e.use.Bytes = satAddInt64(e.use.Bytes, n)
	}
}

// result is the histogram sorted by type id.
func (a *typeAcc) result() ([]TypeUsage, error) {
	if a.tooMany {
		return nil, ErrTooManyTensorTypes
	}
	out := make([]TypeUsage, 0, len(a.by))
	for _, e := range a.by {
		u := e.use
		names := make([]string, 0, len(e.names))
		for n := range e.names {
			names = append(names, n)
		}
		u.Names, u.NamesTruncated = capNames(names, e.namesCut)
		out = append(out, u)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Type < out[j].Type })
	return out, nil
}

// capNames sorts names and lists at most maxNamesOut of them.
func capNames(names []string, cut bool) ([]string, bool) {
	sort.Strings(names)
	if len(names) > maxNamesOut {
		names, cut = names[:maxNamesOut], true
	}
	if len(names) == 0 {
		return nil, cut
	}
	return names, cut
}

func satAddInt64(a int64, b uint64) int64 {
	if b > uint64(math.MaxInt64) || uint64(math.MaxInt64-a) < b {
		return math.MaxInt64
	}
	return a + int64(b)
}

// tensorNamePattern writes a tensor name as its pattern; ok is false for a name
// that is empty or not printable ASCII.
func tensorNamePattern(name string) (string, bool) {
	if name == "" {
		return "", false
	}
	for i := 0; i < len(name); i++ {
		if name[i] < 0x20 || name[i] > 0x7e {
			return "", false
		}
	}
	segs := strings.Split(name, ".")
	for i, s := range segs {
		if s != "" && allASCIIDigits(s) {
			segs[i] = "*"
		}
	}
	return strings.Join(segs, "."), true
}

func allASCIIDigits(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// mergeTypeUsage adds the histogram b of another shard into a (both sorted by
// id), keeping the result sorted.
func mergeTypeUsage(a, b []TypeUsage) []TypeUsage {
	by := map[uint32]TypeUsage{}
	for _, list := range [][]TypeUsage{a, b} {
		for _, u := range list {
			cur, ok := by[u.Type]
			if !ok {
				by[u.Type] = u
				continue
			}
			cur.Tensors += u.Tensors
			if cur.Elements > math.MaxInt64-u.Elements {
				cur.Elements = math.MaxInt64
			} else {
				cur.Elements += u.Elements
			}
			if cur.Bytes > math.MaxInt64-u.Bytes {
				cur.Bytes = math.MaxInt64
			} else {
				cur.Bytes += u.Bytes
			}
			set := map[string]struct{}{}
			for _, n := range cur.Names {
				set[n] = struct{}{}
			}
			for _, n := range u.Names {
				set[n] = struct{}{}
			}
			names := make([]string, 0, len(set))
			for n := range set {
				names = append(names, n)
			}
			cur.Names, cur.NamesTruncated = capNames(names, cur.NamesTruncated || u.NamesTruncated)
			by[u.Type] = cur
		}
	}
	out := make([]TypeUsage, 0, len(by))
	for _, u := range by {
		out = append(out, u)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Type < out[j].Type })
	return out
}

// ReadTypeHistogramModel is the type histogram of a model: every shard of a
// split GGUF (path names shard 00001) or the single file. It uses the table
// walk of ReadQuantProfile, so it fails exactly where that does (a truncated
// table, unusable offsets, an interrupted download, a missing shard); a partial
// histogram is never returned, because a type missing from it would read as
// absent.
func ReadTypeHistogramModel(path string) ([]TypeUsage, error) {
	files, _, err := modelstore.ResolveGGUFShardFiles(path)
	if err != nil {
		return nil, err
	}
	var out []TypeUsage
	for i, file := range files {
		types, err := readTypeHistogramFile(file)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", file, err)
		}
		if i == 0 {
			out = types
		} else {
			out = mergeTypeUsage(out, types)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no tensors in %s", path)
	}
	return out, nil
}

func readTypeHistogramFile(path string) ([]TypeUsage, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	return ReadTypeHistogram(f, st.Size())
}

// TypeUsageMatching returns the entry for a type id in a histogram.
func TypeUsageMatching(types []TypeUsage, id uint32) (TypeUsage, bool) {
	for _, u := range types {
		if u.Type == id {
			return u, true
		}
	}
	return TypeUsage{}, false
}
