package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math/bits"
	"os"
	"sort"
)

// ggmlTypeSize maps a tensor type id to (bytes per block, elements per block).
// Only types whose layout is certain are listed: a type missing here makes
// ExpectedSize report "unknown" rather than guess, so an unfamiliar quant can
// never mark a good file as an incomplete download.
var ggmlTypeSize = map[uint32][2]uint64{
	0: {4, 1}, 1: {2, 1},
	2: {18, 32}, 3: {20, 32}, 6: {22, 32}, 7: {24, 32}, 8: {34, 32}, 9: {36, 32},
	10: {84, 256}, 11: {110, 256}, 12: {144, 256}, 13: {176, 256}, 14: {210, 256}, 15: {292, 256},
	16: {66, 256}, 17: {74, 256}, 18: {98, 256}, 19: {50, 256}, 20: {18, 32},
	21: {110, 256}, 22: {82, 256}, 23: {136, 256},
	24: {1, 1}, 25: {2, 1}, 26: {4, 1}, 27: {8, 1}, 28: {8, 1},
	29: {56, 256}, 30: {2, 1}, 39: {17, 32},
}

const (
	maxTensorCount = 1 << 22
	maxTensorDims  = 8
)

// layoutBytes is the size ggmlTypeSize gives a tensor of ttype with elems
// elements; known is false for a type without a listed layout.
func layoutBytes(ttype uint32, elems uint64) (need uint64, known bool) {
	size, known := ggmlTypeSize[ttype]
	if !known {
		return 0, false
	}
	return (elems + size[1] - 1) / size[1] * size[0], true
}

// offsetOrder returns the indexes of tensors sorted by data offset: a tensor's
// on-disk span runs from its offset to the next one's.
func offsetOrder(tensors []tensorEntry) []int {
	order := make([]int, len(tensors))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return tensors[order[a]].offset < tensors[order[b]].offset })
	return order
}

type countingReader struct {
	r *bufio.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// ExpectedSize returns the file size a GGUF header declares: the data start
// (header end rounded up to general.alignment, default 32) plus the largest
// tensor end (offset + byte size). A file with no tensor data is just its
// unpadded header. ok is false when the size cannot be known
// with certainty: unreadable or corrupt header, a tensor type this package
// has no layout for, or a tensor whose type layout needs more bytes than its
// offset span (the table and the file disagree, as in ReadQuantProfile).
// Callers must treat !ok as "no verdict".
func ExpectedSize(path string) (expected int64, ok bool) {
	f, err := os.Open(path)
	if err != nil {
		return 0, false
	}
	defer f.Close()
	exp, err := expectedSizeFrom(f)
	if err != nil || exp < 0 {
		return 0, false
	}
	return exp, true
}

func expectedSizeFrom(src io.Reader) (int64, error) {
	r := &countingReader{r: bufio.NewReaderSize(src, 1<<20)}
	var buf [8]byte
	u32 := func() (uint32, error) {
		if _, err := io.ReadFull(r, buf[:4]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint32(buf[:4]), nil
	}
	u64 := func() (uint64, error) {
		if _, err := io.ReadFull(r, buf[:8]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint64(buf[:8]), nil
	}
	if _, err := io.ReadFull(r, buf[:4]); err != nil || string(buf[:4]) != "GGUF" {
		return -1, fmt.Errorf("not a GGUF file")
	}
	if v, err := u32(); err != nil || v < 2 || v > 3 {
		return -1, fmt.Errorf("unsupported GGUF version")
	}
	tensorCount, err := u64()
	if err != nil {
		return -1, err
	}
	kvCount, err := u64()
	if err != nil {
		return -1, err
	}
	if tensorCount > maxTensorCount {
		return -1, fmt.Errorf("tensor count %d out of range", tensorCount)
	}
	align := uint64(32)
	for i := uint64(0); i < kvCount; i++ {
		keyLen, err := u64()
		if err != nil {
			return -1, err
		}
		if keyLen > maxGGUFKeyLen {
			return -1, fmt.Errorf("metadata key length %d out of range", keyLen)
		}
		key := make([]byte, keyLen)
		if _, err := io.ReadFull(r, key); err != nil {
			return -1, err
		}
		valType, err := u32()
		if err != nil {
			return -1, err
		}
		if string(key) == "general.alignment" && valType == 4 {
			v, err := u32()
			if err != nil {
				return -1, err
			}
			if v > 0 {
				align = uint64(v)
			}
			continue
		}
		if err := skipGGUFValue(r, valType, 0); err != nil {
			return -1, err
		}
	}
	var maxEnd uint64
	tensors := make([]tensorEntry, 0, min(tensorCount, 1<<14))
	for i := uint64(0); i < tensorCount; i++ {
		nameLen, err := u64()
		if err != nil {
			return -1, err
		}
		if nameLen > maxGGUFKeyLen {
			return -1, fmt.Errorf("tensor name length %d out of range", nameLen)
		}
		if _, err := io.CopyN(io.Discard, r, int64(nameLen)); err != nil {
			return -1, err
		}
		nDims, err := u32()
		if err != nil {
			return -1, err
		}
		if nDims == 0 || nDims > maxTensorDims {
			return -1, fmt.Errorf("tensor dimension count %d out of range", nDims)
		}
		elems := uint64(1)
		overflow := false
		for d := uint32(0); d < nDims; d++ {
			dim, err := u64()
			if err != nil {
				return -1, err
			}
			hi, lo := bits.Mul64(elems, dim)
			if hi != 0 {
				overflow = true
			}
			elems = lo
		}
		ttype, err := u32()
		if err != nil {
			return -1, err
		}
		offset, err := u64()
		if err != nil {
			return -1, err
		}
		nbytes, known := layoutBytes(ttype, elems)
		if !known || overflow {
			return -1, fmt.Errorf("tensor type %d size unknown", ttype)
		}
		if elems > 0 { // an empty tensor owns no bytes and may share an offset
			tensors = append(tensors, tensorEntry{elems: elems, ttype: ttype, offset: offset})
		}
		end := offset + nbytes
		if end < offset || end > 1<<62 {
			return -1, fmt.Errorf("tensor extent out of range")
		}
		if end > maxEnd {
			maxEnd = end
		}
	}
	order := offsetOrder(tensors)
	for k := 0; k+1 < len(order); k++ {
		t := tensors[order[k]]
		span := tensors[order[k+1]].offset - t.offset
		if need, _ := layoutBytes(t.ttype, t.elems); need > span {
			return -1, fmt.Errorf("tensor type %d needs %d bytes but spans %d", t.ttype, need, span)
		}
	}
	headerEnd := uint64(r.n)
	if maxEnd == 0 {
		// No tensor data follows (a metadata-only first shard): split tools
		// write it unpadded, so the header end is the whole file.
		return int64(headerEnd), nil
	}
	dataStart := (headerEnd + align - 1) / align * align
	total := dataStart + maxEnd
	if total < dataStart || total > 1<<62 {
		return -1, fmt.Errorf("expected size out of range")
	}
	return int64(total), nil
}
