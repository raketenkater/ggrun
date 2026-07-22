// Package gguf parses GGUF (GPT-Generated Unified Format) model files natively
// in Go. It supports single-file and split-shard GGUF, recovers gracefully from
// unexpected metadata types, and computes byte accounting that is correct per
// shard for multi-shard models.
package gguf

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// ============================================================================
// Constants
// ============================================================================

// GGUF binary type IDs (per the GGUF v3 spec). Values are stable across v1/v2/v3.
const (
	ggufTypeUint8   = 0
	ggufTypeInt8    = 1
	ggufTypeUint16  = 2
	ggufTypeInt16   = 3
	ggufTypeUint32  = 4
	ggufTypeInt32   = 5
	ggufTypeFloat32 = 6
	ggufTypeBool    = 7
	ggufTypeString  = 8
	ggufTypeArray   = 9
	ggufTypeUint64  = 10
	ggufTypeInt64   = 11
	ggufTypeFloat64 = 12
)

// GGUF version constants.
const (
	ggufVersion1 = 1
	ggufVersion2 = 2
	ggufVersion3 = 3
)

// ggufMagic is the four-byte little-endian file magic ("GGUF" in ASCII).
const ggufMagic uint32 = 0x46554747

// Sanity caps to avoid OOM on corrupt or hostile files. GGUF in the wild has
// at most a few thousand tensors and a few hundred metadata entries; a model
// with a tokenizer string longer than 256 MiB is not a model.
const (
	maxTensors    = 1 << 24  // 16M
	maxKV         = 1 << 20  // 1M
	maxStringLen  = 256 << 20 // 256 MiB
	maxArrayElems = 1 << 28  // 256M elements
)

// ============================================================================
// Public types
// ============================================================================

// Info holds parsed GGUF metadata.
//
// Numeric fields are zero when the corresponding metadata key is absent or
// has an unexpected type. Callers should treat zero as "unknown" rather than
// a real measurement.
type Info struct {
	Architecture       string `json:"arch"`
	Name               string `json:"name"`
	Basename           string `json:"basename"`
	QuantizedBy        string `json:"quantized_by"`
	BlockCount         int    `json:"layers"`
	ContextLength      int    `json:"ctx_train"`
	EmbeddingLength    int    `json:"embd"`
	FeedForwardLength  int    `json:"ff"`
	HeadCountKV        int    `json:"hkv"`
	KeyLength          int    `json:"kl"`
	ValueLength        int    `json:"vl"`
	VocabSize          int    `json:"vocab_size"`
	TokenizerModel     string `json:"tokenizer_model"`
	TokenizerPre       string `json:"tokenizer_pre"`
	TokenizerHash      string `json:"tokenizer_hash"`
	ExpertBytes        int64  `json:"expert_bytes"`
	NonExpertBytes     int64  `json:"non_expert_bytes"`
	TokenEmbdBytes     int64  `json:"token_embd_bytes"`
	OutputBytes        int64  `json:"output_bytes"`
	ShexpBytes         int64  `json:"shexp_bytes"`
	Fused              int    `json:"fused"`
	Experts            int    `json:"experts"`
	ExpertUsed         int    `json:"exp_used"`
	ExpFF              int    `json:"exp_ff"`
	ExpSharedFF        int    `json:"exp_shared_ff"`
	NRot               int    `json:"n_rot"`
	SSM                int    `json:"ssm"`
	FullAttnInterval   int    `json:"full_interval"`
	SlidingWindow      int    `json:"swa"`
	LeadingDense       int    `json:"leading_dense"`
	KVLoraRank         int    `json:"kv_lora"`
	QLoraRank          int    `json:"q_lora"`
	KeyLengthMLA       int    `json:"kl_mla"`
	ValueLengthMLA     int    `json:"vl_mla"`
	HasShexp           bool   `json:"has_shexp"`
	NextNPredictLayers int    `json:"nextn_predict_layers"`
	IsMoE              bool   `json:"is_moe"`

	// Unexported: tensor headers observed during parse. Used by EstimateParams
	// to derive an architecture-agnostic parameter count from tensor dims.
	tensors []tensorInfo
}

// ============================================================================
// ggufScalar — panic-free wrapper around a single metadata value
// ============================================================================

// ggufScalar is a type-safe wrapper around a single GGUF metadata value.
// It carries the original wire kind plus a normalized payload, and exposes
// safe accessors that never panic on the wrong kind. Returning
// (zero, false) instead of panicking is the contract that the metadata
// application code relies on.
type ggufScalar struct {
	k scalarKind
	u uint64
	i int64
	f float64
	s string
	b bool
}

type scalarKind uint8

const (
	scalarNull     scalarKind = iota // type we did not decode (e.g. unhandled array)
	scalarUint                       // any unsigned integer, normalized to uint64
	scalarInt                        // any signed integer, normalized to int64
	scalarFloat                      // any float, normalized to float64
	scalarString                     // GGUF string
	scalarBool                       // GGUF bool
	scalarArrayLen                   // array of any type; u holds the element count
)

func (s ggufScalar) asUint() (uint64, bool) {
	if s.k == scalarUint || s.k == scalarArrayLen {
		return s.u, true
	}
	return 0, false
}

func (s ggufScalar) asInt() (int64, bool) {
	if s.k == scalarInt {
		return s.i, true
	}
	return 0, false
}

func (s ggufScalar) asFloat() (float64, bool) {
	if s.k == scalarFloat {
		return s.f, true
	}
	return 0, false
}

func (s ggufScalar) asString() (string, bool) {
	if s.k == scalarString {
		return s.s, true
	}
	return "", false
}

func (s ggufScalar) asBool() (bool, bool) {
	if s.k == scalarBool {
		return s.b, true
	}
	return false, false
}

func (s ggufScalar) asArrayLen() (uint64, bool) {
	if s.k == scalarArrayLen {
		return s.u, true
	}
	return 0, false
}

// ============================================================================
// Low-level readers — every I/O call is checked, every short read returns an
// error so the caller cannot drift its cursor on a truncated file.
// ============================================================================

func readU8(r io.Reader) (uint8, error) {
	var b [1]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return b[0], nil
}

func readI8(r io.Reader) (int8, error) {
	v, err := readU8(r)
	return int8(v), err
}

func readU16(r io.Reader) (uint16, error) {
	var b [2]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(b[:]), nil
}

func readI16(r io.Reader) (int16, error) {
	v, err := readU16(r)
	return int16(v), err
}

func readU32(r io.Reader) (uint32, error) {
	var b [4]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func readI32(r io.Reader) (int32, error) {
	v, err := readU32(r)
	return int32(v), err
}

func readU64(r io.Reader) (uint64, error) {
	var b [8]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func readI64(r io.Reader) (int64, error) {
	v, err := readU64(r)
	return int64(v), err
}

func readF32(r io.Reader) (float32, error) {
	v, err := readU32(r)
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(v), nil
}

func readF64(r io.Reader) (float64, error) {
	v, err := readU64(r)
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(v), nil
}

// readString reads a length-prefixed GGUF string. The length is bounded by
// maxStringLen to prevent an attacker (or a corrupted file) from forcing an
// allocation of arbitrary size before any content is checked.
func readString(r io.Reader) (string, error) {
	n, err := readU64(r)
	if err != nil {
		return "", err
	}
	if n > maxStringLen {
		return "", fmt.Errorf("gguf: string length %d exceeds max %d", n, maxStringLen)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// readValue decodes a single GGUF metadata value of the given type ID.
// For arrays it skips the contents (only the count is captured, so callers
// can use the count for VocabSize, etc. without buffering the full array).
// Any short read or invalid type returns a descriptive error.
func readValue(r io.Reader, t uint32) (ggufScalar, error) {
	switch t {
	case ggufTypeUint8:
		v, err := readU8(r)
		if err != nil {
			return ggufScalar{}, err
		}
		return ggufScalar{k: scalarUint, u: uint64(v)}, nil
	case ggufTypeInt8:
		v, err := readI8(r)
		if err != nil {
			return ggufScalar{}, err
		}
		return ggufScalar{k: scalarInt, i: int64(v)}, nil
	case ggufTypeUint16:
		v, err := readU16(r)
		if err != nil {
			return ggufScalar{}, err
		}
		return ggufScalar{k: scalarUint, u: uint64(v)}, nil
	case ggufTypeInt16:
		v, err := readI16(r)
		if err != nil {
			return ggufScalar{}, err
		}
		return ggufScalar{k: scalarInt, i: int64(v)}, nil
	case ggufTypeUint32:
		v, err := readU32(r)
		if err != nil {
			return ggufScalar{}, err
		}
		return ggufScalar{k: scalarUint, u: uint64(v)}, nil
	case ggufTypeInt32:
		v, err := readI32(r)
		if err != nil {
			return ggufScalar{}, err
		}
		return ggufScalar{k: scalarInt, i: int64(v)}, nil
	case ggufTypeFloat32:
		v, err := readF32(r)
		if err != nil {
			return ggufScalar{}, err
		}
		return ggufScalar{k: scalarFloat, f: float64(v)}, nil
	case ggufTypeBool:
		v, err := readU8(r)
		if err != nil {
			return ggufScalar{}, err
		}
		return ggufScalar{k: scalarBool, b: v != 0}, nil
	case ggufTypeString:
		s, err := readString(r)
		if err != nil {
			return ggufScalar{}, err
		}
		return ggufScalar{k: scalarString, s: s}, nil
	case ggufTypeArray:
		return readArrayAsLen(r)
	case ggufTypeUint64:
		v, err := readU64(r)
		if err != nil {
			return ggufScalar{}, err
		}
		return ggufScalar{k: scalarUint, u: v}, nil
	case ggufTypeInt64:
		v, err := readI64(r)
		if err != nil {
			return ggufScalar{}, err
		}
		return ggufScalar{k: scalarInt, i: v}, nil
	case ggufTypeFloat64:
		v, err := readF64(r)
		if err != nil {
			return ggufScalar{}, err
		}
		return ggufScalar{k: scalarFloat, f: v}, nil
	default:
		return ggufScalar{}, fmt.Errorf("gguf: unknown metadata type id %d", t)
	}
}

// readArrayAsLen reads an array header, validates the bounds, and skips
// over the element payload. The result is a scalar whose asArrayLen()
// returns the element count. Element contents are not retained.
func readArrayAsLen(r io.Reader) (ggufScalar, error) {
	elemType, err := readU32(r)
	if err != nil {
		return ggufScalar{}, fmt.Errorf("gguf: read array element type: %w", err)
	}
	count, err := readU64(r)
	if err != nil {
		return ggufScalar{}, fmt.Errorf("gguf: read array length: %w", err)
	}
	if count > maxArrayElems {
		return ggufScalar{}, fmt.Errorf("gguf: array length %d exceeds max %d", count, maxArrayElems)
	}
	if err := skipArrayPayload(r, elemType, count); err != nil {
		return ggufScalar{}, fmt.Errorf("gguf: skip array payload: %w", err)
	}
	return ggufScalar{k: scalarArrayLen, u: count}, nil
}

// skipArrayPayload advances r past `count` elements of the given type.
// String elements and nested arrays require per-element reading; numeric
// elements are skipped in a single io.CopyN for speed.
func skipArrayPayload(r io.Reader, elemType uint32, count uint64) error {
	if count == 0 {
		return nil
	}
	switch elemType {
	case ggufTypeUint8, ggufTypeInt8, ggufTypeBool:
		return discardN(r, count)
	case ggufTypeUint16, ggufTypeInt16:
		return discardN(r, 2*count)
	case ggufTypeUint32, ggufTypeInt32, ggufTypeFloat32:
		return discardN(r, 4*count)
	case ggufTypeUint64, ggufTypeInt64, ggufTypeFloat64:
		return discardN(r, 8*count)
	case ggufTypeString:
		for i := uint64(0); i < count; i++ {
			if _, err := readString(r); err != nil {
				return err
			}
		}
		return nil
	case ggufTypeArray:
		return fmt.Errorf("nested arrays are not supported (elem type %d)", elemType)
	default:
		return fmt.Errorf("unknown array element type %d", elemType)
	}
}

func discardN(r io.Reader, n uint64) error {
	// Guard against wrap on 32-bit hosts.
	if n > (1 << 62) {
		return fmt.Errorf("gguf: array payload size %d too large", n)
	}
	_, err := io.CopyN(io.Discard, r, int64(n))
	return err
}

// ============================================================================
// Tensor info + parsed shard
// ============================================================================

// tensorInfo is the parsed view of a single tensor header.
type tensorInfo struct {
	name   string
	dims   [4]uint64
	nDims  uint32
	typeID uint32
	offset uint64
}

// parsedShard is everything Parse needs from a single GGUF file: the file
// size (for span math), the tensor headers, the KV table, and the file
// offset where the tensor data section begins.
type parsedShard struct {
	path           string
	size           int64
	dataSectionOff int64 // file offset where tensor data begins (aligned)
	tensors        []tensorInfo
	metadata       map[string]ggufScalar
}

func readTensorInfo(r io.Reader) (tensorInfo, error) {
	name, err := readString(r)
	if err != nil {
		return tensorInfo{}, fmt.Errorf("read name: %w", err)
	}
	nDims, err := readU32(r)
	if err != nil {
		return tensorInfo{}, fmt.Errorf("read n_dims: %w", err)
	}
	if nDims > 4 {
		return tensorInfo{}, fmt.Errorf("n_dims=%d exceeds 4", nDims)
	}
	var dims [4]uint64
	for d := uint32(0); d < nDims; d++ {
		v, err := readU64(r)
		if err != nil {
			return tensorInfo{}, fmt.Errorf("read dim[%d]: %w", d, err)
		}
		dims[d] = v
	}
	typeID, err := readU32(r)
	if err != nil {
		return tensorInfo{}, fmt.Errorf("read type: %w", err)
	}
	offset, err := readU64(r)
	if err != nil {
		return tensorInfo{}, fmt.Errorf("read offset: %w", err)
	}
	return tensorInfo{
		name:   name,
		dims:   dims,
		nDims:  nDims,
		typeID: typeID,
		offset: offset,
	}, nil
}

// parseShard reads a single GGUF shard and returns its header, KV table, and
// tensor info records. Every I/O call is checked; on error, the returned
// error names the file so a failure deep in the tensor table is still
// attributable to a specific shard.
func parseShard(path string) (*parsedShard, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open %q: %w", path, err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("gguf: stat %q: %w", path, err)
	}

	magic, err := readU32(f)
	if err != nil {
		return nil, fmt.Errorf("gguf: %q: read magic: %w", path, err)
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("gguf: %q: bad magic 0x%08x (want 0x%08x, i.e. \"GGUF\")", path, magic, ggufMagic)
	}

	version, err := readU32(f)
	if err != nil {
		return nil, fmt.Errorf("gguf: %q: read version: %w", path, err)
	}
	if version < ggufVersion1 || version > ggufVersion3 {
		return nil, fmt.Errorf("gguf: %q: unsupported version %d (want %d..%d)", path, version, ggufVersion1, ggufVersion3)
	}

	// GGUF v1 stores tensor_count and kv_count as uint32; v2 and v3 use uint64.
	var tensorCount, kvCount uint64
	if version == ggufVersion1 {
		tc, err := readU32(f)
		if err != nil {
			return nil, fmt.Errorf("gguf: %q: read tensor_count: %w", path, err)
		}
		tensorCount = uint64(tc)
		mc, err := readU32(f)
		if err != nil {
			return nil, fmt.Errorf("gguf: %q: read kv_count: %w", path, err)
		}
		kvCount = uint64(mc)
	} else {
		tc, err := readU64(f)
		if err != nil {
			return nil, fmt.Errorf("gguf: %q: read tensor_count: %w", path, err)
		}
		tensorCount = tc
		mc, err := readU64(f)
		if err != nil {
			return nil, fmt.Errorf("gguf: %q: read kv_count: %w", path, err)
		}
		kvCount = mc
	}

	if tensorCount > maxTensors {
		return nil, fmt.Errorf("gguf: %q: tensor_count %d exceeds max %d", path, tensorCount, maxTensors)
	}
	if kvCount > maxKV {
		return nil, fmt.Errorf("gguf: %q: kv_count %d exceeds max %d", path, kvCount, maxKV)
	}

	metadata := make(map[string]ggufScalar, kvCount)
	for i := uint64(0); i < kvCount; i++ {
		key, err := readString(f)
		if err != nil {
			return nil, fmt.Errorf("gguf: %q: read kv[%d] key: %w", path, i, err)
		}
		typeID, err := readU32(f)
		if err != nil {
			return nil, fmt.Errorf("gguf: %q: read kv[%q] type: %w", path, key, err)
		}
		val, err := readValue(f, typeID)
		if err != nil {
			return nil, fmt.Errorf("gguf: %q: read kv[%q] value: %w", path, key, err)
		}
		metadata[key] = val
	}

	tensors := make([]tensorInfo, 0, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		t, err := readTensorInfo(f)
		if err != nil {
			return nil, fmt.Errorf("gguf: %q: read tensor[%d]: %w", path, i, err)
		}
		tensors = append(tensors, t)
	}

	// Determine the file offset where the tensor data section begins.
	// Per the GGUF spec, tensor offsets are relative to the start of the
	// data section, which begins at the first aligned position after the
	// tensor info table.
	alignment := uint32(32)
	if v, ok := metadata["general.alignment"].asUint(); ok && v > 0 && v <= 1<<20 {
		alignment = uint32(v)
	}
	pos, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("gguf: %q: seek after tensor table: %w", path, err)
	}
	alignI := int64(alignment)
	dataSectionOff := (pos + alignI - 1) / alignI * alignI

	return &parsedShard{
		path:           path,
		size:           fi.Size(),
		dataSectionOff: dataSectionOff,
		tensors:        tensors,
		metadata:       metadata,
	}, nil
}

// ============================================================================
// Split-shard detection
// ============================================================================

// splitShardRe matches names like "model-00001-of-00010.gguf". The capture
// groups are: (1) prefix-before-index, (2) shard index digits, (3) total
// shard count digits, (4) the ".gguf" suffix. The middle two groups
// preserve their original zero-padding so the sibling glob can be built
// without re-padding.
var splitShardRe = regexp.MustCompile(`^(.+)-(\d+)-of-(\d+)(\.gguf)$`)

// shardRef captures the parsed form of a split-shard filename.
type shardRef struct {
	prefix string // absolute path stem, e.g. "/path/to/model"
	index  int    // 1-based shard number
	total  int    // total number of shards
	suffix string // "of-NNNNN.gguf" with the original digit width
}

// detectShards returns the parsed shard reference for path, or ok=false if
// the filename is not a recognized split-shard pattern. Non-split files
// (single .gguf) return ok=false and Parse treats them as a single shard.
func detectShards(path string) (shardRef, bool) {
	dir, base := filepath.Split(path)
	m := splitShardRe.FindStringSubmatch(base)
	if m == nil {
		return shardRef{}, false
	}
	idx, err1 := strconv.Atoi(m[2])
	total, err2 := strconv.Atoi(m[3])
	if err1 != nil || err2 != nil || idx <= 0 || total <= 0 || idx > total {
		return shardRef{}, false
	}
	return shardRef{
		prefix: filepath.Join(dir, m[1]),
		index:  idx,
		total:  total,
		suffix: "of-" + m[3] + m[4], // preserve original padding, e.g. "of-00010.gguf"
	}, true
}

// findSiblingShards globs the directory of a split shard for all matching
// shards, sorts them lexicographically (which is the same as numerically
// when the index field is zero-padded to a fixed width), and returns the
// absolute paths.
func findSiblingShards(s shardRef) ([]string, error) {
	pattern := s.prefix + "-*-" + s.suffix
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("gguf: glob %q: %w", pattern, err)
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("gguf: split shard pattern %q matched no files", pattern)
	}
	sort.Strings(matches)
	return matches, nil
}

// collectShards returns one parsedShard per GGUF file that participates in
// the model. For a single-file GGUF it is a 1-element slice. For a split
// model it is one element per shard, in shard order.
func collectShards(primary string) ([]parsedShard, error) {
	sref, isSplit := detectShards(primary)
	if !isSplit {
		s, err := parseShard(primary)
		if err != nil {
			return nil, err
		}
		return []parsedShard{*s}, nil
	}

	paths, err := findSiblingShards(sref)
	if err != nil {
		return nil, err
	}
	if len(paths) != sref.total {
		return nil, fmt.Errorf("gguf: expected %d shards for %q, found %d (%v)",
			sref.total, primary, len(paths), paths)
	}

	out := make([]parsedShard, 0, len(paths))
	for i, p := range paths {
		s, err := parseShard(p)
		if err != nil {
			return nil, fmt.Errorf("gguf: shard %d/%d (%s): %w", i+1, len(paths), p, err)
		}
		out = append(out, *s)
	}
	return out, nil
}

// aggregateTensors concatenates tensor headers from every shard in order.
// Used by EstimateParams to compute a full-model parameter count.
func aggregateTensors(shards []parsedShard) []tensorInfo {
	var total int
	for _, s := range shards {
		total += len(s.tensors)
	}
	out := make([]tensorInfo, 0, total)
	for _, s := range shards {
		out = append(out, s.tensors...)
	}
	return out
}

// ============================================================================
// Per-shard span accounting
// ============================================================================

// calculateSpans attributes every byte of every tensor to one of the
// non-overlapping categories on Info. The per-shard version here uses each
// shard's own file size rather than a single "primary" file size, which is
// the source of the gross over/under-counting in the previous implementation.
func calculateSpans(shards []parsedShard, alignment uint32, info *Info) {
	for shardIdx := range shards {
		s := &shards[shardIdx]
		if len(s.tensors) == 0 {
			continue
		}

		// Sort by offset; the GGUF spec does not guarantee order in the
		// tensor info table.
		sort.SliceStable(s.tensors, func(i, j int) bool {
			return s.tensors[i].offset < s.tensors[j].offset
		})

		for i := range s.tensors {
			var span uint64
			if i == len(s.tensors)-1 {
				// Last tensor: span goes to end of THIS shard's file.
				// Tensor offsets are relative to the data section start,
				// so the file position of this tensor is dataSectionOff + offset.
				end := uint64(s.size)
				total := uint64(s.dataSectionOff) + s.tensors[i].offset
				if total > end {
					// Truncated shard: span is undefined. Use 0 rather
					// than wrap.
					span = 0
				} else {
					span = end - total
				}
			} else {
				span = s.tensors[i+1].offset - s.tensors[i].offset
			}
			categorizeTensorSpan(s.tensors[i].name, span, info)
		}
	}
}

// categorizeTensorSpan routes a tensor's byte span into the corresponding
// Info field based on the tensor name. The matching is conservative: only
// the most common GGUF naming conventions are recognized. Tensors that
// don't match a category fall into NonExpertBytes.
func categorizeTensorSpan(name string, span uint64, info *Info) {
	lname := strings.ToLower(name)
	switch {
	case strings.Contains(lname, "token_embd.weight"):
		info.TokenEmbdBytes += int64(span)
		info.NonExpertBytes += int64(span)
	case strings.Contains(lname, "output.weight"), strings.Contains(lname, "output_norm"):
		info.OutputBytes += int64(span)
		info.NonExpertBytes += int64(span)
	case strings.Contains(lname, "shexp"):
		info.ShexpBytes += int64(span)
		info.NonExpertBytes += int64(span)
	case strings.Contains(lname, "exps."), strings.Contains(lname, "_exps."):
		// Both the fused "blk.N.ffn_up.exps.weight" form and the
		// per-expert "blk.N.ffn_up_exps.weight" form are routed here.
		info.ExpertBytes += int64(span)
	default:
		info.NonExpertBytes += int64(span)
	}
}

// ============================================================================
// Metadata → Info mapping
// ============================================================================

// applyMetadataToInfo is the single point where the raw KV table gets
// turned into typed Info fields. Every key lookup goes through the
// ggufScalar accessors, so unexpected types are silently skipped rather
// than causing a panic.
func applyMetadataToInfo(meta map[string]ggufScalar, info *Info) {
	if v, ok := meta["general.architecture"].asString(); ok {
		info.Architecture = v
	}
	if v, ok := meta["general.name"].asString(); ok {
		info.Name = v
	}
	if v, ok := meta["general.quantized_by"].asString(); ok {
		info.QuantizedBy = v
	} else if v, ok := meta["general.file_type"].asUint(); ok {
		if name := fileTypeName(v); name != "" {
			info.QuantizedBy = name
		}
	}
	if v, ok := meta["tokenizer.ggml.model"].asString(); ok {
		info.TokenizerModel = v
	}
	if v, ok := meta["tokenizer.ggml.pre"].asString(); ok {
		info.TokenizerPre = v
	}
	if v, ok := meta["tokenizer.ggml.tokens"].asArrayLen(); ok {
		info.VocabSize = int(v)
	}
	if v, ok := meta["tokenizer.ggml.merges"].asArrayLen(); ok && info.VocabSize == 0 {
		// BPE merge count is a usable fallback for vocab when the
		// token list is missing.
		info.VocabSize = int(v)
	}

	// Architecture-prefixed keys: only honor them if we know the arch,
	// to avoid matching keys from a different model.
	arch := info.Architecture
	if arch != "" {
		prefix := arch + "."
		for k, v := range meta {
			if !strings.HasPrefix(k, prefix) {
				continue
			}
			suffix := k[len(prefix):]
			applyArchMeta(suffix, v, info)
		}
	}

	info.TokenizerHash = computeTokenizerHash(info.TokenizerModel, info.VocabSize)
}

// applyArchMeta maps a single architecture-prefixed key (with the "arch."
// prefix already stripped) to the corresponding Info field. Adding support
// for a new field is a one-line change here.
func applyArchMeta(suffix string, val ggufScalar, info *Info) {
	switch suffix {
	// Block-level layout
	case "block_count":
		if v, ok := val.asUint(); ok {
			info.BlockCount = int(v)
		}
	case "context_length":
		if v, ok := val.asUint(); ok {
			info.ContextLength = int(v)
		}
	case "embedding_length":
		if v, ok := val.asUint(); ok {
			info.EmbeddingLength = int(v)
		}
	case "feed_forward_length":
		if v, ok := val.asUint(); ok {
			info.FeedForwardLength = int(v)
		}
	case "leading_dense_block_count":
		if v, ok := val.asUint(); ok {
			info.LeadingDense = int(v)
		}
	case "nextn_predict_layers":
		if v, ok := val.asUint(); ok {
			info.NextNPredictLayers = int(v)
		}

	// Attention
	case "attention.head_count_kv":
		if v, ok := val.asUint(); ok {
			info.HeadCountKV = int(v)
		}
	case "attention.key_length":
		if v, ok := val.asUint(); ok {
			info.KeyLength = int(v)
		}
	case "attention.value_length":
		if v, ok := val.asUint(); ok {
			info.ValueLength = int(v)
		}
	case "attention.sliding_window":
		if v, ok := val.asUint(); ok {
			info.SlidingWindow = int(v)
		}
	case "attention.kv_lora_rank":
		if v, ok := val.asUint(); ok {
			info.KVLoraRank = int(v)
		}
	case "attention.q_lora_rank":
		if v, ok := val.asUint(); ok {
			info.QLoraRank = int(v)
		}
	case "attention.key_length_mla":
		if v, ok := val.asUint(); ok {
			info.KeyLengthMLA = int(v)
		}
	case "attention.value_length_mla":
		if v, ok := val.asUint(); ok {
			info.ValueLengthMLA = int(v)
		}

	// RoPE
	case "rope.dimension_count":
		if v, ok := val.asUint(); ok {
			info.NRot = int(v)
		}

	// MoE
	case "expert_count":
		if v, ok := val.asUint(); ok {
			info.Experts = int(v)
		}
	case "expert_used_count":
		if v, ok := val.asUint(); ok {
			info.ExpertUsed = int(v)
		}
	case "expert_shared_count":
		if v, ok := val.asUint(); ok {
			info.ExpSharedFF = int(v)
			if v > 0 {
				info.HasShexp = true
			}
		}
	case "expert_feed_forward_length":
		if v, ok := val.asUint(); ok {
			info.ExpFF = int(v)
		}
	case "fused_experts":
		if v, ok := val.asUint(); ok {
			info.Fused = int(v)
		}

	// SSM / hybrid
	case "ssm.conv_kernel", "ssm.state_size", "ssm.inner_size":
		// Multiple keys may map to the same field; first one wins.
		if info.SSM == 0 {
			if v, ok := val.asUint(); ok {
				info.SSM = int(v)
			}
		}
	case "ssm.full_attention_interval",
		"ssm.full_attention_block_count",
		"attention.full_attention_interval":
		if v, ok := val.asUint(); ok {
			info.FullAttnInterval = int(v)
		}
	}
}

// ============================================================================
// Public API
// ============================================================================

// Parse reads a GGUF file or the first shard of a split GGUF and returns its
// parsed metadata and tensor accounting.
//
// The path may point to:
//   - a standalone .gguf file, or
//   - any shard of a multi-shard split (e.g. model-00003-of-00010.gguf). All
//     sibling shards are auto-discovered, their tensors are aggregated, and
//     byte spans are computed against each shard's own file size.
//
// Parse does not load tensor payloads; only the header KV table and tensor
// info records are read. Errors include the file path of the offending shard
// when something goes wrong mid-stream.
func Parse(path string) (*Info, error) {
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("gguf: model file path is empty")
	}
	fi, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: stat %q: %w", path, err)
	}
	if fi.IsDir() {
		return nil, fmt.Errorf("gguf: %q is a directory, expected a file", path)
	}

	info := &Info{Basename: filepath.Base(path)}

	shards, err := collectShards(path)
	if err != nil {
		return nil, err
	}

	// All shards have identical metadata; canonical is the first.
	applyMetadataToInfo(shards[0].metadata, info)

	// Alignment defaults to 32 (the spec default) and is overridden by
	// general.alignment if present.
	alignment := uint32(32)
	if v, ok := shards[0].metadata["general.alignment"].asUint(); ok && v > 0 && v <= 1<<20 {
		alignment = uint32(v)
	}

	calculateSpans(shards, alignment, info)

	// MoE is detected from metadata only, never from filename heuristics.
	info.IsMoE = info.Experts > 0 || info.Fused > 0

	info.tensors = aggregateTensors(shards)
	return info, nil
}

// EstimateParams returns a parameter count. When the GGUF was parsed with
// tensor headers (the normal case), the count is derived from the product of
// every tensor's dimensions — accurate for any architecture including MoE,
// MLA, and SSM hybrids. It falls back to a metadata-based estimate for
// vanilla transformers when no tensor information is present.
//
// The count excludes any biases and per-tensor scales; it is an upper bound
// for any model whose tensors follow the conventional "weight" naming.
func (i *Info) EstimateParams() int64 {
	if len(i.tensors) > 0 {
		return estimateFromTensors(i.tensors)
	}
	return estimateFromMetadata(*i)
}

// estimateFromTensors sums the product of dimensions for every tensor.
// For a tied-embedding model, token_embd and output are counted twice
// (which is correct, since both tables exist on disk).
func estimateFromTensors(tensors []tensorInfo) int64 {
	var total int64
	for _, t := range tensors {
		var n int64 = 1
		for d := uint32(0); d < t.nDims; d++ {
			if t.dims[d] == 0 {
				n = 0
				break
			}
			n *= int64(t.dims[d])
		}
		total += n
	}
	return total
}

// estimateFromMetadata approximates a vanilla transformer parameter count.
// It is used only as a fallback when no tensor info is available (e.g.
// when Info is constructed by hand or a parse failure truncated the tensor
// table). The formula is the standard self-attention + FFN block count.
func estimateFromMetadata(i Info) int64 {
	embed := int64(i.EmbeddingLength)
	layers := int64(i.BlockCount)
	ffn := int64(i.FeedForwardLength)
	vocab := int64(i.VocabSize)

	if embed <= 0 || layers <= 0 {
		return 0
	}

	var p int64
	if vocab > 0 {
		p += 2 * vocab * embed // token_embd + (assumed) untied output
	}
	p += layers * (4*embed*embed + 3*embed*ffn)

	if i.IsMoE && i.Experts > 1 && ffn > 0 {
		// Routed experts (shared expert is counted in the FFN term above).
		p += layers * int64(i.Experts-1) * 3 * embed * ffn
	}
	return p
}

// ============================================================================
// Misc helpers
// ============================================================================

// fileTypeName maps a GGUF general.file_type enum to a human-readable
// quantization name. The mapping follows llama.cpp's ggml_type enums.
func fileTypeName(t uint64) string {
	switch t {
	case 0:
		return "F32"
	case 1:
		return "F16"
	case 2:
		return "Q4_0"
	case 3:
		return "Q4_1"
	case 6:
		return "Q5_0"
	case 7:
		return "Q5_1"
	case 8:
		return "Q8_0"
	case 9:
		return "Q8_1"
	case 10:
		return "Q2_K"
	case 11:
		return "Q3_K_S"
	case 12:
		return "Q3_K_M"
	case 13:
		return "Q3_K_L"
	case 14:
		return "Q4_K_S"
	case 15:
		return "Q4_K_M"
	case 16:
		return "Q5_K_S"
	case 17:
		return "Q5_K_M"
	case 18:
		return "Q6_K"
	case 19:
		return "Q8_K"
	case 20:
		return "IQ2_XXS"
	case 21:
		return "IQ2_XS"
	case 22:
		return "IQ3_XXS"
	case 23:
		return "IQ1_S"
	case 24:
		return "IQ4_NL"
	case 25:
		return "IQ3_S"
	case 26:
		return "IQ2_S"
	case 27:
		return "IQ4_XS"
	case 28:
		return "IQ1_M"
	case 29:
		return "BF16"
	case 30:
		return "TQ5_0"
	case 31:
		return "TQ6_0"
	default:
		return ""
	}
}

// computeTokenizerHash returns a stable short hash of (model, vocab_size).
// The full tokenizer hash would require digesting every token string, which
// can be megabytes; this fingerprint is enough to tell two tokenizers apart
// without paying that cost.
func computeTokenizerHash(model string, vocab int) string {
	h := sha256.Sum256([]byte(fmt.Sprintf("%s\x00%d", model, vocab)))
	return hex.EncodeToString(h[:8])
}