package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	modelstore "github.com/raketenkater/ggrun/pkg/models"
)

// QuantProfileVersion changes when QuantProfile's fields or their meaning
// change; it is part of the header-cache value name, so a stored profile of an
// older shape is recomputed rather than misread.
const QuantProfileVersion = 2

// Imatrix evidence, strongest first. A file without either is "unknown", never
// "no": a quantizer can drop the metadata keys, and a K-quant made with an
// importance matrix stores no other trace of it.
const (
	// ImatrixKeys: the header carries quantize.imatrix.* keys, which
	// llama-quantize writes when it is given an importance matrix
	// (tools/quantize/quantize.cpp: LLM_KV_QUANTIZE_IMATRIX_FILE, _DATASET,
	// _N_ENTRIES, _N_CHUNKS).
	ImatrixKeys = "keys"
	// ImatrixTypes: a weight tensor has a type mainline llama.cpp refuses to
	// produce without an importance matrix (src/llama-quant.cpp
	// tensor_requires_imatrix: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS;
	// token embeddings and the output head are exempt).
	ImatrixTypes = "types"
)

// QuantProfile is the real precision of one GGUF file (or, merged, of every
// shard of a split model), read from its tensor table: exact element counts
// and the bytes those elements occupy on disk. File names mislead here: a
// "Q4_K_M" mixes Q4_K, Q6_K and F32 tensors, a "BF16" of a model with MXFP4
// experts stores 4-bit experts.
type QuantProfile struct {
	Version int `json:"version"`
	// Params and Bytes count every tensor. ExpertParams and ExpertBytes are
	// the routed experts (the same test as parse_gguf.py: "_exps.",
	// "_chexps." or ".experts." in the name); shared experts, attention,
	// embeddings and norms are non-expert because every token reads them.
	Params       int64 `json:"params"`
	Bytes        int64 `json:"bytes"`
	ExpertParams int64 `json:"expert_params,omitempty"`
	ExpertBytes  int64 `json:"expert_bytes,omitempty"`
	// Imatrix is ImatrixKeys, ImatrixTypes or "" (unknown).
	Imatrix string `json:"imatrix,omitempty"`
	// FileType is general.file_type (llama_ftype), -1 when absent.
	FileType int `json:"file_type"`
	Shards   int `json:"shards"`
}

// BPW is the parameter-weighted effective bits per weight of the whole file.
func (p QuantProfile) BPW() float64 { return bitsPer(p.Bytes, p.Params) }

// ExpertBPW is the effective bits per weight of the routed experts (0 for a
// dense model).
func (p QuantProfile) ExpertBPW() float64 { return bitsPer(p.ExpertBytes, p.ExpertParams) }

// NonExpertBPW is the effective bits per weight of everything else.
func (p QuantProfile) NonExpertBPW() float64 {
	return bitsPer(p.Bytes-p.ExpertBytes, p.Params-p.ExpertParams)
}

func bitsPer(bytes, params int64) float64 {
	if bytes <= 0 || params <= 0 {
		return 0
	}
	return float64(bytes) * 8 / float64(params)
}

// merge adds one shard's profile.
func (p QuantProfile) merge(o QuantProfile) QuantProfile {
	p.Params += o.Params
	p.Bytes += o.Bytes
	p.ExpertParams += o.ExpertParams
	p.ExpertBytes += o.ExpertBytes
	p.Shards += o.Shards
	if p.Imatrix != ImatrixKeys && o.Imatrix != "" {
		p.Imatrix = o.Imatrix
	}
	if p.FileType < 0 {
		p.FileType = o.FileType
	}
	return p
}

// imatrixRequiredTypes are the tensor types mainline llama-quantize produces
// only with an importance matrix (see ImatrixTypes). These ids are below 40,
// where the backends agree on what an id means.
var imatrixRequiredTypes = map[uint32]bool{16: true, 17: true, 18: true, 19: true, 22: true, 29: true}

var (
	// ErrTensorDataIncomplete: the tensor data does not fit the file (an
	// interrupted download, a cut copy). Such a file has no profile: its
	// offsets would give a nonsense width (a 258 MB part of a 27B model reads
	// as 0.07 bits per weight).
	ErrTensorDataIncomplete = errors.New("tensor data incomplete")
	// ErrTensorLayout: a tensor of a type with a known layout occupies fewer
	// bytes than that layout needs, so the id means something else in the
	// backend that wrote the file, or the file is damaged. No profile.
	ErrTensorLayout = errors.New("tensor smaller than its type's layout")
)

// Bounds for lengths read from the file. A real header has at most tens of
// thousands of tensors; ggml tensors have at most 4 dimensions (maxTensorDims
// is shared with ExpectedSize).
const (
	maxProfileTensors = 1 << 20
	maxProfileKVs     = 1 << 20
	maxTensorElements = 1 << 50
)

type tensorEntry struct {
	name   string
	elems  uint64
	ttype  uint32
	offset uint64
}

// ReadQuantProfile reads one GGUF file's metadata and tensor table from r;
// fileSize is the size of the whole file. Each tensor's bytes are its on-disk
// span: the next tensor's offset minus its own, the last running to the end of
// the file, after a data section that starts at the header end rounded up to
// general.alignment (default 32). Spans are exact for every type, including
// ids whose meaning differs between backends (ids 40 and up: upstream Q2_0 is
// a fork's F8_E4M3_B128), and include the alignment padding a loader maps. The
// type table (ggmlTypeSize, shared with ExpectedSize) only cross-checks: a
// known type in a smaller span is ErrTensorLayout, and data that does not fit
// the file is ErrTensorDataIncomplete. A header without tensors (shard 00001
// of many split models) is valid and contributes only its metadata. Every
// length is bounded before it sizes a read or a skip.
func ReadQuantProfile(src io.Reader, fileSize int64) (QuantProfile, error) {
	p, _, err := readProfileAndTypes(src, fileSize)
	return p, err
}

// ReadTypeHistogram reads one GGUF file's tensor type histogram from r with the
// same walk and the same failures as ReadQuantProfile (see TypeUsage).
func ReadTypeHistogram(src io.Reader, fileSize int64) ([]TypeUsage, error) {
	_, acc, err := readProfileAndTypes(src, fileSize)
	if err != nil {
		return nil, err
	}
	return acc.result()
}

// readProfileAndTypes is the one tensor-table walk behind both.
func readProfileAndTypes(src io.Reader, fileSize int64) (QuantProfile, *typeAcc, error) {
	p := QuantProfile{Version: QuantProfileVersion, FileType: -1, Shards: 1}
	r := &countingReader{r: bufio.NewReaderSize(src, 1<<20)}
	var buf [8]byte
	readFull := func(n int) ([]byte, error) {
		_, err := io.ReadFull(r, buf[:n])
		return buf[:n], err
	}
	readU32 := func() (uint32, error) {
		b, err := readFull(4)
		if err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint32(b), nil
	}
	readU64 := func() (uint64, error) {
		b, err := readFull(8)
		if err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint64(b), nil
	}
	readName := func(what string) (string, error) {
		n, err := readU64()
		if err != nil {
			return "", err
		}
		if n > maxGGUFKeyLen {
			return "", fmt.Errorf("%s length %d exceeds %d", what, n, maxGGUFKeyLen)
		}
		s := make([]byte, n)
		if _, err := io.ReadFull(r, s); err != nil {
			return "", err
		}
		return string(s), nil
	}

	if b, err := readFull(4); err != nil || string(b) != "GGUF" {
		return p, nil, fmt.Errorf("not a GGUF file")
	}
	version, err := readU32()
	if err != nil {
		return p, nil, err
	}
	if version < 2 || version > 3 {
		return p, nil, fmt.Errorf("GGUF version %d is not supported", version)
	}
	tensorCount, err := readU64()
	if err != nil {
		return p, nil, err
	}
	kvCount, err := readU64()
	if err != nil {
		return p, nil, err
	}
	if tensorCount > maxProfileTensors || kvCount > maxProfileKVs {
		return p, nil, fmt.Errorf("header declares %d tensors and %d metadata keys", tensorCount, kvCount)
	}

	align := uint64(32)
	for i := uint64(0); i < kvCount; i++ {
		key, err := readName("metadata key")
		if err != nil {
			return p, nil, err
		}
		valType, err := readU32()
		if err != nil {
			return p, nil, err
		}
		switch {
		case strings.HasPrefix(key, "quantize.imatrix."):
			p.Imatrix = ImatrixKeys
		case key == "general.alignment" && valType == 4:
			v, err := readU32()
			if err != nil {
				return p, nil, err
			}
			if v > 0 {
				align = uint64(v)
			}
			continue
		case key == "general.file_type" && valType != 8 && valType != 9:
			v, err := readSamplingScalar(r, valType)
			if err != nil {
				return p, nil, fmt.Errorf("metadata %q: %w", key, err)
			}
			if n, err := strconv.Atoi(v); err == nil && n >= 0 {
				p.FileType = n
			}
			continue
		}
		if err := skipGGUFValue(r, valType, 0); err != nil {
			return p, nil, fmt.Errorf("metadata %q: %w", key, err)
		}
	}

	tensors := make([]tensorEntry, 0, min(tensorCount, 1<<14))
	acc := newTypeAcc()
	for i := uint64(0); i < tensorCount; i++ {
		name, err := readName("tensor name")
		if err != nil {
			return p, nil, err
		}
		nDims, err := readU32()
		if err != nil {
			return p, nil, err
		}
		if nDims == 0 || nDims > maxTensorDims {
			return p, nil, fmt.Errorf("tensor %q has %d dimensions", name, nDims)
		}
		elements := uint64(1)
		for d := uint32(0); d < nDims; d++ {
			dim, err := readU64()
			if err != nil {
				return p, nil, err
			}
			if dim == 0 {
				elements = 0
				continue
			}
			if dim > maxTensorElements || elements > maxTensorElements/dim {
				return p, nil, fmt.Errorf("tensor %q is larger than any real model", name)
			}
			elements *= dim
		}
		ttype, err := readU32()
		if err != nil {
			return p, nil, err
		}
		offset, err := readU64()
		if err != nil {
			return p, nil, err
		}
		acc.entry(ttype, name, elements)
		if elements > 0 { // an empty tensor owns no bytes and may share an offset
			tensors = append(tensors, tensorEntry{name, elements, ttype, offset})
		}
	}
	if len(tensors) == 0 {
		return p, acc, nil
	}

	dataStart := (uint64(r.n) + align - 1) / align * align
	if fileSize <= 0 || uint64(fileSize) <= dataStart {
		return p, nil, ErrTensorDataIncomplete
	}
	dataSize := uint64(fileSize) - dataStart
	order := offsetOrder(tensors)
	if tensors[order[0]].offset != 0 {
		return p, nil, fmt.Errorf("tensor data does not start at offset 0")
	}
	typesImplyImatrix := false
	for k, i := range order {
		t := tensors[i]
		end := dataSize
		if k+1 < len(order) {
			end = tensors[order[k+1]].offset
		}
		if t.offset >= dataSize || end > dataSize || end <= t.offset {
			return p, nil, fmt.Errorf("tensor %q: %w", t.name, ErrTensorDataIncomplete)
		}
		span := end - t.offset
		acc.addBytes(t.ttype, span)
		if need, known := layoutBytes(t.ttype, t.elems); known {
			if span < need {
				return p, nil, fmt.Errorf("tensor %q type %d: %w (%d of %d bytes)", t.name, t.ttype, ErrTensorLayout, span, need)
			}
			if imatrixRequiredTypes[t.ttype] && t.name != "token_embd.weight" && t.name != "output.weight" {
				typesImplyImatrix = true
			}
		}
		p.Params += int64(t.elems)
		p.Bytes += int64(span)
		if routedExpertTensor(t.name) {
			p.ExpertParams += int64(t.elems)
			p.ExpertBytes += int64(span)
		}
	}
	if p.Imatrix == "" && typesImplyImatrix {
		p.Imatrix = ImatrixTypes
	}
	return p, acc, nil
}

// routedExpertTensor matches parse_gguf.py's routed-expert test.
func routedExpertTensor(name string) bool {
	return strings.Contains(name, "_exps.") || strings.Contains(name, "_chexps.") || strings.Contains(name, ".experts.")
}

// readQuantProfileFile reads one file's profile.
func readQuantProfileFile(path string) (QuantProfile, error) {
	f, err := os.Open(path)
	if err != nil {
		return QuantProfile{}, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return QuantProfile{}, err
	}
	return ReadQuantProfile(f, st.Size())
}

// ReadQuantProfileModel is the profile of a model: every shard of a split
// GGUF (path names shard 00001) or the single file.
func ReadQuantProfileModel(path string) (QuantProfile, error) {
	return quantProfileModel(path, readQuantProfileFile)
}

func quantProfileModel(path string, read func(string) (QuantProfile, error)) (QuantProfile, error) {
	files, _, err := modelstore.ResolveGGUFShardFiles(path)
	if err != nil {
		return QuantProfile{}, err
	}
	var out QuantProfile
	for i, file := range files {
		p, err := read(file)
		if err != nil {
			return QuantProfile{}, err
		}
		if i == 0 {
			out = p
		} else {
			out = out.merge(p)
		}
	}
	if out.Params <= 0 {
		return QuantProfile{}, fmt.Errorf("no tensors in %s", path)
	}
	return out, nil
}

// quantProfileCacheName keys a stored profile by its shape version.
var quantProfileCacheName = "quant-profile-v" + strconv.Itoa(QuantProfileVersion)

// QuantProfile is ReadQuantProfileModel, remembered per shard file (keyed by
// each file's size and modification time), so a model listing does not
// re-read tensor tables on every start.
func (c *HeaderCache) QuantProfile(path string) (QuantProfile, error) {
	return quantProfileModel(path, func(file string) (QuantProfile, error) {
		var readErr error
		p, ok := CachedValue(c, file, quantProfileCacheName, func() (QuantProfile, bool) {
			p, err := readQuantProfileFile(file)
			readErr = err
			return p, err == nil
		})
		if !ok {
			if readErr == nil {
				readErr = fmt.Errorf("no quant profile for %s", file)
			}
			return QuantProfile{}, readErr
		}
		return p, nil
	})
}
