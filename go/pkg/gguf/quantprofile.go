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
const QuantProfileVersion = 1

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
// and the bytes those elements occupy by type. File names mislead here: a
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

// ggmlTypeSize is (elements per block, bytes per block) per ggml type id.
// Ids 0-42 are gguf-py's GGML_QUANT_SIZES (llama.cpp gguf-py/gguf/constants.py,
// mirroring ggml.h block structs); 137+ are ik_llama.cpp quants, taken from
// parse_gguf.py's table. parse_gguf.py's own entries for ids 9-11 and 24-31 do
// not match upstream; it sizes tensors by their on-disk spans first, so those
// entries are only its fallback and are not reused here.
var ggmlTypeSize = map[uint32][2]int64{
	0: {1, 4}, 1: {1, 2}, // F32, F16
	2: {32, 18}, 3: {32, 20}, 6: {32, 22}, 7: {32, 24}, // Q4_0, Q4_1, Q5_0, Q5_1
	8: {32, 34}, 9: {32, 40}, // Q8_0, Q8_1
	10: {256, 84}, 11: {256, 110}, 12: {256, 144}, // Q2_K, Q3_K, Q4_K
	13: {256, 176}, 14: {256, 210}, 15: {256, 292}, // Q5_K, Q6_K, Q8_K
	16: {256, 66}, 17: {256, 74}, 18: {256, 98}, 19: {256, 50}, // IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S
	20: {32, 18}, 21: {256, 110}, 22: {256, 82}, 23: {256, 136}, // IQ4_NL, IQ3_S, IQ2_S, IQ4_XS
	24: {1, 1}, 25: {1, 2}, 26: {1, 4}, 27: {1, 8}, 28: {1, 8}, // I8, I16, I32, I64, F64
	29: {256, 56}, 30: {1, 2}, // IQ1_M, BF16
	34: {256, 54}, 35: {256, 66}, // TQ1_0, TQ2_0
	39: {32, 17}, 40: {64, 36}, 41: {128, 18}, 42: {64, 18}, // MXFP4, NVFP4, Q1_0, Q2_0
	// ik_llama.cpp
	137: {256, 76}, 138: {256, 110}, 139: {256, 144}, 140: {256, 176}, 141: {256, 212}, // IQ2_K..IQ6_K
	144: {256, 136}, 145: {256, 70}, 146: {256, 128}, 152: {256, 168}, // IQ4_KS, IQ2_KS, IQ4_KSS, IQ5_KS
	153: {256, 68}, 154: {256, 100}, 155: {256, 128}, 156: {256, 102}, 157: {256, 86}, // IQ2_KT, IQ3_KT, IQ4_KT, IQ3_KS, IQ2_KL
	337: {256, 76}, 338: {256, 110}, 339: {256, 144}, 340: {256, 176}, 344: {256, 136}, 352: {256, 168}, // _R4
}

// imatrixRequiredTypes are the tensor types mainline llama-quantize produces
// only with an importance matrix (see ImatrixTypes).
var imatrixRequiredTypes = map[uint32]bool{16: true, 17: true, 18: true, 19: true, 22: true, 29: true}

// ErrUnknownTensorType reports a tensor whose type has no size in
// ggmlTypeSize. The file then has no profile: guessing a width would turn an
// estimate into a fake exact number.
var ErrUnknownTensorType = errors.New("unknown tensor type")

// Bounds for lengths read from the file. A real header has at most tens of
// thousands of tensors; ggml tensors have at most 4 dimensions.
const (
	maxProfileTensors = 1 << 20
	maxProfileKVs     = 1 << 20
	maxTensorDims     = 8
	maxTensorElements = 1 << 50
)

// ReadQuantProfile reads one GGUF file's metadata and tensor table from r and
// stops before the tensor data, so a bounded prefix of a remote file is
// enough. Every length is bounded before it sizes a read or a skip.
func ReadQuantProfile(r io.Reader) (QuantProfile, error) {
	p := QuantProfile{Version: QuantProfileVersion, FileType: -1, Shards: 1}
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
		return p, fmt.Errorf("not a GGUF file")
	}
	version, err := readU32()
	if err != nil {
		return p, err
	}
	if version < 2 {
		return p, fmt.Errorf("GGUF version %d is not supported", version)
	}
	tensorCount, err := readU64()
	if err != nil {
		return p, err
	}
	kvCount, err := readU64()
	if err != nil {
		return p, err
	}
	if tensorCount > maxProfileTensors || kvCount > maxProfileKVs {
		return p, fmt.Errorf("header declares %d tensors and %d metadata keys", tensorCount, kvCount)
	}

	for i := uint64(0); i < kvCount; i++ {
		key, err := readName("metadata key")
		if err != nil {
			return p, err
		}
		valType, err := readU32()
		if err != nil {
			return p, err
		}
		switch {
		case strings.HasPrefix(key, "quantize.imatrix."):
			p.Imatrix = ImatrixKeys
		case key == "general.file_type" && valType != 8 && valType != 9:
			v, err := readSamplingScalar(r, valType)
			if err != nil {
				return p, fmt.Errorf("metadata %q: %w", key, err)
			}
			if n, err := strconv.Atoi(v); err == nil && n >= 0 {
				p.FileType = n
			}
			continue
		}
		if err := skipGGUFValue(r, valType, 0); err != nil {
			return p, fmt.Errorf("metadata %q: %w", key, err)
		}
	}

	typesImplyImatrix := false
	for i := uint64(0); i < tensorCount; i++ {
		name, err := readName("tensor name")
		if err != nil {
			return p, err
		}
		nDims, err := readU32()
		if err != nil {
			return p, err
		}
		if nDims == 0 || nDims > maxTensorDims {
			return p, fmt.Errorf("tensor %q has %d dimensions", name, nDims)
		}
		elements := uint64(1)
		for d := uint32(0); d < nDims; d++ {
			dim, err := readU64()
			if err != nil {
				return p, err
			}
			if dim == 0 {
				elements = 0
				continue
			}
			if dim > maxTensorElements || elements > maxTensorElements/dim {
				return p, fmt.Errorf("tensor %q is larger than any real model", name)
			}
			elements *= dim
		}
		ttype, err := readU32()
		if err != nil {
			return p, err
		}
		if _, err := readU64(); err != nil { // data offset
			return p, err
		}
		size, ok := ggmlTypeSize[ttype]
		if !ok {
			return p, fmt.Errorf("tensor %q: %w %d", name, ErrUnknownTensorType, ttype)
		}
		bytes := int64((elements+uint64(size[0])-1)/uint64(size[0])) * size[1]
		p.Params += int64(elements)
		p.Bytes += bytes
		if routedExpertTensor(name) {
			p.ExpertParams += int64(elements)
			p.ExpertBytes += bytes
		}
		if imatrixRequiredTypes[ttype] && name != "token_embd.weight" && name != "output.weight" {
			typesImplyImatrix = true
		}
	}
	if p.Imatrix == "" && typesImplyImatrix {
		p.Imatrix = ImatrixTypes
	}
	if p.Params <= 0 {
		return p, fmt.Errorf("no tensors")
	}
	return p, nil
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
	// A vocabulary is hundreds of thousands of tiny values; buffer them.
	return ReadQuantProfile(bufio.NewReaderSize(f, 1<<20))
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
