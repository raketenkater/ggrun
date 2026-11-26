package recommend

import (
	"fmt"
	"strings"

	"github.com/raketenkater/ggrun/pkg/gguf"
)

// QuantProfile is what the recommender knows about one catalog file's real
// precision. Quant names mislead: files mix tensor types ("Q4_K" files run
// 4.5-4.9 bits per weight), and a "BF16" upload of a model with 4-bit experts
// stores about 4.5. The profile measures instead of reading the name.
type QuantProfile struct {
	// BPW is the effective bits per weight of the whole file. ExpertBPW and
	// NonExpertBPW split it for a MoE when the tensor table was read.
	BPW          float64 `json:"bpw,omitempty"`
	ExpertBPW    float64 `json:"expert_bpw,omitempty"`
	NonExpertBPW float64 `json:"non_expert_bpw,omitempty"`
	// Source says how BPW was derived: QuantSourceHeader, QuantSourceSize,
	// QuantSourceNominal, or "" when it is unknown.
	Source string `json:"source,omitempty"`
	// Imatrix is the evidence that an importance matrix (or a dynamic quant
	// recipe built on one) shaped the file; "" means unknown, never "no".
	Imatrix string `json:"imatrix,omitempty"`
	// FileKey identifies the file (repository, revision, quant and bytes) for
	// per-file results such as a later local KL-divergence measurement.
	FileKey string `json:"file_key,omitempty"`
}

const (
	// QuantSourceHeader: exact, from a downloaded file's tensor table.
	QuantSourceHeader = "header"
	// QuantSourceSize: the file's size over the catalog's parameter count
	// (Artificial Analysis or the Hugging Face GGUF summary). Exact up to the
	// header's few megabytes.
	QuantSourceSize = "size"
	// QuantSourceNominal: the parameter count was itself inferred from quant
	// sizes and nominal widths, so BPW only orders this model's quants.
	QuantSourceNominal = "nominal"
)

// Imatrix evidence beyond the header's (gguf.ImatrixKeys, gguf.ImatrixTypes).
const (
	// ImatrixDynamic: an Unsloth dynamic ("UD-") quant, built with an
	// importance matrix and per-tensor precision choices.
	ImatrixDynamic = "dynamic"
	// ImatrixName: the quant name is a type mainline llama-quantize produces
	// only with an importance matrix.
	ImatrixName = "name"
)

// Imatrixed reports any evidence of importance-matrix or dynamic quantization.
func (p QuantProfile) Imatrixed() bool { return p.Imatrix != "" }

// localQuantProfile reads the profile of a catalog file that is already on
// disk. Only the launcher knows the model directory and the header cache, so
// it supplies the lookup; nil means no local files are consulted.
var localQuantProfile func(c Candidate, q QuantOption) (gguf.QuantProfile, bool)

// SetLocalQuantProfile installs that lookup. nil disables it.
func SetLocalQuantProfile(fn func(c Candidate, q QuantOption) (gguf.QuantProfile, bool)) {
	localQuantProfile = fn
}

// bpwRange bounds a size-derived width: outside it the size or the parameter
// count is wrong (a projector-sized file, a missing shard), not the quant.
const (
	minPlausibleBPW = 1.0
	maxPlausibleBPW = 34.0
)

// quantProfileFor derives one file's profile: the local header when the file
// is on disk, otherwise its catalog size over the model's parameter count.
func quantProfileFor(c Candidate, q QuantOption) QuantProfile {
	p := QuantProfile{FileKey: quantFileKey(c, q), Imatrix: imatrixFromName(q)}
	if localQuantProfile != nil {
		if g, ok := localQuantProfile(c, q); ok && g.BPW() > 0 {
			p.BPW, p.ExpertBPW, p.NonExpertBPW = g.BPW(), g.ExpertBPW(), g.NonExpertBPW()
			p.Source = QuantSourceHeader
			if g.Imatrix != "" {
				p.Imatrix = g.Imatrix
			}
			return p
		}
	}
	if c.TotalParamsB <= 0 {
		return p
	}
	bytes := float64(q.SizeBytes)
	if bytes <= 0 {
		bytes = q.SizeGB * (1 << 30) // size_gb is GiB (size_bytes / 1024^3)
	}
	if bytes <= 0 {
		return p
	}
	bpw := bytes * 8 / (c.TotalParamsB * 1e9)
	if bpw < minPlausibleBPW || bpw > maxPlausibleBPW {
		return p
	}
	p.BPW = bpw
	p.Source = QuantSourceSize
	if c.paramsFromSizes {
		p.Source = QuantSourceNominal
	}
	return p
}

// quantFileKey names a catalog file independently of where it is stored.
func quantFileKey(c Candidate, q QuantOption) string {
	if strings.TrimSpace(c.Repo) == "" || strings.TrimSpace(q.Name) == "" {
		return ""
	}
	rev := c.Revision
	if rev == "" {
		rev = "unpinned"
	}
	return fmt.Sprintf("%s@%s/%s:%d", c.Repo, rev, q.Name, q.SizeBytes)
}

// imatrixRequiredNames are the quant names whose main tensor types mainline
// llama-quantize refuses to produce without an importance matrix
// (src/llama-quant.cpp tensor_requires_imatrix: IQ1_S, IQ1_M, IQ2_XXS,
// IQ2_XS, IQ2_S, IQ3_XXS; IQ2_M files use IQ2_S tensors).
var imatrixRequiredNames = []string{"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS"}

func imatrixFromName(q QuantOption) string {
	name := strings.ToUpper(q.Name)
	if q.Dynamic || strings.HasPrefix(name, "UD-") {
		return ImatrixDynamic
	}
	for _, token := range strings.FieldsFunc(name, func(r rune) bool { return r == '-' || r == '.' || r == ' ' }) {
		for _, req := range imatrixRequiredNames {
			if token == req || strings.HasSuffix(token, "_"+req) {
				return ImatrixName
			}
		}
	}
	return ""
}
