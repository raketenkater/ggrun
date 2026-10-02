package placement

import (
	"fmt"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Where host-resident expert weights execute.
//
// Placing routed experts in host memory (-ot exps=CPU, --n-cpu-moe) decides
// where they are STORED, not where they RUN. Mainline llama.cpp's scheduler
// (ggml/src/ggml-backend.cpp, reviewed at 6b790a9c2) reassigns an operation
// whose weights sit in a host buffer to the first higher-priority backend that
// supports it and whose offload_op accepts its batch; the CUDA backend accepts
// a batch of at least GGML_OP_OFFLOAD_MIN_BATCH, default 32
// (ggml/src/ggml-cuda/ggml-cuda.cu). For MUL_MAT_ID it then copies only the
// experts the batch routed to, on every graph execution -- there is no
// persistent residency. Single-token decode stays below the threshold and
// executes on CPU; a prefill microbatch at or above it streams its touched
// experts over the staging GPU's host link.
//
// That policy belongs to a backend family and its effective settings, not to
// every fork, device type or build. What a --help flag establishes is that
// the switch exists and what it defaults to; it does not prove the threshold,
// operation support for a quantisation, or scheduler order. Hence the
// confidence field, and an unknown mode wherever the family is not reviewed.

type HostExpertExecutionMode string

const (
	HostExpertExecUnknown HostExpertExecutionMode = "unknown"
	HostExpertExecCPU     HostExpertExecutionMode = "cpu"
	HostExpertExecStaged  HostExpertExecutionMode = "gpu-staged"
)

// mainlineCUDAOpOffloadMinBatch is the CUDA backend's default offload
// threshold in the reviewed mainline source.
const mainlineCUDAOpOffloadMinBatch = 32

// HostExpertExecution is the resolved backend policy for host-weight ops.
type HostExpertExecution struct {
	Mode HostExpertExecutionMode `json:"mode"`
	// MinBatch is the smallest per-op batch that is staged; smaller batches
	// (decode, a short final prefill chunk) execute on CPU.
	MinBatch int `json:"min_batch,omitempty"`
	// StagingGPU is the physical index of the first enumerated device, which
	// the scheduler scans first. -1 when nothing is staged or it is unknown.
	StagingGPU int    `json:"staging_gpu"`
	Confidence string `json:"confidence"` // effective-setting, family-default, unknown
	Evidence   string `json:"evidence"`
}

// Fingerprint identifies the policy for performance-evidence scoping.
func (h HostExpertExecution) Fingerprint() string {
	return fmt.Sprintf("%s/min=%d/gpu=%d/%s", h.Mode, h.MinBatch, h.StagingGPU, h.Confidence)
}

// StagedAt reports whether an op over batch tokens is staged onto the GPU.
func (h HostExpertExecution) StagedAt(batch int) bool {
	return h.Mode == HostExpertExecStaged && batch >= h.MinBatch
}

// cAtoi mirrors C atoi, which is how the backend reads the threshold: leading
// whitespace, an optional sign, then digits; anything else stops the parse.
func cAtoi(s string) int {
	s = strings.TrimLeft(s, " \t\n\r\v\f")
	sign, n := 1, 0
	if s != "" && (s[0] == '-' || s[0] == '+') {
		if s[0] == '-' {
			sign = -1
		}
		s = s[1:]
	}
	for _, c := range s {
		if c < '0' || c > '9' || n > 1<<30 {
			break
		}
		n = n*10 + int(c-'0')
	}
	return sign * n
}

// ResolveHostExpertExecution derives the policy from the effective launch
// settings and the backend family. The GPU list must be in the backend's
// enumeration order (ggrun launches with PCI bus ordering and a sorted
// visible-device subset).
func ResolveHostExpertExecution(caps *detect.Capabilities, opts Options) HostExpertExecution {
	out := HostExpertExecution{Mode: HostExpertExecUnknown, StagingGPU: -1, Confidence: "unknown"}
	offload := strings.ToLower(strings.TrimSpace(opts.HostWeightOffload))
	if offload == "off" {
		out.Mode, out.Confidence, out.Evidence = HostExpertExecCPU, "effective-setting", "--no-op-offload"
		return out
	}
	if caps == nil || len(caps.GPUs) == 0 {
		out.Mode, out.Confidence, out.Evidence = HostExpertExecCPU, "effective-setting", "no GPU backend"
		return out
	}
	dialect := strings.ToLower(opts.BackendTag)
	if dialect != "" && dialect != "llama" {
		out.Evidence = "host-weight offload policy not reviewed for backend " + dialect
		return out
	}
	if !strings.Contains(opts.BackendHelp, "--no-op-offload") && !strings.Contains(opts.BackendHelp, "--op-offload") {
		out.Evidence = "backend does not advertise --op-offload"
		return out
	}
	out.Mode, out.StagingGPU = HostExpertExecStaged, caps.GPUs[0].Index
	out.MinBatch, out.Confidence = mainlineCUDAOpOffloadMinBatch, "family-default"
	out.Evidence = fmt.Sprintf("op offload on (%s); CUDA default threshold %d", map[bool]string{true: "--op-offload", false: "backend default"}[offload == "on"], out.MinBatch)
	if opts.OpOffloadMinBatchEnvSet {
		out.MinBatch = cAtoi(opts.OpOffloadMinBatchEnv)
		out.Evidence = fmt.Sprintf("GGML_OP_OFFLOAD_MIN_BATCH=%q", opts.OpOffloadMinBatchEnv)
	}
	return out
}

// HostExpertEstimate is the priced host-expert work of one candidate, in
// seconds per token. Decode and prefill use the same units as DecodeCost.
type HostExpertEstimate struct {
	Execution HostExpertExecution `json:"execution"`
	StoredMB  float64             `json:"stored_mb"`
	// PrefillStaged is true when a full microbatch streams experts to the GPU.
	PrefillStaged bool `json:"prefill_staged"`
	// TouchFraction is the expected fraction of stored experts one microbatch
	// routes to under independent uniform routing -- a sensitivity model, not
	// measured routing, which is correlated and usually touches fewer.
	TouchFraction float64 `json:"touch_fraction"`
	// TransferSecPerToken = touched bytes / (microbatch * copy rate). The copy
	// rate is the lesser of the staging link and host memory bandwidth; a
	// link rate is a ceiling, not observed throughput.
	TransferSecPerToken float64 `json:"transfer_sec_per_token,omitempty"`
	CopyRateMBps        int     `json:"copy_rate_mbps,omitempty"`
	// ComputePriced is false: neither CPU nor GPU expert arithmetic is priced,
	// so a CPU-execution alternative cannot be ranked above staging from this.
	ComputePriced bool `json:"compute_priced"`
}

// stagedExpertTransfer prices one staged microbatch per prompt token. It is
// the only amortisation of the transfer: the result must not be divided by
// another microbatch gain.
func stagedExpertTransfer(caps *detect.Capabilities, exec HostExpertExecution, storedMB, touch float64, ubatch int) (float64, int, bool) {
	if caps == nil || ubatch <= 0 || storedMB <= 0 {
		return 0, 0, false
	}
	link := 0
	for _, gpu := range caps.GPUs {
		if gpu.Index == exec.StagingGPU {
			link = gpu.BandwidthMBps
		}
	}
	if link <= 0 {
		return 0, 0, false
	}
	rate := link
	if host := caps.HostMemoryBandwidthMBps; host > 0 && host < rate {
		rate = host
	}
	return storedMB * touch / (float64(ubatch) * float64(rate)), rate, true
}

// hostExpertAgentCost prices the host-expert terms that differ between two
// ubatch rungs of one MoE plan: prefill (staged transfer per token, or the
// CPU-read prior below the threshold) and decode host reads for cpuLayers of
// host-resident expert layers, weighted as in AgentCost. Everything else in
// the plan is common to the rungs being compared. ok is false when a rate is
// unknown; callers must then keep their previous rule.
func hostExpertAgentCost(caps *detect.Capabilities, model *ModelProfile, exec HostExpertExecution, ubatch, cpuLayers int) (float64, bool) {
	if caps == nil || model == nil || caps.HostMemoryBandwidthMBps <= 0 || ubatch <= 0 {
		return 0, false
	}
	_, moeLayers := moeLayerRange(model)
	routedMB := float64(bytesToMiBCeil(model.ExpertBytes - model.ShexpBytes))
	if moeLayers <= 0 || routedMB <= 0 {
		return 0, false
	}
	cpuMB := routedMB * float64(min(max(0, cpuLayers), moeLayers)) / float64(moeLayers)
	hostBW := float64(caps.HostMemoryBandwidthMBps)
	decode := cpuMB * moeExpertTouchFraction(model, 1) / hostBW
	touch := moeExpertTouchFraction(model, ubatch)
	prefill := cpuMB * touch / hostBW / ubatchPrefillGain(ubatch)
	if exec.StagedAt(ubatch) {
		perToken, _, ok := stagedExpertTransfer(caps, exec, cpuMB, touch, ubatch)
		if !ok {
			return 0, false
		}
		prefill = perToken
	}
	return 0.58*prefill + 0.42*decode, true
}

// StagedPrefillUBatchRaiseCandidates lists larger physical microbatches worth
// measuring for an automatic MoE plan whose host-resident experts are staged
// onto a GPU for prefill: every power of two above the current microbatch up
// to the logical batch, largest first. With the placement unchanged, a staged
// microbatch of B tokens moves touched(B) expert bytes, so the transfer per
// prompt token falls roughly as 1/B even after every expert is touched; there
// is no routing knee to stop at. Memory is the limit, and only the backend's
// exact accounting may decide it. Decode (below the threshold) is unaffected.
//
// It exists because a plan at a small rung never measures larger ones: their
// cold compute estimate at large context is architecture-borrowed and was 3-4x
// high for MiMo-V2.6 (planner: ubatch 256 fits 513k context; backend oracle:
// 723,968 with +3 MiB on the tightest GPU). Only exact backend accounting may
// then admit a rung; this function proposes, it never admits.
func StagedPrefillUBatchRaiseCandidates(caps *detect.Capabilities, opts Options, s *Strategy) []int {
	// Parallel slots share each microbatch: a larger staged prefill batch also
	// delays the other slots' decode tokens, a trade no measurement covers yet.
	if s == nil || !keepsWeightsOnHost(s) || opts.UBatchSizeExplicit || s.Parallel > 1 ||
		s.BatchTuned || s.PerformanceTuned || (s.Draft != nil && s.Draft.Type != DraftNone) {
		return nil
	}
	if !ResolveHostExpertExecution(caps, opts).StagedAt(max(1, s.UBatchSize)) {
		return nil
	}
	var out []int
	for ub := 1; ub <= s.BatchSize; ub *= 2 {
		if ub > s.UBatchSize {
			out = append([]int{ub}, out...)
		}
	}
	return out
}

// keepsWeightsOnHost reports plans whose weights partly live in host memory
// and are therefore staged per microbatch when the backend offloads their
// operations: CPU expert layers of a MoE, or the host layers of a dense model
// that does not fit. Any such plan amortizes the staged bytes over the
// microbatch; nothing here depends on the model family.
func keepsWeightsOnHost(s *Strategy) bool {
	return (s.Type == MoEOffload && s.NCPUMoE > 0) || s.Type == DenseCPUOffload
}

// WithUBatch returns a copy of s at physical microbatch ub. State scoped to
// the old microbatch is dropped: scoped allocation evidence, the resource
// ledger, and the keyed placement-cache path (the raised plan must not be
// written under the old microbatch's key).
func WithUBatch(s *Strategy, model *ModelProfile, ub int) *Strategy {
	if s == nil {
		return nil
	}
	next := cloneStrategy(s)
	next.UBatchSize = ub
	if next.BatchSize < ub {
		next.BatchSize = ub
	}
	next.CheckpointMinStep = checkpointMinStep(model, ub)
	next.ContextAllocationMB, next.ContextAllocationEvidence = 0, ""
	next.ResourceLedger = nil
	next.PlacementCachePath, next.PlacementCacheHit = "", false
	return next
}

// HostExpertAgentCost is the exported comparison used when a launch chooses
// among several admitted placements: staged prefill transfer plus decode host
// reads for cpuLayers host-resident expert layers at ubatch. ok is false when
// the policy is not staged or a rate is unknown.
func HostExpertAgentCost(caps *detect.Capabilities, model *ModelProfile, opts Options, ubatch, cpuLayers int) (float64, bool) {
	exec := ResolveHostExpertExecution(caps, opts)
	if !exec.StagedAt(ubatch) {
		return 0, false
	}
	return hostExpertAgentCost(caps, model, exec, ubatch, cpuLayers)
}
