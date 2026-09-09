package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// The cold compute-buffer formula (firstLaunchComputeBufMBParallel) models a
// dense transformer's activation working set: ubatch x n_embd x n_layer x a
// coefficient. It has NO context term, which is correct for an architecture
// whose attention does not materialise anything proportional to n_ctx.
//
// It is badly wrong for an architecture that does. GLM-5.3-Flash (glm5next)
// carries a "lightning indexer" that scores every query token against the whole
// context, so its graph grows with ubatch x n_ctx. Measured on a 3-GPU rig,
// 2026-09-02, at ctx 1,048,576 -- eight independent probes, perfectly linear in
// ubatch:
//
//	ub 256 -> 17494 MiB   (68.3 MiB/token)
//	ub 128 ->  8879 MiB   (69.4 MiB/token)
//	ub  64 ->  4570 MiB   (71.4 MiB/token)
//
// The dense formula predicted 7.74 MiB/token: 9x low. Placement therefore packed
// weights into VRAM that was not there, every re-plan repeated the same error,
// and ten contained probes over 46 minutes never converged.
//
// Rather than special-casing another architecture name -- which invariant 8
// forbids, and which is how deepseek4's context scaling came to exclude every
// other model -- the excess over the dense estimate is MEASURED and attributed
// to a context-proportional term. A model whose buffer really is context-
// independent measures no excess and is left exactly as it was.

// computeExcessScaling is a measured per-(token x context-position) byte cost
// that the dense formula does not account for.
type computeExcessScaling struct {
	// BytesPerTokenCtx is the excess in bytes per (ubatch token x context
	// position). Zero means no excess was observed and the dense estimate stands.
	BytesPerTokenCtx float64
	// Evidence names where the calibration came from, for the ledger line.
	Evidence string
}

// scaledComputeBufMB returns the dense estimate plus the measured excess for
// this exact (ubatch, context). It is monotonic in both, and reduces to the
// dense estimate when nothing was measured.
func (c computeExcessScaling) scaledComputeBufMB(denseMB, uBatch, contextSize int) int {
	if c.BytesPerTokenCtx <= 0 || uBatch <= 0 || contextSize <= 0 {
		return denseMB
	}
	excessMB := c.BytesPerTokenCtx * float64(uBatch) * float64(contextSize) / (1024 * 1024)
	if excessMB < 0 {
		return denseMB
	}
	return denseMB + int(excessMB)
}

// MeasuredComputeExcess derives the context-proportional excess for one model
// from any probe record that OBSERVED a compute buffer (guarded or live
// allocation, never an oracle prediction). It is scoped to the model artifact,
// backend/features, GPU set, and slots, and
// deliberately NOT to context or ubatch: the whole point is that one
// measurement transfers to every other context and microbatch, which the
// ctx/ubatch-keyed probe cache cannot do -- every re-plan changed the key and
// went cold again, which is why ten measurements taught the planner nothing.
func MeasuredComputeExcess(cacheDir string, model *ModelProfile, gpus []detect.GPU, parallel int, backendTag string) computeExcessScaling {
	if cacheDir == "" || model == nil {
		return computeExcessScaling{}
	}
	entries, err := os.ReadDir(cacheDir)
	if err != nil {
		return computeExcessScaling{}
	}
	best := computeExcessScaling{}
	for _, ent := range entries {
		if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".probe") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(cacheDir, ent.Name()))
		if err != nil {
			continue
		}
		scope, ok := matchingRelatedProbeScope(ent.Name(), string(data), model, gpus, parallel, backendTag)
		if !ok {
			continue
		}
		ctx, ubatch := scope.Context, scope.UBatch
		schema := 0
		observed := false
		maxCompute := 0
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			switch {
			case strings.HasPrefix(line, "PROBE_CACHE_SCHEMA="):
				schema, _ = strconv.Atoi(strings.TrimPrefix(line, "PROBE_CACHE_SCHEMA="))
			case strings.HasPrefix(line, "PROBED_COMPUTE_BUF_EVIDENCE="):
				observed = observedAllocationEvidence(strings.TrimPrefix(line, "PROBED_COMPUTE_BUF_EVIDENCE="))
			case strings.HasPrefix(line, "PROBED_COMPUTE_BUF_MB_CUDA"):
				var dev, v int
				if _, err := fmt.Sscanf(line, "PROBED_COMPUTE_BUF_MB_CUDA%d=%d", &dev, &v); err == nil && v > maxCompute {
					maxCompute = v
				}
			}
		}
		// Only an OBSERVED buffer for this model on this GPU set can calibrate.
		// An oracle prediction is the very estimate under test.
		if schema < probeMetadataIntegritySchema || !observed || maxCompute <= 0 {
			continue
		}
		dense := firstLaunchComputeBufMBParallel(model, ubatch, max(1, scope.Parallel))
		excessMB := float64(maxCompute - dense)
		if excessMB <= 0 {
			// The dense model already covers this architecture. Recording a zero is
			// as informative as recording a positive value: it says "no ctx term".
			continue
		}
		bytesPerTokenCtx := excessMB * 1024 * 1024 / (float64(ubatch) * float64(ctx))
		if bytesPerTokenCtx > best.BytesPerTokenCtx {
			best = computeExcessScaling{
				BytesPerTokenCtx: bytesPerTokenCtx,
				Evidence: fmt.Sprintf("measured %d MiB at ubatch %d / ctx %d (dense estimate %d MiB)",
					maxCompute, ubatch, ctx, dense),
			}
		}
	}
	return best
}
