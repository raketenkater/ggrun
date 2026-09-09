package placement

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Runtime graph growth is the driver-side allocation a backend never accounts
// for: the CUDA graph executable instantiated at warmup, plus whatever the
// allocator rounds up. No no-alloc oracle predicts it -- llama-fit-params
// builds the graph structure but never instantiates the executable -- so it is
// the one quantity that requires a real load to learn.
//
// Until now ggrun learned it exactly one way: by crashing. Every production
// caller was RecordRuntimeGraphGrowthFromOOM; the non-OOM recorder had no
// caller outside tests. The consequence, measured on this rig 2026-09-04: 23 of
// 24 GLM probe entries carried no growth at all, and the single entry that did
// held one device's figure salvaged from an OOM.
//
// That breaks the whole measure-then-pack loop. ComputeExpertSeats refuses to
// offer a device any seats without measured growth (correctly -- a default
// margin there is the static reserve invariant 8 forbids), so a rig that never
// crashes never earns the evidence needed to pack tighter, and holds its
// conservative slack forever. Measured on the same rig: 17.8 GB of slack held
// across three devices on "gguf-and-probe-estimate" evidence.
//
// This file closes the loop by learning the same quantity from a launch that
// worked. Growth is what the device actually holds minus everything the backend
// itself accounted for:
//
//	growth = live_used - system_overhead - companions - (model + KV + compute)
//
// All four subtracted terms are measured, not modelled: the first from the
// driver, the second from ggrun's own system probe, the third from recorded
// companion measurements, and the last from the backend's own log rows.

// serveGrowthLedger is one device's backend-accounted bytes, parsed from the
// server log the launch just produced.
type serveGrowthLedger struct {
	ModelMB   int
	KVMB      int
	ComputeMB int
}

func (l serveGrowthLedger) total() int { return l.ModelMB + l.KVMB + l.ComputeMB }

var (
	serveModelBufRe   = regexp.MustCompile(`CUDA(\d+) model buffer size *= *([0-9.]+)`)
	serveKVBufRe      = regexp.MustCompile(`CUDA(\d+) KV buffer size *= *([0-9.]+)`)
	serveComputeBufRe = regexp.MustCompile(`CUDA(\d+) compute buffer size *= *([0-9.]+)`)
)

// parseServeGrowthLedgers reads the backend's own per-device accounting.
//
// Last-writer-wins per device: a launch that re-planned writes several rounds
// of buffer rows into one log, and only the final round describes the process
// that is actually serving. Summing them would inflate the accounted total and
// hide real growth.
func parseServeGrowthLedgers(logData string) map[int]serveGrowthLedger {
	out := map[int]serveGrowthLedger{}
	apply := func(re *regexp.Regexp, set func(*serveGrowthLedger, int)) {
		for _, m := range re.FindAllStringSubmatch(logData, -1) {
			dev, err := strconv.Atoi(m[1])
			if err != nil {
				continue
			}
			mb, err := strconv.ParseFloat(m[2], 64)
			if err != nil || mb < 0 {
				continue
			}
			l := out[dev]
			set(&l, int(mb+0.5))
			out[dev] = l
		}
	}
	apply(serveModelBufRe, func(l *serveGrowthLedger, v int) { l.ModelMB = v })
	apply(serveKVBufRe, func(l *serveGrowthLedger, v int) { l.KVMB = v })
	apply(serveComputeBufRe, func(l *serveGrowthLedger, v int) { l.ComputeMB = v })
	return out
}

// serveGrowthFloorMB is the smallest delta worth recording. Below it the figure
// is indistinguishable from driver rounding and sampling jitter between the
// backend's accounting and nvidia-smi, and recording it would present noise as
// evidence.
//
// This is not a safety margin: it gates whether a measurement is believable at
// all, and a device below it simply records nothing.
const serveGrowthFloorMB = 16

// serveVRAMSampler is indirected so a test can supply device measurements
// without a GPU, the same way cmd/ggrun indirects runtimeVRAMUsedMB across its
// admission boundary. Production always reads the driver.
var serveVRAMSampler = QueryVRAMUsed

// RecordRuntimeGraphGrowthFromServe learns runtime graph growth from a launch
// that reached a serving state, and returns the per-device figures it stored.
//
// Deliberately conservative about what it will claim:
//
//   - a device with no backend accounting in the log is skipped, because the
//     whole delta would otherwise be attributed to growth;
//   - a non-positive delta is skipped rather than recorded as zero, since zero
//     growth is a claim that would let a packer fill the device completely;
//   - the result is recorded as exact (not estimated), because every term is
//     measured -- which is what makes it eligible for transfer by
//     RelatedModelRuntimeGraphGrowth.
//
// logData is the backend log's CONTENTS, matching its siblings in the same
// post-launch block (ParseComputeBuffersByGPU, RunPostLaunchModelProbe): the
// launch holds the log in memory as p.LogBuf and never guarantees a file on
// disk, so a path-taking signature here would read nothing and record nothing.
func RecordRuntimeGraphGrowthFromServe(
	cacheDir string, model *ModelProfile, strategy *Strategy, backendTag string,
	gpus []detect.GPU, logData string, companionVRAMByGPU map[int]int,
) map[int]int {
	if model == nil || strategy == nil || len(gpus) == 0 || strings.TrimSpace(logData) == "" {
		return nil
	}
	// Cache buffers are outside the model/KV/compute log rows. Subtracting a
	// planned cache map would mix estimates with measurements and can understate
	// driver growth. Until actual per-device cache buffers are reported, keep the
	// residual unattributed; complete observed allocation remains available.
	if strategy.HotExpertCacheSlots > 0 || ObserveHotExpertCache(logData).Enabled {
		return nil
	}
	ledgers := parseServeGrowthLedgers(logData)
	if len(ledgers) == 0 {
		return nil
	}
	overhead := SystemCUDAOverheadByGPU(cacheDir, gpus)

	growth := map[int]int{}
	for _, g := range gpus {
		ledger, ok := ledgers[g.Index]
		if !ok || ledger.total() <= 0 {
			continue // no accounting for this device: the delta would be unattributable
		}
		live := serveVRAMSampler(g.Index)
		if live <= 0 {
			continue
		}
		delta := live - overhead[g.Index] - companionVRAMByGPU[g.Index] - ledger.total()
		if delta < serveGrowthFloorMB {
			continue
		}
		growth[g.Index] = delta
	}
	if len(growth) == 0 {
		return nil
	}
	if err := RecordRuntimeGraphGrowth(cacheDir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, backendTag, gpus, strategy.Parallel, growth); err != nil {
		fmt.Fprintf(os.Stderr, "[launch] warning: could not persist measured graph growth: %v\n", err)
		return nil
	}
	parts := make([]string, 0, len(gpus))
	for _, g := range gpus {
		if v, ok := growth[g.Index]; ok {
			parts = append(parts, fmt.Sprintf("CUDA%d=%dMB", g.Index, v))
		}
	}
	fmt.Fprintf(os.Stderr,
		"[launch] measured runtime graph growth from this serve: %s (learned without an OOM)\n",
		strings.Join(parts, ", "))
	return growth
}
