package placement

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// The host prompt cache decides how much of a conversation's prefix survives to
// its next turn, and ggrun sized it from a KV formula that does not describe it.
// A saved entry is target state + draft state + checkpoints; on this project KV
// for a whole 1M context estimated 9238 MiB while one entry for a 126k-token
// prompt measured 6494. Sized from the estimate, the budget held one entry, so a
// four-slot server evicted one conversation to admit the next and re-prefilled
// ~126k tokens on the turn after. Measured 17.5% cache-read on that server
// against 88.3% on a single-slot server sharing the same box and client.
//
// The backend already knows the true figure and prints it, so ggrun reads it
// rather than modelling it -- the same choice already made for compute buffers
// and runtime graph growth.

// promptCacheStateRe matches the server's own cache accounting:
//
//	srv  update: - cache state: 3 prompts, 19484.211 MiB (limits: 9761.000 MiB, 0 tokens, 4096 est)
//
// It is logged at trace level, one above llama.cpp's default of info, so a
// launch that wants this measurement must raise verbosity.
var promptCacheStateRe = regexp.MustCompile(
	`cache state:\s*(\d+)\s+prompts,\s*([0-9.]+)\s*MiB\s*\(limits:\s*([0-9.]+)\s*MiB,\s*(\d+)\s*tokens`)

// promptCacheEntryRe matches the eviction and skip lines, which carry an entry's
// size directly. These appear at warning level, so they are visible by default,
// but only when the cache is already under pressure -- by then the re-prefill
// has been paid.
var promptCacheEntryRe = regexp.MustCompile(`entry \(size = ([0-9.]+) MiB\)`)

// promptCacheSkipRe matches an entry rejected for exceeding the whole budget.
// This one matters more than it looks: the backend skips silently as far as the
// cache is concerned, so the prompt is never stored and every turn re-prefills
// with no eviction line to explain why.
var promptCacheSkipRe = regexp.MustCompile(
	`prompt state size ([0-9.]+) MiB exceeds cache size limit ([0-9.]+) MiB`)

// Some compatible backends (notably ik_llama) log the sequence state size
// before allocating a prompt-cache entry, but do not reject an over-limit
// entry before copying it. If that copy exhausts the memory scope there is no
// cache-state, eviction, or skip line for the next launch to learn from.
//
// A cache entry is the sequence state plus every context checkpoint copied
// with the prompt. Both quantities are printed before the copy, so reconstruct
// the full entry from those backend-authoritative figures. This also lets a
// healthy first save teach the next launch without waiting for cache pressure.
var promptCacheSaveRe = regexp.MustCompile(
	`saving prompt with length\s+\d+,\s+total state size\s*=\s*([0-9.]+)\s*MiB`)

var promptCheckpointCreateRe = regexp.MustCompile(
	`created context checkpoint\s+(\d+)\s+of\s+\d+.*size\s*=\s*([0-9.]+)\s*MiB`)

// PromptCacheObservation is what a backend log revealed about the prompt cache.
type PromptCacheObservation struct {
	// BytesPerToken is total cached bytes over total cached tokens. Zero when
	// the log had no trace-level cache state to read.
	BytesPerToken float64
	// LargestEntryMB is the biggest single entry seen, from eviction or skip
	// lines. Useful on its own: the budget must exceed one entry or nothing is
	// ever cached.
	LargestEntryMB float64
	// Prompts and LimitMB are the last reported cache occupancy and budget.
	Prompts int
	LimitMB float64
	// Skipped counts entries rejected for exceeding the budget outright.
	Skipped int
	// Evicted counts entries dropped to make room.
	Evicted int
	// LargestCheckpointMB is one active checkpoint's backend-reported size.
	// It is separate from LargestEntryMB because active slot checkpoints and
	// prompt-cache copies coexist and must be charged independently.
	LargestCheckpointMB float64
}

// Measured reports whether the observation carries a usable per-token cost.
func (o PromptCacheObservation) Measured() bool { return o.BytesPerToken > 0 }

// ObservePromptCache extracts prompt-cache accounting from backend log output.
// It reads whatever is present: the trace-level state line gives the per-token
// cost directly, while eviction and skip lines still bound the entry size when
// verbosity was left at the default.
func ObservePromptCache(logText string) PromptCacheObservation {
	var obs PromptCacheObservation
	if strings.TrimSpace(logText) == "" {
		return obs
	}

	// Later lines describe a warmer cache, so the last state wins.
	for _, m := range promptCacheStateRe.FindAllStringSubmatch(logText, -1) {
		prompts, err1 := strconv.Atoi(m[1])
		sizeMB, err2 := strconv.ParseFloat(m[2], 64)
		limitMB, err3 := strconv.ParseFloat(m[3], 64)
		tokens, err4 := strconv.Atoi(m[4])
		if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
			continue
		}
		obs.Prompts, obs.LimitMB = prompts, limitMB
		if tokens > 0 && sizeMB > 0 {
			obs.BytesPerToken = sizeMB * 1024 * 1024 / float64(tokens)
		}
	}

	for _, m := range promptCacheEntryRe.FindAllStringSubmatch(logText, -1) {
		obs.Evicted++
		if v, err := strconv.ParseFloat(m[1], 64); err == nil && v > obs.LargestEntryMB {
			obs.LargestEntryMB = v
		}
	}
	for _, m := range promptCacheSkipRe.FindAllStringSubmatch(logText, -1) {
		obs.Skipped++
		if v, err := strconv.ParseFloat(m[1], 64); err == nil && v > obs.LargestEntryMB {
			obs.LargestEntryMB = v
		}
	}

	// Reconstruct saved-entry size in log order. Use the largest checkpoint
	// ordinal and size observed in the run: a branch can reduce the current count
	// immediately before the first cache save, but the cache must still be able
	// to hold the larger state this same workload already demonstrated. Combining
	// the maxima can overestimate a little; that is bounded by computeCRAM's host
	// ceiling, while underestimating can make the backend die during the very save
	// needed to produce better evidence.
	checkpointCount := 0
	checkpointSizeMB := 0.0
	for _, line := range strings.Split(logText, "\n") {
		if m := promptCheckpointCreateRe.FindStringSubmatch(line); m != nil {
			count, err1 := strconv.Atoi(m[1])
			sizeMB, err2 := strconv.ParseFloat(m[2], 64)
			if err1 == nil && err2 == nil && count >= 0 && sizeMB > 0 {
				if count > checkpointCount {
					checkpointCount = count
				}
				if sizeMB > checkpointSizeMB {
					checkpointSizeMB = sizeMB
					obs.LargestCheckpointMB = sizeMB
				}
			}
		}
		if m := promptCacheSaveRe.FindStringSubmatch(line); m != nil {
			stateMB, err := strconv.ParseFloat(m[1], 64)
			if err != nil || stateMB <= 0 {
				continue
			}
			entryMB := stateMB + float64(checkpointCount)*checkpointSizeMB
			if entryMB > obs.LargestEntryMB {
				obs.LargestEntryMB = entryMB
			}
		}
	}
	return obs
}

// promptCacheBudgetMB sizes the host prompt cache from a measured per-token cost
// so that every slot can keep one conversation resident.
//
// Slots is the right multiplier because a slot is exactly what gets evicted: a
// conversation returning to a slot whose prefix was dropped re-prefills from
// zero. Beyond one entry per slot there is no further hit rate to buy, so this
// deliberately does not grow to fill available RAM -- that memory is wanted by
// the expert layers and by the scope the backend runs inside.
//
// Returns 0 when nothing was measured, so the caller keeps its derived value
// rather than acting on a guess dressed as a measurement.
func promptCacheBudgetMB(bytesPerToken float64, promptTokens, slots int) int {
	if bytesPerToken <= 0 || promptTokens <= 0 {
		return 0
	}
	if slots < 1 {
		slots = 1
	}
	perEntryMB := bytesPerToken * float64(promptTokens) / (1024 * 1024)
	// One spare entry: a conversation is stored before the one it replaces is
	// dropped, so a budget of exactly n_slots entries still thrashes at the edge.
	return int(perEntryMB * float64(slots+1))
}

// measuredPromptCacheBudgetMB sizes the cache from whichever measurement is
// available, preferring the directly observed entry size.
//
// The entry size is preferred because it needs no assumption: the per-token
// route has to guess how long a typical cached turn is, and on this project
// that guess was the whole error -- KV for a full 1M context estimated 9238 MiB
// while one real entry measured 6494. It is also the cheaper measurement, since
// the backend prints entry sizes at warning level and the per-token state only
// at trace.
//
// Returns 0 when nothing was measured, so the caller keeps its derived value
// rather than acting on a guess dressed as a measurement.
func measuredPromptCacheBudgetMB(s *Strategy, slots int) int {
	if s == nil {
		return 0
	}
	if slots < 1 {
		slots = 1
	}
	if s.MeasuredPromptCacheEntryMB > 0 {
		// One spare entry, for the same reason as below: a conversation is
		// stored before the one it replaces is dropped.
		return int(s.MeasuredPromptCacheEntryMB * float64(slots+1))
	}
	return promptCacheBudgetMB(s.MeasuredPromptCacheBPT, s.PromptCacheTypicalTokens, slots)
}

// LoadMeasuredPromptCache copies any stored prompt-cache measurement onto a
// strategy so CRAM sizing can use it. Absent measurements leave the strategy
// untouched, which keeps the derived budget in place.
func LoadMeasuredPromptCache(cacheDir string, model *ModelProfile, s *Strategy, backendTag string, gpus []detect.GPU) {
	if s == nil || model == nil {
		return
	}
	pc := loadProbeCache(cacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendTag, gpus, s.Parallel)
	if pc == nil {
		return
	}
	if pc.PromptCacheEntryMB > 0 {
		s.MeasuredPromptCacheEntryMB = pc.PromptCacheEntryMB
	}
	if pc.PromptCacheBytesPerToken > 0 {
		s.MeasuredPromptCacheBPT = pc.PromptCacheBytesPerToken
	}
	if pc.CheckpointMB > 0 {
		s.MeasuredCheckpointMB = pc.CheckpointMB
	}
}

// ApplyMeasuredCheckpointObservation raises a strategy's planned host footprint
// when the running backend reports a checkpoint larger than the estimate used
// at launch. This runs before the post-canary cgroup resize, so the controller
// never tightens memory.max below the active checkpoints a long agent session
// can still create. It returns the old and new total checkpoint reserves.
func ApplyMeasuredCheckpointObservation(model *ModelProfile, s *Strategy, obs PromptCacheObservation) (beforeMB, afterMB int, changed bool) {
	if model == nil || s == nil || obs.LargestCheckpointMB <= 0 {
		return 0, 0, false
	}
	beforeMB = checkpointFootprintMB(model, s.MaxCheckpoints, s.Parallel, s.ContextAllocationMB, s.ContextSize, s.MeasuredCheckpointMB)
	if obs.LargestCheckpointMB <= s.MeasuredCheckpointMB {
		return beforeMB, beforeMB, false
	}
	s.MeasuredCheckpointMB = obs.LargestCheckpointMB
	afterMB = checkpointFootprintMB(model, s.MaxCheckpoints, s.Parallel, s.ContextAllocationMB, s.ContextSize, s.MeasuredCheckpointMB)
	if afterMB <= beforeMB {
		return beforeMB, afterMB, false
	}
	if s.PlannedHostFootprintMB > 0 {
		s.PlannedHostFootprintMB += afterMB - beforeMB
	}
	return beforeMB, afterMB, true
}

// RecordPromptCacheObservation stores whatever a backend log revealed about the
// prompt cache, keyed exactly like the other probes.
//
// The eviction and skip lines it reads only appear once the cache is already
// too small, so this is self-correcting rather than predictive: the first run
// with an undersized budget pays the re-prefill and the next one does not.
func RecordPromptCacheObservation(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, obs PromptCacheObservation) error {
	if model == nil || ctxSize <= 0 || ubatch <= 0 {
		return nil
	}
	if obs.LargestEntryMB <= 0 && obs.BytesPerToken <= 0 && obs.LargestCheckpointMB <= 0 {
		return nil
	}
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	compute := map[int]int{}
	growth := map[int]int{}
	estimated := map[int]bool{}
	kvPerLayerMB := 0
	if pc != nil {
		for k, v := range pc.ComputeBufByGPU {
			compute[k] = v
		}
		for k, v := range pc.RuntimeGraphGrowthByGPU {
			growth[k] = v
		}
		for k, v := range pc.RuntimeGraphGrowthEstimatedByGPU {
			estimated[k] = v
		}
		kvPerLayerMB = pc.KVPerLayerMB
		// Keep the largest entry ever seen: a run that happened to cache only
		// short turns must not shrink a budget a long one proved necessary.
		if pc.PromptCacheEntryMB > obs.LargestEntryMB {
			obs.LargestEntryMB = pc.PromptCacheEntryMB
		}
		if pc.CheckpointMB > obs.LargestCheckpointMB {
			obs.LargestCheckpointMB = pc.CheckpointMB
		}
	}
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel,
		compute, growth, estimated, kvPerLayerMB,
		probeMeasurements{BytesPerToken: obs.BytesPerToken, EntryMB: obs.LargestEntryMB, CheckpointMB: obs.LargestCheckpointMB})
}

// RecordMeasuredPromptCache stores a per-token prompt-cache cost read from
// backend output, keyed exactly like the other probes so a different context,
// KV type, slot count or backend cannot inherit it.
func RecordMeasuredPromptCache(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, bytesPerToken float64) error {
	if model == nil || bytesPerToken <= 0 || ctxSize <= 0 || ubatch <= 0 {
		return nil
	}
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	compute := map[int]int{}
	growth := map[int]int{}
	estimated := map[int]bool{}
	kvPerLayerMB := 0
	if pc != nil {
		for k, v := range pc.ComputeBufByGPU {
			compute[k] = v
		}
		for k, v := range pc.RuntimeGraphGrowthByGPU {
			growth[k] = v
		}
		for k, v := range pc.RuntimeGraphGrowthEstimatedByGPU {
			estimated[k] = v
		}
		kvPerLayerMB = pc.KVPerLayerMB
	}
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel,
		compute, growth, estimated, kvPerLayerMB, probeMeasurements{BytesPerToken: bytesPerToken})
}

// MeasuredPromptCacheBytesPerToken returns a stored measurement, or 0 when this
// shape has never been observed.
func MeasuredPromptCacheBytesPerToken(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) float64 {
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	if pc == nil {
		return 0
	}
	return pc.PromptCacheBytesPerToken
}

// LargestRoutedExpertLayerMB is the size of the biggest routed-expert layer,
// which is the unit placement actually moves between GPU and CPU.
//
// It exists so a CUDA out-of-memory abort can be answered with a real quantity.
// The abort at graph instantiation carries no allocation size, and the previous
// response to that was to reserve a tenth of the card -- a number derived from
// nothing, which either overshoots and withholds several layers permanently or
// undershoots and crashes again. Moving exactly one layer is the smallest step
// that changes the outcome, and its cost is already known exactly from the GGUF
// ledger, so the retry converges instead of guessing.
//
// Returns 0 when the ledger is unavailable, so the caller can fall back rather
// than reserve zero.
func LargestRoutedExpertLayerMB(model *ModelProfile) int {
	if model == nil {
		return 0
	}
	layers := model.RoutedExpertLayerBytes
	if len(layers) == 0 {
		layers = model.ExpertLayerBytes
	}
	var maxBytes int64
	for _, b := range layers {
		if b > maxBytes {
			maxBytes = b
		}
	}
	if maxBytes <= 0 {
		return 0
	}
	return int((maxBytes + (1 << 20) - 1) / (1 << 20))
}

// RefreshRuntimeCachePolicy returns a copy of s whose host prompt-cache budget
// and checkpoint limits are re-sized from the current measurements. Placement,
// context and microbatch are kept: a prompt-cache measurement changes a host
// RAM budget, not what the GPUs hold, and re-planning everything for it threw
// away a verified (or oracle-raised) plan on every Claude Code relaunch.
func RefreshRuntimeCachePolicy(caps *detect.Capabilities, model *ModelProfile, s *Strategy, opts Options) *Strategy {
	if caps == nil || model == nil || s == nil {
		return s
	}
	next := cloneStrategy(s)
	LoadMeasuredPromptCache(opts.CacheDir, model, next, backendCacheTag(opts), caps.GPUs)
	totalSizeMB := model.TotalSizeMB
	if totalSizeMB <= 0 {
		totalSizeMB = int((model.SizeBytes + 1048575) / 1048576)
	}
	applyRuntimeCachePolicy(model, next, caps, totalSizeMB, computeKVTotalMBForStrategy(model, next), opts)
	return next
}
