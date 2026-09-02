package placement

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// hotExpertCacheDefaultInserts is the reviewed backend contract's conservative
// upload throttle. It changes transfer pressure, not cache capacity. The first
// core integration keeps it fixed so one experiment changes one dimension.
const hotExpertCacheDefaultInserts = 2

const (
	hotExpertCacheAlignmentBytes = int64(256)
	hotExpertCacheMiB            = int64(1024 * 1024)
	hotExpertTelemetryMinSteps   = uint64(512)
)

var hotExpertOTLayerPattern = regexp.MustCompile(`blk\\\.\(([^)]*)\)\\\.`)
var hotExpertEnabledPattern = regexp.MustCompile(`MoE expert cache enabled: ([0-9]+) layers x ([0-9]+) slots, ([0-9]+) inserts/step, ([0-9]+(?:\.[0-9]+)?) MiB device memory`)
var hotExpertTelemetryPattern = regexp.MustCompile(`moe-cache: steps=([0-9]+) hits=([0-9]+) misses=([0-9]+) hit-rate=([0-9]+(?:\.[0-9]+)?)%`)

// HotExpertCacheObservation is the backend's startup acknowledgement of the
// exact cache it actually allocated. A healthy HTTP server is insufficient:
// the reference backend intentionally degrades to cache-off when allocation
// fails, which must not be benchmarked or persisted as a cache-on candidate.
type HotExpertCacheObservation struct {
	Enabled   bool
	Layers    int
	Slots     int
	Inserts   int
	DeviceMiB float64
	Reason    string
}

// HotExpertCacheTelemetry is the last periodic runtime report emitted by the
// reviewed backend. Startup acknowledgement proves only that memory was
// allocated; non-zero hits plus the identical live A/B prove that the graph
// exercised the cache and that doing so helped this concrete model/workload.
type HotExpertCacheTelemetry struct {
	Observed bool
	Steps    uint64
	Hits     uint64
	Misses   uint64
	HitRate  float64
	Reason   string
}

// ObserveHotExpertCache parses the stable capability log contract.
func ObserveHotExpertCache(logData string) HotExpertCacheObservation {
	if match := hotExpertEnabledPattern.FindStringSubmatch(logData); len(match) == 5 {
		layers, _ := strconv.Atoi(match[1])
		slots, _ := strconv.Atoi(match[2])
		inserts, _ := strconv.Atoi(match[3])
		deviceMiB, _ := strconv.ParseFloat(match[4], 64)
		return HotExpertCacheObservation{
			Enabled: true, Layers: layers, Slots: slots, Inserts: inserts, DeviceMiB: deviceMiB,
		}
	}
	lower := strings.ToLower(logData)
	for _, marker := range []string{
		"failed to allocate moe cache buffer",
		"no host-resident expert layers found - disabled",
		"moe expert cache" + " disabled",
	} {
		if strings.Contains(lower, marker) {
			return HotExpertCacheObservation{Reason: marker}
		}
	}
	return HotExpertCacheObservation{Reason: "backend emitted no hot-expert enable acknowledgement"}
}

// ValidateHotExpertCacheObservation proves that a cache-on Strategy became the
// cache-on process it claims to be. The controller's memory plan includes
// tables/alignment while the backend reports only the three cache tensors, so
// the observed number may be lower but must never exceed the planned ceiling.
func ValidateHotExpertCacheObservation(s *Strategy, logData string) error {
	if s == nil || s.HotExpertCacheSlots <= 0 {
		return nil
	}
	observation := ObserveHotExpertCache(logData)
	if !observation.Enabled {
		return fmt.Errorf("hot-expert cache was not enabled: %s", observation.Reason)
	}
	if observation.Slots != s.HotExpertCacheSlots ||
		observation.Inserts != s.HotExpertCacheInserts ||
		observation.Layers != s.HotExpertCacheLayers {
		return fmt.Errorf(
			"hot-expert cache acknowledgement differs from plan: got %d layers/%d slots/%d inserts, want %d/%d/%d",
			observation.Layers, observation.Slots, observation.Inserts,
			s.HotExpertCacheLayers, s.HotExpertCacheSlots, s.HotExpertCacheInserts,
		)
	}
	plannedMiB := 0
	for _, value := range s.HotExpertCacheVRAMByGPU {
		plannedMiB += value
	}
	if observation.DeviceMiB <= 0 || observation.DeviceMiB > float64(plannedMiB)+1 ||
		math.IsNaN(observation.DeviceMiB) || math.IsInf(observation.DeviceMiB, 0) {
		return fmt.Errorf(
			"hot-expert cache device allocation %.1f MiB is outside the planned %d MiB ceiling",
			observation.DeviceMiB, plannedMiB,
		)
	}
	return nil
}

// ObserveHotExpertCacheTelemetry returns the last complete aggregate report.
// Taking the last line makes a long explicit calibration retain the most
// representative window while keeping the parser independent of log prefixes.
func ObserveHotExpertCacheTelemetry(logData string) HotExpertCacheTelemetry {
	matches := hotExpertTelemetryPattern.FindAllStringSubmatch(logData, -1)
	if len(matches) == 0 {
		return HotExpertCacheTelemetry{Reason: "backend emitted no hot-expert runtime telemetry"}
	}
	match := matches[len(matches)-1]
	steps, stepsErr := strconv.ParseUint(match[1], 10, 64)
	hits, hitsErr := strconv.ParseUint(match[2], 10, 64)
	misses, missesErr := strconv.ParseUint(match[3], 10, 64)
	hitRate, rateErr := strconv.ParseFloat(match[4], 64)
	if stepsErr != nil || hitsErr != nil || missesErr != nil || rateErr != nil {
		return HotExpertCacheTelemetry{Reason: "backend emitted malformed hot-expert runtime telemetry"}
	}
	return HotExpertCacheTelemetry{
		Observed: true,
		Steps:    steps,
		Hits:     hits,
		Misses:   misses,
		HitRate:  hitRate,
	}
}

// ValidateHotExpertCacheTelemetry requires enough decode work, internally
// consistent counters, and at least one hit. It intentionally does not impose
// a universal hit-rate threshold: the identical end-to-end agent A/B and its
// regression gates decide whether the cache is beneficial on this model.
func ValidateHotExpertCacheTelemetry(s *Strategy, logData string) (HotExpertCacheTelemetry, error) {
	if s == nil || s.HotExpertCacheSlots <= 0 {
		return HotExpertCacheTelemetry{}, nil
	}
	telemetry := ObserveHotExpertCacheTelemetry(logData)
	if err := validateHotExpertCacheTelemetryValue(telemetry); err != nil {
		return telemetry, err
	}
	return telemetry, nil
}

func validateHotExpertCacheTelemetryValue(telemetry HotExpertCacheTelemetry) error {
	if !telemetry.Observed {
		return fmt.Errorf("hot-expert cache activity is unverified: %s", telemetry.Reason)
	}
	if telemetry.Steps < hotExpertTelemetryMinSteps {
		return fmt.Errorf(
			"hot-expert telemetry covered %d decode steps, need at least %d",
			telemetry.Steps, hotExpertTelemetryMinSteps,
		)
	}
	if telemetry.Hits > ^uint64(0)-telemetry.Misses || telemetry.Hits+telemetry.Misses == 0 {
		return fmt.Errorf("hot-expert telemetry has no valid lookup counters")
	}
	if math.IsNaN(telemetry.HitRate) || math.IsInf(telemetry.HitRate, 0) || telemetry.HitRate < 0 || telemetry.HitRate > 100 {
		return fmt.Errorf("hot-expert telemetry reported invalid hit rate %.3f%%", telemetry.HitRate)
	}
	expected := 100 * float64(telemetry.Hits) / float64(telemetry.Hits+telemetry.Misses)
	// The backend prints one decimal place; 0.11 percentage points covers only
	// that rounding and does not hide materially inconsistent counters.
	if math.Abs(expected-telemetry.HitRate) > 0.11 {
		return fmt.Errorf(
			"hot-expert telemetry counters imply %.1f%% hit rate, backend reported %.1f%%",
			expected, telemetry.HitRate,
		)
	}
	if telemetry.Hits == 0 {
		return fmt.Errorf("hot-expert cache completed %d lookups without a hit", telemetry.Misses)
	}
	return nil
}

type hotExpertCacheShape struct {
	perSlotBytesByGPU map[int]int64
	fixedBytesByGPU   map[int]int64
	layersByGPU       map[int]int
	layers            int
}

func backendSupportsHotExpertCache(help string) bool {
	return backendHelpSupportsExactFlag(help, "--moe-expert-cache") &&
		backendHelpSupportsExactFlag(help, "--moe-expert-cache-inserts")
}

// BackendSupportsHotExpertCache exposes the exact help-surface probe to the
// command layer and tests. A fork name is never capability evidence.
func BackendSupportsHotExpertCache(help string) bool {
	return backendSupportsHotExpertCache(help)
}

func hotExpertRequestedSlots(opts Options) int {
	if opts.HotExpertCacheSlots > 0 {
		return opts.HotExpertCacheSlots
	}
	value := strings.TrimSpace(strings.ToLower(opts.HotExperts))
	n, _ := strconv.Atoi(value)
	if n > 0 {
		return n
	}
	return 0
}

func hotExpertOptimizerRequested(opts Options) bool {
	policy := strings.TrimSpace(opts.HotExperts)
	return strings.EqualFold(policy, "auto") || strings.EqualFold(policy, "on")
}

func hotExpertRequired(opts Options) bool {
	return strings.EqualFold(strings.TrimSpace(opts.HotExperts), "on")
}

func clearHotExpertCache(s *Strategy) {
	if s == nil {
		return
	}
	s.HotExpertCacheSlots = 0
	s.HotExpertCacheInserts = 0
	s.HotExpertCacheLayers = 0
	s.HotExpertCacheVRAMByGPU = nil
	s.HotExpertCacheLayersByGPU = nil
	s.HotExpertCacheEvidence = ""
	s.HotExpertCacheSteps = 0
	s.HotExpertCacheHits = 0
	s.HotExpertCacheMisses = 0
	s.HotExpertCacheHitRate = 0
}

// WithoutHotExpertCache returns a cache-free deep copy suitable for deriving
// the original calibration scope after a verified cache-on winner fails during
// a later lifecycle. It prevents that old decision from reapplying the same
// failed winner on every launch.
func WithoutHotExpertCache(s *Strategy) *Strategy {
	copy := cloneStrategy(s)
	clearHotExpertCache(copy)
	return copy
}

func cloneIntMap(values map[int]int) map[int]int {
	if len(values) == 0 {
		return nil
	}
	out := make(map[int]int, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}

func hotExpertPinnedLayers(ot string) (whole, partial map[int]bool, err error) {
	whole = map[int]bool{}
	partial = map[int]bool{}
	for _, raw := range strings.Split(ot, ",") {
		part := strings.TrimSpace(raw)
		if part == "" || (!strings.Contains(part, "=CUDA") && !strings.Contains(part, "=Vulkan")) {
			continue
		}
		if !strings.Contains(part, "exps") {
			continue
		}
		match := hotExpertOTLayerPattern.FindStringSubmatch(part)
		if len(match) != 2 {
			return nil, nil, fmt.Errorf("cannot classify expert override %q", part)
		}
		// Only the exact rule emitted by this engine proves that routed gate/up/down
		// weights are all resident on the device. Router-auxiliary names alone do
		// not prove complete tensors; an unfamiliar older rule therefore fails
		// closed and leaves the stable placement cache-free.
		isWhole := strings.Contains(part, expertTensorPattern)
		isPartial := strings.Contains(part, `ffn_(gate_up|up_gate|gate|up)_(ch|)exps`)
		if !isWhole && !isPartial {
			return nil, nil, fmt.Errorf("cannot prove whether expert override %q leaves complete host tensors", part)
		}
		for _, rawLayer := range strings.Split(match[1], "|") {
			layer, parseErr := strconv.Atoi(rawLayer)
			if parseErr != nil {
				return nil, nil, fmt.Errorf("invalid expert layer %q in override", rawLayer)
			}
			if isWhole {
				whole[layer] = true
			} else {
				partial[layer] = true
			}
		}
	}
	return whole, partial, nil
}

func hotExpertHasCPUCatchAll(ot string) bool {
	last := ""
	for _, raw := range strings.Split(ot, ",") {
		if part := strings.TrimSpace(raw); part != "" {
			last = part
		}
	}
	// ggrun's OT contract is first-match-wins: GPU whole/partial exceptions
	// precede one final host catch-all. Any other order describes different
	// residency than the controller's layer classifier.
	return strings.EqualFold(last, "exps=CPU")
}

func hotExpertCacheShapeFor(caps *detect.Capabilities, model *ModelProfile, s *Strategy, opts Options) (*hotExpertCacheShape, error) {
	if caps == nil || model == nil || s == nil {
		return nil, fmt.Errorf("hot experts require complete hardware, model, and placement data")
	}
	if !backendSupportsHotExpertCache(opts.BackendHelp) {
		return nil, fmt.Errorf("selected backend does not advertise the hot-expert cache capability")
	}
	if !model.IsMoE || s.Type != MoEOffload || s.NCPUMoE <= 0 {
		return nil, fmt.Errorf("hot experts require host-resident routed MoE layers")
	}
	if s.MMapRequired {
		return nil, fmt.Errorf("hot experts are disabled in the mmap last-resort lane")
	}
	if s.Parallel > 1 {
		return nil, fmt.Errorf("the reviewed hot-expert backend supports only single-slot decode")
	}
	if mode := strings.ToLower(strings.TrimSpace(s.SplitMode)); mode != "" && mode != "layer" {
		return nil, fmt.Errorf("hot-expert router ownership is proven only for layer split mode, not %q", s.SplitMode)
	}
	if s.Draft != nil && s.Draft.Type != DraftNone {
		return nil, fmt.Errorf("the reviewed hot-expert backend bypasses multi-token speculative decode")
	}
	if model.Fused != 0 {
		return nil, fmt.Errorf("the reviewed hot-expert backend does not support fused gate/up expert tensors")
	}
	if model.NumExperts <= 0 {
		return nil, fmt.Errorf("model metadata does not expose the routed expert count")
	}
	if model.NumLayers <= 0 || len(model.RoutedExpertLayerBytes) < model.NumLayers {
		return nil, fmt.Errorf("exact per-layer routed-expert tensor bytes are unavailable")
	}
	if !hotExpertHasCPUCatchAll(s.OTString) {
		return nil, fmt.Errorf("placement does not expose an exact host-expert catch-all override")
	}

	runtimeCaps, err := restrictGPUs(caps, opts.GPUs)
	if err != nil || runtimeCaps == nil || len(runtimeCaps.GPUs) == 0 {
		if err == nil {
			err = fmt.Errorf("no GPU is selected")
		}
		return nil, err
	}
	whole, partial, err := hotExpertPinnedLayers(s.OTString)
	if err != nil {
		return nil, err
	}
	layerOwners, _ := layerDeviceAssignments(s.TensorSplit, model.NumLayers)
	defaultGPU := s.MainGPU
	if defaultGPU < 0 {
		defaultGPU = runtimeCaps.GPUs[0].Index
	}
	shape := &hotExpertCacheShape{
		perSlotBytesByGPU: map[int]int64{},
		fixedBytesByGPU:   map[int]int64{},
		layersByGPU:       map[int]int{},
	}
	start, count := moeLayerRange(model)
	for layer := start; layer < start+count; layer++ {
		if whole[layer] || partial[layer] {
			continue
		}
		layerBytes := model.RoutedExpertLayerBytes[layer]
		if layerBytes <= 0 {
			return nil, fmt.Errorf("routed expert bytes are missing for host layer %d", layer)
		}
		owner := defaultGPU
		if layer < len(layerOwners) {
			ordinal := layerOwners[layer]
			if ordinal >= 0 && ordinal < len(runtimeCaps.GPUs) {
				owner = runtimeCaps.GPUs[ordinal].Index
			}
		}
		if gpuOrdinal(runtimeCaps.GPUs, owner) < 0 {
			return nil, fmt.Errorf("host expert layer %d resolves to unavailable GPU %d", layer, owner)
		}
		// The backend allocates three cache tensors with n_slots+1 expert
		// slices. The extra slice is the permanent zero/dummy slot. GGUF tensor
		// bytes give the authoritative combined gate+up+down slice stride.
		perExpert := ceilDivInt64(layerBytes, int64(model.NumExperts))
		if perExpert <= 0 {
			return nil, fmt.Errorf("cannot derive expert slice bytes for layer %d", layer)
		}
		tableBytes := alignInt64(int64(model.NumExperts)*4, hotExpertCacheAlignmentBytes)
		alignmentGuard := 4 * hotExpertCacheAlignmentBytes // three tensors + device table
		shape.perSlotBytesByGPU[owner] += perExpert
		shape.fixedBytesByGPU[owner] += perExpert + tableBytes + alignmentGuard
		shape.layersByGPU[owner]++
		shape.layers++
	}
	if shape.layers == 0 {
		return nil, fmt.Errorf("no complete host-resident expert layer has a GPU router owner")
	}
	return shape, nil
}

func ceilDivInt64(value, divisor int64) int64 {
	if value <= 0 || divisor <= 0 {
		return 0
	}
	return (value + divisor - 1) / divisor
}

func alignInt64(value, alignment int64) int64 {
	if value <= 0 || alignment <= 0 {
		return value
	}
	return ceilDivInt64(value, alignment) * alignment
}

func hotExpertCacheLayout(shape *hotExpertCacheShape, slots int) (map[int]int, error) {
	if shape == nil || slots <= 0 {
		return nil, fmt.Errorf("hot-expert slots must be positive")
	}
	out := make(map[int]int, len(shape.perSlotBytesByGPU))
	for gpu, perSlot := range shape.perSlotBytesByGPU {
		fixed := shape.fixedBytesByGPU[gpu]
		if perSlot <= 0 || fixed < 0 {
			return nil, fmt.Errorf("hot-expert cache has invalid byte geometry on GPU %d", gpu)
		}
		maxInt64 := int64(^uint64(0) >> 1)
		if int64(slots) > (maxInt64-fixed)/perSlot {
			return nil, fmt.Errorf("hot-expert cache byte geometry overflows on GPU %d", gpu)
		}
		total := fixed + int64(slots)*perSlot
		out[gpu] = int(ceilDivInt64(total, hotExpertCacheMiB))
	}
	return out, nil
}

func hotExpertCacheMaxSlots(model *ModelProfile, shape *hotExpertCacheShape, ledger ResourceLedger) int {
	if model == nil || shape == nil || !ledger.Exact || !ledger.Fits {
		return 0
	}
	slack := make(map[int]int, len(ledger.Devices))
	for _, device := range ledger.Devices {
		slack[device.GPU] = device.SlackMB
	}
	maxSlots := model.NumExperts
	for gpu, perSlot := range shape.perSlotBytesByGPU {
		available := int64(slack[gpu]) * hotExpertCacheMiB
		fixed := shape.fixedBytesByGPU[gpu]
		if perSlot <= 0 || available <= fixed {
			return 0
		}
		slots := int((available - fixed) / perSlot)
		if slots < maxSlots {
			maxSlots = slots
		}
	}
	if maxSlots > model.NumExperts {
		maxSlots = model.NumExperts
	}
	minUseful := model.ExpertUsedCount
	if minUseful <= 0 {
		minUseful = 1
	}
	if maxSlots < minUseful {
		return 0
	}
	return maxSlots
}

func assignHotExpertCache(s *Strategy, shape *hotExpertCacheShape, slots int) error {
	layout, err := hotExpertCacheLayout(shape, slots)
	if err != nil {
		return err
	}
	s.HotExpertCacheSlots = slots
	s.HotExpertCacheInserts = hotExpertCacheDefaultInserts
	s.HotExpertCacheLayers = shape.layers
	s.HotExpertCacheVRAMByGPU = layout
	s.HotExpertCacheLayersByGPU = cloneIntMap(shape.layersByGPU)
	s.HotExpertCacheEvidence = "exact GGUF routed-expert slices + backend dummy/table/alignment contract"
	return nil
}

func hotExpertMinUsefulSlots(model *ModelProfile) int {
	if model == nil || model.ExpertUsedCount <= 0 {
		return 1
	}
	return model.ExpertUsedCount
}

func hotExpertCacheCandidate(caps *detect.Capabilities, model *ModelProfile, base *Strategy, opts Options) (*Strategy, error) {
	if !hotExpertOptimizerRequested(opts) || base == nil || base.HotExpertCacheSlots > 0 {
		return nil, nil
	}
	shape, err := hotExpertCacheShapeFor(caps, model, base, opts)
	if err != nil {
		return nil, err
	}
	baseLedger := BuildResourceLedger(caps, model, base, opts)
	if !baseLedger.Exact || !baseLedger.Fits {
		return nil, fmt.Errorf("exact cache-free allocation evidence is unavailable")
	}
	slots := hotExpertCacheMaxSlots(model, shape, baseLedger)
	if slots > 0 {
		candidate := cloneStrategy(base)
		if err := assignHotExpertCache(candidate, shape, slots); err != nil {
			return nil, err
		}
		ledger := BuildResourceLedger(caps, model, candidate, opts)
		if !ledger.Exact || !ledger.Fits {
			return nil, fmt.Errorf("calculated %d-slot cache does not fit the exact composed ledger", slots)
		}
		candidate.ResourceLedger = &ledger
		return candidate, nil
	}
	// Leftover VRAM after a packed GPU-expert layout is not the policy when
	// auto is on: free whole GPU expert layers until a useful cache fits, then
	// let the ordinary A/B decide against the packed cache-free baseline.
	return hotExpertPriorityCandidate(caps, model, base, opts, baseLedger)
}

func hotExpertPriorityCandidate(caps *detect.Capabilities, model *ModelProfile, base *Strategy, opts Options, baseLedger ResourceLedger) (*Strategy, error) {
	minUseful := hotExpertMinUsefulSlots(model)
	candidate := cloneStrategy(base)
	clearHotExpertCache(candidate)
	slack := hotExpertSlackByGPU(baseLedger)
	maxDrops := 0
	if whole, _, err := hotExpertPinnedLayers(candidate.OTString); err == nil {
		maxDrops = len(whole)
	}
	for drop := 0; drop <= maxDrops; drop++ {
		shape, err := hotExpertCacheShapeFor(caps, model, candidate, opts)
		if err != nil {
			return nil, err
		}
		adjusted := hotExpertLedgerWithSlack(baseLedger, slack)
		// Slot arithmetic may use the exact source ledger plus conservative GGUF
		// demotion deltas, but the resulting placement itself is not exact until
		// the backend admits it. Do not leak this temporary planning authority to
		// the candidate ResourceLedger below.
		slotLedger := adjusted
		slotLedger.Exact = baseLedger.Exact
		slots := hotExpertCacheMaxSlots(model, shape, slotLedger)
		if slots >= minUseful {
			if err := assignHotExpertCache(candidate, shape, slots); err != nil {
				return nil, err
			}
			applyHotExpertCacheLedger(&adjusted, candidate, false)
			if !adjusted.Fits {
				return nil, fmt.Errorf("priority %d-slot cache does not fit the demoted per-device ledger", slots)
			}
			if drop > 0 {
				candidate.HotExpertCacheEvidence += fmt.Sprintf("; priority: demoted %d GPU expert layer(s) so a useful cache is a first-class challenger", drop)
			}
			candidate.ResourceLedger = &adjusted
			return candidate, nil
		}
		gpu := hotExpertTightestGPU(shape, slack)
		layer, ok := hotExpertHighestPinnedLayerOnGPU(candidate.OTString, gpu)
		if !ok {
			break
		}
		if !hotExpertDemotePinnedLayer(candidate, layer) {
			break
		}
		freed := 0
		if layer >= 0 && layer < len(model.RoutedExpertLayerBytes) {
			freed = bytesToMiBCeil(model.RoutedExpertLayerBytes[layer])
		}
		if freed < 1 {
			freed = 1
		}
		slack[gpu] += freed
	}
	return nil, fmt.Errorf("residual per-device VRAM cannot hold the minimum %d useful slots", minUseful)
}

func hotExpertSlackByGPU(ledger ResourceLedger) map[int]int {
	out := make(map[int]int, len(ledger.Devices))
	for _, device := range ledger.Devices {
		out[device.GPU] = device.SlackMB
	}
	return out
}

func hotExpertLedgerWithSlack(base ResourceLedger, slack map[int]int) ResourceLedger {
	out := base
	out.Devices = append([]DeviceResourceLedger(nil), base.Devices...)
	demotedHostMB := 0
	for i := range out.Devices {
		if extra, ok := slack[out.Devices[i].GPU]; ok {
			freed := extra - out.Devices[i].SlackMB
			if freed > 0 {
				out.Devices[i].ModelMB -= freed
				if out.Devices[i].ModelMB < 0 {
					out.Devices[i].ModelMB = 0
				}
				out.Devices[i].RequiredMB -= freed
				if out.Devices[i].RequiredMB < 0 {
					out.Devices[i].RequiredMB = 0
				}
				demotedHostMB += freed
			}
			out.Devices[i].SlackMB = extra
		}
	}
	if demotedHostMB > 0 {
		out.Host.ModelMB += demotedHostMB
		out.Host.RequiredMB += demotedHostMB
		out.Host.SlackMB -= demotedHostMB
	}
	// The source allocation was exact, but changing expert residency changes
	// backend buffer alignment and CUDA-host allocation. GGUF tensor bytes are a
	// conservative planning derivation, not observed allocation evidence. The
	// contained preflight/live admission must restore Exact before promotion.
	out.Exact = false
	if out.Evidence != "" {
		out.Evidence += "+derived-expert-demotion"
	} else {
		out.Evidence = "derived-expert-demotion"
	}
	out.Fits = out.Host.SlackMB >= 0
	for _, device := range out.Devices {
		if device.SlackMB < 0 {
			out.Fits = false
		}
	}
	return out
}

func hotExpertTightestGPU(shape *hotExpertCacheShape, slack map[int]int) int {
	if shape == nil || len(shape.perSlotBytesByGPU) == 0 {
		return -1
	}
	tightest := -1
	bestSlots := int(^uint(0) >> 1)
	for gpu, perSlot := range shape.perSlotBytesByGPU {
		available := int64(slack[gpu]) * hotExpertCacheMiB
		fixed := shape.fixedBytesByGPU[gpu]
		slots := 0
		if perSlot > 0 && available > fixed {
			slots = int((available - fixed) / perSlot)
		}
		if tightest < 0 || slots < bestSlots || (slots == bestSlots && gpu < tightest) {
			tightest = gpu
			bestSlots = slots
		}
	}
	return tightest
}

func hotExpertHighestPinnedLayerOnGPU(ot string, gpu int) (int, bool) {
	if gpu < 0 {
		return -1, false
	}
	whole, _, err := hotExpertPinnedLayers(ot)
	if err != nil {
		return -1, false
	}
	highest := -1
	for _, part := range strings.Split(ot, ",") {
		part = strings.TrimSpace(part)
		dev, ok := hotExpertCUDAIndex(part)
		if !ok || dev != gpu || !strings.Contains(part, expertTensorPattern) {
			continue
		}
		match := hotExpertOTLayerPattern.FindStringSubmatch(part)
		if len(match) != 2 {
			continue
		}
		for _, raw := range strings.Split(match[1], "|") {
			layer, parseErr := strconv.Atoi(raw)
			if parseErr != nil || !whole[layer] {
				continue
			}
			if layer > highest {
				highest = layer
			}
		}
	}
	if highest < 0 {
		return -1, false
	}
	return highest, true
}

func hotExpertCUDAIndex(part string) (int, bool) {
	idx := strings.LastIndex(part, "=")
	if idx < 0 {
		return 0, false
	}
	dev := strings.TrimSpace(part[idx+1:])
	if !strings.HasPrefix(strings.ToUpper(dev), "CUDA") {
		return 0, false
	}
	n, err := strconv.Atoi(strings.TrimPrefix(strings.ToUpper(dev), "CUDA"))
	return n, err == nil
}

func hotExpertDemotePinnedLayer(s *Strategy, layer int) bool {
	if s == nil || layer < 0 || strings.TrimSpace(s.OTString) == "" {
		return false
	}
	var parts []string
	found := false
	demotedGPU := -1
	for _, raw := range strings.Split(s.OTString, ",") {
		part := strings.TrimSpace(raw)
		if part == "" {
			continue
		}
		if !strings.Contains(part, expertTensorPattern) {
			parts = append(parts, part)
			continue
		}
		match := hotExpertOTLayerPattern.FindStringSubmatch(part)
		if len(match) != 2 {
			parts = append(parts, part)
			continue
		}
		var keep []string
		for _, rawLayer := range strings.Split(match[1], "|") {
			if strings.TrimSpace(rawLayer) == strconv.Itoa(layer) {
				found = true
				if gpu, ok := hotExpertCUDAIndex(part); ok {
					demotedGPU = gpu
				}
				continue
			}
			keep = append(keep, rawLayer)
		}
		if len(keep) == 0 {
			continue
		}
		parts = append(parts, strings.Replace(part, match[1], strings.Join(keep, "|"), 1))
	}
	if !found {
		return false
	}
	s.OTString = strings.Join(parts, ",")
	s.NCPUMoE++
	if demotedGPU >= 0 {
		hotExpertRemoveVRAMLedgerLayer(s, demotedGPU)
	}
	return true
}

func hotExpertRemoveVRAMLedgerLayer(s *Strategy, gpu int) {
	if s == nil || gpu < 0 || len(s.VRAMLedger) == 0 {
		return
	}
	ledger := append([]GPULedgerEntry(nil), s.VRAMLedger...)
	for i := range ledger {
		if ledger[i].GPU == gpu && ledger[i].ExpertLayers > 0 {
			ledger[i].ExpertLayers--
			break
		}
	}
	s.VRAMLedger = ledger
}

// finalizeHotExpertCache applies a requested cache after the stable placement
// is complete. Auto/on build leftover or demoted cache-on from exact cache-free
// evidence and return that layout so standard launch actually emits
// --moe-expert-cache. Auto may restore packed after runtime rejection; on and a
// positive slot count fail closed instead of silently serving cache-free.
func finalizeHotExpertCache(caps *detect.Capabilities, model *ModelProfile, opts Options, s *Strategy) (*Strategy, error) {
	if s == nil {
		return nil, nil
	}
	s.BackendSupportsHotExpertCache = backendSupportsHotExpertCache(opts.BackendHelp)

	// A verified automatic winner already carries its exact slots. Revalidate
	// the capability and model layout rather than erasing the measured result.
	requested := hotExpertRequestedSlots(opts)
	priorSlots := s.HotExpertCacheSlots
	priorEvidence := s.HotExpertCacheEvidence
	priorTelemetry := HotExpertCacheTelemetry{
		Observed: s.HotExpertCacheSteps > 0 || s.HotExpertCacheHits > 0 || s.HotExpertCacheMisses > 0,
		Steps:    s.HotExpertCacheSteps,
		Hits:     s.HotExpertCacheHits,
		Misses:   s.HotExpertCacheMisses,
		HitRate:  s.HotExpertCacheHitRate,
	}
	if requested <= 0 && s.HotExpertCacheSlots <= 0 {
		if !hotExpertOptimizerRequested(opts) {
			return s, nil
		}
		candidate, err := hotExpertCacheCandidate(caps, model, s, opts)
		if candidate != nil {
			candidate.BackendSupportsHotExpertCache = s.BackendSupportsHotExpertCache
			return candidate, nil
		}
		if err != nil {
			s.OptimizationExclusions = append(s.OptimizationExclusions, "hot-experts: "+err.Error())
			if hotExpertRequired(opts) {
				return nil, fmt.Errorf("hot experts required but no cache-on placement was admitted: %w", err)
			}
		}
		if hotExpertRequired(opts) {
			return nil, fmt.Errorf("hot experts required but the optimizer produced no cache-on placement")
		}
		return s, nil
	}
	if requested <= 0 {
		requested = s.HotExpertCacheSlots
	}
	if model == nil || model.NumExperts <= 0 || requested > model.NumExperts {
		available := 0
		if model != nil {
			available = model.NumExperts
		}
		return nil, fmt.Errorf("hot experts requested %d cache slots per layer, but the model exposes %d routed experts", requested, available)
	}
	if !s.BackendSupportsHotExpertCache {
		return nil, fmt.Errorf("hot experts requested, but the selected backend does not expose --moe-expert-cache and --moe-expert-cache-inserts")
	}

	base := cloneStrategy(s)
	clearHotExpertCache(base)
	baseOpts := opts
	baseOpts.HotExperts = "off"
	baseOpts.HotExpertCacheSlots = 0
	shape, err := hotExpertCacheShapeFor(caps, model, base, baseOpts)
	if err != nil {
		return nil, fmt.Errorf("hot experts unavailable: %w", err)
	}
	baseLedger := BuildResourceLedger(caps, model, base, baseOpts)
	if !baseLedger.Exact {
		return nil, fmt.Errorf("hot experts need an allocation-measured baseline before %d slots can be admitted; launch once with auto/off first", requested)
	}
	clearHotExpertCache(s)
	if err := assignHotExpertCache(s, shape, requested); err != nil {
		return nil, err
	}
	// A verified automatic winner already proved runtime activity. Recomputing
	// its geometry must not erase that evidence before the direct-start record is
	// saved again. Numeric first-use requests have no prior cache and therefore
	// cannot enter this branch.
	if priorSlots == requested && validateHotExpertCacheTelemetryValue(priorTelemetry) == nil {
		s.HotExpertCacheSteps = priorTelemetry.Steps
		s.HotExpertCacheHits = priorTelemetry.Hits
		s.HotExpertCacheMisses = priorTelemetry.Misses
		s.HotExpertCacheHitRate = priorTelemetry.HitRate
		if priorEvidence != "" {
			s.HotExpertCacheEvidence = priorEvidence
		}
	}
	// Admission deliberately composes the allocation-measured cache-free
	// baseline with the exact GGUF cache charge. The backend's no-alloc path
	// cannot instantiate this cache (weights have no data buffers there), so
	// waiting for a cache-on probe would make first use impossible.
	ledger := BuildResourceLedger(caps, model, s, baseOpts)
	if !ledger.Exact || !ledger.Fits {
		clearHotExpertCache(s)
		return nil, fmt.Errorf("hot-expert cache with %d slots does not fit the complete per-device ledger", requested)
	}
	s.ResourceLedger = &ledger
	return s, nil
}

func applyHotExpertCacheLedger(ledger *ResourceLedger, s *Strategy, allocationAlreadyIncludesCache bool) {
	if ledger == nil || s == nil || s.HotExpertCacheSlots <= 0 || allocationAlreadyIncludesCache {
		return
	}
	for i := range ledger.Devices {
		charge := s.HotExpertCacheVRAMByGPU[ledger.Devices[i].GPU]
		if charge <= 0 {
			continue
		}
		ledger.Devices[i].HotExpertCacheMB += charge
		ledger.Devices[i].RequiredMB += charge
		ledger.Devices[i].SlackMB -= charge
		if ledger.Devices[i].SlackMB < 0 {
			ledger.Fits = false
		}
	}
	if ledger.Evidence != "" {
		ledger.Evidence += "+gguf-hot-expert-cache"
	} else {
		ledger.Evidence = "gguf-hot-expert-cache"
	}
}
