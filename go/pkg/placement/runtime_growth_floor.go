package placement

import "github.com/raketenkater/ggrun/pkg/detect"

// runtimeGrowthFloor keeps exact backend namespaces intact. The only cross-build
// input is an explicitly supplied, provenance-checked parent's serving reserve.
func runtimeGrowthFloor(model *ModelProfile, s *Strategy, opts Options, gpus []detect.GPU) map[int]int {
	out := map[int]int{}
	merge := func(values map[int]int) {
		for gpu, value := range values {
			if old, known := out[gpu]; value >= 0 && (!known || value > old) {
				out[gpu] = value
			}
		}
	}
	merge(RelatedModelRuntimeGraphGrowth(opts.CacheDir, model, gpus, max(1, s.Parallel), backendCacheTag(opts)))
	if opts.HotExpertCacheSlots > 0 {
		free := opts
		free.HotExpertCacheSlots = 0
		merge(RelatedModelRuntimeGraphGrowth(opts.CacheDir, model, gpus, max(1, s.Parallel), backendCacheTag(free)))
	}
	if opts.RuntimeGrowthBaseBackendTag != "" {
		parent := opts
		parent.BackendCacheTag = opts.RuntimeGrowthBaseBackendTag
		parent.HotExpertCacheSlots = 0
		merge(relatedRuntimeGraphGrowth(opts.CacheDir, model, gpus, max(1, s.Parallel), backendCacheTag(parent), true))
	}
	return out
}

// Add only the portion not already present in an observed allocation. When the
// compute/overhead breakdown is missing, retain that aggregate and reserve the
// floor separately: an unlabelled peak cannot prove how much growth it includes.
func applyObservedRuntimeFloor(d *DeviceResourceLedger, growth, compute, overhead int, breakdownKnown bool) {
	included := 0
	if breakdownKnown {
		included = max(0, d.GraphMB-compute-overhead)
	}
	included = min(included, growth)
	d.GraphMB -= included
	d.RuntimeMB, d.RuntimeMeasured = growth, true
	d.RequiredMB += growth - included
	d.SlackMB -= growth - included
}
