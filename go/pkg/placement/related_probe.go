package placement

import (
	"path/filepath"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// relatedProbeScope identifies the measured shape of a neighboring probe.
// Only context, ubatch, and KV settings may vary for these prediction inputs;
// the artifact, backend/features, hardware, and slot count must match.
type relatedProbeScope struct {
	Context  int
	UBatch   int
	Parallel int
}

func matchingRelatedProbeScope(name, content string, model *ModelProfile, gpus []detect.GPU, parallel int, backendTag string) (relatedProbeScope, bool) {
	var scope relatedProbeScope
	if model == nil {
		return scope, false
	}
	fields := map[string]string{}
	modelName := ""
	headerSeen := false
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# Probe cache for ") {
			modelName = strings.TrimPrefix(line, "# Probe cache for ")
		}
		if !strings.HasPrefix(line, "# ctx=") {
			continue
		}
		if headerSeen {
			return scope, false
		}
		headerSeen = true
		for _, field := range strings.Fields(strings.TrimPrefix(line, "# ")) {
			key, value, ok := strings.Cut(field, "=")
			if !ok {
				return scope, false
			}
			if _, duplicate := fields[key]; duplicate {
				return scope, false
			}
			fields[key] = value
		}
	}
	for _, key := range []string{"ctx", "ubatch", "kv_quality", "kv_placement", "backend", "gpu_sig", "parallel"} {
		if _, present := fields[key]; !present {
			return scope, false
		}
	}
	var err error
	if scope.Context, err = strconv.Atoi(fields["ctx"]); err != nil || scope.Context <= 0 {
		return scope, false
	}
	if scope.UBatch, err = strconv.Atoi(fields["ubatch"]); err != nil || scope.UBatch <= 0 {
		return scope, false
	}
	if scope.Parallel, err = strconv.Atoi(fields["parallel"]); err != nil || scope.Parallel < 0 {
		return scope, false
	}
	if backendTag == "" {
		backendTag = "llama"
	}
	recordedBackend := fields["backend"]
	if recordedBackend == "" {
		recordedBackend = "llama"
	}
	if modelName != filepath.Base(model.Path) || fields["gpu_sig"] != gpuSignatureHash(gpus) ||
		recordedBackend != backendTag || probeParallelKey(scope.Parallel) != probeParallelKey(parallel) {
		return scope, false
	}
	// The existing filename hashes the complete artifact identity, including
	// shard stats and model geometry. Reconstructing it validates old records
	// without accepting a same-basename replacement or discarding valid probes.
	expected := probeCachePath(".", model, scope.Context, scope.UBatch,
		fields["kv_quality"], fields["kv_placement"], backendTag, gpus, probeParallelKey(scope.Parallel))
	return scope, name == filepath.Base(expected)
}
