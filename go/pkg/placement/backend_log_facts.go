package placement

import "strings"

// BackendLogFacts is what the planner's own log parsers read from a backend's
// load log. It exists for the backend contract check (review I-05): a backend
// whose log wording changes would otherwise leave these silently zero and the
// planner falling back to estimates.
type BackendLogFacts struct {
	HostBuffersMB int     // CPU / CUDA_Host model, KV, compute and output buffers
	HostModelMB   int     // the model-weights share of HostBuffersMB
	KVTotalMB     float64 // KV across devices and regions
}

// ParseBackendLogFacts runs the planner's parsers on a load log.
func ParseBackendLogFacts(log string) BackendLogFacts {
	// The weights share is what the same parser reads once the KV, compute
	// and output lines are set aside: its default branch is the model buffer.
	var weights []string
	for _, line := range strings.Split(log, "\n") {
		if strings.Contains(line, "KV buffer size =") || strings.Contains(line, "compute buffer size =") || strings.Contains(line, "output buffer size =") {
			continue
		}
		weights = append(weights, line)
	}
	return BackendLogFacts{
		HostBuffersMB: parseHostBuffersFromLog(log),
		HostModelMB:   parseHostBuffersFromLog(strings.Join(weights, "\n")),
		KVTotalMB:     parseKVBufferTotalMB(log),
	}
}
