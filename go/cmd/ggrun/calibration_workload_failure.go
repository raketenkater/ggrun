package main

import (
	"errors"
	"strings"

	"github.com/raketenkater/ggrun/pkg/server"
)

// candidatePostStartCUDAOOM classifies only an OOM in the portion of a
// candidate log after its most recent model-loaded marker. EOF and cancellation
// remain inconclusive unless this candidate's own backend log proves the OOM.
func candidatePostStartCUDAOOM(p *server.Process, cause error) (class, reason string, ok bool) {
	if p == nil || p.LogBuf == nil {
		return "", "", false
	}
	return candidatePostStartCUDAOOMFromLog(p.LogBuf.String(), cause)
}

func candidatePostStartCUDAOOMFromLog(logData string, cause error) (class, reason string, ok bool) {
	lines := strings.Split(logData, "\n")
	loadedAt, loaded := runtimeGrowthWindowStart(lines)
	if !loaded || loadedAt < 0 || loadedAt >= len(lines) {
		return "", "", false
	}
	postStart := strings.Join(lines[loadedAt:], "\n")
	if !strings.Contains(strings.ToLower(postStart), "cuda error: out of memory") {
		return "", "", false
	}
	err := startupExactAdmissionFailure(postStart, cause)
	if err == nil {
		return "", "", false
	}
	class, reason = exactAdmissionFailureEvidence(err)
	if class == "" || reason == "" || !errors.Is(err, cause) {
		return "", "", false
	}
	return class, reason, true
}
