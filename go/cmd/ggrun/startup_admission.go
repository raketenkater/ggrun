package main

import (
	"fmt"
	"github.com/raketenkater/ggrun/pkg/recovery"
	"strings"
)

// startupExactAdmissionFailure classifies a failed candidate without choosing
// a recovery configuration or inventing an allocation size.
func startupExactAdmissionFailure(logData string, cause error) error {
	if device, mb, _, ok := startupLogCUDAOOMDetailed(logData); ok {
		return exactAdmissionError(exactAdmissionCUDAOOM, fmt.Sprintf(" on device %d allocating %d MiB", device, mb), cause)
	}
	// CUDA graph instantiation and VMM failures can omit the requested size.
	// They still prove that this argv failed admission. Keep size unknown;
	// neither a guessed VRAM fraction nor a tensor-buffer charge belongs here.
	lines := strings.Split(logData, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if !strings.Contains(strings.ToLower(lines[i]), "cuda error: out of memory") {
			continue
		}
		detail := " (allocation size unreported)"
		for j := i + 1; j < len(lines) && j <= i+3; j++ {
			if device, ok := recovery.ParseCUDADevice(lines[j]); ok {
				detail = fmt.Sprintf(" on device %d (allocation size unreported)", device)
				break
			}
		}
		return exactAdmissionError(exactAdmissionCUDAOOM, detail, cause)
	}
	return nil
}
