//go:build windows

package detect

import "runtime"

// detectPhysicalCores returns physical CPU cores on Windows.
// Uses logical cores / 2 as approximation (HT assumption).
func detectPhysicalCores() int {
	n := runtime.NumCPU()
	if n >= 4 {
		return n / 2
	}
	return n
}

// detectRAMFreeMB returns available RAM on Windows.
// Uses a simple fallback — proper WMI query can be added later.
func detectRAMFreeMB() int {
	return 8192 // safe default for modern Windows systems
}
