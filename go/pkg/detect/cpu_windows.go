//go:build windows

package detect

import (
	"runtime"
	"syscall"
	"unsafe"
)

var procGlobalMemoryStatusEx = syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")

type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

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
func detectRAMFreeMB() int {
	ram := detectRAMWindows()
	if ram.FreeMB > 0 {
		return ram.FreeMB
	}
	return 8192 // conservative fallback when the Windows API is unavailable
}

func detectRAMWindows() RAMInfo {
	var mem memoryStatusEx
	mem.Length = uint32(unsafe.Sizeof(mem))
	ok, _, _ := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&mem)))
	if ok == 0 {
		return RAMInfo{}
	}
	return RAMInfo{
		TotalMB: int(mem.TotalPhys / 1024 / 1024),
		FreeMB:  int(mem.AvailPhys / 1024 / 1024),
	}
}
