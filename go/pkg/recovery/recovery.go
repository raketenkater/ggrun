package recovery

import (
	"errors"
	"io"
	"math"
	"regexp"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// FailureType classifies how a server load failed.
type FailureType string

var errPlacementPromotion = errors.New("placement promotion restart")

const (
	FailureOOM          FailureType = "oom"
	FailurePinnedFail   FailureType = "pinned_fail"
	FailurePinnedCap    FailureType = "pinned_cap_exceeded"
	FailurePinnedHang   FailureType = "pinned_hang"
	FailureCUDAOOM      FailureType = "cuda_oom"
	FailureRAMOOM       FailureType = "ram_oom"
	FailureUnknownModel FailureType = "unknown_model"
	// FailureBackendCapability is a deterministic mismatch between the launch
	// args and what the active backend build supports — e.g. a CPU-only
	// llama-server told to offload to CUDA0 ("unknown buffer type"). Restarting
	// the identical command can never succeed, so the launcher fails fast.
	FailureBackendCapability FailureType = "backend_capability"
	FailureUnknown           FailureType = "unknown"
)

// Launcher wraps server startup with crash recovery and fallback.
type Launcher struct {
	BinaryPath    string
	Args          []string
	FallbackPath  string // mainline llama-server if ik_llama fails
	MaxRestarts   int
	BackoffBase   time.Duration
	HealthTimeout time.Duration
	KeepAlive     bool
	OnLog         func(string)
	OnFailure     func(FailureType, string)
	OnRestart     func(int, time.Duration)
	OnFallback    func(string)
	OnCUDAOOM     func(device int, allocMB int, args []string) ([]string, *placement.CacheEntry, bool)
	OnPromote     func(logPath string, args []string) ([]string, bool)

	// Quiet keeps the backend's stdout out of the terminal (it still goes to the
	// per-run log file). Used by Claude Code mode, where ggrun hands the terminal
	// to the `claude` client and backend logs must not bleed into its UI.
	Quiet bool

	// LogPath, if set, is where the backend log is written (a stable, discoverable
	// location like .logs/ggrun-server-<port>.log). Empty falls back to a temp file.
	LogPath string

	PlacementCachePath string
	// SuccessEntry, if set, is the computed placement to persist to
	// PlacementCachePath the first time the server loads healthy — so a launch
	// that lands right is reused verbatim next time (OOM-recovery overwrites it
	// with the corrected placement if it has to intervene).
	SuccessEntry     *placement.CacheEntry
	ProbeCacheDir    string
	ProbeModel       *placement.ModelProfile
	ProbeCtxSize     int
	ProbeUBatchSize  int
	ProbeKVQuality   string
	ProbeKVPlacement string
	ProbeBackendTag  string
	ProbeGPUs        []detect.GPU

	lastLogPath string // log written by the most recent runOnce
}

type startupLogCapture struct {
	mu    sync.Mutex
	file  io.Writer
	live  atomic.Bool
	buf   []byte
	start time.Time
}

type startupStreamWriter struct {
	capture *startupLogCapture
	term    io.Writer
}

var startupSpinnerFrames = []string{"|", "/", "-", "\\"}

const startupProgressLogEvery = 10 * time.Second

type startupProgress struct {
	done  int64
	total int64
}

type startupProgressTracker struct {
	pid   int
	paths map[string]int64
	total int64
	// Backend initialization may reopen or seek the same shard, so the current
	// fd offsets can move backwards. Never regress the user-visible progress.
	maxDone int64
}

var startupSplitGGUFName = regexp.MustCompile(`(?i)^(.*)-([0-9]+)-of-([0-9]+)\.gguf$`)

// oomPattern matches OOM markers on word boundaries so model names like
// "Bloom" or words like "room" in log output don't classify as OOM.
var (
	oomPattern        = regexp.MustCompile(`(?i)\b(oom|out of memory)\b`)
	ramOOMPattern     = regexp.MustCompile(`(?i)\bram\b.*\boom\b|\boom\b.*\bram\b`)
	cudaOOMPattern    = regexp.MustCompile(`(?i)allocating\s+([0-9]+(?:\.[0-9]+)?)\s+MiB\s+on device\s+(\d+):\s+cudaMalloc failed: out of memory`)
	cudaDevicePattern = regexp.MustCompile(`(?i)\bcurrent device:\s*(\d+)\b`)
)

// ParseCUDAOOM extracts the CUDA device and allocation size from a llama.cpp
// cudaMalloc OOM line.
func ParseCUDAOOM(line string) (device int, allocMB int, ok bool) {
	m := cudaOOMPattern.FindStringSubmatch(line)
	if m == nil {
		return 0, 0, false
	}
	alloc, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return 0, 0, false
	}
	device, err = strconv.Atoi(m[2])
	if err != nil {
		return 0, 0, false
	}
	return device, int(math.Ceil(alloc)), true
}

// ParseCUDADevice extracts the device from the diagnostic emitted by newer
// llama.cpp CUDA VMM allocation failures. Those failures abort in cuMemCreate
// and report "current device" on a separate line, but omit the requested byte
// count, so ParseCUDAOOM cannot recognize them.
func ParseCUDADevice(line string) (device int, ok bool) {
	m := cudaDevicePattern.FindStringSubmatch(line)
	if m == nil {
		return 0, false
	}
	device, err := strconv.Atoi(m[1])
	if err != nil {
		return 0, false
	}
	return device, true
}
