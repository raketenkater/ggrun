package placement

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Hot experts cannot be sized until the cache-free baseline for the exact
// launch shape has been allocation-measured, and only a completed launch
// records that. finalizeHotExpertCache therefore bootstraps: it serves the
// cache-free plan once and promises the cache on "the next launch of this
// shape".
//
// That promise was never kept on this rig. Measured 2026-09-07 across four
// consecutive launches with identical user settings:
//
//	--n-cpu-moe  41 -> 40 -> 41 -> 40
//	distinct allocation-placement identities for ctx 287744 / ubatch 256: 4
//
// The launch that measures shape A records evidence which changes the next
// plan to shape B, whose baseline is unmeasured, so it bootstraps and records
// evidence that flips the plan back to A. Satisfying the precondition is what
// destroys it, and hot experts is structurally unreachable: it has never once
// engaged, through every fix to the cache-sizing code, because none of that
// code is on this path.
//
// The pin closes the loop. When a bootstrap serves a cache-free baseline, it
// records the topology it is about to measure; the next launch of the same
// scope reproduces that topology instead of re-deriving one, so the
// measurement applies by construction. The pin is consumed as soon as the
// cache engages, and it never invents a plan -- it can only replay one this
// planner already produced and a real launch already measured.

// HotExpertBootstrapPin is the topology a bootstrap launch measured.
type HotExpertBootstrapPin struct {
	// Shape coordinates the pin is only valid under. A pin from a different
	// context or microbatch describes a different allocation entirely.
	ContextSize int
	UBatch      int
	Parallel    int

	// The MoE topology to reproduce.
	NCPUMoE     int
	OTString    string
	TensorSplit []float64
	SplitMode   string

	RecordedAt time.Time
}

// Valid reports whether the pin carries a topology worth replaying.
func (p HotExpertBootstrapPin) Valid() bool {
	return p.ContextSize > 0 && p.UBatch > 0 && p.NCPUMoE > 0
}

// AppliesTo reports whether this pin describes the shape the planner is
// currently working in. The pin replays a MoE topology, not a whole plan, so
// the coordinates that decide allocation must already agree.
func (p HotExpertBootstrapPin) AppliesTo(s *Strategy) bool {
	if !p.Valid() || s == nil {
		return false
	}
	return s.ContextSize == p.ContextSize &&
		s.UBatchSize == p.UBatch &&
		max(1, s.Parallel) == max(1, p.Parallel)
}

// Describe renders the pin for a launch banner.
func (p HotExpertBootstrapPin) Describe() string {
	return fmt.Sprintf("n-cpu-moe %d at ctx %d / ubatch %d", p.NCPUMoE, p.ContextSize, p.UBatch)
}

func hotExpertPinPath(cacheDir, modelBasename string) string {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	safe := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '.', r == '_':
			return r
		default:
			return '_'
		}
	}, modelBasename)
	return filepath.Join(cacheDir, "hot_expert_bootstrap_"+safe+".pin")
}

// RecordHotExpertBootstrapPin stores the topology a bootstrap launch is about
// to measure, so the next launch of the same scope reproduces it.
func RecordHotExpertBootstrapPin(cacheDir, modelPath string, s *Strategy) error {
	if s == nil || s.ContextSize <= 0 || s.UBatchSize <= 0 || s.NCPUMoE <= 0 {
		return nil
	}
	path := hotExpertPinPath(cacheDir, CalibrationModelBasename(modelPath))
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	release, err := acquirePlacementLock(path+".lock", 5*time.Second)
	if err != nil {
		return err
	}
	defer release()
	split := make([]string, len(s.TensorSplit))
	for i, v := range s.TensorSplit {
		split[i] = strconv.FormatFloat(v, 'g', -1, 64)
	}
	body := fmt.Sprintf("# Hot-expert bootstrap topology for %s\n"+
		"PIN_CTX=%d\nPIN_UBATCH=%d\nPIN_PARALLEL=%d\nPIN_NCPUMOE=%d\n"+
		"PIN_SPLIT_MODE=%s\nPIN_TENSOR_SPLIT=%s\nPIN_AT=%d\nPIN_OT=%s\n",
		CalibrationModelBasename(modelPath),
		s.ContextSize, s.UBatchSize, max(1, s.Parallel), s.NCPUMoE,
		s.SplitMode, strings.Join(split, ","), time.Now().Unix(), s.OTString)
	return atomicWriteFile(path, []byte(body), 0o644)
}

// MeasuredHotExpertBootstrapPin returns the pinned topology, or a zero value.
func MeasuredHotExpertBootstrapPin(cacheDir, modelPath string) HotExpertBootstrapPin {
	data, err := os.ReadFile(hotExpertPinPath(cacheDir, CalibrationModelBasename(modelPath)))
	if err != nil {
		return HotExpertBootstrapPin{}
	}
	var p HotExpertBootstrapPin
	for _, line := range strings.Split(string(data), "\n") {
		key, raw, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		// OT strings contain '=' (blk...=CUDA1), so only trim the key side and
		// keep the value verbatim.
		switch key {
		case "PIN_CTX":
			p.ContextSize, _ = strconv.Atoi(strings.TrimSpace(raw))
		case "PIN_UBATCH":
			p.UBatch, _ = strconv.Atoi(strings.TrimSpace(raw))
		case "PIN_PARALLEL":
			p.Parallel, _ = strconv.Atoi(strings.TrimSpace(raw))
		case "PIN_NCPUMOE":
			p.NCPUMoE, _ = strconv.Atoi(strings.TrimSpace(raw))
		case "PIN_SPLIT_MODE":
			p.SplitMode = strings.TrimSpace(raw)
		case "PIN_TENSOR_SPLIT":
			for _, f := range strings.Split(strings.TrimSpace(raw), ",") {
				if v, convErr := strconv.ParseFloat(f, 64); convErr == nil {
					p.TensorSplit = append(p.TensorSplit, v)
				}
			}
		case "PIN_AT":
			if v, convErr := strconv.ParseInt(strings.TrimSpace(raw), 10, 64); convErr == nil {
				p.RecordedAt = time.Unix(v, 0)
			}
		case "PIN_OT":
			p.OTString = strings.TrimRight(raw, "\r")
		}
	}
	return p
}

// ClearHotExpertBootstrapPin removes the pin once it has served its purpose.
//
// Called when the cache actually engages: the pin exists only to carry one
// measurement forward to the launch that consumes it, and a stale pin would
// keep replaying an old topology after the evidence has moved on.
func ClearHotExpertBootstrapPin(cacheDir, modelPath string) error {
	err := os.Remove(hotExpertPinPath(cacheDir, CalibrationModelBasename(modelPath)))
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// applyHotExpertBootstrapPin replays the pinned MoE topology onto a freshly
// computed strategy.
//
// Only the topology fields move. Everything the fit stage decided about memory
// stays as computed, and the caller still runs exact admission over the result,
// so a pin that no longer fits is rejected by the ordinary path rather than
// trusted because it was recorded.
func applyHotExpertBootstrapPin(s *Strategy, p HotExpertBootstrapPin) bool {
	if s == nil || !p.AppliesTo(s) {
		return false
	}
	if s.NCPUMoE == p.NCPUMoE && s.OTString == p.OTString {
		return false // already the pinned shape; nothing to replay
	}
	s.NCPUMoE = p.NCPUMoE
	s.OTString = p.OTString
	if len(p.TensorSplit) > 0 {
		s.TensorSplit = append([]float64(nil), p.TensorSplit...)
	}
	if p.SplitMode != "" {
		s.SplitMode = p.SplitMode
	}
	return true
}
