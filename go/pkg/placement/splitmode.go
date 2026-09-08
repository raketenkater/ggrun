package placement

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Split-mode eligibility.
//
// ggrun emits --split-mode layer everywhere. That is the right answer on the
// development rig, but it is right for a reason that does not generalise, and
// the reason was worth finding: both alternatives abort at model load, and
// neither abort inspects the GPU topology at all.
//
//	--split-mode row     -> "failed to load model", core dumped
//	--split-mode tensor  -> "failed to load model", core dumped
//
// Traced to the backend the launch actually runs (a GLM fork, arch glm5next):
// row asks the device registry for "ggml_backend_split_buffer_type" through
// get_proc_address and throws when it comes back null, which it does for CUDA
// because split buffers are implemented only in the SYCL backend; tensor is
// refused by a per-architecture denylist that ends in `default: return true`,
// and glm5next is one of its explicit `return false` cases.
//
// So "layer is correct" is a fact about this architecture and this backend
// build, not about three mismatched cards. A Llama- or Qwen-family model on the
// same GPUs would be offered tensor parallelism, and shipping layer as a
// universal default would deny those users a mode their model supports.
//
// Invariant 8 forbids encoding that reasoning as a rule: "row is SYCL-only"
// is upstream implementation detail that a backend update can change under us,
// and a denylist read from one fork's source is not a claim ggrun can make about
// a binary it did not build. So nothing here is hardcoded. Eligibility is
// probed against the backend's own no-alloc fit oracle and cached per
// (backend build, model architecture, GPU topology); an unprobed mode stays
// unknown and unusable rather than being assumed either way.
//
// This component reports eligibility only. Invariant 7 reserves promotion for
// repeated live A/B evidence, so a mode that loads becomes a candidate the
// optimizer may consider -- never a default it inherits from a probe.

// SplitModeSupport is one --split-mode value's eligibility on this combination.
type SplitModeSupport struct {
	Mode string
	// Supported is true only when the backend loaded the model in this mode.
	Supported bool
	// Probed is false when no attempt has been made, or when the attempt could
	// not be carried out (missing oracle, timeout). An unprobed mode is unknown,
	// which is not the same as unsupported.
	Probed bool
	// Reason carries the backend's own first error line when it refused, so a
	// launch record explains the refusal instead of restating our guess.
	Reason string
}

// Eligible reports whether the optimizer may consider this mode as a candidate.
func (s SplitModeSupport) Eligible() bool { return s.Probed && s.Supported }

// SplitModeReport is the eligibility picture for one launch.
type SplitModeReport struct {
	Modes []SplitModeSupport
	// Key identifies the (backend build, architecture, topology) this was
	// probed against. Evidence recorded under one key says nothing about
	// another, which is what keeps a backend upgrade from inheriting stale
	// capability claims (invariant 9).
	Key string
}

// EligibleModes lists the modes the optimizer may consider, always including
// layer, which every backend implements and which needs no probe.
func (r SplitModeReport) EligibleModes() []string {
	out := []string{"layer"}
	for _, m := range r.Modes {
		if m.Mode != "layer" && m.Eligible() {
			out = append(out, m.Mode)
		}
	}
	return out
}

// Summary renders the report for a launch record.
func (r SplitModeReport) Summary() string {
	if len(r.Modes) == 0 {
		return "split modes unprobed"
	}
	parts := make([]string, 0, len(r.Modes))
	for _, m := range r.Modes {
		switch {
		case !m.Probed:
			parts = append(parts, m.Mode+" unknown")
		case m.Supported:
			parts = append(parts, m.Mode+" ok")
		case m.Reason != "":
			parts = append(parts, fmt.Sprintf("%s refused (%s)", m.Mode, m.Reason))
		default:
			parts = append(parts, m.Mode+" refused")
		}
	}
	return strings.Join(parts, "; ")
}

// probeSplitMode runs the fit oracle in one mode and reports whether the model
// loaded. Replaced in tests; running a real backend is the only honest way to
// answer this, so there is no pure-Go fallback that could quietly guess.
var probeSplitMode = func(ctx context.Context, fitBin, modelPath, mode string, extra []string) (bool, string) {
	args := append([]string{"-m", modelPath, "--split-mode", mode}, extra...)
	cmd := exec.CommandContext(ctx, fitBin, args...)
	out, err := cmd.CombinedOutput()
	if err == nil {
		return true, ""
	}
	return false, firstBackendError(string(out))
}

// firstBackendError picks the line a reader needs out of a failed load. The
// backend prints its diagnosis and then a generic "failed to load model"; the
// diagnosis is the useful half.
func firstBackendError(out string) string {
	var generic string
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		low := strings.ToLower(line)
		switch {
		case strings.Contains(low, "does not support split buffers"),
			strings.Contains(low, "not implemented for architecture"),
			strings.Contains(low, "requires flash_attn"),
			strings.Contains(low, "needs >="):
			return trimTo(line, 160)
		case strings.Contains(low, "failed to load model"),
			strings.Contains(low, "error loading model"):
			if generic == "" {
				generic = trimTo(line, 160)
			}
		}
	}
	return generic
}

func trimTo(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

// SplitModeProbeKey identifies the combination a capability answer belongs to.
// The backend build is identified by size and modification time rather than a
// content hash: a rebuilt oracle must invalidate its own evidence, and hashing
// a 100 MB binary on every launch to learn that is not worth the read.
func SplitModeProbeKey(fitBin, arch string, gpus []detect.GPU) string {
	h := sha256.New()
	fmt.Fprintf(h, "splitmode-v1|%s|%s|", filepath.Base(fitBin), strings.ToLower(strings.TrimSpace(arch)))
	if fi, err := os.Stat(fitBin); err == nil {
		fmt.Fprintf(h, "%d|%d|", fi.Size(), fi.ModTime().UnixNano())
	} else {
		fmt.Fprintf(h, "nostat|")
	}
	fmt.Fprintf(h, "%s", gpuSignatureHash(gpus))
	return hex.EncodeToString(h.Sum(nil))[:24]
}

func splitModeCachePath(cacheDir, key string) string {
	return filepath.Join(cacheDir, "splitmode-"+key+".caps")
}

// LoadSplitModeSupport returns cached eligibility for this combination.
func LoadSplitModeSupport(cacheDir, fitBin, arch string, gpus []detect.GPU) (SplitModeReport, bool) {
	key := SplitModeProbeKey(fitBin, arch, gpus)
	data, err := os.ReadFile(splitModeCachePath(cacheDir, key))
	if err != nil {
		return SplitModeReport{Key: key}, false
	}
	report := SplitModeReport{Key: key}
	for _, line := range strings.Split(string(data), "\n") {
		parts := strings.SplitN(strings.TrimSpace(line), "|", 3)
		if len(parts) < 2 || parts[0] == "" {
			continue
		}
		m := SplitModeSupport{Mode: parts[0], Probed: true, Supported: parts[1] == "1"}
		if len(parts) == 3 {
			m.Reason = parts[2]
		}
		report.Modes = append(report.Modes, m)
	}
	if len(report.Modes) == 0 {
		return report, false
	}
	sort.Slice(report.Modes, func(i, j int) bool { return report.Modes[i].Mode < report.Modes[j].Mode })
	return report, true
}

// RecordSplitModeSupport persists eligibility for this combination.
func RecordSplitModeSupport(cacheDir, fitBin, arch string, gpus []detect.GPU, report SplitModeReport) error {
	if cacheDir == "" {
		return nil
	}
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return err
	}
	key := SplitModeProbeKey(fitBin, arch, gpus)
	var b strings.Builder
	for _, m := range report.Modes {
		if !m.Probed {
			continue
		}
		flag := "0"
		if m.Supported {
			flag = "1"
		}
		fmt.Fprintf(&b, "%s|%s|%s\n", m.Mode, flag, strings.ReplaceAll(m.Reason, "\n", " "))
	}
	return os.WriteFile(splitModeCachePath(cacheDir, key), []byte(b.String()), 0o644)
}

// DetectSplitModeSupport answers which split modes this backend can load this
// model in, preferring cached evidence and probing only what is missing.
//
// layer is never probed: it is the mode every backend implements and the one
// ggrun already emits, so spending an oracle run to confirm it would add launch
// latency for an answer that cannot change the plan.
//
// A probe failure leaves the mode unprobed rather than unsupported. That
// distinction matters: "we could not ask" must not harden into "your hardware
// cannot do this" and then persist.
func DetectSplitModeSupport(cacheDir, fitBin, modelPath, arch string, gpus []detect.GPU,
	extra []string, timeout time.Duration,
) SplitModeReport {
	key := SplitModeProbeKey(fitBin, arch, gpus)
	if cached, ok := LoadSplitModeSupport(cacheDir, fitBin, arch, gpus); ok {
		return cached
	}
	report := SplitModeReport{Key: key}
	report.Modes = append(report.Modes, SplitModeSupport{Mode: "layer", Supported: true, Probed: true})
	if fitBin == "" || modelPath == "" || len(gpus) < 2 {
		// One device cannot be split across, so the alternatives are not merely
		// unsupported, they are meaningless. Report layer and probe nothing.
		return report
	}
	if timeout <= 0 {
		timeout = 90 * time.Second
	}
	for _, mode := range []string{"row", "tensor"} {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		ok, reason := probeSplitMode(ctx, fitBin, modelPath, mode, extra)
		timedOut := ctx.Err() != nil
		cancel()
		m := SplitModeSupport{Mode: mode, Supported: ok, Probed: true, Reason: reason}
		if timedOut && !ok {
			m.Probed = false
			m.Reason = "probe timed out"
		}
		report.Modes = append(report.Modes, m)
	}
	sort.Slice(report.Modes, func(i, j int) bool { return report.Modes[i].Mode < report.Modes[j].Mode })
	_ = RecordSplitModeSupport(cacheDir, fitBin, arch, gpus, report)
	return report
}
