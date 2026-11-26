package placement

import (
	"fmt"
	"strings"

	"github.com/raketenkater/ggrun/pkg/detect"
)

// Dense CPU offload is admitted by a pooled check (all free VRAM plus free RAM
// against the whole plan), so KV the plan assigns to a GPU was never compared
// with that GPU. The per-device ledger is the account that does: a device that
// carries KV and has negative slack cannot hold its share, whatever the pool
// says. The automatic context search treats such a candidate as not fitting,
// so it lowers the context or moves KV to the host; dry run refuses it.

// DenseOffloadKVOverflow lists the devices of a dense CPU-offload plan whose
// ledger carries KV and has negative slack. Other plan types return nil.
func DenseOffloadKVOverflow(s *Strategy, ledger *ResourceLedger) []DeviceResourceLedger {
	if s == nil || ledger == nil || s.Type != DenseCPUOffload {
		return nil
	}
	var over []DeviceResourceLedger
	for _, device := range ledger.Devices {
		if device.ContextMB > 0 && device.SlackMB < 0 {
			over = append(over, device)
		}
	}
	return over
}

// FormatKVOverflow names each overflowing device with its KV and slack.
func FormatKVOverflow(over []DeviceResourceLedger) string {
	parts := make([]string, 0, len(over))
	for _, device := range over {
		parts = append(parts, fmt.Sprintf("CUDA%d KV %d MiB, slack %d MiB", device.GPU, device.ContextMB, device.SlackMB))
	}
	return strings.Join(parts, "; ")
}

// denseOffloadKVFitError prices one resolved dense CPU-offload candidate on
// the per-device ledger and refuses it when KV on a GPU does not fit there.
func denseOffloadKVFitError(caps *detect.Capabilities, model *ModelProfile, s *Strategy, opts Options) error {
	if s == nil || s.Type != DenseCPUOffload || s.KVPlacement == "cpu" {
		return nil
	}
	ledger := BuildResourceLedger(caps, model, s, opts)
	if over := DenseOffloadKVOverflow(s, &ledger); len(over) > 0 {
		return fmt.Errorf("dense CPU offload KV does not fit its GPU at ctx %d (%s)", s.ContextSize, FormatKVOverflow(over))
	}
	return nil
}
