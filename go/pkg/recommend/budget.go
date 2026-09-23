package recommend

import "github.com/raketenkater/ggrun/pkg/detect"

// PlanningCapabilities applies recommendation limits to installed capacity.
// This is an advisory view, never live allocation evidence. Busy processes must
// not change what this machine can run after they stop. GPU selection is applied
// by the caller before headroom is distributed.
func PlanningCapabilities(caps *detect.Capabilities, ramBudgetMB, ramLimitPercent, vramHeadroomMB, ramHeadroomMB int) *detect.Capabilities {
	if caps == nil {
		return nil
	}
	out := *caps
	out.GPUs = append([]detect.GPU(nil), caps.GPUs...)
	// An explicit ceiling overrides the percentage, as it does at launch.
	if ramBudgetMB > 0 {
		out.RAM.TotalMB = min(out.RAM.TotalMB, ramBudgetMB)
	} else if ramLimitPercent > 0 && ramLimitPercent < 100 {
		out.RAM.TotalMB = out.RAM.TotalMB * ramLimitPercent / 100
	}
	out.RAM.TotalMB = max(0, out.RAM.TotalMB-ramHeadroomMB)
	out.RAM.FreeMB = out.RAM.TotalMB
	return detect.ApplyVRAMHeadroom(&out, vramHeadroomMB)
}
