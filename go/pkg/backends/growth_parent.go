package backends

import (
	"encoding/hex"
	"os"
	"strings"
)

// HotExpertGrowthParent returns a reproducible parent for reserve-only evidence.
// It grants neither allocation authority nor performance equivalence to the
// overlay. Only an ordered reviewed hot-expert patch prefix may differ.
func HotExpertGrowthParent(overlay Backend, installed []Backend) *Backend {
	if overlay.BaseTag == "" || len(overlay.Features) != 1 || overlay.Features[0] != "hot-experts" {
		return nil
	}
	for _, base := range installed {
		if base.Tag != overlay.BaseTag || base.Tag == overlay.Tag || len(base.Features) != 0 {
			continue
		}
		if _, err := hex.DecodeString(base.Commit); err != nil {
			return nil
		}
		if len(base.Commit) != 40 || base.Commit != overlay.Commit || strings.TrimSpace(base.GitURL) == "" || base.GitURL != overlay.GitURL {
			return nil
		}
		if stat, err := os.Stat(base.Path); err != nil || !stat.Mode().IsRegular() {
			return nil
		}
		recipe, err := ComposeRecipe(base, "hot-experts")
		if err != nil {
			return nil
		}
		names := recipe.PatchNames()
		// Old reviewed overlays may lack later telemetry/dedup patches. They still
		// need the original feature patch, and unknown/reordered patches refuse.
		basePatchCount := len(names) - len(FeatureByName("hot-experts").Patches)
		if len(base.AppliedPatches) != basePatchCount || len(overlay.AppliedPatches) <= basePatchCount || len(overlay.AppliedPatches) > len(names) {
			return nil
		}
		for i, name := range overlay.AppliedPatches {
			if name != names[i] {
				return nil
			}
		}
		for i, name := range base.AppliedPatches {
			if i >= len(names) || name != names[i] {
				return nil
			}
		}
		copy := base
		return &copy
	}
	return nil
}
