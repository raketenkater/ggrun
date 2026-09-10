package main

import (
	"fmt"
	"github.com/raketenkater/ggrun/pkg/placement"
	"strconv"
	"strings"
)

// Required capabilities are checked before placement, so an unsupported
// backend cannot send a valid model through memory recovery or the safe floor.
func validateRequiredHotExpertBackend(req *launchRequest, be *backendInfo) error {
	if req == nil || req.HotExpertsRuntimeDisabled {
		return nil
	}
	policy := strings.ToLower(strings.TrimSpace(req.HotExperts))
	slots, _ := strconv.Atoi(policy)
	if policy != "on" && slots <= 0 {
		return nil
	}
	if be == nil || !placement.BackendSupportsHotExpertCache(be.Help) {
		return fmt.Errorf("hot experts required, but the selected backend does not expose the cache capability; select or install a compatible hot-expert backend before launch")
	}
	return nil
}
