package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/controller"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

const backendCapabilitySchema = 2

type backendFlagCapability struct {
	Reason   string `json:"reason"`
	HasValue bool   `json:"has_value,omitempty"`
}

type backendCapabilityRecord struct {
	SchemaVersion           int                               `json:"schema_version"`
	Scope                   string                            `json:"scope"`
	BackendIdentity         string                            `json:"backend_identity"`
	ModelIdentity           string                            `json:"model_identity"`
	Architecture            string                            `json:"architecture"`
	Disabled                map[string]backendFlagCapability  `json:"disabled"`
	CPUExpertMMapCapability placement.CPUExpertMMapCapability `json:"cpu_expert_mmap_capability,omitempty"`
	CPUExpertMMapEvidence   string                            `json:"cpu_expert_mmap_evidence,omitempty"`
	SparseLookupCapability  placement.SparseLookupCapability  `json:"sparse_lookup_capability,omitempty"`
	SparseLookupEvidence    string                            `json:"sparse_lookup_evidence,omitempty"`
	UpdatedAt               string                            `json:"updated_at"`
}

func backendCapabilityScope(model *placement.ModelProfile, be *backendInfo) (scope, modelIdentity, backendIdentity string) {
	if model != nil {
		modelIdentity = placement.SpecTargetIdentity(model)
	}
	if be != nil {
		backendIdentity = evidenceBackendCacheTag(be)
	}
	return controller.ScopeKey("backend-capability-v1", backendIdentity, modelIdentity), modelIdentity, backendIdentity
}

func backendCapabilityPath(cacheDir string, model *placement.ModelProfile, be *backendInfo) string {
	if strings.TrimSpace(cacheDir) == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	scope, _, _ := backendCapabilityScope(model, be)
	return filepath.Join(cacheDir, "backend-capabilities", "capability-"+scope+".json")
}

func loadBackendCapabilityRecord(cacheDir string, model *placement.ModelProfile, be *backendInfo) backendCapabilityRecord {
	path := backendCapabilityPath(cacheDir, model, be)
	data, err := os.ReadFile(path)
	if err != nil {
		return backendCapabilityRecord{}
	}
	var record backendCapabilityRecord
	if json.Unmarshal(data, &record) != nil {
		return backendCapabilityRecord{}
	}
	scope, modelIdentity, backendIdentity := backendCapabilityScope(model, be)
	if record.SchemaVersion != backendCapabilitySchema || record.Scope != scope ||
		record.ModelIdentity != modelIdentity || record.BackendIdentity != backendIdentity {
		return backendCapabilityRecord{}
	}
	return record
}

func applyCachedBackendCapabilities(req *launchRequest, cacheDir string, model *placement.ModelProfile, be *backendInfo) {
	if req == nil || be == nil {
		return
	}
	record := loadBackendCapabilityRecord(cacheDir, model, be)
	if record.CPUExpertMMapCapability != "" {
		be.CPUExpertMMapCapability = placement.NormalizeCPUExpertMMapCapability(string(record.CPUExpertMMapCapability))
		be.CPUExpertMMapEvidence = record.CPUExpertMMapEvidence
	}
	if record.SparseLookupCapability != "" {
		be.SparseLookupCapability = placement.NormalizeSparseLookupCapability(string(record.SparseLookupCapability))
		be.SparseLookupEvidence = record.SparseLookupEvidence
	}
	flags := make([]string, 0, len(record.Disabled))
	for flag := range record.Disabled {
		flags = append(flags, flag)
	}
	sort.Strings(flags)
	for _, flag := range flags {
		capability := record.Disabled[flag]
		if userExplicitBackendFlag(req, flag) {
			fmt.Printf("[launch] cached backend rejection for %s is ignored because this launch explicitly supplies it.\n", flag)
			continue
		}
		if disableBackendFlagWithArity(req, flag, capability.Reason, capability.HasValue) {
			if flag == "--swa-full" {
				req.ExtraArgs = setPassthroughBoolFlag(req.ExtraArgs, flag, false)
			}
			fmt.Printf("[launch] applying measured backend capability: %s disabled for this exact model/backend build.\n", flag)
		}
	}
}

func persistBackendCapability(cacheDir string, model *placement.ModelProfile, be *backendInfo, flag, reason string, hasValue bool) error {
	path := backendCapabilityPath(cacheDir, model, be)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	release, err := acquireBackendCapabilityLock(path+".lock", 5*time.Second)
	if err != nil {
		return err
	}
	defer release()
	record := loadBackendCapabilityRecord(cacheDir, model, be)
	scope, modelIdentity, backendIdentity := backendCapabilityScope(model, be)
	if record.Disabled == nil {
		record.Disabled = make(map[string]backendFlagCapability)
	}
	record.SchemaVersion, record.Scope = backendCapabilitySchema, scope
	record.ModelIdentity, record.BackendIdentity = modelIdentity, backendIdentity
	if model != nil {
		record.Architecture = model.ModelArch
	}
	record.Disabled[flag] = backendFlagCapability{Reason: reason, HasValue: hasValue}
	record.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return err
	}
	return atomicBackendCapabilityWrite(path, append(data, '\n'))
}

// persistBackendCPUExpertMMapCapability records live evidence for the exact
// model/backend build. A contradiction is sticky: the next launch must price
// CPU experts as resident instead of repeating a huge anonymous allocation
// under an mmap plan that cannot actually reclaim it.
func persistBackendCPUExpertMMapCapability(cacheDir string, model *placement.ModelProfile, be *backendInfo, capability placement.CPUExpertMMapCapability, evidence string) error {
	path := backendCapabilityPath(cacheDir, model, be)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	release, err := acquireBackendCapabilityLock(path+".lock", 5*time.Second)
	if err != nil {
		return err
	}
	defer release()
	record := loadBackendCapabilityRecord(cacheDir, model, be)
	scope, modelIdentity, backendIdentity := backendCapabilityScope(model, be)
	if record.Disabled == nil {
		record.Disabled = make(map[string]backendFlagCapability)
	}
	record.SchemaVersion, record.Scope = backendCapabilitySchema, scope
	record.ModelIdentity, record.BackendIdentity = modelIdentity, backendIdentity
	if model != nil {
		record.Architecture = model.ModelArch
	}
	record.CPUExpertMMapCapability = placement.NormalizeCPUExpertMMapCapability(string(capability))
	record.CPUExpertMMapEvidence = strings.TrimSpace(evidence)
	record.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return err
	}
	return atomicBackendCapabilityWrite(path, append(data, '\n'))
}

// persistBackendSparseLookupCapability records a live lookup-tier observation
// for the exact model/backend build, like the CPU-expert record above.
func persistBackendSparseLookupCapability(cacheDir string, model *placement.ModelProfile, be *backendInfo, capability placement.SparseLookupCapability, evidence string) error {
	path := backendCapabilityPath(cacheDir, model, be)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	release, err := acquireBackendCapabilityLock(path+".lock", 5*time.Second)
	if err != nil {
		return err
	}
	defer release()
	record := loadBackendCapabilityRecord(cacheDir, model, be)
	scope, modelIdentity, backendIdentity := backendCapabilityScope(model, be)
	if record.Disabled == nil {
		record.Disabled = make(map[string]backendFlagCapability)
	}
	record.SchemaVersion, record.Scope = backendCapabilitySchema, scope
	record.ModelIdentity, record.BackendIdentity = modelIdentity, backendIdentity
	if model != nil {
		record.Architecture = model.ModelArch
	}
	record.SparseLookupCapability = placement.NormalizeSparseLookupCapability(string(capability))
	record.SparseLookupEvidence = strings.TrimSpace(evidence)
	record.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return err
	}
	return atomicBackendCapabilityWrite(path, append(data, '\n'))
}

func acquireBackendCapabilityLock(path string, timeout time.Duration) (func(), error) {
	deadline := time.Now().Add(timeout)
	for {
		file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			_, _ = fmt.Fprintf(file, "%d\n", os.Getpid())
			_ = file.Close()
			return func() { _ = os.Remove(path) }, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return nil, err
		}
		if info, statErr := os.Stat(path); statErr == nil && time.Since(info.ModTime()) > 2*time.Minute {
			_ = os.Remove(path)
			continue
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("timed out waiting for backend capability lock")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func atomicBackendCapabilityWrite(path string, data []byte) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".backend-capability-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if runtime.GOOS == "windows" {
		_ = os.Remove(path)
	}
	return os.Rename(name, path)
}

var repairableGeneratedBackendFlags = map[string]bool{
	"--swa-full": true, "-khad": true, "-muge": true, "-ger": true, "-mqkv": true,
	"--run-time-repack": true, "--ctx-checkpoints": true, "--ctx-checkpoints-interval": true,
	"--checkpoint-min-step": true, "--metrics": true, "-lv": true, "--verbosity": true,
	"--log-verbosity": true, "--defrag-thold": true,
}

func repairableGeneratedBackendFlag(flag string) bool { return repairableGeneratedBackendFlags[flag] }

func rejectedFlagHasValue(args []string, flag string) bool {
	for index, arg := range args {
		if arg != flag || index+1 >= len(args) {
			continue
		}
		next := args[index+1]
		if !strings.HasPrefix(next, "-") {
			return true
		}
		_, err := strconv.ParseFloat(next, 64)
		return err == nil
	}
	return false
}

// validateAndRepairBackendArgs is a bounded parser-only controller loop. It can
// remove only optional flags generated by ggrun, never user input or core model,
// memory, flash-attention, placement, or quality options.
func validateAndRepairBackendArgs(req *launchRequest, cfg *config.Config, model *placement.ModelProfile,
	be *backendInfo, caps *detect.Capabilities, strategy *placement.Strategy, args []string,
) (*placement.Strategy, []string, error) {
	repaired := make(map[string]bool)
	// One parser attempt per unique allowlisted flag, plus the final successful
	// validation after the last removal. This remains finite while supporting a
	// backend dialect that rejects every optional optimization ggrun can emit.
	for attempt := 0; attempt <= len(repairableGeneratedBackendFlags); attempt++ {
		err := validateBackendLaunchArgs(be, args)
		if err == nil {
			return strategy, args, nil
		}
		var rejected *backendArgValidationError
		if !errors.As(err, &rejected) || rejected.Flag == "" || !repairableGeneratedBackendFlag(rejected.Flag) {
			return strategy, args, err
		}
		if userExplicitBackendFlag(req, rejected.Flag) {
			return strategy, args, fmt.Errorf("backend rejected explicitly supplied %s; refusing to change user input: %w", rejected.Flag, err)
		}
		if repaired[rejected.Flag] {
			return strategy, args, fmt.Errorf("backend argument repair repeated %s without progress: %w", rejected.Flag, err)
		}
		if !hasArg(args, rejected.Flag) {
			return strategy, args, err
		}
		hasValue := rejectedFlagHasValue(args, rejected.Flag)
		reason := "backend parser rejected ggrun-generated flag: " + rejected.Diagnostic
		if !disableBackendFlagWithArity(req, rejected.Flag, reason, hasValue) {
			return strategy, args, fmt.Errorf("backend argument repair repeated without progress: %w", err)
		}
		repaired[rejected.Flag] = true
		if persistErr := persistBackendCapability(cfg.CacheDir, model, be, rejected.Flag, reason, hasValue); persistErr != nil {
			fmt.Fprintf(os.Stderr, "[launch] warning: could not persist backend capability: %v\n", persistErr)
		}
		if rejected.Flag == "--swa-full" {
			req.ExtraArgs = setPassthroughBoolFlag(req.ExtraArgs, rejected.Flag, false)
			opts := placementOptionsFromRequest(req, model, be, cfg.CacheDir)
			opts.SkipPlacementCache = true
			next, computeErr := placement.Compute(caps, model, opts)
			if computeErr != nil {
				return strategy, args, fmt.Errorf("backend flag repair re-plan failed: %w", computeErr)
			}
			strategy = applyCalibrationDecision(req, cfg, model, be, caps, next)
			claudeCodeSlotAdjust(strategy, model, req.ClaudeCode, req.ParallelSet, req.BatchSizeSet, req.UBatchSizeSet)
		}
		nextArgs := buildLaunchServerArgs(req, cfg, be, caps, model, strategy)
		if formatCommand(nextArgs) == formatCommand(args) {
			return strategy, args, fmt.Errorf("backend argument repair for %s made no argv change", rejected.Flag)
		}
		fmt.Fprintf(os.Stderr, "[launch] backend parser rejected generated %s; removed it before model load and remembered this exact capability scope\n", rejected.Flag)
		args = nextArgs
	}
	return strategy, args, fmt.Errorf("backend argument repair exceeded %d unique generated flags", len(repairableGeneratedBackendFlags))
}
