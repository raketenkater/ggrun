package main

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/advisor"
	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/chattemplate"
	"github.com/raketenkater/ggrun/pkg/claudeauto"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/gguf"
	"github.com/raketenkater/ggrun/pkg/libhub"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/server"
)

type claudeAutoRuntime struct {
	reviewer     *server.Process
	reviewerLog  io.Closer
	reviewerPort int
	reviewerGPU  int
	companion    *claudeCompanionProfile
	router       *claudeauto.Router
}

// workerRouteEnabled is deliberately separate from reviewer availability.
// Qwen3.5-2B is a review-only profile: its process is a valid classifier
// destination, but advertising it as local-fast would send ordinary cheap-tier
// background work into a model the user selected only for security reviews.
func (r *claudeAutoRuntime) workerRouteEnabled() bool {
	return r != nil && r.reviewerPort > 0 && r.companion != nil && r.companion.ServesWorkers
}

func claudeAutoReviewerNeeded(extraArgs []string) bool {
	if disabledEnv("GGRUN_CLAUDE_AUTO_REVIEWER") {
		return false
	}
	permissionArgs := claudeCodePermissionArgs(extraArgs)
	// "inherit" can still resolve to Auto in settings.json. Starting the small
	// reviewer is harmless when no classifier calls arrive and keeps inheritance
	// functional when the user's configured default is Auto.
	return permissionArgs == nil || (len(permissionArgs) == 2 && permissionArgs[1] == "auto")
}

// claudeCompanionNeeded includes the cheap-tier worker role. Claude Code uses
// that lane in every permission mode, while only Auto mode emits classifier
// requests. The existing disable switch remains authoritative for users who
// prefer to spend no separate companion memory.
func claudeCompanionNeeded(extraArgs []string) bool {
	return !disabledEnv("GGRUN_CLAUDE_AUTO_REVIEWER")
}

func disabledEnv(key string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "0", "false", "no", "off", "disabled":
		return true
	default:
		return false
	}
}

// claudeReviewerCompanionName is kept stable so existing Qwen measurements
// remain useful when NanoBeige is not installed.
const claudeReviewerCompanionName = "claude-auto-reviewer"

const claudeNanoCompanionName = "claude-worker-nanbeige42"

// claudeSmallReviewerCompanionName names the small/light Qwen3.5-2B review-only
// lane. It is distinct from the worker/reviewer so the placement ledger keeps
// separate VRAM footprints for the two Qwen profiles.
const claudeSmallReviewerCompanionName = "claude-auto-small-reviewer"

type claudeCompanionProfile struct {
	Name           string
	DisplayName    string
	ModelPath      string
	BackendPath    string
	ModelMarker    string
	MeasurementKey string
	KVType         string
	// ContextTokens and ReservationVRAMMB are one indivisible capacity
	// contract. The measurement key includes both context and KV type so a 64k
	// sample can never under-reserve a later 128k/256k reviewer.
	ContextTokens     int
	ReservationVRAMMB int
	NanoBeige         bool
	// ServesWorkers authorizes ordinary local-fast traffic in addition to
	// classifier/security reviews. It is false for the Qwen3.5-2B review-only
	// profile; review availability alone must never imply worker capability.
	ServesWorkers bool
	// Artifact is the pinned model this profile runs. The launcher installs
	// exactly this one: ReservationVRAMMB describes THIS model, so installing
	// another artifact under it under-reserves the companion.
	Artifact claudeauto.ModelSpec
}

// claudeReviewerArtifactSpec returns the pinned artifact a profile runs. The
// NanoBeige worker is provisioned through the advisor, not the reviewer cache,
// so it has no pinned reviewer spec.
func claudeReviewerArtifactSpec(p *claudeCompanionProfile) claudeauto.ModelSpec {
	if p == nil || p.NanoBeige {
		return claudeauto.ModelSpec{}
	}
	if p.Artifact.Name != "" {
		return p.Artifact
	}
	return claudeauto.DefaultReviewerSpec()
}

func (p *claudeCompanionProfile) companionMeasurementKey() string {
	if p == nil {
		return ""
	}
	if key := strings.TrimSpace(p.MeasurementKey); key != "" {
		return key
	}
	return p.Name
}

func (p *claudeCompanionProfile) reviewerContextTokens() int {
	if p != nil && p.ContextTokens > 0 {
		return p.ContextTokens
	}
	return claudeReviewerContextTokens
}

// The Qwen3.5-4B worker/reviewer runs at 128k. Its measured/derived Q8 KV is
// 1088 MiB per 64k (8 attention layers, 4 KV heads, 256-wide K/V), so doubling
// the old 64k profile adds ~1088 MiB. Six GiB covers weights, KV, graph, and
// allocator jitter until the context-scoped live measurement supersedes it.
const claudeReviewerReservationVRAMMB = 6144

// The 2B review-only model can afford its full native 262k context. The live
// 64k process occupies ~2050 MiB and its measured Q8 KV is 408 MiB; scaling KV
// to 262k predicts ~3274 MiB total. Four GiB keeps the complete control plane
// independent of the busy main model with conservative allocator headroom.
const claudeSmallReviewerReservationVRAMMB = 4096

// claudeNanoReservationVRAMMB covers the measured 8843 MiB peak for the real
// 64k/Q8 worker profile on the RTX 3090 Ti, rounded up before per-host samples
// replace it. Nanbeige's 44 logical KV layers make it materially larger than
// its 2.4 GiB weight file; reserving only the weights would corrupt placement.
const claudeNanoReservationVRAMMB = 11264

const (
	claudeReviewerContextTokens      = 131072
	claudeSmallReviewerContextTokens = 262144
	claudeNanoReviewerContextTokens  = 131072
)

var (
	claudeNanoArtifactReady = func(path string) bool {
		return advisor.VerifyArtifact(path, advisor.DefaultArtifact) == nil
	}
	claudeNanoBackend    = reviewedSupportBackend
	claudeNanoGPUCapable = func(path string) bool {
		_, err := claudeReviewerGPUDevice(path, claudeReviewerBackendEnv(path, nil))
		return err == nil
	}
)

func resolveClaudeCompanionProfile(req *launchRequest, cacheDir string) *claudeCompanionProfile {
	if req != nil && req.ReviewerProfile != nil {
		return req.ReviewerProfile
	}
	appHome := ""
	if req != nil {
		appHome = strings.TrimSpace(req.AppHome)
	}
	if appHome == "" {
		appHome = backends.AppHome()
	}

	var profile *claudeCompanionProfile
	// An explicit reviewer model/backend override (env) retains its historical
	// precedence and always outranks both ggrun's automatic choice and the
	// --claude-reviewer flag. Model capability is a separate question: an
	// arbitrary GGUF is safe to use for the narrow classifier lane only. Never
	// advertise it as the local-fast worker merely because it was supplied in an
	// environment variable. A backend-only override still runs ggrun's pinned,
	// worker-verified 4B artifact and keeps that artifact's worker capability.
	modelOverride := strings.TrimSpace(os.Getenv("GGRUN_CLAUDE_REVIEWER_MODEL")) != ""
	backendOverride := strings.TrimSpace(os.Getenv("GGRUN_CLAUDE_REVIEWER_BIN")) != ""
	envOverride := modelOverride || backendOverride
	override := ""
	if req != nil {
		override = strings.TrimSpace(req.ClaudeReviewerOverride)
	}
	// --claude-reviewer nanbeige forces the NanoBeige4.2 MoE worker. It is an
	// explicit opt-in: auto mode no longer defaults to NanoBeige. The readiness
	// checks always gate — a forced NanoBeige with no installed artifact degrades
	// to the Qwen profile rather than launching a broken worker.
	forceNano := !envOverride && override == claudeReviewerNanbeige
	// --claude-reviewer qwen2b forces the small/light Qwen3.5-2B review-only
	// profile (the cheap safety-review lane). It is the one role with no worker
	// duty: only the classifier/security-monitor lane targets the small reviewer.
	forceSmall := !envOverride && override == claudeReviewerQwen2B
	// auto (the default) and --claude-reviewer qwen both resolve to the Qwen3.5-4B
	// worker/reviewer: the big/fast dense companion that handles BOTH the
	// safety-review calls and the cheap-tier worker calls (the user's measured
	// best). NanoBeige is never the automatic choice — it needs the explicit
	// --claude-reviewer nanbeige opt-in above.
	if forceNano && cacheDir != "" && claudeNanoArtifactReady(advisor.DefaultModelPath(cacheDir)) {
		modelPath := advisor.DefaultModelPath(cacheDir)
		if backendPath, err := claudeNanoBackend(); err == nil && claudeNanoGPUCapable(backendPath) {
			profile = &claudeCompanionProfile{
				Name: claudeNanoCompanionName, DisplayName: "Nanbeige4.2-3B",
				ModelPath: modelPath, BackendPath: backendPath, ModelMarker: modelPath,
				MeasurementKey:    claudeNanoCompanionName + "-ctx131072-kv-q4_0",
				KVType:            "q4_0",
				ContextTokens:     claudeNanoReviewerContextTokens,
				ReservationVRAMMB: claudeNanoReservationVRAMMB, NanoBeige: true,
				ServesWorkers: true,
			}
		}
	}
	if profile == nil && forceSmall {
		modelPath := claudeauto.SmallReviewerModelPath(appHome)
		profile = &claudeCompanionProfile{
			Name: claudeSmallReviewerCompanionName, DisplayName: claudeauto.DefaultSmallReviewerDisplayName,
			ModelPath: modelPath, ModelMarker: modelPath,
			// The 2B profile owns its own measurement key so it never collides with
			// the 4B footprint or the stale legacy-name measurement.
			MeasurementKey:    "claude-reviewer-qwen35-2b-q4-k-m-ctx262144-kv-q8_0",
			KVType:            "q8_0",
			ContextTokens:     claudeSmallReviewerContextTokens,
			ReservationVRAMMB: claudeSmallReviewerReservationVRAMMB,
			ServesWorkers:     false,
			Artifact:          claudeauto.SmallReviewerSpec(),
		}
	}
	// Fallback (env override, degraded NanoBeige, or any unmapped value): the
	// Qwen3.5-4B worker/reviewer profile. A model env override carries its own
	// model path through the helper functions, but remains review-only because
	// ggrun has not verified the arbitrary artifact as a general worker.
	if profile == nil {
		modelPath := claudeauto.ReviewerModelPath(appHome)
		profile = &claudeCompanionProfile{
			Name: claudeReviewerCompanionName, DisplayName: claudeauto.DefaultReviewerDisplayName,
			ModelPath: modelPath, ModelMarker: modelPath,
			// The Qwen3.5-4B profile owns its own measurement key so it never
			// inherits the stale 2B footprint stored under the legacy name.
			MeasurementKey:    "claude-reviewer-qwen35-4b-q4-k-m-ctx131072-kv-q8_0",
			KVType:            "q8_0",
			ContextTokens:     claudeReviewerContextTokens,
			ReservationVRAMMB: claudeReviewerReservationVRAMMB,
			ServesWorkers:     !modelOverride,
			Artifact:          claudeauto.DefaultReviewerSpec(),
		}
	}
	if req != nil {
		req.ReviewerProfile = profile
	}
	return profile
}

// claudeReviewerReservation builds the placement companion for the Auto reviewer
// when the launch needs one. The GPU preference mirrors the legacy walk —
// least-valuable (slowest-link, smallest) first, main GPU last — but expressed
// as data so the planner owns the final seat and the main model packs around it.
// CPU fallback is deliberately disallowed here: placement has no companion-RAM
// ledger, so a hidden 3-4 GiB CPU worker could invalidate strict no-mmap math.
func claudeReviewerReservation(req *launchRequest, caps *detect.Capabilities, cacheDir string) *placement.CompanionReservation {
	if req == nil || !req.ClaudeCode || req.ClaudeReviewerDisabled || !claudeCompanionNeeded(nil) || req.CPUMode {
		return nil
	}
	if caps == nil || len(caps.GPUs) == 0 {
		return nil
	}
	// Prefer what the reviewer actually took on a previous launch. The constant
	// below is a conservative bound and always overshoots -- measured 2114 MiB
	// against 2600 reserved -- and it overshoots on the least valuable GPU by
	// design, which is where withheld VRAM is worth the most.
	profile := resolveClaudeCompanionProfile(req, cacheDir)
	vramMB := profile.ReservationVRAMMB
	if cacheDir != "" {
		if measured := placement.MeasuredCompanionVRAMMB(cacheDir, profile.companionMeasurementKey()); measured > 0 {
			vramMB = measured
		}
	}
	// A reviewer left behind by a previous ggrun is already counted in the VRAM
	// nvidia-smi reports as used, so reserving the full amount on top charges one
	// process twice. Measured on this project: 2096 MiB resident from the old run
	// plus a 2600 MiB reservation, ~4.7 GB withheld for a 2.1 GB helper, which
	// cost the 3060 two expert layers -- it took 4 where an unoccupied card took
	// 6. Subtracting what is already resident is safe because the reviewer is
	// started before the main model loads, so the seat is occupied continuously
	// whether the old process or the new one holds it.
	if resident := residentReviewerVRAM(profile.ModelMarker); resident > 0 {
		vramMB -= resident
		if vramMB < 0 {
			vramMB = 0
		}
	}
	if vramMB <= 0 {
		return nil
	}
	return &placement.CompanionReservation{
		Name:          profile.Name,
		VRAMMB:        vramMB,
		GPUPreference: claudeReviewerGPUCandidates(caps, req),
		AllowCPU:      false,
	}
}

// recordReviewerVRAM stores what the reviewer actually occupies so the next
// launch reserves that instead of the constant. It runs after the health check,
// when weights and the CUDA context are resident; the stored value only ever
// grows, because a sample taken before the reviewer's KV fills would shrink the
// reservation and overrun on the next long conversation.
func recordReviewerVRAM(cfg *config.Config, p *server.Process, profile *claudeCompanionProfile) {
	if cfg == nil || p == nil || p.Cmd == nil || p.Cmd.Process == nil || profile == nil {
		return
	}
	usedMB := placement.QueryVRAMUsedByPID(p.Cmd.Process.Pid)
	if usedMB <= 0 {
		return
	}
	if err := placement.RecordCompanionVRAM(cfg.CacheDir, profile.companionMeasurementKey(), usedMB); err != nil {
		return
	}
	if usedMB < profile.ReservationVRAMMB {
		fmt.Printf("[claude-code] Auto reviewer measured at %d MiB; releasing %d MiB the %d MiB reservation withheld\n",
			usedMB, profile.ReservationVRAMMB-usedMB, profile.ReservationVRAMMB)
	}
}

// residentReviewerVRAMMB sums the VRAM held by reviewer processes that are
// already running, so a reservation is not added on top of memory the hardware
// scan has already reported as in use.
//
// Matching is on the reviewer's own model directory, which no other process
// loads, rather than on a PID ggrun does not own: the leftover belongs to a
// previous launch that has not finished exiting.
// residentReviewerVRAM is indirected so tests can describe a host without
// depending on whatever happens to be running on the machine.
var residentReviewerVRAM = residentReviewerVRAMMB

func residentReviewerVRAMMB(modelMarker string) int {
	if strings.TrimSpace(modelMarker) == "" {
		return 0
	}
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return 0
	}
	total := 0
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil || pid <= 0 {
			continue
		}
		raw, err := os.ReadFile(filepath.Join("/proc", e.Name(), "cmdline"))
		if err != nil || !strings.Contains(string(raw), modelMarker) {
			continue
		}
		total += placement.QueryVRAMUsedByPID(pid)
	}
	return total
}

// startClaudeAutoReviewer launches the pinned Auto reviewer on the seat the
// placement planner returned in companionPlacements. When the planner ran, its
// single decision is authoritative: a GPU seat is tried once (no fallback walk
// that could land the reviewer on the GPU the plan deliberately kept free), a
// CPU seat (-1) goes straight to the CPU path. Without a plan (reviewer needed
// but no companion reservation supplied) it falls back to the legacy
// least-valuable-GPU-first walk.
func startClaudeAutoReviewer(req *launchRequest, cfg *config.Config, caps *detect.Capabilities, companionPlacements []placement.CompanionPlacement) (*claudeAutoRuntime, error) {
	if req == nil || !req.ClaudeCode || req.ClaudeReviewerDisabled || !claudeCompanionNeeded(nil) {
		return nil, nil
	}
	if req.CPUMode || caps == nil || len(caps.GPUs) == 0 {
		req.ClaudeReviewerDisabled = true
		return nil, nil
	}
	appHome := ""
	if cfg != nil {
		appHome = strings.TrimSpace(cfg.AppHome)
	}
	if appHome == "" {
		appHome = backends.AppHome()
	}
	cacheDir := ""
	modelDir := ""
	if cfg != nil {
		cacheDir = cfg.CacheDir
		modelDir = cfg.ModelDir
	}
	profile := resolveClaudeCompanionProfile(req, cacheDir)
	modelPath := profile.ModelPath
	if profile.NanoBeige {
		if err := advisor.VerifyArtifact(modelPath, advisor.DefaultArtifact); err != nil {
			return nil, fmt.Errorf("prepare local Auto worker: %w", err)
		}
	} else {
		var err error
		// Install the artifact THIS profile runs. Always ensuring the 4B meant
		// --claude-reviewer qwen2b planned a 2600 MB seat and then launched the
		// 4600 MB model in it, and the 2B was never downloaded at all.
		modelPath, err = claudeauto.EnsureReviewerSpec(context.Background(), appHome, modelDir,
			claudeReviewerArtifactSpec(profile), os.Stdout)
		if err != nil {
			return nil, fmt.Errorf("prepare local Auto reviewer %s: %w", profile.DisplayName, err)
		}
	}
	var be *backendInfo
	if profile.BackendPath != "" {
		be = detectBackend(profile.BackendPath)
	} else {
		be = findClaudeReviewerBackend(caps)
	}
	if be == nil {
		return nil, fmt.Errorf("local Auto needs a compatible llama-server for %s; none was found", profile.DisplayName)
	}
	port, err := freeLoopbackPort()
	if err != nil {
		return nil, err
	}
	logWriter, logCloser := claudeReviewerLog(cfg, port)

	candidates := claudeReviewerGPUCandidates(caps, req)
	planned := false
	for _, cp := range companionPlacements {
		if cp.Name == profile.Name {
			planned = true
			if cp.GPU >= 0 {
				candidates = []int{cp.GPU}
			} else {
				candidates = nil
			}
		}
	}

	var lastErr error
	for _, gpu := range candidates {
		env := claudeReviewerBackendEnv(be.Path, []string{fmt.Sprintf("CUDA_VISIBLE_DEVICES=%d", gpu)})
		device, probeErr := claudeReviewerDeviceForGPU(be.Path, env, claudeReviewerPhysicalGPU(caps, gpu))
		if probeErr != nil {
			lastErr = probeErr
			if planned {
				if logCloser != nil {
					_ = logCloser.Close()
				}
				return nil, fmt.Errorf("select local Auto reviewer device on planned GPU %d: %w", gpu, probeErr)
			}
			fmt.Fprintf(os.Stderr, "[claude-code] Auto reviewer backend cannot use GPU %d: %v\n", gpu, probeErr)
			continue
		}
		// CUDA_VISIBLE_DEVICES is required in addition to --device. Without it,
		// llama.cpp initializes contexts on every GPU even though all reviewer
		// tensors live on the selected device (observed: +262 MiB on the main
		// CUDA0 during a DeepSeek-V4 run). Ask the backend for the device name it
		// exposes after isolation instead of assuming every fork uses CUDA0.
		args := claudeReviewerArgsWithContextAndKV(be.Path, modelPath, port, device, be.Help, profile.KVType, cacheDir, profile.reviewerContextTokens())
		p, err := server.StartWithTimeoutToEnv(args, port, 5*time.Minute, logWriter, logWriter, env)
		if err == nil {
			fmt.Printf("[claude-code] Auto worker/reviewer ready on GPU %d (PID %d, %s, ctx %dk)\n", gpu, p.Cmd.Process.Pid, profile.DisplayName, profile.reviewerContextTokens()/1024)
			recordReviewerVRAM(cfg, p, profile)
			return &claudeAutoRuntime{reviewer: p, reviewerLog: logCloser, reviewerPort: port, reviewerGPU: gpu, companion: profile}, nil
		}
		lastErr = err
		if planned {
			// The planner already accounted for this GPU's free VRAM; a failed
			// load means the plan's data was stale, so report it instead of
			// silently moving onto a GPU the plan reserved for the model.
			if logCloser != nil {
				_ = logCloser.Close()
			}
			return nil, fmt.Errorf("start local Auto reviewer on planned GPU %d: %w", gpu, err)
		}
		fmt.Fprintf(os.Stderr, "[claude-code] Auto reviewer did not fit GPU %d; trying the next device.\n", gpu)
	}

	// CPU is slower, but it preserves autonomous/fail-closed behavior on systems
	// whose GPUs are already full. It is also the normal path on CPU-only hosts.
	args := claudeReviewerArgsWithContextAndKV(be.Path, modelPath, port, "", be.Help, profile.KVType, cacheDir, profile.reviewerContextTokens())
	p, err := server.StartWithTimeoutToEnv(args, port, 5*time.Minute, logWriter, logWriter, claudeReviewerBackendEnv(be.Path, claudeReviewerCPUEnv()))
	if err != nil {
		if logCloser != nil {
			_ = logCloser.Close()
		}
		if lastErr != nil {
			return nil, fmt.Errorf("start local Auto reviewer (GPU: %v; CPU: %w)", lastErr, err)
		}
		return nil, fmt.Errorf("start local Auto reviewer: %w", err)
	}
	fmt.Printf("[claude-code] Auto worker/reviewer ready on CPU (PID %d, %s, ctx %dk)\n", p.Cmd.Process.Pid, profile.DisplayName, profile.reviewerContextTokens()/1024)
	return &claudeAutoRuntime{reviewer: p, reviewerLog: logCloser, reviewerPort: port, reviewerGPU: -1, companion: profile}, nil
}

func claudeReviewerArgs(binary, modelPath string, port int, device, help string) []string {
	return claudeReviewerArgsWithKV(binary, modelPath, port, device, help, "q8_0", "")
}

// reviewerCatalogTemplate resolves a corrected chat template for a reviewer
// model from the data-driven catalog. It matches on the reviewer GGUF's own
// architecture/basename (read from the model file), only applying when the
// embedded template carries the raise_exception marker. cacheDir is where the
// corrected template is materialized; an empty cacheDir (legacy callers) skips
// the override entirely rather than writing to an unknown location.
func reviewerCatalogTemplate(cacheDir, modelPath string) string {
	if strings.TrimSpace(cacheDir) == "" || strings.TrimSpace(modelPath) == "" {
		return ""
	}
	embedded := gguf.ChatTemplate(modelPath)
	arch, basename := reviewerModelIdentity(modelPath)
	entry, ok := chattemplate.Resolve(arch, basename, embedded, true)
	if !ok {
		return ""
	}
	path, err := chattemplate.Materialize(cacheDir, entry)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[claude-code] warning: chat-template override for reviewer %s unavailable: %v\n", entry.Name, err)
		return ""
	}
	return path
}

// reviewerModelIdentity returns the GGUF architecture and file basename for a
// reviewer model. The architecture is probed from the filename (reviewer
// artifacts carry the family in their name); unknown names yield "" and the
// catalog falls through to basename matching.
func reviewerModelIdentity(modelPath string) (string, string) {
	basename := strings.ToLower(filepath.Base(modelPath))
	return reviewerArchFromName(basename), basename
}

// reviewerArchFromName guesses a GGUF architecture from a model filename when
// the full header parse is too heavy for the reviewer arg builder. Known
// reviewer models carry the family in the name; unknown names yield "" and the
// catalog falls through to basename matching.
func reviewerArchFromName(basename string) string {
	switch {
	case strings.Contains(basename, "nanbeige"):
		return "nanbeige"
	case strings.Contains(basename, "qwen3"):
		return "qwen35"
	default:
		return ""
	}
}

func claudeReviewerArgsWithKV(binary, modelPath string, port int, device, help, kvType string, cacheDir string) []string {
	return claudeReviewerArgsWithContextAndKV(binary, modelPath, port, device, help, kvType, cacheDir, claudeReviewerContextTokens)
}

// claudeReviewerArgsWithContextAndKV emits the exact capacity contract that
// placement reserved. Keeping context in this builder prevents the router,
// reservation, and backend from describing three different reviewer windows.
func claudeReviewerArgsWithContextAndKV(binary, modelPath string, port int, device, help, kvType string, cacheDir string, contextTokens int) []string {
	if contextTokens <= 0 {
		contextTokens = claudeReviewerContextTokens
	}
	args := []string{
		binary, "-m", modelPath,
		"--host", "127.0.0.1", "--port", strconv.Itoa(port),
		"--ctx-size", strconv.Itoa(contextTokens), "--parallel", "1",
		"--alias", "local",
		"--temp", "0", "--presence-penalty", "0", "--repeat-penalty", "1",
	}
	// A broken chat template (e.g. a Jinja `raise_exception` macro the backend
	// Jinja engine cannot parse — "Unable to generate parser for this template.
	// System message must be at the beginning") 400s every request under --jinja.
	// Tool calls REQUIRE --jinja (the backend returns "tools param requires
	// --jinja flag" without it), so the fix is NOT to drop --jinja but to serve
	// a corrected template from the data-driven chat-template catalog via
	// --chat-template-file. The worker/reviewer never receives user chat-template
	// overrides (its args are built from scratch here), so the catalog applies
	// directly whenever the reviewer model's arch/basename has a catalog entry.
	args = append(args, "--jinja")
	if tpl := reviewerCatalogTemplate(cacheDir, modelPath); tpl != "" {
		args = append(args, "--chat-template-file", tpl)
	}
	// Exact-flag matching, not substring: mainline's --reasoning-format and
	// --reasoning-budget both contain "--reasoning" but reject `--reasoning off`,
	// and an unknown flag kills the server at startup. --reasoning-budget 0 is
	// mainline's way of disabling thinking, so a recent fork still gets a
	// non-thinking classifier rather than a dead one.
	switch {
	case helpHasExactFlag(help, "--reasoning"):
		args = append(args, "--reasoning", "off")
	case helpHasExactFlag(help, "--reasoning-budget"):
		args = append(args, "--reasoning-budget", "0")
	}
	// The classifier carries a large policy prompt, so its KV cache is a
	// meaningful part of the reviewer footprint. The small Qwen classifier keeps
	// its established Q8 profile; Nanbeige uses Q4 because its 44 logical KV
	// layers otherwise consume nearly nine GiB at 64k. Keep the compatibility
	// guard for older user-provided llama-server binaries.
	if strings.Contains(help, "--cache-type-k") && strings.Contains(help, "--cache-type-v") {
		if strings.TrimSpace(kvType) == "" {
			kvType = "q8_0"
		}
		args = append(args, "--cache-type-k", kvType, "--cache-type-v", kvType)
	}
	if device != "" {
		args = append(args, "--device", device, "--split-mode", "none", "-ngl", "999", "-mg", "0")
	} else {
		args = append(args, "-ngl", "0")
	}
	return args
}

// helpHasExactFlag reports whether a backend's --help advertises exactly this
// flag, so "--reasoning" cannot be satisfied by "--reasoning-format".
func helpHasExactFlag(help, flag string) bool {
	for _, field := range strings.Fields(help) {
		trimmed := strings.Trim(field, " ,;[]()")
		if trimmed == flag || strings.HasPrefix(field, flag+"=") {
			return true
		}
	}
	return false
}

func claudeReviewerListDevices(binary string, env []string) (string, error) {
	args := []string{binary, "--list-devices"}
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Env = server.OverrideEnv(server.ChildEnv(os.Environ(), args), env)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("probe %s --list-devices: %w: %s", binary, err, strings.TrimSpace(string(out)))
	}
	return string(out), nil
}

func claudeReviewerPhysicalGPU(caps *detect.Capabilities, index int) *detect.GPU {
	if caps == nil {
		return nil
	}
	for i := range caps.GPUs {
		if caps.GPUs[i].Index == index {
			return &caps.GPUs[i]
		}
	}
	return nil
}

// claudeReviewerDeviceForGPU names the backend device that is physical GPU
// gpu. A CUDA backend is isolated by CUDA_VISIBLE_DEVICES and reports it
// directly. A Vulkan backend ignores that isolation and enumerates every card
// in its own order, so the device is matched by name, and only when exactly
// one Vulkan device carries it: fresh NVIDIA installs pair the ik_llama CUDA
// server, which cannot run the reviewer, with a mainline Vulkan server.
func claudeReviewerDeviceForGPU(binary string, env []string, gpu *detect.GPU) (string, error) {
	device, err := claudeReviewerGPUDevice(binary, env)
	if err == nil || gpu == nil || strings.TrimSpace(gpu.Name) == "" {
		return device, err
	}
	out, listErr := claudeReviewerListDevices(binary, env)
	if listErr != nil {
		return "", listErr
	}
	var matches []string
	for _, line := range strings.Split(out, "\n") {
		name, desc, ok := strings.Cut(strings.TrimSpace(line), ":")
		if !ok || !strings.HasPrefix(name, "Vulkan") {
			continue
		}
		desc = strings.TrimSpace(desc)
		if i := strings.LastIndex(desc, " ("); i >= 0 {
			desc = desc[:i]
		}
		if strings.EqualFold(strings.TrimSpace(desc), strings.TrimSpace(gpu.Name)) {
			matches = append(matches, name)
		}
	}
	if len(matches) == 1 {
		return matches[0], nil
	}
	if len(matches) > 1 {
		return "", fmt.Errorf("backend %s exposes %d Vulkan devices named %q; cannot tell which is GPU %d", binary, len(matches), gpu.Name, gpu.Index)
	}
	return "", err
}

func claudeReviewerGPUDevice(binary string, env []string) (string, error) {
	out, err := claudeReviewerListDevices(binary, env)
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Fields(strings.TrimSpace(line))
		if len(fields) == 0 {
			continue
		}
		name := strings.TrimSuffix(fields[0], ":")
		if strings.HasPrefix(name, "CUDA") {
			return name, nil
		}
	}
	return "", fmt.Errorf("backend %s advertises no CUDA device after GPU isolation: %s", binary, strings.TrimSpace(out))
}

func claudeReviewerCPUEnv() []string {
	// GPU-enabled llama-server binaries initialize their accelerator backend even
	// with -ngl 0. Hide accelerators so a CPU fallback still starts when every
	// device is full, which is precisely when this path is needed.
	return []string{
		"CUDA_VISIBLE_DEVICES=-1",
		"HIP_VISIBLE_DEVICES=-1",
		"ROCR_VISIBLE_DEVICES=-1",
	}
}

func claudeReviewerBackendEnv(binary string, env []string) []string {
	if libPath, ok := libhub.StableLibraryPath(binary); ok {
		return libhub.ApplyHubToChildEnv(env, libPath)
	}
	return env
}

func findClaudeReviewerBackend(caps *detect.Capabilities) *backendInfo {
	seen := map[string]bool{}
	var candidates []string
	// GGRUN_CLAUDE_REVIEWER_BIN pairs with GGRUN_CLAUDE_REVIEWER_MODEL: a
	// reviewer model whose architecture only a fork understands (Nanbeige's
	// looped transformer, say) is unusable without also naming the binary that
	// can load it, and the resolution below deliberately prefers ggrun's own
	// mainline bundle over any fork.
	if bin := strings.TrimSpace(os.Getenv("GGRUN_CLAUDE_REVIEWER_BIN")); bin != "" {
		candidates = append(candidates, bin)
	}
	// Prefer ggrun's maintained mainline binary over an arbitrary LLAMA_SERVER
	// or architecture fork selected for the main model.
	appHome := backends.AppHome()
	candidates = append(candidates,
		filepath.Join(appHome, ".bin", "llama-server-cuda"),
		filepath.Join(appHome, ".bin", "llama-server-cuda.exe"),
		filepath.Join(appHome, ".bin", "llama-server"),
		filepath.Join(appHome, ".bin", "llama-server.exe"),
	)
	if caps != nil {
		for _, be := range caps.Backends {
			candidates = append(candidates, be.Path)
		}
	}
	candidates = append(candidates, backendSearchPaths()...)
	var fallback *backendInfo
	for _, path := range candidates {
		if path == "" || seen[path] {
			continue
		}
		seen[path] = true
		if _, err := os.Stat(path); err != nil {
			continue
		}
		be := detectBackend(path)
		if !be.IsIK && (helpHasExactFlag(be.Help, "--reasoning") || helpHasExactFlag(be.Help, "--reasoning-budget")) {
			if fallback == nil {
				fallback = be
			}
			env := claudeReviewerBackendEnv(be.Path, nil)
			if _, err := claudeReviewerGPUDevice(be.Path, env); err == nil {
				return be
			}
		}
	}
	// A CPU-only or Vulkan mainline backend can still run the reviewer via the
	// CPU fallback when no CUDA-capable mainline backend is installed.
	return fallback
}

// claudeReviewerGPUCandidates preserves the largest GPU for the main model and
// tries the least valuable remaining accelerator first. The reviewer is small
// and bursty, while a large model benefits continuously from keeping its faster
// secondary GPU available for expert or dense tensors. A failed real load moves
// to the next candidate; no estimated memory cushion is used.
func claudeReviewerGPUCandidates(caps *detect.Capabilities, req *launchRequest) []int {
	if caps == nil || len(caps.GPUs) == 0 || (req != nil && req.CPUMode) {
		return nil
	}
	if req != nil && strings.TrimSpace(req.GPUsFlag) != "" {
		// CUDA_VISIBLE_DEVICES takes physical device IDs here. Preserve the user's
		// selected order, reject nonexistent devices, and never silently substitute
		// CUDA0 for a sparse selection such as --gpus 1,2.
		parts := strings.Split(req.GPUsFlag, ",")
		available := map[int]bool{}
		for _, gpu := range caps.GPUs {
			available[gpu.Index] = true
		}
		seen := map[int]bool{}
		out := make([]int, 0, len(parts))
		for _, part := range parts {
			if idx, err := strconv.Atoi(strings.TrimSpace(part)); err == nil && available[idx] && !seen[idx] {
				seen[idx] = true
				out = append(out, idx)
			}
		}
		return out
	}
	gpus := append([]detect.GPU(nil), caps.GPUs...)
	reserved := 0
	for i := 1; i < len(gpus); i++ {
		if gpus[i].VRAMTotalMB > gpus[reserved].VRAMTotalMB ||
			(gpus[i].VRAMTotalMB == gpus[reserved].VRAMTotalMB && gpus[i].BandwidthMBps > gpus[reserved].BandwidthMBps) {
			reserved = i
		}
	}
	mainGPU := gpus[reserved]
	gpus = append(gpus[:reserved], gpus[reserved+1:]...)
	sort.SliceStable(gpus, func(i, j int) bool {
		if gpus[i].BandwidthMBps != gpus[j].BandwidthMBps {
			return gpus[i].BandwidthMBps < gpus[j].BandwidthMBps
		}
		if gpus[i].VRAMTotalMB != gpus[j].VRAMTotalMB {
			return gpus[i].VRAMTotalMB < gpus[j].VRAMTotalMB
		}
		return gpus[i].Index < gpus[j].Index
	})
	gpus = append(gpus, mainGPU)
	out := make([]int, 0, len(gpus))
	for _, gpu := range gpus {
		out = append(out, gpu.Index)
	}
	return out
}

func freeLoopbackPort() (int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, fmt.Errorf("allocate local Auto reviewer port: %w", err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	if err := ln.Close(); err != nil {
		return 0, err
	}
	return port, nil
}

func claudeReviewerLog(cfg *config.Config, port int) (io.Writer, io.Closer) {
	dir := os.TempDir()
	if cfg != nil && cfg.LogDir != "" {
		dir = cfg.LogDir
	}
	path := filepath.Join(dir, fmt.Sprintf("ggrun-claude-reviewer-%d.log", port))
	f, err := os.Create(path)
	if err != nil {
		return io.Discard, nil
	}
	fmt.Printf("[claude-code] Auto reviewer logs -> %s\n", path)
	return f, f
}

// claudeRouterMetricsPath keeps one launch's per-request evidence next to that
// launch's reviewer log.
func claudeRouterMetricsPath(cfg *config.Config, port int) string {
	dir := os.TempDir()
	if cfg != nil && cfg.LogDir != "" {
		dir = cfg.LogDir
	}
	return filepath.Join(dir, fmt.Sprintf("ggrun-claude-requests-%d.jsonl", port))
}

func (r *claudeAutoRuntime) startRouter(cfg *config.Config, mainHost string, mainPort int, supportsVision bool, maxMainActive int, serverArgs []string) error {
	if r == nil {
		return nil
	}
	host := mainHost
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	mainBaseURL := fmt.Sprintf("http://%s:%d", host, mainPort)
	reviewerBaseURL := mainBaseURL
	if r.reviewerPort > 0 {
		reviewerBaseURL = fmt.Sprintf("http://127.0.0.1:%d", r.reviewerPort)
	}
	router, err := claudeauto.StartRouter(
		mainBaseURL,
		reviewerBaseURL,
		supportsVision,
		maxMainActive,
	)
	if err != nil {
		return err
	}
	r.router = router
	// Point Claude Code's cheap tiers at the companion only when the selected
	// profile explicitly serves worker traffic. A seated Qwen3.5-2B remains a
	// perfectly valid separate safety reviewer, but local-fast falls through to
	// the main model because that profile is review-only.
	router.SetCompanion("local", r.workerRouteEnabled())
	// Tell the router the exact context that was reserved and launched for this
	// profile. A fixed 64k threshold made the healthy 2B reviewer reject real
	// 65k+ Claude control prompts and queue them behind a busy p1 main model.
	reviewerContext := claudeReviewerContextTokens
	if r.companion != nil {
		reviewerContext = r.companion.reviewerContextTokens()
	}
	router.SetReviewerContext(reviewerContext)
	// The pass-cost decomposition needs to know how many tokens a prefill pass
	// carried, which is the micro-batch the backend was launched with.
	router.SetUBatch(argIntValue(serverArgs, "-ub", "--ubatch-size"))
	// A host-expert MoE can allocate two slots yet perform worse when two long
	// prefills contend for the same DDR/PCIe path. Keep the physical slots, but
	// schedule their active phases from observable request/TTFB boundaries.
	hostExpertLayers := argIntValue(serverArgs, "-ncmoe", "--n-cpu-moe")
	phaseAware := maxMainActive > 1 && hostExpertLayers > 0
	router.SetPhaseAwareAdmission(phaseAware, 8192)
	// Poll the backend's own counters so /ggrun/router can report measured
	// throughput rather than request wall-clock. Five seconds is well below any
	// human-visible status refresh and negligible load on the backend.
	router.StartBackendPolling(5 * time.Second)
	// Recording is evidence, not a dependency: a metrics failure must not stop
	// the user's launch.
	metricsPath := claudeRouterMetricsPath(cfg, router.Port())
	if err := router.EnableMetrics(metricsPath); err != nil {
		fmt.Fprintf(os.Stderr, "[claude-code] per-request metrics disabled: %v\n", err)
	} else {
		fmt.Printf("[claude-code] per-request metrics -> %s\n", metricsPath)
	}
	if r.reviewerPort > 0 && r.workerRouteEnabled() {
		fmt.Printf("[claude-code] Auto router ready on %s (coding -> main model, cheap work + safety -> local companion)\n", router.URL())
	} else if r.reviewerPort > 0 {
		fmt.Printf("[claude-code] Auto router ready on %s (coding + cheap work -> main model, safety -> local reviewer)\n", router.URL())
	} else {
		fmt.Printf("[claude-code] agent gateway ready on %s\n", router.URL())
	}
	if maxMainActive > 0 {
		if phaseAware {
			fmt.Printf("[claude-code] agent admission: %d physical lane(s), phase-aware active scheduling (cold host-expert prefills serialize; cache-hot appends may overlap decode)\n", maxMainActive)
		} else {
			fmt.Printf("[claude-code] agent admission: %d active main-model request(s); queued cancellations remain client-controlled\n", maxMainActive)
		}
	}
	return nil
}

// claudeMainMaxActive is how many main-model requests ggrun lets reach the
// backend at once. It is the real concurrency of the server: the backend's slot
// count only caps what this admits, so `--parallel 4` with a limit of 1 leaves
// three slots permanently idle. Measured on this project across 339 requests,
// the backend spent 75922.93 s with exactly one request in flight and 0.05 s
// with two, against a 48-minute median queue for 3.4 minutes of service.
//
// The default stays 1 for host-offloaded models because that is the only value
// with a measurement behind it -- 4-way decode returned 3.3 tok/s aggregate
// against 4.13 single-stream. Everything arguing for more (86% of wall clock is
// prefill, which batches rather than competes) is untested, and a default is not
// where an untested hypothesis belongs. --claude-max-active is how that gets
// measured; whichever wins should become the default, ideally calibrated rather
// than constant.
//
// That reasoning covers the *default*. It must not survive an explicit
// --parallel, which is the user asking for that many lanes: allocating the slots
// and then admitting one is strictly worse than never allocating them, because
// --ctx-size is split across slots either way. Observed live -- `--parallel 2`
// produced two slots of 131072, one of them permanently idle, at the same
// throughput a single 262144 slot had delivered.
func claudeMainMaxActive(req *launchRequest, strategy *placement.Strategy, pending ...*placement.CalibrationDecision) int {
	if req == nil || !req.ClaudeCode || strategy == nil {
		return 0
	}
	limit := defaultClaudeMainMaxActive(req, strategy, pending...)
	if req.ClaudeMaxActiveSet {
		limit = req.ClaudeMaxActive
	}
	// Admission above the slot count does not buy concurrency, it relocates the
	// queue into llama.cpp -- which serves FIFO and knows nothing about the
	// safety lane, so a permission review would start waiting behind bulk work.
	if limit > strategy.Parallel && strategy.Parallel > 0 {
		limit = strategy.Parallel
	}
	if limit < 0 {
		limit = 0
	}
	return limit
}

func defaultClaudeMainMaxActive(req *launchRequest, strategy *placement.Strategy, pending ...*placement.CalibrationDecision) int {
	// An explicit --parallel is an instruction, not a hint. Every slot costs
	// context whether or not it is ever used, so honouring the flag by allocating
	// slots while refusing to admit into them is the one outcome nobody asked
	// for. The conservative host-offload default below still applies whenever
	// ggrun chose the slot count itself.
	if req != nil && req.ParallelSet && strategy.Parallel > 1 {
		return strategy.Parallel
	}
	if strategy.Type == placement.MoEOffload || strategy.Type == placement.DenseCPUOffload {
		// Host-offloaded concurrency is hardware/model dependent, so the cold
		// default remains serialized. Once the exact agent workload has compared
		// a challenger (including a baseline win), admit only the number of lanes
		// that comparison actually represented. The optional pending decision is
		// the first-launch bridge: it lets the selected process pass its router
		// lifecycle gates under the same concurrency that was benchmarked without
		// prematurely marking it reusable performance evidence.
		measured := strategy.PerformanceTuned
		if len(pending) > 0 && automaticCalibrationEvidenceValid(pending[0]) {
			measured = true
		}
		if measured {
			return min(max(1, strategy.Parallel), max(1, requestWorkloadConcurrency(req)))
		}
		return 1
	}
	// A single slot can only serve one request, but the limit still matters:
	// without it no scheduler is constructed at all, so lane priority, affinity
	// and aging are switched off and llama.cpp queues the fan-out FIFO instead.
	if strategy.Parallel <= 1 {
		return 1
	}
	return 0
}

// setMessageDelimiters forwards the backend's chat role markers to the router.
func (r *claudeAutoRuntime) setMessageDelimiters(delims []claudeauto.MessageDelimiter) {
	if r == nil || r.router == nil {
		return
	}
	r.router.SetMessageDelimiters(delims)
}

func (r *claudeAutoRuntime) clientPort(fallback int) int {
	if r != nil && r.router != nil && r.router.Port() > 0 {
		return r.router.Port()
	}
	return fallback
}

func (r *claudeAutoRuntime) isRunning() bool {
	return r != nil && r.reviewer != nil && r.reviewer.IsRunning()
}

func (r *claudeAutoRuntime) stop() {
	if r == nil {
		return
	}
	if r.router != nil {
		_ = r.router.Close()
		r.router = nil
	}
	if r.reviewer != nil {
		_ = r.reviewer.Stop()
		r.reviewer = nil
	}
	if r.reviewerLog != nil {
		_ = r.reviewerLog.Close()
		r.reviewerLog = nil
	}
}
