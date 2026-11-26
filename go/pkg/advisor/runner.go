package advisor

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/backends"
	"github.com/raketenkater/ggrun/pkg/boundexec"
	"github.com/raketenkater/ggrun/pkg/libhub"
	"github.com/raketenkater/ggrun/pkg/server"
)

type Runner struct {
	BackendPath    string
	ModelPath      string
	Artifact       Artifact
	CacheDir       string
	Online         bool
	StartupTimeout time.Duration
	RequestTimeout time.Duration
	MemoryMaxMB    int
	HTTPClient     *http.Client
}

type RunReport struct {
	Started         bool          `json:"started"`
	OnlineSources   int           `json:"online_sources"`
	Duration        time.Duration `json:"duration"`
	ReleaseVerified bool          `json:"release_verified"`
	BackendLogTail  string        `json:"backend_log_tail,omitempty"`
}

// ErrResourceReleaseUnverified is a hard controller boundary. Callers may
// degrade to deterministic reasoning after ordinary advisor failures, but they
// must not start the main model while this error is present.
var ErrResourceReleaseUnverified = errors.New("support expert resource release was not verified")

type resourceSnapshot struct {
	AvailableRAMMB int
	GPUUsedMB      map[int]int
}

func (runner Runner) Analyze(ctx context.Context, incident Incident) (Decision, RunReport, error) {
	startedAt := time.Now()
	report := RunReport{}
	if err := incident.Normalize(); err != nil {
		return Decision{}, report, err
	}
	if strings.TrimSpace(runner.BackendPath) == "" || strings.TrimSpace(runner.ModelPath) == "" {
		return Decision{}, report, errors.New("support expert requires a backend and installed model")
	}
	artifact := runner.Artifact
	if strings.TrimSpace(artifact.Name) == "" {
		artifact = DefaultArtifact
	}
	if !strings.EqualFold(strings.TrimSpace(artifact.Architecture), "nanbeige") {
		return Decision{}, report, errors.New("support-expert artifact manifest must declare nanbeige architecture")
	}
	if err := VerifyArtifact(runner.ModelPath, artifact); err != nil {
		return Decision{}, report, fmt.Errorf("verify support-expert artifact: %w", err)
	}
	if supported, probed := backends.BackendSupportsArch(runner.BackendPath, "nanbeige"); !probed || !supported {
		return Decision{}, report, errors.New("support-expert backend has no verified nanbeige architecture support; update or install a compatible mainline backend")
	}
	if sameFile(runner.BackendPath, runner.ModelPath) {
		return Decision{}, report, errors.New("invalid support-expert backend/model paths")
	}
	incident.Knowledge = append(incident.Knowledge, BundledKnowledge(incident)...)
	if runner.Online {
		// Deterministic evidence chain: Official issue index -> pinned model card
		// -> bundled offline knowledge. Each tier that completes without new
		// evidence (or fails fetch) records a "research_degraded" settings note
		// instead of aborting; the analysis continues with whatever it has.
		if degraded := researchKnowledge(ctx, runner.HTTPClient, &incident, &report); len(degraded) > 0 {
			if incident.Settings == nil {
				incident.Settings = map[string]string{}
			}
			incident.Settings["research_degraded"] = strings.Join(degraded, "; ")
		}
	}
	if len(incident.Knowledge) > 12 {
		incident.Knowledge = incident.Knowledge[:12]
	}

	baseline := captureResources()
	port, err := freeLoopbackPort()
	if err != nil {
		return Decision{}, report, err
	}
	startupTimeout := runner.StartupTimeout
	if startupTimeout <= 0 {
		startupTimeout = 5 * time.Minute
	}
	memoryMax := runner.MemoryMaxMB
	if memoryMax <= 0 {
		memoryMax = 8192
	}
	args := []string{
		runner.BackendPath, "-m", runner.ModelPath,
		"--host", "127.0.0.1", "--port", strconv.Itoa(port),
		"--ctx-size", "16384", "-ngl", "0", "--parallel", "1",
		"-b", "512", "-ub", "128", "--no-mmap", "--alias", "ggrun-support",
	}
	// NanoBeige is a thinking model and this backend enables --jinja by default,
	// so llama.cpp routes everything between <think> and </think> into
	// message.reasoning_content and leaves message.content empty. On 2026-08-02
	// that consumed the entire generation budget on a real incident and the
	// controller decoded an empty body ("EOF"), wasting three minutes for no
	// decision. The advisor wants a verdict, never a deliberation. The Auto
	// reviewer already disables thinking this way; do the same here, gated on
	// the backend's own help so an older fork gets a working helper instead of
	// one killed at startup by an unknown flag.
	args = append(args, reasoningOffArgs(backendHelp(runner.BackendPath))...)
	var logs bytes.Buffer
	process, startErr := server.StartWithTimeoutToOptions(args, port, startupTimeout, &logs, &logs, server.StartOptions{
		EnvOverrides: supportBackendEnv(runner.BackendPath),
		MemoryHighMB: memoryMax * 7 / 8,
		MemoryMaxMB:  memoryMax,
	})
	if startErr != nil {
		// StartWithTimeoutToOptions already performs a best-effort stop when
		// readiness fails. Repeat the idempotent stop here and, critically, keep
		// the returned process in the release check. A nil process would make
		// processGone true even if an unusually stubborn backend survived the
		// first teardown attempt.
		if process != nil {
			_ = process.Stop()
		}
		report.BackendLogTail = tailText(logs.String(), 4000)
		report.ReleaseVerified = waitForResourceRelease(process, baseline, 30*time.Second)
		report.Duration = time.Since(startedAt)
		if !report.ReleaseVerified {
			return Decision{}, report, fmt.Errorf("%w after failed helper startup: %v", ErrResourceReleaseUnverified, startErr)
		}
		return Decision{}, report, fmt.Errorf("start support expert: %w", startErr)
	}
	report.Started = true

	decision, analyzeErr := runner.query(ctx, port, incident)
	stopErr := process.Stop()
	report.BackendLogTail = tailText(logs.String(), 4000)
	report.ReleaseVerified = waitForResourceRelease(process, baseline, 30*time.Second)
	report.Duration = time.Since(startedAt)
	if err := completedRunError(report.ReleaseVerified, analyzeErr, stopErr); err != nil {
		return Decision{}, report, err
	}
	return decision, report, nil
}

// researchBase is a test seam for the deterministic chain: it points the
// Official and ModelCard tiers at their live hosts by default and lets tests
// redirect both to local httptest endpoints without touching the exported
// Researcher methods.
var researchBase = struct {
	Official  string
	ModelCard string
}{
	Official:  defaultOfficialSearchBase,
	ModelCard: defaultModelCardBase,
}

// researchKnowledge runs the deterministic online evidence chain
// (Official issue index -> pinned model card) and appends whatever it yields to
// incident.Knowledge. Each tier that could not contribute returns a human-
// readable reason; those are joined and returned so the caller can record a
// "research_degraded" settings note. The bundled offline knowledge is not part
// of this chain — it is always appended before research starts, so degradation
// only ever means "online tiers were thin", never "no knowledge at all".
func researchKnowledge(ctx context.Context, client *http.Client, incident *Incident, report *RunReport) []string {
	if incident == nil {
		return nil
	}
	researcher := Researcher{Client: client}
	var degraded []string

	// Tier 1: official ggml-org/llama.cpp issue/PR index.
	sources, err := researcher.ResearchOfficial(ctx, *incident, researchBase.Official)
	if err == nil {
		if len(sources) > 0 {
			incident.Knowledge = append(incident.Knowledge, sources...)
			if report != nil {
				report.OnlineSources = len(sources)
			}
		} else {
			degraded = append(degraded, "official issue index: no sources")
		}
	} else {
		degraded = append(degraded, "official issue index: "+err.Error())
	}

	// Tier 2: pinned model card for the incident's architecture.
	cardSources, cardErr := researcher.ResearchModelCard(ctx, *incident, researchBase.ModelCard)
	if cardErr == nil {
		if len(cardSources) > 0 {
			incident.Knowledge = append(incident.Knowledge, cardSources...)
		} else {
			degraded = append(degraded, "model card: not pinned for this architecture")
		}
	} else {
		degraded = append(degraded, "model card: "+cardErr.Error())
	}
	return degraded
}

func supportBackendEnv(binary string) []string {
	env := []string{
		"CUDA_VISIBLE_DEVICES=", "ROCR_VISIBLE_DEVICES=", "HIP_VISIBLE_DEVICES=",
		"GGML_VK_VISIBLE_DEVICES=", "ZE_AFFINITY_MASK=",
	}
	// The helper may be a different backend family from the main model. Resolve
	// its own shared-library tree explicitly so a stale/global IK or mainline
	// hub cannot make an otherwise valid NanoBeige helper fail at exec time.
	if libraryPath, ok := libhub.StableLibraryPath(binary); ok {
		env = libhub.ApplyHubToChildEnv(env, libraryPath)
	}
	return env
}

// completedRunError deliberately gives the resource-release boundary higher
// precedence than query or Stop errors. The support expert is optional, but a
// process/resource leak is not: callers must not restart the main model until
// teardown is proven, even when the helper also returned an ordinary error.
func completedRunError(releaseVerified bool, analyzeErr, stopErr error) error {
	if !releaseVerified {
		return fmt.Errorf("%w: helper stopped but RAM/VRAM did not return to its pre-run baseline", ErrResourceReleaseUnverified)
	}
	if analyzeErr != nil {
		return analyzeErr
	}
	if stopErr != nil {
		return fmt.Errorf("stop support expert: %w", stopErr)
	}
	return nil
}

func (runner Runner) query(ctx context.Context, port int, incident Incident) (Decision, error) {
	incidentJSON, err := json.Marshal(incident)
	if err != nil {
		return Decision{}, err
	}
	system := `You are ggrun's constrained support expert and candidate ranker.
Use only the typed incident and supplied knowledge. The deterministic controller owns safety and execution.
Return exactly one JSON object with schema_version, action, confidence, rationale, and evidence_codes.
Use only an action listed in allowed_actions. Fill only fields required by that action.
Never emit commands, flags, paths, URLs, repositories, code, prose outside JSON, or changes to context/model quantization/user quality.`
	body := map[string]interface{}{
		"model": "ggrun-support",
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": string(incidentJSON)},
		},
		"temperature": 0.1,
		"max_tokens":  2048,
		"stream":      false,
		"response_format": map[string]interface{}{
			"type": "json_object",
		},
	}
	data, err := json.Marshal(body)
	if err != nil {
		return Decision{}, err
	}
	requestCtx := ctx
	timeout := runner.RequestTimeout
	if timeout <= 0 {
		timeout = 4 * time.Minute
	}
	requestCtx, cancel := context.WithTimeout(requestCtx, timeout)
	defer cancel()
	request, err := http.NewRequestWithContext(requestCtx, http.MethodPost,
		fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port), bytes.NewReader(data))
	if err != nil {
		return Decision{}, err
	}
	request.Header.Set("Content-Type", "application/json")
	client := runner.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	response, err := client.Do(request)
	if err != nil {
		return Decision{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		return Decision{}, fmt.Errorf("support expert returned HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(detail)))
	}
	var decoded struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content string `json:"content"`
				// A thinking model that never closes its reasoning block leaves
				// content empty and puts everything here. Reading it back turns
				// an unexplained "EOF" into a diagnosable failure, and recovers
				// the answer when the JSON did land inside the reasoning trace.
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&decoded); err != nil {
		return Decision{}, err
	}
	if len(decoded.Choices) == 0 {
		return Decision{}, errors.New("support expert returned no decision")
	}
	choice := decoded.Choices[0]
	content := strings.TrimSpace(choice.Message.Content)
	if content == "" {
		content = extractJSONObject(choice.Message.ReasoningContent)
	}
	if content == "" {
		if choice.FinishReason == "length" {
			return Decision{}, errors.New("support expert hit its generation limit before emitting a decision; " +
				"the helper is reasoning instead of answering (check --reasoning-budget)")
		}
		return Decision{}, errors.New("support expert returned an empty decision body")
	}
	return DecodeDecision([]byte(content), incident)
}

func freeLoopbackPort() (int, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port, nil
}

func captureResources() resourceSnapshot {
	return resourceSnapshot{AvailableRAMMB: availableRAMMB(), GPUUsedMB: gpuUsedMB()}
}

func availableRAMMB() int {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) >= 2 {
			kb, _ := strconv.Atoi(fields[1])
			return kb / 1024
		}
	}
	return 0
}

func gpuUsedMB() map[int]int {
	out, err := boundexec.Output(boundexec.ProbeTimeout, "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits")
	if err != nil {
		boundexec.Report(err, "VRAM use not sampled")
		return nil
	}
	result := map[int]int{}
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Split(line, ",")
		if len(fields) != 2 {
			continue
		}
		index, err1 := strconv.Atoi(strings.TrimSpace(fields[0]))
		used, err2 := strconv.Atoi(strings.TrimSpace(fields[1]))
		if err1 == nil && err2 == nil {
			result[index] = used
		}
	}
	return result
}

func waitForResourceRelease(process *server.Process, baseline resourceSnapshot, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for {
		processGone := process == nil || !process.IsRunning()
		current := captureResources()
		ramOK := baseline.AvailableRAMMB <= 0 || current.AvailableRAMMB <= 0 || current.AvailableRAMMB >= baseline.AvailableRAMMB-512
		gpuOK := true
		for index, before := range baseline.GPUUsedMB {
			if after := current.GPUUsedMB[index]; after > before+64 {
				gpuOK = false
				break
			}
		}
		if processGone && ramOK && gpuOK {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func sameFile(left, right string) bool {
	leftInfo, leftErr := os.Stat(left)
	rightInfo, rightErr := os.Stat(right)
	return leftErr == nil && rightErr == nil && os.SameFile(leftInfo, rightInfo)
}

// backendHelp returns the backend's --help text, or "" when it cannot be
// probed. An empty result is "unknown", not "unsupported": callers must fall
// back to passing nothing rather than passing a flag that could kill startup.
func backendHelp(binary string) string {
	if strings.TrimSpace(binary) == "" {
		return ""
	}
	cmd := exec.Command(binary, "--help")
	cmd.Env = append(os.Environ(), supportBackendEnv(binary)...)
	out, err := cmd.CombinedOutput()
	if err != nil && len(out) == 0 {
		return ""
	}
	return string(out)
}

// reasoningOffArgs picks whichever spelling this backend advertises for
// "generate an answer, not a deliberation". Exact-flag matching matters:
// --reasoning-format and --reasoning-budget both contain "--reasoning" but
// reject `--reasoning off`, and an unknown flag aborts argument parsing.
func reasoningOffArgs(help string) []string {
	switch {
	case helpHasExactFlag(help, "--reasoning"):
		return []string{"--reasoning", "off"}
	case helpHasExactFlag(help, "--reasoning-budget"):
		return []string{"--reasoning-budget", "0"}
	}
	return nil
}

func helpHasExactFlag(help, flag string) bool {
	for _, field := range strings.Fields(help) {
		trimmed := strings.Trim(field, " ,;[]()")
		if trimmed == flag || strings.HasPrefix(field, flag+"=") {
			return true
		}
	}
	return false
}

// extractJSONObject returns the first balanced top-level JSON object in text,
// or "" when there is none. It exists only to rescue a decision that a thinking
// helper emitted inside its reasoning trace; the result still goes through the
// full DecodeDecision/ValidateDecision path, so nothing here widens what the
// controller will accept.
func extractJSONObject(text string) string {
	start := strings.IndexByte(text, '{')
	if start < 0 {
		return ""
	}
	depth, inString, escaped := 0, false, false
	for i := start; i < len(text); i++ {
		c := text[i]
		switch {
		case escaped:
			escaped = false
		case c == '\\' && inString:
			escaped = true
		case c == '"':
			inString = !inString
		case inString:
		case c == '{':
			depth++
		case c == '}':
			depth--
			if depth == 0 {
				return strings.TrimSpace(text[start : i+1])
			}
		}
	}
	return ""
}

func tailText(value string, maxBytes int) string {
	if maxBytes > 0 && len(value) > maxBytes {
		value = value[len(value)-maxBytes:]
	}
	return strings.TrimSpace(value)
}

func DefaultModelPath(cacheDir string) string { return ArtifactPath(cacheDir, DefaultArtifact) }
