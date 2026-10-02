// Package claudeauto provides the local safety-review path used by Claude
// Code's Auto permission mode. Claude sends these reviews to the same model ID
// as normal coding turns, so ggrun routes the distinctive security-monitor
// request to a small dedicated model and leaves every other request on the
// user's main model.
package claudeauto

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/raketenkater/ggrun/pkg/backendmetrics"
)

const (
	// ClassifierMarker is part of Claude Code's Auto-mode system prompt. Keep
	// this exact and narrow so ordinary user prompts mentioning "security" do
	// not leave the main coding model.
	ClassifierMarker = "You are a security monitor for autonomous AI coding agents."

	DefaultReviewerDisplayName = "Qwen3.5-4B"
	DefaultReviewerFile        = "Qwen3.5-4B-Q4_K_M-00001-of-00001.gguf"
	// DefaultReviewerSize and DefaultReviewerSHA pin the exact Q4_K_M artifact,
	// so an upstream branch update cannot silently change local permission
	// decisions.
	DefaultReviewerSize = int64(2740937888)
	DefaultReviewerSHA  = "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
	// DefaultReviewerURL is the hub file with exactly that size and SHA. Its
	// remote name differs from DefaultReviewerFile, the local name: deriving
	// the URL from the local name pointed at a file the hub never had (404).
	DefaultReviewerURL = "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"

	// DefaultReviewerLocalDir is the model-directory subfolder that already holds
	// the pinned Q4_K_M artifact (models/Qwen3.5-4B-Q4_K_M/), so first use does
	// not download the same bytes into the private reviewer cache.
	DefaultReviewerLocalDir = "Qwen3.5-4B-Q4_K_M"

	// DefaultSmallReviewer* pins the small/light review-only Qwen3.5-2B profile.
	// Auto mode routes the classifier (security-review) lane to the 4B worker/
	// reviewer; the 2B is the explicit --claude-reviewer qwen2b choice for hosts
	// that want the cheapest possible safety-review lane. Size and SHA256 are the
	// real values of the locally installed artifact at
	// .cache/claude-reviewer/Qwen_Qwen3.5-2B-Q4_K_M.gguf.
	DefaultSmallReviewerDisplayName = "Qwen3.5-2B"
	DefaultSmallReviewerFile        = "Qwen_Qwen3.5-2B-Q4_K_M.gguf"
	DefaultSmallReviewerSize        = int64(1396198496)
	DefaultSmallReviewerSHA         = "57a1085840f497d764a7fc5d346922dbde961efb54cc792ea81d694fd846a1d8"
	DefaultSmallReviewerURL         = "https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/main/" + DefaultSmallReviewerFile

	maxRoutedRequestBytes = 16 << 20

	// Route labels recorded per request.
	routeMain                 = "main"
	routeReviewer             = "reviewer"
	routeReviewerStopStripped = "reviewer/stop-stripped-verdict"

	textOnlyImagePlaceholder = "[Image omitted by ggrun: this local model was launched without an mmproj. Use a text extraction/OCR tool or relaunch with --vision.]"
)

// ModelSpec pins a reviewer artifact so an upstream branch update cannot
// silently change local permission decisions.
type ModelSpec struct {
	URL    string
	Name   string
	Size   int64
	SHA256 string
}

var defaultModel = ModelSpec{
	URL: DefaultReviewerURL, Name: DefaultReviewerFile,
	Size: DefaultReviewerSize, SHA256: DefaultReviewerSHA,
}

// smallModel pins the 2B review-only artifact. It exists so the qwen2b profile
// installs the model it actually runs: the profile is planned with the smaller
// reservation, so installing the 4B under it leaves the companion badly
// under-reserved.
var smallModel = ModelSpec{
	URL: DefaultSmallReviewerURL, Name: DefaultSmallReviewerFile,
	Size: DefaultSmallReviewerSize, SHA256: DefaultSmallReviewerSHA,
}

// DefaultReviewerSpec is the pinned Qwen3.5-4B worker/reviewer artifact.
func DefaultReviewerSpec() ModelSpec { return defaultModel }

// SmallReviewerSpec is the pinned Qwen3.5-2B review-only artifact.
func SmallReviewerSpec() ModelSpec { return smallModel }

// ReviewerModelPath returns the user override, or ggrun's private reviewer
// cache path. The reviewer is deliberately kept out of the normal model list.
func ReviewerModelPath(appHome string) string {
	if path := strings.TrimSpace(os.Getenv("GGRUN_CLAUDE_REVIEWER_MODEL")); path != "" {
		return path
	}
	return filepath.Join(appHome, ".cache", "claude-reviewer", DefaultReviewerFile)
}

// SmallReviewerModelPath returns the small/light review-only Qwen3.5-2B
// artifact path. The small reviewer has no env override: the single
// GGRUN_CLAUDE_REVIEWER_MODEL override selects the worker/reviewer, and the 2B
// is a distinct, deliberately small profile for the review-only lane.
func SmallReviewerModelPath(appHome string) string {
	return filepath.Join(appHome, ".cache", "claude-reviewer", DefaultSmallReviewerFile)
}

// LocalReviewerModelPath looks for the pinned artifact under the model
// directory's matching subfolder (e.g. models/Qwen3.5-4B-Q4_K_M/), where it is
// already present and verified, so first use does not re-download it into the
// private reviewer cache. The subfolder holds a plain GGUF or a split-name
// symlink to it; validatePinnedGGUF runs on the returned path.
func LocalReviewerModelPath(modelDir string) (string, bool) {
	if strings.TrimSpace(modelDir) == "" {
		return "", false
	}
	dir := filepath.Join(modelDir, DefaultReviewerLocalDir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", false
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(strings.ToLower(name), ".gguf") {
			continue
		}
		// Prefer the file that carries the pinned basename, falling back to the
		// first GGUF in the folder (e.g. a single-split symlink target).
		if strings.EqualFold(name, DefaultReviewerFile) {
			return filepath.Join(dir, name), true
		}
	}
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
			return filepath.Join(dir, e.Name()), true
		}
	}
	return "", false
}

// EnsureReviewerModel validates a user-supplied model, or downloads and
// verifies the pinned default artifact on first use.
func EnsureReviewerModel(ctx context.Context, appHome string, progress io.Writer) (string, error) {
	return EnsureReviewerModelWithModelDir(ctx, appHome, "", progress)
}

// EnsureReviewerModelWithModelDir is EnsureReviewerModel plus a local model
// directory: when the pinned artifact is already installed there (matching the
// folder name in DefaultReviewerLocalDir), it is verified and used directly
// instead of downloading a private copy.
func EnsureReviewerModelWithModelDir(ctx context.Context, appHome, modelDir string, progress io.Writer) (string, error) {
	return EnsureReviewerSpec(ctx, appHome, modelDir, defaultModel, progress)
}

// EnsureReviewerSpec validates or downloads one pinned reviewer artifact. The
// caller passes the spec for the profile it is about to launch, because the
// profile's VRAM reservation describes THAT model: installing a different
// artifact under it under-reserves the companion and OOMs a card the planner
// believed had room.
func EnsureReviewerSpec(ctx context.Context, appHome, modelDir string, spec ModelSpec, progress io.Writer) (string, error) {
	isDefault := spec.Name == defaultModel.Name
	path := reviewerSpecPath(appHome, spec)
	// The single GGRUN_CLAUDE_REVIEWER_MODEL override selects the worker/reviewer
	// only. It must not silently satisfy a request for the 2B review-only lane.
	if isDefault && strings.TrimSpace(os.Getenv("GGRUN_CLAUDE_REVIEWER_MODEL")) != "" {
		if err := validateGGUF(path, 0); err != nil {
			return "", fmt.Errorf("GGRUN_CLAUDE_REVIEWER_MODEL: %w", err)
		}
		return path, nil
	}
	if err := validatePinnedGGUF(path, spec); err == nil {
		return path, nil
	}
	// The model directory already holds the pinned artifact: use it without
	// downloading (the 4B ships with ggrun's model set). Only the 4B has a
	// known model-set folder; the 2B always lives in the reviewer cache.
	if isDefault {
		if local, ok := LocalReviewerModelPath(modelDir); ok {
			if err := validatePinnedGGUF(local, spec); err == nil {
				return local, nil
			}
		}
	}
	if progress != nil {
		fmt.Fprintf(progress, "[claude-code] downloading pinned local Auto reviewer %s (%.1f GB)...\n",
			spec.Name, float64(spec.Size)/(1024*1024*1024))
	}
	if err := downloadModel(ctx, reviewerHTTPClient(), spec, path, progress); err != nil {
		return "", err
	}
	return path, nil
}

// reviewerSpecPath is where a pinned reviewer artifact lives. The 4B honours
// GGRUN_CLAUDE_REVIEWER_MODEL through ReviewerModelPath; every other pinned
// artifact sits in the private reviewer cache under its own name.
func reviewerSpecPath(appHome string, spec ModelSpec) string {
	if spec.Name == defaultModel.Name {
		return ReviewerModelPath(appHome)
	}
	return filepath.Join(appHome, ".cache", "claude-reviewer", spec.Name)
}

func validateGGUF(path string, wantSize int64) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if wantSize > 0 && info.Size() != wantSize {
		return fmt.Errorf("wrong size for %s: got %d, want %d", path, info.Size(), wantSize)
	}
	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return fmt.Errorf("read GGUF header: %w", err)
	}
	if string(magic[:]) != "GGUF" {
		return fmt.Errorf("%s is not a GGUF file", path)
	}
	return nil
}

func validatePinnedGGUF(path string, spec ModelSpec) error {
	if err := validateGGUF(path, spec.Size); err != nil {
		return err
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return fmt.Errorf("verify Auto reviewer: %w", err)
	}
	if got := hex.EncodeToString(h.Sum(nil)); !strings.EqualFold(got, spec.SHA256) {
		return fmt.Errorf("Auto reviewer checksum mismatch (got %s)", got)
	}
	return nil
}

const (
	// A multi-GB artifact on a slow link is not a hung request, so the download
	// has no whole-request deadline. Progress is policed by a stall watchdog
	// instead: bytes must keep arriving, but they may take as long as they take.
	// The previous 30-minute client timeout meant the pinned 2.55 GiB reviewer
	// could only ever install on a link sustaining ~1.5 MB/s; anything slower
	// failed at the deadline and threw away every byte it had fetched.
	reviewerDownloadAttempts = 4
	reviewerStallTimeout     = 2 * time.Minute
	reviewerHeaderTimeout    = 60 * time.Second
)

// reviewerHTTPClient builds the client used for pinned reviewer downloads.
func reviewerHTTPClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			DialContext:           (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
			TLSHandshakeTimeout:   30 * time.Second,
			ResponseHeaderTimeout: reviewerHeaderTimeout,
			ExpectContinueTimeout: time.Second,
		},
	}
}

// downloadModel fetches a pinned artifact, resuming an interrupted attempt
// rather than starting over. The partial file keeps a stable name so a download
// interrupted in an earlier run is continued instead of re-fetched, and the
// artifact is only verified and installed once it is complete.
func downloadModel(ctx context.Context, client *http.Client, spec ModelSpec, dest string, progress io.Writer) error {
	if client == nil {
		client = reviewerHTTPClient()
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return fmt.Errorf("create reviewer cache: %w", err)
	}
	part := dest + ".part"

	var lastErr error
	for attempt := 1; attempt <= reviewerDownloadAttempts; attempt++ {
		have := partialSize(part)
		if have > spec.Size {
			// A stale or foreign part file: start clean rather than resume into
			// the middle of something that is not this artifact.
			_ = os.Remove(part)
			have = 0
		}
		if have == spec.Size {
			break
		}
		if have > 0 && progress != nil {
			fmt.Fprintf(progress, "[claude-code] resuming Auto reviewer download at %d%%\n", int(have*100/spec.Size))
		}
		lastErr = fetchIntoPart(ctx, client, spec, part, have, progress)
		if lastErr == nil {
			break
		}
		if ctx.Err() != nil {
			return fmt.Errorf("download Auto reviewer: %w", ctx.Err())
		}
		if attempt < reviewerDownloadAttempts && progress != nil {
			fmt.Fprintf(progress, "[claude-code] Auto reviewer download interrupted (%v); retrying\n", lastErr)
		}
	}
	if size := partialSize(part); size != spec.Size {
		if lastErr != nil {
			return fmt.Errorf("download Auto reviewer: %w", lastErr)
		}
		return fmt.Errorf("download Auto reviewer: got %d bytes, want %d", size, spec.Size)
	}
	// Verify the assembled file rather than a running hash: with resume the bytes
	// come from more than one response, and what matters is what landed on disk.
	sum, err := fileSHA256(part)
	if err != nil {
		return fmt.Errorf("verify Auto reviewer: %w", err)
	}
	if !strings.EqualFold(sum, spec.SHA256) {
		_ = os.Remove(part)
		return fmt.Errorf("download Auto reviewer: checksum mismatch (got %s); the partial download was discarded", sum)
	}
	if err := os.Rename(part, dest); err != nil {
		return fmt.Errorf("install Auto reviewer: %w", err)
	}
	if progress != nil {
		fmt.Fprintln(progress, "[claude-code] local Auto reviewer downloaded and verified.")
	}
	return nil
}

// partialSize is how many bytes of a resumable download are already on disk.
func partialSize(path string) int64 {
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return info.Size()
}

// fetchIntoPart appends to the partial file, asking the server to resume from
// what is already there. A server that ignores Range answers 200 with the whole
// artifact, so the file is truncated first and the transfer starts over.
func fetchIntoPart(ctx context.Context, client *http.Client, spec ModelSpec, part string, have int64, progress io.Writer) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, spec.URL, nil)
	if err != nil {
		return err
	}
	if have > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", have))
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		have = 0 // Range ignored (or not requested): this is the whole artifact.
	case http.StatusPartialContent:
		if have == 0 {
			return fmt.Errorf("unexpected HTTP 206 for a fresh download")
		}
	case http.StatusRequestedRangeNotSatisfiable:
		// Already have everything the server will give us; let the caller verify.
		return nil
	default:
		return fmt.Errorf("HTTP %s", resp.Status)
	}

	flags := os.O_CREATE | os.O_WRONLY
	if have > 0 {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	// Watchdog: bytes must keep arriving. A connection that goes quiet is
	// cancelled so the retry loop can resume, instead of hanging until the
	// process is killed.
	body, stop := stallGuard(ctx, resp.Body)
	defer stop()

	var w io.Writer = f
	if progress != nil {
		w = io.MultiWriter(f, &progressWriter{out: progress, total: spec.Size, done: have, next: have + (256 << 20)})
	}
	if _, err := io.Copy(w, body); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	return f.Close()
}

// stallGuard wraps a response body so a transfer that stops producing bytes for
// reviewerStallTimeout fails instead of hanging forever.
func stallGuard(ctx context.Context, body io.Reader) (io.Reader, func()) {
	guarded, cancel := context.WithCancel(ctx)
	counter := &stallCounter{}
	done := make(chan struct{})
	go func() {
		ticker := time.NewTicker(reviewerStallTimeout / 4)
		defer ticker.Stop()
		var lastSeen int64
		idle := time.Duration(0)
		for {
			select {
			case <-done:
				return
			case <-guarded.Done():
				return
			case <-ticker.C:
				now := counter.read.Load()
				if now != lastSeen {
					lastSeen, idle = now, 0
					continue
				}
				idle += reviewerStallTimeout / 4
				if idle >= reviewerStallTimeout {
					cancel()
					return
				}
			}
		}
	}()
	counter.r = body
	return counter, func() { close(done); cancel() }
}

type stallCounter struct {
	r    io.Reader
	read atomic.Int64
}

func (c *stallCounter) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	if n > 0 {
		c.read.Add(int64(n))
	}
	return n, err
}

// fileSHA256 hashes a completed download so resumed transfers are verified
// against what actually landed on disk.
func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

type progressWriter struct {
	out   io.Writer
	total int64
	done  int64
	next  int64
}

func (w *progressWriter) Write(p []byte) (int, error) {
	w.done += int64(len(p))
	if w.done >= w.next {
		pct := 0
		if w.total > 0 {
			pct = int(w.done * 100 / w.total)
		}
		fmt.Fprintf(w.out, "[claude-code] Auto reviewer download: %d%%\n", pct)
		w.next += 256 << 20
	}
	return len(p), nil
}

// IsClassifierRequest reports whether a /v1/messages body is Claude Code's
// hidden Auto safety review.
func IsClassifierRequest(body []byte) bool {
	var request struct {
		System json.RawMessage `json:"system"`
	}
	if json.Unmarshal(body, &request) != nil || len(request.System) == 0 {
		return false
	}
	var blocks []struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(request.System, &blocks) == nil {
		for _, block := range blocks {
			if strings.Contains(block.Text, ClassifierMarker) {
				return true
			}
		}
		return false
	}
	var system string
	return json.Unmarshal(request.System, &system) == nil && strings.Contains(system, ClassifierMarker)
}

// Router exposes a loopback-only endpoint and sends classifier requests to the
// reviewer while transparently streaming all normal traffic to the main model.
type Router struct {
	server         *http.Server
	ln             net.Listener
	port           int
	sched          *scheduler
	maxMainActive  int
	companionAlias string
	hasCompanion   bool
	mainBaseURL    string
	// separateReviewer reports whether the reviewer runs on its own backend
	// distinct from the main model. When false (no-room fallback), classifier
	// requests route to the main model with a visible notice — never silently.
	separateReviewer bool
	// reviewerContext is the separate reviewer's context window in tokens. A
	// classifier prompt estimated to exceed it is routed to the main model with
	// a visible notice (reviewer overflow) instead of being sent where it cannot
	// fit. Non-positive disables the overflow fallback.
	reviewerContext int
	// Claude Code issues a classifier request per tool call, so an unguarded
	// notice would repeat on every one of them and bury the rest of the run's
	// output. Each fallback explains itself the first time it happens and stays
	// quiet after that; the route is still recorded on every request.
	noRoomNotice   sync.Once
	overflowNotice sync.Once
	// reviewerFailNotice covers the runtime case: a seated reviewer that could
	// not answer. Announced once rather than per tool call, like the others.
	reviewerFailNotice sync.Once
	// utilityReviewerNotice announces the review-only-reviewer utility lane
	// once; the route decision itself is per request.
	utilityReviewerNotice sync.Once
	// utilityFailNotice announces that a preferred companion failed and the
	// request was preserved by retrying it on the main model.
	utilityFailNotice sync.Once
	// msgDelimiters are the chat role markers read from the backend's own
	// startup output. They are injected into main-route requests that carry
	// none, because a sliding-window model can only reuse a prefix from a
	// context checkpoint and llama.cpp only creates one at a user-message
	// boundary it can find.
	msgDelimiters []MessageDelimiter
	ubatch        int
	backendMu     sync.Mutex
	backend       map[string]any
	pollOnce      sync.Once
	pollStop      chan struct{}
	stopPollOnce  sync.Once
	mainActive    atomic.Int64
	mainQueued    atomic.Int64
	metrics       *metricsSink
}

// SetCompanion enables the cheap-tier lane. alias is the model name the
// companion backend was launched with; hasSeparateBackend must be false when
// the companion URL is the main model, so cheap-tier work is not routed into a
// lane that leads back to the same server.
func (r *Router) SetCompanion(alias string, hasSeparateBackend bool) {
	if r == nil {
		return
	}
	r.companionAlias = strings.TrimSpace(alias)
	r.hasCompanion = hasSeparateBackend
}

// SetReviewerContext sets the separate reviewer's context window in tokens.
// When a separate reviewer is seated and a classifier prompt is estimated to
// exceed this window, the router falls back to the main model (self-classify)
// with a visible notice instead of forwarding a prompt the reviewer cannot
// accept. A non-positive value disables the overflow fallback.
func (r *Router) SetReviewerContext(tokens int) {
	if r != nil {
		r.reviewerContext = tokens
	}
}

// SetPhaseAwareAdmission enables the conservative active-lane policy for
// host-offloaded multi-slot models. Long cold prompts serialize because their
// CPU-expert/PCIe passes make concurrent decode collapse; small and cache-hot
// appends may use the otherwise idle slot after the active request emits its
// first generated SSE delta.
func (r *Router) SetPhaseAwareAdmission(enabled bool, coldPromptTokens int) {
	if r != nil && r.sched != nil {
		r.sched.setPhaseAware(enabled, coldPromptTokens)
	}
}

// estimatedPromptTokens conservatively estimates how many context tokens a
// routed request consumes from its JSON body size. The estimate must err on the
// high side so an oversized prompt is caught rather than silently sent into an
// overflow: ~2 bytes per token. The previous ~3-byte divisor under-counted
// code-dense and escaped-JSON prompts — real reviews of 65,675 tokens arrived
// in bodies the estimate scored below the 65,536 window and overflowed the
// reviewer with a 400. The conservative false-positive case is harmless: a
// prompt that would actually fit is sent to the main model early instead of
// risking a reviewer overflow.
func estimatedPromptTokens(body []byte) int {
	return (len(body) + 1) / 2
}

// utilityEnabled reports whether cheap-tier requests have somewhere to go.
func (r *Router) utilityEnabled() bool {
	return r != nil && r.hasCompanion
}

// StartRouter starts the local request router on an automatically selected
// loopback port. Text-only routes replace Anthropic image blocks with a text
// notice so one image-producing tool result cannot poison every later turn.
func StartRouter(mainBaseURL, reviewerBaseURL string, supportsVision bool, maxMainActive int) (*Router, error) {
	mainURL, err := url.Parse(mainBaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse main model URL: %w", err)
	}
	reviewerURL, err := url.Parse(reviewerBaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse reviewer URL: %w", err)
	}
	mainProxy := httputil.NewSingleHostReverseProxy(mainURL)
	reviewerProxy := httputil.NewSingleHostReverseProxy(reviewerURL)
	mainProxy.ErrorHandler = proxyError
	reviewerProxy.ErrorHandler = proxyError
	// Classifier traffic marks its attempts on the request context so a failed
	// review can be retried on the main model; utility traffic carries no
	// attempt and keeps surfacing errors directly.
	installReviewerFallbackHooks(reviewerProxy)
	router := &Router{maxMainActive: maxMainActive, mainBaseURL: strings.TrimRight(mainBaseURL, "/")}
	router.separateReviewer = strings.TrimRight(reviewerBaseURL, "/") != strings.TrimRight(mainBaseURL, "/")
	if maxMainActive > 0 {
		router.sched = newScheduler(maxMainActive)
	}

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/ggrun/router" {
			w.Header().Set("Content-Type", "application/json")
			status := map[string]any{
				"active": router.mainActive.Load(),
				"queued": router.mainQueued.Load(),
				"limit":  int64(router.maxMainActive),
			}
			if admission := router.sched.status(); admission != nil {
				status["admission"] = admission
			}
			if summary := router.MetricsSummary(); summary != nil {
				status["metrics"] = summary
			}
			// Served from the last polled snapshot, never fetched inline: the
			// status endpoint must not add load to a backend that is already
			// saturated, and must not block on it either.
			if backend, ok := router.BackendSnapshot(); ok {
				status["backend"] = backend
			}
			_ = json.NewEncoder(w).Encode(status)
			return
		}
		if r.URL.Path != "/v1/messages" {
			mainProxy.ServeHTTP(w, r)
			return
		}
		body, err := io.ReadAll(io.LimitReader(r.Body, maxRoutedRequestBytes+1))
		if err != nil {
			http.Error(w, "read request body", http.StatusBadRequest)
			return
		}
		if len(body) > maxRoutedRequestBytes {
			http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		r.ContentLength = int64(len(body))
		if !supportsVision {
			body = sanitizeTextOnlyImages(body)
			r.Body = io.NopCloser(bytes.NewReader(body))
			r.ContentLength = int64(len(body))
		}
		if IsClassifierRequest(body) {
			// A classifier/security request is normally the separate reviewer's
			// job. Two user-approved (Option A) cases route it to the MAIN model
			// instead, each with a visible notice — never silently:
			//   1) no-room: no separate reviewer was seated (reviewer proxy
			//      targets the main model), so the main model classifies its own
			//      request.
			//   2) overflow: a separate reviewer IS seated, but this prompt
			//      exceeds the reviewer's context window, so it cannot accept it.
			proxy, route := reviewerProxy, routeReviewer
			switch {
			case !router.separateReviewer:
				router.noRoomNotice.Do(func() {
					fmt.Fprintf(os.Stderr, "[claude-code] no separate Auto reviewer seated; the main model is classifying its own requests (self-classify, user-approved)\n")
				})
				proxy, route = mainProxy, routeMain
			case router.reviewerContext > 0 && estimatedPromptTokens(body) > router.reviewerContext:
				router.overflowNotice.Do(func() {
					fmt.Fprintf(os.Stderr, "[claude-code] an Auto review prompt exceeded the reviewer context window (%d tokens); such reviews go to the main model instead (self-classify, reviewer overflow)\n", router.reviewerContext)
				})
				proxy, route = mainProxy, routeMain
			}
			// Try the separate reviewer first. A reviewer that crashed, never
			// came up, or rejected this prompt (a context that will not hold it,
			// a template it cannot apply) must not take the review down with it:
			// nothing is written to the client until the response is known good,
			// so a failed attempt falls through to the main model instead of
			// returning 502.
			if route == routeReviewer {
				alias := MainAlias
				if router.utilityEnabled() {
					alias = router.companionAlias
				}
				if router.tryReviewer(w, r, reviewerProxy, reviewerClassifierBody(body, alias)) {
					return
				}
				router.reviewerFailNotice.Do(func() {
					fmt.Fprintf(os.Stderr, "[claude-code] the Auto reviewer could not answer a review; routing it to the main model instead (self-classify, reviewer unavailable)\n")
				})
				proxy, route = mainProxy, routeMain
			}
			body = retargetModel(body, MainAlias)
			r.Body = io.NopCloser(bodyReader(body))
			r.ContentLength = int64(len(body))
			router.serve(w, r, proxy, route, body)
			return
		}
		// Decide the cheap-tier destination before injecting any main-backend
		// metadata. A companion has a different template and must never receive
		// message delimiters parsed from the main model. With no companion, rewrite
		// local-fast back to the actual main alias before forwarding.
		if IsUtilityRequest(body) {
			if router.utilityEnabled() {
				companionBody := utilityBody(body, router.companionAlias)
				if router.tryReviewerUtility(w, r, reviewerProxy, companionBody) {
					return
				}
				router.utilityFailNotice.Do(func() {
					fmt.Fprintf(os.Stderr, "[claude-code] the local companion could not answer cheap-tier work; retrying it on the main model\n")
				})
			}
			// No worker-serving companion. A seated review-only reviewer is
			// still a real second backend: cheap-tier calls (the permission
			// classifier, haiku-tier background work) on the main model queue
			// behind long foreground streams on a single slot and time out, so
			// prefer the reviewer whenever the prompt fits its context. Too
			// large for the reviewer falls through to the main model as before.
			if !router.utilityEnabled() && router.separateReviewer && (router.reviewerContext <= 0 || estimatedPromptTokens(body) <= router.reviewerContext) {
				router.utilityReviewerNotice.Do(func() {
					fmt.Fprintf(os.Stderr, "[claude-code] no worker companion; cheap-tier calls (classifier, background) go to the seated reviewer instead of queueing behind main-model work\n")
				})
				reviewerBody := utilityBody(body, MainAlias)
				if router.tryReviewerUtility(w, r, reviewerProxy, reviewerBody) {
					return
				}
				router.utilityFailNotice.Do(func() {
					fmt.Fprintf(os.Stderr, "[claude-code] the seated reviewer could not answer cheap-tier work; retrying it on the main model\n")
				})
			}
			body = utilityBody(body, MainAlias)
			r.Body = io.NopCloser(bodyReader(body))
			r.ContentLength = int64(len(body))
		}
		// Main-model work only: the reviewer is a different model with its own
		// template, and its conversations are short enough that a missing
		// checkpoint costs little.
		if delims := router.messageDelimiters(); len(delims) > 0 {
			if injected := InjectMessageDelimiters(body, delims); len(injected) > 0 {
				body = injected
				r.Body = io.NopCloser(bodyReader(body))
				r.ContentLength = int64(len(body))
			}
		}
		router.serve(w, r, mainProxy, routeMain, body)
	})
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("listen for Claude Auto router: %w", err)
	}
	router.ln = ln
	router.port = ln.Addr().(*net.TCPAddr).Port
	router.server = &http.Server{Handler: handler, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		_ = router.server.Serve(ln)
	}()
	return router, nil
}

// serve proxies one routed request and records its timing. Only the main route
// is admission-controlled: the reviewer lane stays free so a safety review never
// waits behind coding work.
func (r *Router) serve(w http.ResponseWriter, req *http.Request, proxy *httputil.ReverseProxy, route string, body []byte) {
	start := time.Now()
	conversation := conversationKey(body)
	var queue time.Duration
	if route == routeMain && r != nil && r.sched != nil {
		r.mainQueued.Add(1)
		admission := r.sched.acquireRequest(req.Context(), conversation, laneOf(body), estimatedPromptTokens(body))
		r.mainQueued.Add(-1)
		if admission == nil {
			r.record(route, body, start, time.Since(start), nil)
			return
		}
		queue = time.Since(start)
		r.mainActive.Add(1)
		defer func() {
			r.mainActive.Add(-1)
			admission.release()
		}()
		metered := &meteredWriter{ResponseWriter: w, start: start, onDecode: admission.markDecode}
		proxy.ServeHTTP(metered, req)
		r.record(route, body, start, queue, metered)
		return
	}
	metered := &meteredWriter{ResponseWriter: w, start: start}
	proxy.ServeHTTP(metered, req)
	r.record(route, body, start, queue, metered)
}

// record stores one request's timing. A nil metered writer means the caller gave
// up while queued, which is exactly the starvation case worth counting.
func (r *Router) record(route string, body []byte, start time.Time, queue time.Duration, metered *meteredWriter) {
	if r == nil || r.metrics == nil {
		return
	}
	rec := RequestRecord{
		Time:         start.UTC().Format(time.RFC3339Nano),
		Route:        route,
		Conversation: conversationKey(body),
		Stream:       isStreamRequest(body),
		QueueMS:      queue.Milliseconds(),
		TotalMS:      time.Since(start).Milliseconds(),
		RequestBytes: int64(len(body)),
		Aborted:      true,
	}
	if metered != nil {
		rec.Status = metered.status
		rec.ResponseBytes = metered.written
		rec.Usage = metered.usage
		rec.TTFBMS = metered.ttfb.Milliseconds()
		rec.DecodeStartMS = metered.decodeStart.Milliseconds()
		rec.Aborted = metered.status == 0 && metered.written == 0
	}
	if rec.Aborted {
		if metered == nil {
			rec.CancelPhase = "queue"
		} else {
			rec.CancelPhase = "service"
		}
		switch {
		case rec.TotalMS >= 50_000 && rec.TotalMS <= 75_000:
			rec.DeadlineBucket = "~60s"
		case rec.TotalMS >= 550_000 && rec.TotalMS <= 650_000:
			rec.DeadlineBucket = "~600s"
		}
	}
	r.metrics.record(rec)
}

// sanitizeTextOnlyImages replaces only Anthropic message image blocks. It
// handles both direct user images and images nested in tool_result content,
// which is how Claude Code returns Read results for GIF/PNG/JPEG files.
// Unrelated JSON objects (including tool schemas) are deliberately untouched.
func sanitizeTextOnlyImages(body []byte) []byte {
	var payload map[string]any
	if json.Unmarshal(body, &payload) != nil {
		return body
	}
	messages, ok := payload["messages"].([]any)
	if !ok {
		return body
	}
	replaced := 0
	for _, rawMessage := range messages {
		message, ok := rawMessage.(map[string]any)
		if !ok {
			continue
		}
		content, ok := message["content"].([]any)
		if !ok {
			continue
		}
		message["content"], replaced = sanitizeContentBlocks(content, replaced)
	}
	if replaced == 0 {
		return body
	}
	sanitized, err := json.Marshal(payload)
	if err != nil {
		return body
	}
	return sanitized
}

func sanitizeContentBlocks(blocks []any, replaced int) ([]any, int) {
	for i, rawBlock := range blocks {
		block, ok := rawBlock.(map[string]any)
		if !ok {
			continue
		}
		typeName, _ := block["type"].(string)
		switch typeName {
		case "image":
			blocks[i] = map[string]any{"type": "text", "text": textOnlyImagePlaceholder}
			replaced++
		case "tool_result":
			if nested, ok := block["content"].([]any); ok {
				block["content"], replaced = sanitizeContentBlocks(nested, replaced)
			}
		}
	}
	return blocks, replaced
}

func proxyError(w http.ResponseWriter, _ *http.Request, err error) {
	if errors.Is(err, context.Canceled) {
		return
	}
	http.Error(w, "local model unavailable", http.StatusBadGateway)
}

// Port returns the loopback port Claude Code should use.
func (r *Router) Port() int {
	if r == nil {
		return 0
	}
	return r.port
}

// URL returns the router's loopback base URL.
func (r *Router) URL() string {
	return "http://127.0.0.1:" + strconv.Itoa(r.Port())
}

// Close stops accepting routed requests and drains active requests briefly.
func (r *Router) Close() error {
	if r == nil || r.server == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if r.pollStop != nil {
		r.stopPollOnce.Do(func() { close(r.pollStop) })
	}
	err := r.server.Shutdown(ctx)
	if closeErr := r.metrics.close(); err == nil {
		err = closeErr
	}
	return err
}

// SetMessageDelimiters records the chat role markers for the main model. Safe to
// call once the backend has started and its startup output has been read.
func (r *Router) SetMessageDelimiters(delims []MessageDelimiter) {
	if r == nil {
		return
	}
	r.backendMu.Lock()
	defer r.backendMu.Unlock()
	r.msgDelimiters = append([]MessageDelimiter(nil), delims...)
}

func (r *Router) messageDelimiters() []MessageDelimiter {
	if r == nil {
		return nil
	}
	r.backendMu.Lock()
	defer r.backendMu.Unlock()
	return r.msgDelimiters
}

// StartBackendPolling samples the model server's own timing counters on a
// timer and caches the result.
//
// The backend's counters are the authority for throughput: router timings
// measure whole HTTP requests, and a cancelled call or a client that stops
// reading keeps one open long after the model has finished with it. ggrun
// derived 0.26 tok/s that way while the backend reported 4.13 for the same
// traffic.
//
// Polling is deliberately decoupled from the status endpoint. A status line
// refreshing once a second must not translate into a request per second
// against a saturated backend, and must never block on one.
func (r *Router) StartBackendPolling(interval time.Duration) {
	if r == nil || r.mainBaseURL == "" || interval <= 0 {
		return
	}
	r.pollOnce.Do(func() {
		r.pollStop = make(chan struct{})
		go func() {
			ticker := time.NewTicker(interval)
			defer ticker.Stop()
			for {
				r.refreshBackendSnapshot()
				select {
				case <-ticker.C:
				case <-r.pollStop:
					return
				}
			}
		}()
	})
}

// backendPollTimeout must exceed how long the backend can take to answer while
// it is working, not how long the answer "should" take.
//
// /metrics is not a static read: llama.cpp posts a SERVER_TASK_TYPE_METRICS
// task onto the same queue as inference and blocks until the task loop reaches
// it. High priority does not preempt the batch in flight, so the floor is one
// microbatch -- 3.58 s at -ub 512 and 143 t/s on this project. The deadline was
// 3 s, below that floor, so a poll issued while the backend was busy could not
// succeed by construction. Measured live: 1.41 s, timeout, 3.66 s.
//
// Two consequences, and the second is the reason this matters. Each expired
// poll left a cancelled task behind -- 281 in 33 minutes, 2756 task ids
// consumed against 7 real inference tasks. And the snapshot only ever refreshed
// when the backend was idle enough to answer inside 3 s, which biased the
// counters this router treats as the authority on throughput toward exactly the
// moments when nothing was running.
//
// Polling is sequential -- refreshBackendSnapshot returns before the next tick
// is awaited -- so a generous deadline cannot stack requests. It only lowers the
// sample rate while the backend is busy, which beats sampling nothing.
const backendPollTimeout = 30 * time.Second

// backendPollClient is passed explicitly because Fetch's nil-client default
// carries its own 5 s timeout, and http.Client.Timeout is independent of the
// context: whichever fires first wins. Raising only the context deadline would
// have left the effective limit at 5 s and changed nothing.
var backendPollClient = &http.Client{Timeout: backendPollTimeout}

func (r *Router) refreshBackendSnapshot() {
	ctx, cancel := context.WithTimeout(context.Background(), backendPollTimeout)
	defer cancel()
	m, err := backendmetrics.Fetch(ctx, backendPollClient, r.mainBaseURL)
	if err != nil || !m.Available() {
		return
	}
	snap := map[string]any{
		"prompt_tokens_per_s": m.PromptTokensPerSecond(),
		"decode_tokens_per_s": m.DecodeTokensPerSecond(),
		"prompt_tokens":       m.PromptTokens,
		"predicted_tokens":    m.PredictedTokens,
		"decode_calls":        m.DecodeCalls,
		"busy_slots_per_call": m.BusySlotsPerCall,
	}
	if pc, ok := backendmetrics.PassCostFrom(m, r.ubatch); ok {
		snap["pass_cost"] = map[string]any{
			"fixed_ms":    pc.FixedMS,
			"marginal_ms": pc.MarginalMS,
			"fixed_share": pc.FixedShare(),
			"ubatch":      pc.UBatch,
			"projected_tokens_per_s": map[string]float64{
				"1": pc.ProjectedTokensPerSecond(1),
				"4": pc.ProjectedTokensPerSecond(4),
				"8": pc.ProjectedTokensPerSecond(8),
			},
		}
	}
	r.backendMu.Lock()
	r.backend = snap
	r.backendMu.Unlock()
}

// BackendSnapshot returns the most recent polled backend metrics.
func (r *Router) BackendSnapshot() (map[string]any, bool) {
	if r == nil {
		return nil, false
	}
	r.backendMu.Lock()
	defer r.backendMu.Unlock()
	if r.backend == nil {
		return nil, false
	}
	return r.backend, true
}

// SetUBatch records the micro-batch the backend was launched with, which the
// pass-cost decomposition needs to know how many tokens a prefill pass carried.
func (r *Router) SetUBatch(n int) {
	if r != nil && n > 0 {
		r.ubatch = n
	}
}
