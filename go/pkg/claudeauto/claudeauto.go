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
	"time"
)

const (
	// ClassifierMarker is part of Claude Code's Auto-mode system prompt. Keep
	// this exact and narrow so ordinary user prompts mentioning "security" do
	// not leave the main coding model.
	ClassifierMarker = "You are a security monitor for autonomous AI coding agents."

	DefaultReviewerDisplayName = "Qwen3.5-2B"
	DefaultReviewerFile        = "Qwen_Qwen3.5-2B-Q4_K_M.gguf"
	DefaultReviewerSize        = int64(1396198496)
	DefaultReviewerSHA         = "57a1085840f497d764a7fc5d346922dbde961efb54cc792ea81d694fd846a1d8"
	DefaultReviewerURL         = "https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/7d26695454df6de5fbcce2e58681e62dae06ce43/" + DefaultReviewerFile

	maxRoutedRequestBytes = 16 << 20
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

// ReviewerModelPath returns the user override, or ggrun's private reviewer
// cache path. The reviewer is deliberately kept out of the normal model list.
func ReviewerModelPath(appHome string) string {
	if path := strings.TrimSpace(os.Getenv("GGRUN_CLAUDE_REVIEWER_MODEL")); path != "" {
		return path
	}
	return filepath.Join(appHome, ".cache", "claude-reviewer", DefaultReviewerFile)
}

// EnsureReviewerModel validates a user-supplied model, or downloads and
// verifies the pinned default artifact on first use.
func EnsureReviewerModel(ctx context.Context, appHome string, progress io.Writer) (string, error) {
	path := ReviewerModelPath(appHome)
	if strings.TrimSpace(os.Getenv("GGRUN_CLAUDE_REVIEWER_MODEL")) != "" {
		if err := validateGGUF(path, 0); err != nil {
			return "", fmt.Errorf("GGRUN_CLAUDE_REVIEWER_MODEL: %w", err)
		}
		return path, nil
	}
	if err := validatePinnedGGUF(path, defaultModel); err == nil {
		return path, nil
	}
	if progress != nil {
		fmt.Fprintf(progress, "[claude-code] downloading pinned local Auto reviewer (%.1f GB)...\n", float64(defaultModel.Size)/(1024*1024*1024))
	}
	client := &http.Client{Timeout: 30 * time.Minute}
	if err := downloadModel(ctx, client, defaultModel, path, progress); err != nil {
		return "", err
	}
	return path, nil
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

func downloadModel(ctx context.Context, client *http.Client, spec ModelSpec, dest string, progress io.Writer) error {
	if client == nil {
		client = http.DefaultClient
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return fmt.Errorf("create reviewer cache: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(dest), ".reviewer-*.part")
	if err != nil {
		return fmt.Errorf("create reviewer download: %w", err)
	}
	tmpPath := tmp.Name()
	keep := false
	defer func() {
		tmp.Close()
		if !keep {
			_ = os.Remove(tmpPath)
		}
	}()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, spec.URL, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("download Auto reviewer: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download Auto reviewer: HTTP %s", resp.Status)
	}
	h := sha256.New()
	w := io.MultiWriter(tmp, h)
	if progress != nil {
		w = io.MultiWriter(tmp, h, &progressWriter{out: progress, total: spec.Size, next: 256 << 20})
	}
	n, err := io.Copy(w, resp.Body)
	if err != nil {
		return fmt.Errorf("download Auto reviewer: %w", err)
	}
	if n != spec.Size {
		return fmt.Errorf("download Auto reviewer: got %d bytes, want %d", n, spec.Size)
	}
	gotSHA := hex.EncodeToString(h.Sum(nil))
	if !strings.EqualFold(gotSHA, spec.SHA256) {
		return fmt.Errorf("download Auto reviewer: checksum mismatch (got %s)", gotSHA)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("sync Auto reviewer: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close Auto reviewer: %w", err)
	}
	if err := os.Rename(tmpPath, dest); err != nil {
		return fmt.Errorf("install Auto reviewer: %w", err)
	}
	keep = true
	if progress != nil {
		fmt.Fprintln(progress, "[claude-code] local Auto reviewer downloaded and verified.")
	}
	return nil
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
	server *http.Server
	ln     net.Listener
	port   int
}

// StartRouter starts the local request router on an automatically selected
// loopback port.
func StartRouter(mainBaseURL, reviewerBaseURL string) (*Router, error) {
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

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
		if IsClassifierRequest(body) {
			reviewerProxy.ServeHTTP(w, r)
			return
		}
		mainProxy.ServeHTTP(w, r)
	})
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("listen for Claude Auto router: %w", err)
	}
	router := &Router{ln: ln, port: ln.Addr().(*net.TCPAddr).Port}
	router.server = &http.Server{Handler: handler, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		_ = router.server.Serve(ln)
	}()
	return router, nil
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
	return r.server.Shutdown(ctx)
}
