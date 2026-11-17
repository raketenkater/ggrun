package daemon

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/raketenkater/ggrun/pkg/server"
)

// Daemon holds a persistent llama-server process and exposes a control API.
type Daemon struct {
	addr        string
	mu          sync.Mutex
	process     *server.Process
	config      Config
	http        *http.Server
	startServer func([]string, int, time.Duration) (*server.Process, error)
}

// Config holds daemon settings.
type Config struct {
	ModelPath   string   `json:"model_path"`
	ServerArgs  []string `json:"server_args"`
	Port        int      `json:"port"`
	ControlPort int      `json:"control_port"`
	MemoryMaxMB int      `json:"memory_max_mb,omitempty"`
	// MemoryHighMB is the cgroup reclaim threshold for the managed backend.
	// When zero and MemoryMaxMB is set, the daemon derives it the same way the
	// launcher does: for an mmap-backed plan the plan budget stays on the soft
	// boundary and the hard ceiling moves up to the host reclaim ceiling.
	MemoryHighMB int `json:"memory_high_mb,omitempty"`

	// ControlToken authenticates every control route. New generates one when it
	// is empty, so browser-originated localhost POSTs cannot mutate the daemon.
	ControlToken string `json:"-"`
	// TokenFile is where Start writes ControlToken (mode 0600). Empty uses
	// <user config dir>/ggrun/daemon-<control port>.token.
	TokenFile string `json:"-"`

	// AllowExplicitServerArgs keeps trusted tests/owners able to provide argv
	// directly. The public control API defaults to model-path reloads only.
	AllowExplicitServerArgs bool `json:"-"`

	// StartupTimeoutSecs caps how long handleStart/handleReload waits for
	// the underlying llama-server to become healthy. Cold-cache loads of
	// big MoE models (e.g. MiniMax M2.7 @ 94 GB) routinely take 2-3 min.
	// Zero falls back to the daemon default (300s).
	StartupTimeoutSecs int `json:"startup_timeout_secs,omitempty"`

	// ComputeArgs, if set, rebuilds the full llama-server argv from a model
	// path. /reload calls it when handed a model_path with no explicit
	// server_args, so swaps get the same auto-placement as the initial
	// launch. Not serialized — injected by the daemon's owner.
	ComputeArgs func(modelPath string, port int) ([]string, error) `json:"-"`
	// ComputeMemory recomputes the cgroup reclaim/hard limits for the exact
	// argv returned by ComputeArgs. A reload can switch backend loader or move a
	// model between resident and mmap placement, so retaining the old model's
	// memory.high policy is unsafe. Not serialized — injected by the owner.
	ComputeMemory func(serverArgs []string) (highMB, maxMB int, err error) `json:"-"`
}

// startupTimeout returns the configured wait or the daemon default.
func (c Config) startupTimeout() time.Duration {
	if c.StartupTimeoutSecs > 0 {
		return time.Duration(c.StartupTimeoutSecs) * time.Second
	}
	return 300 * time.Second
}

// New creates a new daemon instance.
func New(cfg Config) *Daemon {
	if cfg.ControlPort == 0 {
		cfg.ControlPort = 9090
	}
	if cfg.ControlToken == "" {
		cfg.ControlToken = generateControlToken()
	}
	return &Daemon{
		// The control API is still loopback-only even though every route also
		// requires a bearer token. Defense in depth matters for localhost pivots.
		addr:   net.JoinHostPort("127.0.0.1", fmt.Sprintf("%d", cfg.ControlPort)),
		config: cfg,
	}
}

// Start launches the control API server.
func (d *Daemon) Start() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/status", d.handleStatus)
	mux.HandleFunc("/start", d.handleStart)
	mux.HandleFunc("/stop", d.handleStop)
	mux.HandleFunc("/reload", d.handleReload)
	mux.HandleFunc("/config", d.handleConfig)

	srv := &http.Server{Addr: d.addr, Handler: mux}
	d.mu.Lock()
	d.http = srv
	token := d.config.ControlToken
	d.mu.Unlock()
	fmt.Printf("[daemon] control API on %s\n", d.addr)
	// The token is a credential: it goes to an owner-only file, never to the
	// terminal or a log that other users or tools may read.
	path, werr := d.writeControlToken(token)
	if werr != nil {
		return fmt.Errorf("write daemon control token: %w", werr)
	}
	fmt.Printf("[daemon] control token in %s (owner-only)\n", path)
	err := srv.ListenAndServe()
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

// controlTokenPath is the file Start writes the control token to.
func (d *Daemon) controlTokenPath() (string, error) {
	if d.config.TokenFile != "" {
		return d.config.TokenFile, nil
	}
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "ggrun", fmt.Sprintf("daemon-%d.token", d.config.ControlPort)), nil
}

func (d *Daemon) writeControlToken(token string) (string, error) {
	path, err := d.controlTokenPath()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return "", err
	}
	// Replace rather than rewrite in place, so an existing file with wider
	// permissions never holds the new token.
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".*")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return "", err
	}
	if _, err := tmp.WriteString(token + "\n"); err != nil {
		tmp.Close()
		return "", err
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		return "", err
	}
	return path, nil
}

// Close stops the managed model process and closes the control API. It is safe
// to call during signal-driven shutdown even if no model has been started.
func (d *Daemon) Close() error {
	d.mu.Lock()
	var processErr error
	if d.process != nil {
		processErr = d.process.Stop()
		d.process = nil
	}
	srv := d.http
	d.http = nil
	d.mu.Unlock()

	var httpErr error
	if srv != nil {
		httpErr = srv.Close()
	}
	return errors.Join(processErr, httpErr)
}

func (d *Daemon) handleStatus(w http.ResponseWriter, r *http.Request) {
	if !d.authorize(w, r) {
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "{\"error\":\"method not allowed\"}", http.StatusMethodNotAllowed)
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	status := map[string]interface{}{
		"running":     d.process != nil && d.process.IsRunning(),
		"config":      d.safeConfigLocked(),
		"server_port": d.config.Port,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (d *Daemon) handleStart(w http.ResponseWriter, r *http.Request) {
	if !d.authorize(w, r) {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "{\"error\":\"method not allowed\"}", http.StatusMethodNotAllowed)
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.process != nil && d.process.IsRunning() {
		http.Error(w, `{"error":"already running"}`, http.StatusConflict)
		return
	}
	p, err := d.start(d.config, d.config.ServerArgs, d.config.Port, d.config.startupTimeout())
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), http.StatusInternalServerError)
		return
	}
	d.process = p
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "started"})
}

func (d *Daemon) handleStop(w http.ResponseWriter, r *http.Request) {
	if !d.authorize(w, r) {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "{\"error\":\"method not allowed\"}", http.StatusMethodNotAllowed)
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.process == nil {
		http.Error(w, `{"error":"not running"}`, http.StatusConflict)
		return
	}
	if err := d.process.Stop(); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), http.StatusInternalServerError)
		return
	}
	d.process = nil
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "stopped"})
}

func (d *Daemon) handleReload(w http.ResponseWriter, r *http.Request) {
	if !d.authorize(w, r) {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "{\"error\":\"method not allowed\"}", http.StatusMethodNotAllowed)
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	var newCfg Config
	if err := json.NewDecoder(io.LimitReader(r.Body, maxControlBodyBytes)).Decode(&newCfg); err != nil {
		http.Error(w, `{"error":"invalid json"}`, http.StatusBadRequest)
		return
	}
	next := d.config
	if newCfg.ModelPath != "" {
		next.ModelPath = newCfg.ModelPath
	}
	if newCfg.Port > 0 {
		next.Port = newCfg.Port
	}
	if newCfg.StartupTimeoutSecs > 0 {
		next.StartupTimeoutSecs = newCfg.StartupTimeoutSecs
	}
	if len(newCfg.ServerArgs) > 0 {
		if !d.config.AllowExplicitServerArgs {
			http.Error(w, `{"error":"server_args reload is disabled; send model_path to recompute placement"}`, http.StatusBadRequest)
			return
		}
		next.ServerArgs = newCfg.ServerArgs
	} else if newCfg.ModelPath != "" && next.ComputeArgs != nil {
		// Bare model swap — let ggrun compute placement for it.
		args, err := next.ComputeArgs(next.ModelPath, next.Port)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":"compute placement: %s"}`, err.Error()), http.StatusInternalServerError)
			return
		}
		next.ServerArgs = args
	}
	if next.ComputeMemory != nil {
		highMB, maxMB, err := next.ComputeMemory(next.ServerArgs)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":"compute memory scope: %s"}`, err.Error()), http.StatusInternalServerError)
			return
		}
		next.MemoryHighMB, next.MemoryMaxMB = highMB, maxMB
	}

	// A reload is an apply-and-run operation. When the daemon is idle this starts
	// the newly selected model; when it is already serving, it replaces it.
	wasRunning := d.process != nil && d.process.IsRunning()
	if wasRunning {
		if err := d.process.Stop(); err != nil {
			http.Error(w, fmt.Sprintf(`{"error":"stop old model: %s"}`, err.Error()), http.StatusInternalServerError)
			return
		}
		d.process = nil
		// Small delay to free ports
		time.Sleep(500 * time.Millisecond)
	}
	p, err := d.start(next, next.ServerArgs, next.Port, next.startupTimeout())
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), http.StatusInternalServerError)
		return
	}
	d.process = p
	d.config = next

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "reloaded"})
}

func (c Config) memoryHighMaxMB(args []string) (highMB, maxMB int) {
	_ = args
	maxMB = c.MemoryMaxMB
	highMB = c.MemoryHighMB
	if highMB > 0 {
		// An explicit reclaim threshold (the mmap reclaim band) is carried
		// through as configured; the caller has already placed the host
		// ceiling in MemoryMaxMB.
		return highMB, maxMB
	}
	// No explicit band: the soft threshold equals the hard ceiling, which is
	// exactly the daemon's pre-existing behavior (a single MemoryMax cap). A
	// zero max leaves both zero — no cgroup memory limit, as before.
	highMB = maxMB
	return highMB, maxMB
}

func (d *Daemon) start(cfg Config, args []string, port int, timeout time.Duration) (*server.Process, error) {
	if d.startServer != nil {
		return d.startServer(args, port, timeout)
	}
	highMB, maxMB := cfg.memoryHighMaxMB(args)
	return server.StartWithTimeoutToOptions(
		args,
		port,
		timeout,
		os.Stdout,
		os.Stderr,
		server.StartOptions{MemoryHighMB: highMB, MemoryMaxMB: maxMB},
	)
}

func (d *Daemon) handleConfig(w http.ResponseWriter, r *http.Request) {
	if !d.authorize(w, r) {
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "{\"error\":\"method not allowed\"}", http.StatusMethodNotAllowed)
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(d.safeConfigLocked())
}

const maxControlBodyBytes = 1 << 20

func generateControlToken() string {
	var b [32]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(fmt.Sprintf("generate daemon control token: %v", err))
	}
	return base64.RawURLEncoding.EncodeToString(b[:])
}

func (d *Daemon) authorize(w http.ResponseWriter, r *http.Request) bool {
	d.mu.Lock()
	token := d.config.ControlToken
	d.mu.Unlock()
	if token == "" {
		http.Error(w, `{"error":"daemon control token is not configured"}`, http.StatusUnauthorized)
		return false
	}
	got := r.Header.Get("X-GGRUN-Daemon-Token")
	if got == "" {
		got = bearerToken(r.Header.Get("Authorization"))
	}
	if subtle.ConstantTimeCompare([]byte(got), []byte(token)) != 1 {
		http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
		return false
	}
	return true
}

func bearerToken(header string) string {
	if len(header) < len("Bearer ") || !strings.EqualFold(header[:len("Bearer ")], "Bearer ") {
		return ""
	}
	return strings.TrimSpace(header[len("Bearer "):])
}

func (d *Daemon) safeConfigLocked() map[string]interface{} {
	return map[string]interface{}{
		"model_path":           d.config.ModelPath,
		"port":                 d.config.Port,
		"control_port":         d.config.ControlPort,
		"memory_max_mb":        d.config.MemoryMaxMB,
		"startup_timeout_secs": d.config.StartupTimeoutSecs,
	}
}
