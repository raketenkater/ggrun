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

	// ControlToken authenticates every control route. New generates one when it
	// is empty, so browser-originated localhost POSTs cannot mutate the daemon.
	ControlToken string `json:"-"`

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
	fmt.Printf("[daemon] control token: %s\n", token)
	err := srv.ListenAndServe()
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
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
	p, err := d.start(d.config.ServerArgs, d.config.Port, d.config.startupTimeout())
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
	p, err := d.start(next.ServerArgs, next.Port, next.startupTimeout())
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), http.StatusInternalServerError)
		return
	}
	d.process = p
	d.config = next

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "reloaded"})
}

func (d *Daemon) start(args []string, port int, timeout time.Duration) (*server.Process, error) {
	if d.startServer != nil {
		return d.startServer(args, port, timeout)
	}
	return server.StartWithTimeoutToOptions(
		args,
		port,
		timeout,
		os.Stdout,
		os.Stderr,
		server.StartOptions{MemoryMaxMB: d.config.MemoryMaxMB},
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
