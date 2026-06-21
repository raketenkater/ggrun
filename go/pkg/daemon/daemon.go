package daemon

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/raketenkater/ggrun/pkg/server"
)

// Daemon holds a persistent llama-server process and exposes a control API.
type Daemon struct {
	addr    string
	process *server.Process
	config  Config
}

// Config holds daemon settings.
type Config struct {
	ModelPath   string   `json:"model_path"`
	ServerArgs  []string `json:"server_args"`
	Port        int      `json:"port"`
	ControlPort int      `json:"control_port"`

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
	return &Daemon{
		addr:   fmt.Sprintf(":%d", cfg.ControlPort),
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
	fmt.Printf("[daemon] control API on %s\n", d.addr)
	return srv.ListenAndServe()
}

func (d *Daemon) handleStatus(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"running":     d.process != nil && d.process.IsRunning(),
		"config":      d.config,
		"server_port": d.config.Port,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (d *Daemon) handleStart(w http.ResponseWriter, r *http.Request) {
	if d.process != nil && d.process.IsRunning() {
		http.Error(w, `{"error":"already running"}`, http.StatusConflict)
		return
	}
	p, err := server.StartWithTimeout(d.config.ServerArgs, d.config.Port, d.config.startupTimeout())
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), http.StatusInternalServerError)
		return
	}
	d.process = p
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "started"})
}

func (d *Daemon) handleStop(w http.ResponseWriter, r *http.Request) {
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
	var newCfg Config
	if err := json.NewDecoder(r.Body).Decode(&newCfg); err != nil {
		http.Error(w, `{"error":"invalid json"}`, http.StatusBadRequest)
		return
	}
	if newCfg.ModelPath != "" {
		d.config.ModelPath = newCfg.ModelPath
	}
	if newCfg.Port > 0 {
		d.config.Port = newCfg.Port
	}
	if newCfg.StartupTimeoutSecs > 0 {
		d.config.StartupTimeoutSecs = newCfg.StartupTimeoutSecs
	}
	if len(newCfg.ServerArgs) > 0 {
		// Caller supplied explicit args — trust them verbatim.
		d.config.ServerArgs = newCfg.ServerArgs
	} else if newCfg.ModelPath != "" && d.config.ComputeArgs != nil {
		// Bare model swap — let ggrun compute placement for it.
		args, err := d.config.ComputeArgs(d.config.ModelPath, d.config.Port)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":"compute placement: %s"}`, err.Error()), http.StatusInternalServerError)
			return
		}
		d.config.ServerArgs = args
	}

	// Restart if running
	wasRunning := d.process != nil && d.process.IsRunning()
	if wasRunning {
		d.process.Stop()
		// Small delay to free ports
		time.Sleep(500 * time.Millisecond)
		p, err := server.StartWithTimeout(d.config.ServerArgs, d.config.Port, d.config.startupTimeout())
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), http.StatusInternalServerError)
			return
		}
		d.process = p
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "reloaded"})
}

func (d *Daemon) handleConfig(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(d.config)
}
