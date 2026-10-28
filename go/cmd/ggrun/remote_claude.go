package main

import (
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/raketenkater/ggrun/clientkit"
	"github.com/raketenkater/ggrun/pkg/config"
	"github.com/raketenkater/ggrun/pkg/placement"
	"github.com/raketenkater/ggrun/pkg/remoteclaude"
)

type remoteClaudeOptions struct {
	Configured bool
	Headless   bool
	Host       string
	Port       int
	URL        string
	TokenFile  string
}

// Remote-only flags are removed before ordinary launch parsing. Backend flags
// after -- are never consumed, and the original argv stays available for logs.
func extractRemoteClaudeArgs(args []string) ([]string, remoteClaudeOptions, error) {
	o := remoteClaudeOptions{Host: "127.0.0.1", Port: 8080}
	var out []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--" {
			out = append(out, args[i:]...)
			break
		}
		k, v, eq := strings.Cut(a, "=")
		if k == "--headless" {
			o.Headless = true
			if eq {
				var err error
				o.Headless, err = strconv.ParseBool(v)
				if err != nil {
					return nil, o, fmt.Errorf("--headless requires true or false")
				}
			}
			continue
		}
		switch k {
		case "--gateway-host", "--gateway-port", "--gateway-url", "--gateway-token-file":
			o.Configured = true
			if !eq {
				if i+1 >= len(args) || strings.HasPrefix(args[i+1], "--") {
					return nil, o, fmt.Errorf("%s requires a value", k)
				}
				i++
				v = args[i]
			}
			switch k {
			case "--gateway-host":
				o.Host = v
			case "--gateway-url":
				o.URL = v
			case "--gateway-token-file":
				o.TokenFile = v
			case "--gateway-port":
				n, e := strconv.Atoi(v)
				if e != nil || n < 1 || n > 65535 {
					return nil, o, fmt.Errorf("invalid gateway port")
				}
				o.Port = n
			}
		default:
			out = append(out, a)
		}
	}
	return out, o, nil
}

func validateRemoteClaude(req *launchRequest) error {
	o := req.RemoteClaude
	if !o.Headless {
		if o.Configured {
			return fmt.Errorf("gateway flags require --claude-code --headless")
		}
		return nil
	}
	if !req.ClaudeCode {
		return fmt.Errorf("--headless requires --claude-code")
	}
	if req.ClaudeResume != "" || req.ClaudeResumeForce {
		return fmt.Errorf("headless mode serves laptop sessions; resume with ggrun-client claude --resume on the laptop")
	}
	if req.Benchmark || req.WorkerBenchmark {
		return fmt.Errorf("headless serving cannot be combined with a one-shot benchmark")
	}
	if o.Port == req.Port {
		return fmt.Errorf("gateway port must differ from the private model port")
	}
	if strings.TrimSpace(o.Host) == "" {
		return fmt.Errorf("gateway host must name a listener address")
	}
	if o.URL != "" {
		if _, err := clientkit.BaseURL(o.URL, true); err != nil {
			return err
		}
	}
	for _, arg := range req.ExtraArgs {
		if arg == "--host" || strings.HasPrefix(arg, "--host=") {
			return fmt.Errorf("headless mode keeps the backend private; set --gateway-host instead of a backend --host passthrough")
		}
	}
	// Headless owns a separate network gateway. A model listener inherited from
	// older wildcard config must not expose the unauthenticated backend.
	req.Host = "127.0.0.1"
	return nil
}

type remoteClaudeService struct {
	listener   net.Listener
	server     *http.Server
	handler    atomic.Value
	options    remoteClaudeOptions
	key        string
	stopStatus func()
}
type handlerBox struct{ handler http.Handler }

func (s *remoteClaudeService) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if h := s.handler.Load(); h != nil {
		h.(handlerBox).handler.ServeHTTP(w, r)
		return
	}
	w.Header().Set("Retry-After", "5")
	http.Error(w, "ggrun model is loading or recovering", http.StatusServiceUnavailable)
}
func (s *remoteClaudeService) close() {
	if s == nil {
		return
	}
	s.stopServingStatus()
	if s.server != nil {
		_ = s.server.Close()
	}
	if s.listener != nil {
		_ = s.listener.Close()
	}
}
func (s *remoteClaudeService) recovering() {
	s.unavailable("ggrun model is recovering")
}

func (s *remoteClaudeService) stopping() {
	s.unavailable("ggrun server is stopping")
}

func (s *remoteClaudeService) stopServingStatus() {
	if s != nil && s.stopStatus != nil {
		s.stopStatus()
		s.stopStatus = nil
	}
}

func (s *remoteClaudeService) unavailable(message string) {
	if s != nil {
		s.stopServingStatus()
		s.handler.Store(handlerBox{http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Retry-After", "5")
			http.Error(w, message, http.StatusServiceUnavailable)
		})})
	}
}

func prepareRemoteClaude(req *launchRequest, cfg *config.Config) (*remoteClaudeService, error) {
	if !req.RemoteClaude.Headless {
		return nil, nil
	}
	o := req.RemoteClaude
	if o.TokenFile == "" {
		o.TokenFile = filepath.Join(cfg.CacheDir, "remote-claude-token")
	}
	key, err := clientkit.Credential(o.TokenFile, true)
	if err != nil {
		return nil, fmt.Errorf("gateway credential: %w", err)
	}
	ln, err := net.Listen("tcp", net.JoinHostPort(o.Host, strconv.Itoa(o.Port)))
	if err != nil {
		return nil, fmt.Errorf("reserve gateway before model load: %w", err)
	}
	s := &remoteClaudeService{listener: ln, options: o, key: key}
	// Reserve the stable port before expensive loads. It answers loading until
	// the exact gateway canary has verified the private serving path.
	s.server = &http.Server{Handler: remoteclaude.Authenticate(key, s), ReadHeaderTimeout: 10 * time.Second}
	go func() { _ = s.server.Serve(ln) }()
	return s, nil
}

func remoteClaudeProfile(req *launchRequest, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, r *claudeAutoRuntime, args []string) clientkit.Profile {
	slots := max(1, strategy.Parallel)
	modified := int64(0)
	if st, err := os.Stat(req.ModelPath); err == nil {
		modified = st.ModTime().UnixNano()
	}
	id := sha256.Sum256([]byte(fmt.Sprintf("%s|%d|%d|%s", req.ModelPath, model.SizeBytes, modified, scopedProbeBackendTagForStrategy(req, model, be, strategy))))
	p := claudeClientProfile(args, strategy.ContextSize)
	p.Slots = slots
	p.ServerVersion = version
	p.ModelID = hex.EncodeToString(id[:])
	p.ModelName = model.Name
	p.Vision = hasArg(args, "--mmproj")
	p.SeparateWorker = r != nil && r.workerRouteEnabled()
	return p
}

func (s *remoteClaudeService) activate(req *launchRequest, model *placement.ModelProfile, strategy *placement.Strategy, be *backendInfo, r *claudeAutoRuntime, args []string) error {
	if s == nil {
		return nil
	}
	if r == nil || r.router == nil {
		return fmt.Errorf("remote Claude mode requires the verified agent gateway")
	}
	h, err := remoteclaude.NewHandler(r.router.URL(), s.key, remoteClaudeProfile(req, model, strategy, be, r, args))
	if err != nil {
		return err
	}
	s.handler.Store(handlerBox{h})
	u := s.options.URL
	if u == "" {
		if addr, ok := s.listener.Addr().(*net.TCPAddr); ok && addr.IP.IsUnspecified() {
			u = "https://YOUR-SERVER"
		} else {
			u = "http://" + s.listener.Addr().String()
		}
	}
	connectionArgs := []string{"ggrun-client", "connect", u}
	if _, err := clientkit.BaseURL(u, false); err != nil {
		connectionArgs = append(connectionArgs, "--allow-http")
	}
	fmt.Printf("[remote] Claude gateway listening on %s\n", s.listener.Addr())
	fmt.Printf("[remote] Laptop client: https://github.com/raketenkater/ggrun-client\n[remote] Connect: %s\n", displayCommand(connectionArgs))
	if s.options.URL == "" {
		fmt.Println("[remote] Use the reachable server/proxy URL on the laptop; loopback addresses are local to this machine.")
	}
	fmt.Printf("[remote] Credential is private: use %s to display it explicitly.\n", displayCommand([]string{"ggrun", "gateway-token", "--file", s.options.TokenFile}))
	fmt.Println("[remote] Laptop disconnects leave the model loaded. Ctrl+C here stops serving.")
	s.stopServingStatus()
	s.stopStatus = startRemoteClaudeStatus(os.Stderr, r.router.AdmissionStatus)
	return nil
}

func cmdGatewayToken(args []string) {
	fs := flag.NewFlagSet("gateway-token", flag.ContinueOnError)
	file := fs.String("file", "", "private gateway credential file")
	if err := fs.Parse(args); err != nil {
		if err != flag.ErrHelp {
			os.Exit(2)
		}
		return
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "Usage: ggrun gateway-token [--file PATH]")
		os.Exit(2)
	}
	if *file == "" {
		*file = filepath.Join(loadConfigOrExit().CacheDir, "remote-claude-token")
	}
	key, err := clientkit.Credential(*file, false)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Gateway credential:", err)
		os.Exit(1)
	}
	fmt.Println(key)
}

func printDryRunRemoteClaudeNotice(req *launchRequest) {
	fmt.Printf("[remote] A real launch serves an authenticated Claude gateway on %s without starting Claude Code on this machine.\n", net.JoinHostPort(req.RemoteClaude.Host, strconv.Itoa(req.RemoteClaude.Port)))
	if req.ClaudeReviewerDisabled {
		fmt.Println("[remote] Reviews use the main model; no separate reviewer is requested.")
	} else {
		fmt.Println("[remote] The normal server-side reviewer/worker policy remains enabled.")
	}
}
