package main

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/raketenkater/ggrun/pkg/claudesession"
	"github.com/raketenkater/ggrun/pkg/config"
)

// Claude Code's workflow resume cache is keyed by session ID on disk, so the
// session ID is the whole resume handle. ggrun mints it at launch with
// --session-id instead of discovering it afterwards by scanning transcripts,
// and stores it with the launch shape that produced it.

// claudeSessionSpec carries the resume decision from the launch path into the
// client invocation.
type claudeSessionSpec struct {
	ID       string
	Resume   bool
	Workflow *claudesession.Workflow
	Cached   int
	// RouterPort is this launch's loopback gateway, 0 when the client talks
	// to the backend directly.
	RouterPort int
}

// claudeProjectsDir is where Claude Code keeps per-project session state.
func claudeProjectsDir() string {
	if dir := os.Getenv("CLAUDE_CONFIG_DIR"); dir != "" {
		return filepath.Join(dir, "projects")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".claude", "projects")
}

// newClaudeSessionSpec mints a session ID for a fresh launch.
func newClaudeSessionSpec() (*claudeSessionSpec, error) {
	id, err := claudesession.NewSessionID()
	if err != nil {
		return nil, err
	}
	return &claudeSessionSpec{ID: id}, nil
}

// isLatestResumeValue reports whether a --claude-resume value means "the newest
// recoverable session" rather than a specific session ID. The claudeResumeSpec
// model guard needs the same distinction: an explicit ID names its backend, a
// "latest" pick does not.
func isLatestResumeValue(value string) bool {
	value = strings.TrimSpace(value)
	return value == "" || strings.EqualFold(value, "latest") || strings.EqualFold(value, "last")
}

// resolveClaudeResume loads a recorded session for --claude-resume. The value
// is a session ID, or "latest" for the newest session in this directory.
//
// A record is only worth resuming when it has something to reopen: a workflow
// journal, a transcript, or a recorded workflow pointer. A session that was
// launched live but never ran a turn records none of these, so "latest" skips it
// to the newest recoverable session and an explicit ID is refused with a
// message instead of reopening an empty conversation.
func resolveClaudeResume(cacheDir, workDir, value string) (claudesession.Record, error) {
	if isLatestResumeValue(value) {
		return claudesession.LatestRecoverable(cacheDir, workDir)
	}
	rec, err := claudesession.Load(cacheDir, value)
	if err != nil {
		return rec, err
	}
	if !rec.Recoverable() {
		return rec, fmt.Errorf("session %s has no transcript or workflow to resume", rec.SessionID)
	}
	return rec, nil
}

// claudeResumeSpec turns a recorded session into a launch spec, refusing only
// when the proposed launch cannot hold the recorded conversation.
//
// The recorded model is informational — what the session ran under THEN. A
// resume reopens the conversation against whatever model this launch starts
// now, so a recorded model that differs is announced rather than blocking, and
// the launch uses the current model even when the recorded one has since been
// removed. The "recorded model no longer present" error is reserved for an
// explicit resume by session ID, which names that backend and cannot be
// reopened by anything else; "latest"/empty picks have no such commitment.
//
// explicit is true when the user named a specific session ID (rather than
// "latest"), and currentModel is the path of the model this launch is starting,
// when it is known, so a difference from the recorded model can be reported.
func claudeResumeSpec(rec claudesession.Record, serverArgs []string, force bool, explicit bool, currentModel string) (*claudeSessionSpec, error) {
	// Resuming reopens the conversation against the current model. A record that
	// points at a different model is not an error — it just means the session's
	// earlier turns ran on another backend — but the user should know the
	// conversation is being continued on a different model.
	if rec.ModelPath != "" && currentModel != "" && rec.ModelPath != currentModel {
		fmt.Printf("[claude-code] resuming session %s recorded under %s using current model %s\n",
			rec.SessionID, filepath.Base(rec.ModelPath), filepath.Base(currentModel))
	}
	// Only an explicit resume commits to the recorded backend. If that model is
	// gone, the alternative is an opaque "missing shard" failure from deep inside
	// the loader; name the stale path here instead. A "latest" pick is a request
	// for the newest conversation, not for a specific model, so it launches the
	// current model and is not blocked by a recorded model that has disappeared.
	if explicit && !rec.ModelPathExists() {
		return nil, fmt.Errorf("recorded model no longer present: %s", rec.ModelPath)
	}
	if mismatches := rec.ShapeMismatches(serverArgs); len(mismatches) > 0 && !force {
		var lines []string
		for _, m := range mismatches {
			lines = append(lines, "  "+m.String())
		}
		return nil, fmt.Errorf(
			"session %s was recorded with a larger slot than this launch provides:\n%s\n"+
				"A conversation that fit before may not fit now, and it would fail mid-run "+
				"rather than here. Relaunch with at least the recorded context per slot, or "+
				"pass --claude-resume-force to truncate.",
			rec.SessionID, strings.Join(lines, "\n"))
	}
	spec := &claudeSessionSpec{ID: rec.SessionID, Resume: true, Workflow: rec.Workflow}
	if spec.Workflow != nil && spec.Workflow.RunID != "" {
		journal := claudesession.JournalPath(claudeProjectsDir(), rec.WorkDir, rec.SessionID, spec.Workflow.RunID)
		spec.Cached = claudesession.CachedAgents(journal)
		return spec, nil
	}
	// The run ID is assigned inside Claude Code, so discover the session's
	// newest recoverable run instead of requiring it to have been recorded.
	if wf, cached := claudesession.LatestRun(claudeProjectsDir(), rec.WorkDir, rec.SessionID); wf != nil {
		spec.Workflow, spec.Cached = wf, cached
	}
	return spec, nil
}

// recordClaudeSession stores the session and the exact shape it ran under.
// Recording is evidence, not a dependency: a failure must not stop a launch.
func recordClaudeSession(cacheDir string, spec *claudeSessionSpec, modelPath, backend string, port int, launchArgs, serverArgs []string) {
	if spec == nil || cacheDir == "" {
		return
	}
	workDir, err := os.Getwd()
	if err != nil {
		return
	}
	rec := claudesession.Record{
		SessionID:  spec.ID,
		WorkDir:    workDir,
		ModelPath:  modelPath,
		Backend:    backend,
		Port:       port,
		RouterPort: spec.RouterPort,
		LaunchArgs: launchArgs,
		ServerArgs: serverArgs,
		Workflow:   spec.Workflow,
	}
	if err := claudesession.Save(cacheDir, rec); err != nil {
		fmt.Fprintf(os.Stderr, "[claude-code] could not record session for resume: %v\n", err)
	}
}

// claudeLaunchSession resolves the session for this launch: either a recorded
// one being resumed, or a freshly minted ID that is recorded for next time.
// clientPort is the port Claude Code talks to; when it differs from the backend
// port it is ggrun's router, recorded so a later resume can attach through it.
func claudeLaunchSession(cfg *config.Config, req *launchRequest, serverArgs []string, clientPort int) (*claudeSessionSpec, error) {
	cacheDir := ""
	if cfg != nil {
		cacheDir = cfg.CacheDir
	}
	routerPort := 0
	if req != nil && clientPort > 0 && clientPort != req.Port {
		routerPort = clientPort
	}
	if req != nil && req.ClaudeResume != "" {
		workDir, err := os.Getwd()
		if err != nil {
			return nil, err
		}
		rec, err := resolveClaudeResume(cacheDir, workDir, req.ClaudeResume)
		if err != nil {
			return nil, err
		}
		// An explicit session ID owns its recorded backend (its recorded model
		// must still exist); "latest"/empty picks are requests for the newest
		// conversation and are reopened against this launch's current model.
		spec, err := claudeResumeSpec(rec, serverArgs, req.ClaudeResumeForce, !isLatestResumeValue(req.ClaudeResume), req.ModelPath)
		if err != nil {
			return nil, err
		}
		// The recorded shape stays as it was; only the live gateway changes.
		spec.RouterPort = routerPort
		if routerPort > 0 && cacheDir != "" {
			rec.RouterPort = routerPort
			if saveErr := claudesession.Save(cacheDir, rec); saveErr != nil {
				fmt.Fprintf(os.Stderr, "[claude-code] could not update session record: %v\n", saveErr)
			}
		}
		return spec, nil
	}
	spec, err := newClaudeSessionSpec()
	if err != nil {
		// A missing session ID only costs the ability to resume later, so it
		// must not block the launch itself.
		fmt.Fprintf(os.Stderr, "[claude-code] could not mint a session id: %v\n", err)
		return nil, nil
	}
	spec.RouterPort = routerPort
	var (
		modelPath  string
		backend    string
		port       int
		launchArgs []string
	)
	if req != nil {
		modelPath, backend, port = req.ModelPath, req.Backend, req.Port
		// The argv as given, so `ggrun claude resume` reproduces this launch
		// exactly instead of re-deriving flags that may since have changed.
		launchArgs = claudeStripResumeArgs(req.OriginalArgs)
	}
	recordClaudeSession(cacheDir, spec, modelPath, backend, port, launchArgs, serverArgs)
	return spec, nil
}

// claudeSessionArgs prepends the session flags to the client invocation.
//
// A resume must reuse the original session ID. --fork-session mints a new one,
// which moves the journal path and silently discards every cached agent, so it
// is refused rather than quietly honoured.
func claudeSessionArgs(spec *claudeSessionSpec, extraArgs, args []string) ([]string, error) {
	if spec == nil || spec.ID == "" {
		return args, nil
	}
	if hasArg(extraArgs, "--session-id") || hasArg(extraArgs, "--resume") || hasArg(extraArgs, "-r") {
		// The user pinned their own session; do not fight them for it.
		return args, nil
	}
	if !spec.Resume {
		return append([]string{"--session-id", spec.ID}, args...), nil
	}
	if hasArg(extraArgs, "--fork-session") {
		return nil, fmt.Errorf(
			"--fork-session cannot be combined with a workflow resume: forking mints a new " +
				"session ID, which moves the workflow journal path and discards every cached agent")
	}
	return append([]string{"--resume", spec.ID}, args...), nil
}

// portServesLLM reports whether an HTTP endpoint answers on the port. A plain
// TCP connect is not enough to call the port "serving": the recorded port may be
// held by an unrelated listener, and resuming against that would hand Claude a
// garbage base URL. The OpenAI-compatible surface is what the client actually
// talks to, so it is the honest probe.
func portServesLLM(port int) bool {
	if port <= 0 {
		return false
	}
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	conn, err := net.DialTimeout("tcp", addr, 300*time.Millisecond)
	if err != nil {
		return false
	}
	_ = conn.Close()
	client := &http.Client{Timeout: 1 * time.Second}
	for _, path := range []string{"/v1/models", "/health"} {
		resp, err := client.Get(fmt.Sprintf("http://%s%s", addr, path))
		if err == nil {
			resp.Body.Close()
			return true
		}
	}
	return false
}

// portServesClaudeRouter reports whether ggrun's loopback router answers on the
// port. Its status endpoint is the identity check: a reused ephemeral port held
// by anything else does not serve /ggrun/router.
func portServesClaudeRouter(port int) bool {
	if port <= 0 {
		return false
	}
	client := &http.Client{Timeout: 1 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/ggrun/router", port))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	var status map[string]any
	if resp.StatusCode != http.StatusOK || json.NewDecoder(resp.Body).Decode(&status) != nil {
		return false
	}
	_, ok := status["limit"]
	return ok
}

// claudeResumeClientPort picks where a live resume points Claude Code. The
// recorded router carries admission control and Auto reviewer routing; the raw
// backend has neither, so falling back to it is announced rather than silent.
func claudeResumeClientPort(rec claudesession.Record) int {
	if portServesClaudeRouter(rec.RouterPort) {
		fmt.Printf("[claude-code] Resuming through the running ggrun router on port %d.\n", rec.RouterPort)
		return rec.RouterPort
	}
	fmt.Fprintf(os.Stderr, "[claude-code] warning: no ggrun router is running for this server; Claude Code connects to the backend on port %d directly, "+
		"without agent admission control or Auto reviewer routing.\n", rec.Port)
	return rec.Port
}

// claudeResumeAgainstLive reopens a recorded session against a backend that is
// already serving on the recorded port, instead of relaunching the recorded
// launch shape only to die at guardPortFree. The server stays owned by whatever
// started it; this process runs the client and exits with its code.
func claudeResumeAgainstLive(cacheDir string, rec claudesession.Record) int {
	fmt.Printf("[claude-code] A server is already running on the recorded port %d; resuming session %s against it.\n",
		rec.Port, rec.SessionID)
	spec := &claudeSessionSpec{ID: rec.SessionID, Resume: true, Workflow: rec.Workflow}
	if spec.Workflow == nil {
		if wf, cached := claudesession.LatestRun(claudeProjectsDir(), rec.WorkDir, rec.SessionID); wf != nil {
			spec.Workflow, spec.Cached = wf, cached
		}
	}
	port := claudeResumeClientPort(rec)
	// The Workflow hook and status line ride on --settings, as on a normal launch.
	clientArgs, _ := claudeCodeProgressClientArgs(nil, port)
	code := runClaudeCodeClient("127.0.0.1", port, rec.ServerArgs, clientArgs, spec, 0)
	if code == -1 {
		// `claude` isn't installed; match the normal launch path and print the
		// copy-paste recipe pointed at the live server instead of exiting blank.
		printClaudeCodeRecipe("127.0.0.1", port, rec.ServerArgs, 0)
		return -1
	}
	// Record on exit as well as on launch: the workflow run ID is assigned inside
	// Claude Code, so only now is the resume handle complete. The live server is
	// not owned by this process, so it is not stopped here.
	refreshClaudeSessionRecord(cacheDir, spec, rec.ModelPath, rec.Backend, rec.Port,
		claudeStripResumeArgs(rec.LaunchArgs), rec.ServerArgs)
	return code
}

// claudeResumePrompt asks Claude Code to continue the recorded workflow from
// its journal. Cached agents replay without a model call; anything that was
// still in flight when the session stopped re-runs.
func claudeResumePrompt(spec *claudeSessionSpec) string {
	if spec == nil || spec.Workflow == nil || spec.Workflow.RunID == "" {
		return ""
	}
	wf := spec.Workflow
	var b strings.Builder
	fmt.Fprintf(&b, "Resume the interrupted workflow run %s.", wf.RunID)
	if wf.ScriptPath != "" {
		fmt.Fprintf(&b, " Call Workflow({scriptPath: %q, resumeFromRunId: %q}).", wf.ScriptPath, wf.RunID)
	} else {
		fmt.Fprintf(&b, " Call Workflow with resumeFromRunId: %q and the same script and args as before.", wf.RunID)
	}
	b.WriteString(" Do not change the script or args: agents whose prompt and options are unchanged replay from cache," +
		" and any edit re-runs everything after the first changed call.")
	return b.String()
}

// holdInterruptForClaude keeps Ctrl+C from killing ggrun while Claude Code owns
// the terminal.
//
// The client runs in the foreground, so Ctrl+C reaches the whole process group.
// Claude Code handles it itself -- the first interrupt cancels its current turn
// rather than ending the session -- but ggrun installs no handler until after
// the client returns, so the default action would terminate ggrun mid-session,
// orphan the backend and skip the session record that makes the run resumable.
//
// signal.Notify is used rather than signal.Ignore on purpose: an ignored
// disposition survives exec and would leave Claude Code unable to see Ctrl+C at
// all, whereas a caught signal is reset to default in the child.
func holdInterruptForClaude() func() {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	done := make(chan struct{})
	go func() {
		for {
			select {
			case <-sigCh:
				// Absorbed: the interrupt belongs to Claude Code.
			case <-done:
				return
			}
		}
	}()
	// Idempotent: the caller defers this and may also release it explicitly on
	// an early return, and a second close would panic.
	var once sync.Once
	return func() {
		once.Do(func() {
			signal.Stop(sigCh)
			close(done)
		})
	}
}

// refreshClaudeSessionRecord re-records the session once Claude Code exits, so
// a workflow started during the session is part of the resume handle. The run
// ID is assigned inside Claude Code and cannot be known at launch.
func refreshClaudeSessionRecord(cacheDir string, spec *claudeSessionSpec, modelPath, backend string, port int, launchArgs, serverArgs []string) {
	if spec == nil || spec.ID == "" || cacheDir == "" {
		return
	}
	workDir, err := os.Getwd()
	if err != nil {
		return
	}
	if wf, cached := claudesession.LatestRun(claudeProjectsDir(), workDir, spec.ID); wf != nil {
		spec.Workflow, spec.Cached = wf, cached
		fmt.Printf("[claude-code] Session %s recorded: workflow %s has %d completed agents cached. "+
			"Resume with: ggrun claude resume\n", spec.ID, wf.RunID, cached)
	}
	// Recording on exit also repairs a launch-time record that could not be
	// written, so a resumable session is not lost to a transient error.
	if spec.Resume {
		// A resumed session keeps its original recorded shape; only the
		// workflow pointer is refreshed above.
		if rec, err := claudesession.Load(cacheDir, spec.ID); err == nil {
			rec.Workflow = spec.Workflow
			if saveErr := claudesession.Save(cacheDir, rec); saveErr != nil {
				fmt.Fprintf(os.Stderr, "[claude-code] could not update session record: %v\n", saveErr)
			}
			return
		}
	}
	recordClaudeSession(cacheDir, spec, modelPath, backend, port, launchArgs, serverArgs)
}

// claudeStripResumeArgs removes resume flags from a recorded launch so
// replaying it cannot chain a resume of a resume.
func claudeStripResumeArgs(args []string) []string {
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		switch {
		case args[i] == "--claude-resume":
			i++ // also drop its value
		case strings.HasPrefix(args[i], "--claude-resume="):
		case args[i] == "--claude-resume-force":
		default:
			out = append(out, args[i])
		}
	}
	return out
}

// cmdClaude implements `ggrun claude list|resume`.
func cmdClaude(args []string) {
	sub := "resume"
	if len(args) > 0 {
		sub = args[0]
		args = args[1:]
	}
	switch sub {
	case "help", "--help", "-h":
		fmt.Fprintln(os.Stderr, "Usage: ggrun claude [list | resume [session-id|latest] [--force] [flag overrides...]]")
		fmt.Fprintln(os.Stderr, "  e.g. ggrun claude resume latest --spec dflash")
	case "list", "ls", "sessions":
		cmdClaudeList()
	case "resume", "continue":
		target, force, overrides := parseClaudeResumeArgs(args)
		cmdClaudeResume(target, force, overrides)
	default:
		fmt.Fprintln(os.Stderr, "Usage: ggrun claude [list | resume [session-id|latest] [--force] [flag overrides...]]")
		fmt.Fprintln(os.Stderr, "  e.g. ggrun claude resume latest --spec dflash")
		os.Exit(2)
	}
}

// parseClaudeResumeArgs splits `ggrun claude resume` arguments into the target
// session and the shape-override flag. The override matters because placement
// legitimately recomputes between launches -- a reboot alone can move an expert
// layer -- so a refused resume must have a recovery path that does not require
// reconstructing the long-form launch command by hand.
// Anything else is an override applied on top of the recorded launch. A session
// records the flags it was started with, and some of them are decisions rather
// than facts: the long-running Laguna workflow here was recorded with --spec off
// because speculative decoding could not load its drafter yet. Once the fork was
// fixed there was no way to resume that session with speculation on, since the
// only alternative was reconstructing a launch line of thirty-odd flags by hand
// and losing the session binding that makes resume worth using.
func parseClaudeResumeArgs(args []string) (target string, force bool, overrides []string) {
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch arg {
		case "--force", "-f", "--claude-resume-force":
			force = true
			continue
		}
		if strings.HasPrefix(arg, "-") {
			overrides = append(overrides, arg)
			// ggrun spells valued flags "--flag value", so a following token
			// that is not itself a flag belongs to this one.
			if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
				overrides = append(overrides, args[i+1])
				i++
			}
			continue
		}
		if target == "" {
			target = arg
		}
	}
	return target, force, overrides
}

// claudeApplyResumeOverrides layers override flags onto a recorded launch,
// replacing a flag's value where it was already set and appending it otherwise.
// Replacing rather than appending matters: ggrun takes the first occurrence of
// several flags, so an appended --spec would be read as a duplicate and the
// recorded "off" would still win.
func claudeApplyResumeOverrides(recorded, overrides []string) []string {
	if len(overrides) == 0 {
		return recorded
	}
	out := append([]string(nil), recorded...)
	for i := 0; i < len(overrides); i++ {
		flag := overrides[i]
		value := ""
		hasValue := false
		if i+1 < len(overrides) && !strings.HasPrefix(overrides[i+1], "-") {
			value, hasValue = overrides[i+1], true
			i++
		}
		at := -1
		for j, tok := range out {
			if tok == flag {
				at = j
				break
			}
		}
		switch {
		case at < 0 && hasValue:
			out = append(out, flag, value)
		case at < 0:
			out = append(out, flag)
		case !hasValue:
			// A bare flag already present needs nothing.
		case at+1 < len(out) && !strings.HasPrefix(out[at+1], "-"):
			out[at+1] = value
		default:
			// Recorded as a bare flag but overridden with a value.
			rest := append([]string{value}, out[at+1:]...)
			out = append(out[:at+1], rest...)
		}
	}
	return out
}

func cmdClaudeList() {
	cfg := loadConfigOrExit()
	workDir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	records, err := claudesession.List(cfg.CacheDir, workDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if len(records) == 0 {
		fmt.Printf("No recorded Claude Code sessions for %s\n", workDir)
		return
	}
	fmt.Printf("Recorded Claude Code sessions for %s:\n\n", workDir)
	for _, rec := range records {
		wf, cached := claudesession.LatestRun(claudeProjectsDir(), rec.WorkDir, rec.SessionID)
		fmt.Printf("  %s  %s  %s\n", rec.SessionID,
			rec.Recorded.Local().Format("2006-01-02 15:04"), filepath.Base(rec.ModelPath))
		if wf != nil {
			fmt.Printf("      workflow %s (%s): %d completed agents cached\n", wf.RunID, wf.Name, cached)
		}
	}
	fmt.Println("\nResume the newest with: ggrun claude resume")
}

// cmdClaudeResume relaunches the recorded backend shape and reopens the
// session. It replays the recorded launch argv rather than re-deriving flags,
// because a placement or KV default that changed since would silently
// reinterpret the cached conversation.
func cmdClaudeResume(target string, force bool, overrides []string) {
	cfg := loadConfigOrExit()
	workDir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	rec, err := resolveClaudeResume(cfg.CacheDir, workDir, target)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		fmt.Fprintln(os.Stderr, "List recorded sessions with: ggrun claude list")
		os.Exit(1)
	}
	if wf, cached := claudesession.LatestRun(claudeProjectsDir(), rec.WorkDir, rec.SessionID); wf != nil {
		fmt.Printf("[claude-code] Session %s, workflow %s (%s): %d completed agents will replay from cache.\n",
			rec.SessionID, wf.RunID, wf.Name, cached)
	}
	// The recorded launch replays the model the session was recorded under. If
	// that model is gone, the launch would die deep inside the loader with an
	// opaque "missing shard" error. Name the stale path up front instead.
	if !rec.ModelPathExists() {
		fmt.Fprintf(os.Stderr, "Error: recorded model no longer present: %s\n", rec.ModelPath)
		fmt.Fprintln(os.Stderr, "The session record points at a model that has since been removed or renamed.")
		fmt.Fprintln(os.Stderr, "List recorded sessions with: ggrun claude list")
		os.Exit(1)
	}
	launchArgs := claudeStripResumeArgs(rec.LaunchArgs)
	if len(launchArgs) == 0 {
		if rec.ModelPath == "" {
			fmt.Fprintf(os.Stderr, "Error: session %s has no recorded launch to reproduce\n", rec.SessionID)
			os.Exit(1)
		}
		launchArgs = []string{rec.ModelPath, "--claude-code"}
	}
	if len(overrides) > 0 {
		launchArgs = claudeApplyResumeOverrides(launchArgs, overrides)
		fmt.Printf("[claude-code] Overriding recorded launch: %s\n", strings.Join(overrides, " "))
	}
	launchArgs = append(launchArgs, "--claude-resume", rec.SessionID)
	if force {
		launchArgs = append(launchArgs, "--claude-resume-force")
	}
	// The recorded launch will die at guardPortFree if its port is already
	// serving. The session's own backend may have survived, or a fresh launch of
	// the same model may be running: either way the recorded conversation can be
	// reopened against the live server without paying for a second load.
	if rec.Port > 0 && portServesLLM(rec.Port) {
		code := claudeResumeAgainstLive(cfg.CacheDir, rec)
		os.Exit(code)
	}
	cmdLaunch(launchArgs)
}

// describeClaudeResume reports what a resume will actually recover, so the cost
// is visible before the backend is loaded.
func describeClaudeResume(spec *claudeSessionSpec) string {
	if spec == nil || !spec.Resume {
		return ""
	}
	if spec.Workflow == nil || spec.Workflow.RunID == "" {
		return fmt.Sprintf("[claude-code] Resuming session %s (no recorded workflow run).", spec.ID)
	}
	return fmt.Sprintf(
		"[claude-code] Resuming session %s, workflow %s: %d completed agents replay from cache; "+
			"agents still running when it stopped re-run.",
		spec.ID, spec.Workflow.RunID, spec.Cached)
}
