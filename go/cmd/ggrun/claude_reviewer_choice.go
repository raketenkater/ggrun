package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/detect"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// The self-classify answer is a user policy, not placement evidence: when the
// user agrees at launch that the main model classifies its own Auto checks,
// a relaunch of the same scope must not seat the reviewer first and re-plan
// around it. The record lives beside, not inside, the verified-config cache so
// replacing or retiring a verified config never forgets the answer.
const (
	claudeReviewerChoiceSchemaVersion = 1
	claudeReviewerChoiceSelfClassify  = "self-classify"
	claudeReviewerChoiceDir           = "claude-reviewer-choice"
)

type claudeReviewerChoiceRecord struct {
	SchemaVersion int    `json:"schema_version"`
	Scope         string `json:"scope"`
	Choice        string `json:"choice"`
	AnsweredAt    string `json:"answered_at,omitempty"`
}

// claudeReviewerChoiceScope is the launch scope the answer applies to: the
// exact model artifact, the hardware, and the GPU set and CPU mode the user
// launched with. A different model, card or GPU restriction asks again.
func claudeReviewerChoiceScope(req *launchRequest, model *placement.ModelProfile, caps *detect.Capabilities) string {
	if req == nil || model == nil || caps == nil {
		return ""
	}
	gpus, _ := parseGPUIndices(req.GPUsFlag)
	sum := sha256.Sum256([]byte(strings.Join([]string{
		fmt.Sprintf("v%d", claudeReviewerChoiceSchemaVersion),
		placement.SpecTargetIdentity(model),
		placement.SpecHardwareIdentity(caps),
		fmt.Sprint(gpus),
		fmt.Sprintf("cpu=%t", req.CPUMode),
	}, "\x00")))
	return hex.EncodeToString(sum[:])
}

func claudeReviewerChoicePath(cacheDir, scope string) string {
	if cacheDir == "" || len(scope) < 16 {
		return ""
	}
	return filepath.Join(cacheDir, claudeReviewerChoiceDir, scope[:16]+".json")
}

// claudeReviewerChoiceExplicit reports an explicit reviewer choice for this
// launch: any --claude-reviewer value or a set GGRUN_CLAUDE_AUTO_REVIEWER.
// Either overrides the remembered answer.
func claudeReviewerChoiceExplicit(req *launchRequest) bool {
	if req != nil && req.ClaudeReviewerOverride != "" {
		return true
	}
	return strings.TrimSpace(os.Getenv("GGRUN_CLAUDE_AUTO_REVIEWER")) != ""
}

// loadClaudeReviewerChoice returns the remembered choice for scope, "" when
// none is saved. A corrupt, foreign-schema or foreign-scope record is an error
// and the caller falls back to the question (or its non-interactive default).
func loadClaudeReviewerChoice(cacheDir, scope string) (string, error) {
	path := claudeReviewerChoicePath(cacheDir, scope)
	if path == "" {
		return "", nil
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	var rec claudeReviewerChoiceRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return "", fmt.Errorf("unreadable reviewer choice %s: %w", path, err)
	}
	if rec.SchemaVersion != claudeReviewerChoiceSchemaVersion || rec.Scope != scope {
		return "", fmt.Errorf("reviewer choice %s belongs to another scope or schema", path)
	}
	if rec.Choice != claudeReviewerChoiceSelfClassify {
		return "", fmt.Errorf("reviewer choice %s has unknown choice %q", path, rec.Choice)
	}
	return rec.Choice, nil
}

func saveClaudeReviewerChoice(cacheDir, scope, choice string) error {
	path := claudeReviewerChoicePath(cacheDir, scope)
	if path == "" {
		return fmt.Errorf("no cache directory or launch scope")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(claudeReviewerChoiceRecord{
		SchemaVersion: claudeReviewerChoiceSchemaVersion,
		Scope:         scope,
		Choice:        choice,
		AnsweredAt:    time.Now().UTC().Format(time.RFC3339),
	}, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".choice-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(append(data, '\n')); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// forgetClaudeSelfClassifyForExplicitReviewer drops the remembered answer when
// a real launch names a reviewer with --claude-reviewer: the newest explicit
// choice wins for later launches too. --claude-reviewer off needs no record,
// and the env only overrides the record for the launch it is set on.
func forgetClaudeSelfClassifyForExplicitReviewer(req *launchRequest, cacheDir string, model *placement.ModelProfile, caps *detect.Capabilities, output io.Writer) {
	if req == nil || !req.ClaudeCode || req.ClaudeReviewerOverride == "" || req.ClaudeReviewerOverride == claudeReviewerOff {
		return
	}
	path := claudeReviewerChoicePath(cacheDir, claudeReviewerChoiceScope(req, model, caps))
	if path == "" {
		return
	}
	if err := os.Remove(path); err == nil {
		fmt.Fprintf(output, "[claude-code] --claude-reviewer %s replaces the saved self-classify choice for this launch scope\n", req.ClaudeReviewerOverride)
	}
}

// applyRememberedClaudeSelfClassify disables the separate reviewer before the
// first plan when the user already chose self-classify for this scope. It runs
// before the reviewer reservation is built, so the plan, the verified-config
// lookup and the launch all see the reviewer-free request. An explicit
// --claude-reviewer value or GGRUN_CLAUDE_AUTO_REVIEWER overrides it. It never
// asks and never writes, so dry-run and non-interactive launches can use it.
func applyRememberedClaudeSelfClassify(req *launchRequest, cacheDir string, model *placement.ModelProfile, caps *detect.Capabilities, output io.Writer) bool {
	if req == nil || !req.ClaudeCode || req.ClaudeReviewerDisabled || req.CPUMode || claudeReviewerChoiceExplicit(req) {
		return false
	}
	choice, err := loadClaudeReviewerChoice(cacheDir, claudeReviewerChoiceScope(req, model, caps))
	if err != nil {
		fmt.Fprintf(output, "[claude-code] ignoring the saved reviewer choice: %v\n", err)
		return false
	}
	if choice != claudeReviewerChoiceSelfClassify {
		return false
	}
	req.ClaudeReviewerDisabled = true
	req.claudeSelfClassifyRemembered = true
	fmt.Fprintln(output, "[claude-code] using your saved choice for this launch scope: no separate Auto reviewer, the main model classifies its own checks (--claude-reviewer auto seats the reviewer again)")
	return true
}

// rememberClaudeSelfClassify saves an interactive self-classify approval for
// this scope. A save failure only warns: the launch already has its answer.
func rememberClaudeSelfClassify(req *launchRequest, cacheDir string, model *placement.ModelProfile, caps *detect.Capabilities, output io.Writer) {
	if req == nil || !req.claudeSelfClassifyApproved || req.claudeSelfClassifyRemembered || !req.ClaudeReviewerDisabled {
		return
	}
	scope := claudeReviewerChoiceScope(req, model, caps)
	if err := saveClaudeReviewerChoice(cacheDir, scope, claudeReviewerChoiceSelfClassify); err != nil {
		fmt.Fprintf(output, "[claude-code] warning: self-classify approved for this launch, but the choice could not be saved: %v\n", err)
		return
	}
	req.claudeSelfClassifyRemembered = true
	fmt.Fprintln(output, "[claude-code] self-classify choice saved for this launch scope; relaunches will not seat a separate reviewer (--claude-reviewer auto seats it again)")
}
