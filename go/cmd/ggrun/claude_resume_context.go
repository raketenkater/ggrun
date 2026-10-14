package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/raketenkater/ggrun/pkg/claudesession"
	"github.com/raketenkater/ggrun/pkg/placement"
)

// resumeContextHeadroom is room for the next turn on top of what the resumed
// conversation already holds: the new prompt, tool results and the reply.
const resumeContextHeadroom = 32768

// resumeContextGranularity rounds the requested context so a session that grew
// by a few tokens does not change the launch identity.
const resumeContextGranularity = 4096

// resumeContextNeed is the context per slot a resumed conversation of the given
// size needs to take its next turn.
func resumeContextNeed(tokens int) int {
	if tokens <= 0 {
		return 0
	}
	need := tokens + resumeContextHeadroom
	return (need + resumeContextGranularity - 1) / resumeContextGranularity * resumeContextGranularity
}

// ctxFlagIsAutomatic reports whether the user left context to the planner.
func ctxFlagIsAutomatic(flag string) bool {
	switch strings.ToLower(strings.TrimSpace(flag)) {
	case "", "fit", "auto":
		return true
	}
	return false
}

// resumeSessionTokens measures the conversation a --claude-resume launch will
// reopen. ok is false when the session or its transcript cannot be read.
var resumeSessionTokens = func(cacheDir, value string) (string, int, bool) {
	workDir, err := os.Getwd()
	if err != nil {
		return "", 0, false
	}
	rec, err := resolveClaudeResume(cacheDir, workDir, value)
	if err != nil {
		return "", 0, false
	}
	tokens, ok := claudesession.CurrentContextTokens(claudeProjectsDir(), rec.WorkDir, rec.SessionID)
	return rec.SessionID, tokens, ok
}

// ensureResumeContext makes an automatic-context launch large enough for the
// session it resumes. The planner sizes context from free memory, not from the
// conversation, so a launch that learned a new reserve could come up below a
// session's size: a 155,588-token session was rejected by a 144,384-token
// slot on every turn (2026-10-14). The conversation size is measured from the
// transcript; when the planned slot is short, the plan is recomputed with that
// context as an explicit request, which moves weights rather than shrinking
// the session. An explicit --ctx is the user's and is only warned about.
func ensureResumeContext(req *launchRequest, cacheDir string, model *placement.ModelProfile, strategy *placement.Strategy,
	compute func(*launchRequest) (*placement.Strategy, error),
) *placement.Strategy {
	if req == nil || strategy == nil || !req.ClaudeCode || req.ClaudeResume == "" || req.ClaudeResumeForce || compute == nil {
		return strategy
	}
	session, tokens, ok := resumeSessionTokens(cacheDir, req.ClaudeResume)
	if !ok || tokens <= 0 {
		return strategy
	}
	need := resumeContextNeed(tokens)
	slots := max(strategy.Parallel, 1)
	if strategy.ContextSize/slots >= need {
		return strategy
	}
	if model != nil && model.CTXTrain > 0 && need > model.CTXTrain {
		fmt.Fprintf(os.Stderr, "[claude-code] session %s is %d tokens; with room for a reply it needs %d, more than this model's %d. Compact it in Claude Code (/compact) first.\n",
			session, tokens, need, model.CTXTrain)
		return strategy
	}
	if !ctxFlagIsAutomatic(req.CtxFlag) {
		fmt.Fprintf(os.Stderr, "[claude-code] session %s is %d tokens and needs about %d per slot; --ctx %s gives %d. Raise --ctx to resume it.\n",
			session, tokens, need, req.CtxFlag, strategy.ContextSize/slots)
		return strategy
	}
	fmt.Printf("[claude-code] session %s is %d tokens; the plan gives %d per slot; planning %d per slot so it can resume\n",
		session, tokens, strategy.ContextSize/slots, need)
	candidate := *req
	candidate.CtxFlag = strconv.Itoa(need * slots)
	planned, err := compute(&candidate)
	if err != nil && slots > 1 && !req.ParallelSet {
		// One conversation is being resumed; one slot of the needed size is the
		// smallest plan that holds it.
		candidate.CtxFlag = strconv.Itoa(need)
		candidate.Parallel, candidate.ParallelSet = 1, true
		planned, err = compute(&candidate)
	}
	if err != nil || planned == nil {
		fmt.Fprintf(os.Stderr, "[claude-code] could not plan %d tokens per slot (%v); launching with %d, which this session does not fit\n",
			need, err, strategy.ContextSize/slots)
		return strategy
	}
	*req = candidate
	return planned
}
