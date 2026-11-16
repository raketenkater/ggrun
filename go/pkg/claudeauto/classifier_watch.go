package claudeauto

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
	"sync/atomic"
)

// classifierWarnAfter is how many Bash tool calls may pass without a single
// recognised classifier request before the router says so.
const classifierWarnAfter = 3

// classifierWatch detects that Auto mode's reviews stopped being recognised
// (review I-04): ggrun finds them by a marker in Claude Code's private system
// prompt, and a reworded prompt would silently send every review to the main
// model's queue.
type classifierWatch struct {
	expect     atomic.Bool
	classifier atomic.Int64
	bash       atomic.Int64
	once       sync.Once
	out        io.Writer
}

// ExpectClassifier tells the router that the session it serves runs in Auto
// mode, so Bash calls without any classifier request are suspicious.
func (r *Router) ExpectClassifier(on bool) {
	if r != nil {
		r.watch.expect.Store(on)
	}
}

func (w *classifierWatch) sawClassifier() { w.classifier.Add(1) }

// sawMain counts the Bash calls the request's last assistant turn made.
func (w *classifierWatch) sawMain(body []byte) {
	if !w.expect.Load() || w.classifier.Load() > 0 {
		return
	}
	n := lastTurnToolUses(body, "Bash")
	if n == 0 || w.bash.Add(int64(n)) < classifierWarnAfter || w.classifier.Load() > 0 {
		return
	}
	w.once.Do(func() {
		out := w.out
		if out == nil {
			out = os.Stderr
		}
		fmt.Fprintf(out, "[claude-code] warning: Auto mode is on, but %d Bash calls passed without one recognised permission review. "+
			"Unless a rule or hook allowed them, Claude Code may have changed its review prompt and reviews are queueing on the main model; "+
			"please report it with your Claude Code version.\n", w.bash.Load())
	})
}

// lastTurnToolUses counts tool_use blocks named name in the request's last
// assistant message: the calls whose results this request carries.
func lastTurnToolUses(body []byte, name string) int {
	var req struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if json.Unmarshal(body, &req) != nil {
		return 0
	}
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role != "assistant" {
			continue
		}
		var blocks []struct {
			Type string `json:"type"`
			Name string `json:"name"`
		}
		if json.Unmarshal(req.Messages[i].Content, &blocks) != nil {
			return 0
		}
		n := 0
		for _, b := range blocks {
			if b.Type == "tool_use" && b.Name == name {
				n++
			}
		}
		return n
	}
	return 0
}
