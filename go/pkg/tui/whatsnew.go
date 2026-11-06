package tui

import (
	"os"
	"path/filepath"
	"strings"
)

// whatsNewID names the current announcement. A new feature gets a new ID;
// each ID is shown once per user, then recorded in the cache directory.
const whatsNewID = "ggrun-client-1"

const whatsNewText = "New: ggrun-client — use this machine's model from Claude Code on another computer.\n" +
	"  Set [x] Claude Code to \"Remote client\" and launch; on the other computer install\n" +
	"  github.com/raketenkater/ggrun-client and run ggrun-client.  (GGRUN_NO_WHATS_NEW=1 hides this)"

// takeWhatsNew returns the announcement once and records it as seen, so it
// is not repeated on later starts. Recording happens on read: a session that
// opened the TUI has had its chance to see it.
func takeWhatsNew(cacheDir string) string {
	if v := strings.TrimSpace(os.Getenv("GGRUN_NO_WHATS_NEW")); v != "" && v != "0" {
		return ""
	}
	if cacheDir == "" {
		return ""
	}
	path := filepath.Join(cacheDir, "whats-new-seen")
	if b, err := os.ReadFile(path); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			if strings.TrimSpace(line) == whatsNewID {
				return ""
			}
		}
	}
	if os.MkdirAll(cacheDir, 0o755) != nil {
		return ""
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return "" // Never show a banner that would repeat forever.
	}
	defer f.Close()
	if _, err = f.WriteString(whatsNewID + "\n"); err != nil {
		return ""
	}
	return whatsNewText
}
