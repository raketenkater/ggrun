package tui

import (
	"os"
	"path/filepath"
	"strings"
)

// whatsNewID names the current announcement. A new feature gets a new ID;
// each ID is shown once per user, then recorded in the cache directory.
const whatsNewID = "ggrun-client-1"

const whatsNewText = "New: use this model from Claude Code on another computer — [x] Claude Code → Remote client, then ggrun-client (github.com/raketenkater/ggrun-client)"

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

// withWhatsNew inserts the one-line announcement at offset when the screen has
// a spare row for it. A screen taller than the terminal loses its top lines,
// so the banner must never push the title out of view.
func withWhatsNew(view string, at int, banner string, width, height int) string {
	if banner == "" || at < 0 || at > len(view) {
		return view
	}
	if width > 4 && len([]rune(banner))+2 > width {
		r := []rune(banner)
		banner = string(r[:width-3]) + "…"
	}
	if height > 0 && strings.Count(view, "\n")+2 > height {
		return view
	}
	return view[:at] + "  " + recommendStyle.Render(banner) + "\n" + view[at:]
}

// whatsNewRows is the screen row the banner takes from the model list.
func (m Model) whatsNewRows() int {
	if m.whatsNew == "" {
		return 0
	}
	return 1
}
