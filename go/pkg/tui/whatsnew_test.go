package tui

import (
	"strings"
	"testing"
)

func TestWhatsNewShowsOncePerAnnouncement(t *testing.T) {
	t.Setenv("GGRUN_NO_WHATS_NEW", "")
	dir := t.TempDir()
	first := takeWhatsNew(dir)
	if !strings.Contains(first, "ggrun-client") || !strings.Contains(first, "Remote client") || strings.Contains(first, "\n") {
		t.Fatalf("announcement missing: %q", first)
	}
	if again := takeWhatsNew(dir); again != "" {
		t.Fatalf("announcement repeated: %q", again)
	}
}

func TestWhatsNewOptOutAndNoCache(t *testing.T) {
	t.Setenv("GGRUN_NO_WHATS_NEW", "1")
	if s := takeWhatsNew(t.TempDir()); s != "" {
		t.Fatalf("opt-out ignored: %q", s)
	}
	t.Setenv("GGRUN_NO_WHATS_NEW", "")
	if s := takeWhatsNew(""); s != "" {
		t.Fatal("shown without a place to record it, so it would repeat forever")
	}
}

func TestMainViewShowsAnnouncement(t *testing.T) {
	m := Model{width: 120, height: 40, whatsNew: whatsNewText}
	if !strings.Contains(m.viewMain(), "New: use this model") {
		t.Fatal("main view does not show the announcement")
	}
	m.whatsNew = ""
	if strings.Contains(m.viewMain(), "New: use this model") {
		t.Fatal("announcement shown after it was seen")
	}
}

func TestWhatsNewNeverPushesTheTitleOffScreen(t *testing.T) {
	view := "═══ ggrun ═══\nversion\nrest\n"
	at := len("═══ ggrun ═══\nversion\n")
	if got := withWhatsNew(view, at, whatsNewText, 200, 40); !strings.Contains(got, "New: use this model") || !strings.HasPrefix(got, "═══ ggrun ═══") {
		t.Fatalf("banner not inserted under the version line: %q", got)
	}
	if got := withWhatsNew(view, at, whatsNewText, 200, 4); got != view {
		t.Fatalf("banner inserted into a full screen: %q", got)
	}
	if got := withWhatsNew(view, at, whatsNewText, 40, 40); strings.Count(got, "\n") != strings.Count(view, "\n")+1 {
		t.Fatalf("narrow terminal banner is not one line: %q", got)
	}
}
