package backends

import (
	"strings"
	"time"
)

// Staleness is what `ggrun backend status` reports for one registration. It is
// computed from local data only (the embedded catalog, the manifest and the
// checkout), so it never reaches the network and never affects a launch.
type Staleness struct {
	Tag    string
	Recipe string // catalog entry the tag names, "" for a hand-added backend

	Kind          string
	SupersededBy  string
	Evidence      []string
	Expires       string
	ReviewOverdue bool // Expires has passed

	PinnedCommit   string // the recipe's pin
	BuiltCommit    string // commit recorded at registration
	BuiltOffPin    bool   // BuiltCommit differs from the pin (both known)
	CheckoutCommit string // HEAD of the source checkout, "" when unreadable
	CheckoutOffPin bool   // CheckoutCommit differs from the pin (both known)

	BuiltAt       time.Time
	BuiltAtSource string // "recorded" or "binary file time"
	Age           time.Duration
	LastBuild     *BuildRecord
}

// SameCommit compares two commit ids, allowing one to be an abbreviation.
func SameCommit(a, b string) bool {
	a, b = strings.ToLower(strings.TrimSpace(a)), strings.ToLower(strings.TrimSpace(b))
	if a == "" || b == "" {
		return false
	}
	return strings.HasPrefix(a, b) || strings.HasPrefix(b, a)
}

// AssessBackend builds the report for one registration. recipe is its catalog
// entry (nil when it has none), checkoutHead the source checkout's HEAD ("" if
// unknown) and binModTime the binary's modification time (zero if unknown).
func AssessBackend(be Backend, recipe *Recipe, checkoutHead string, binModTime, now time.Time) Staleness {
	s := Staleness{Tag: be.Tag, BuiltCommit: be.Commit, CheckoutCommit: checkoutHead, LastBuild: be.Build}
	if recipe != nil {
		s.Recipe = recipe.Name
		s.Kind, s.SupersededBy, s.Evidence, s.Expires = recipe.Kind, recipe.SupersededBy, recipe.Evidence, recipe.Expires
		s.PinnedCommit = recipe.Commit
		if recipe.Expires != "" {
			if day, err := time.Parse("2006-01-02", recipe.Expires); err == nil && now.After(day.Add(24*time.Hour)) {
				s.ReviewOverdue = true
			}
		}
		if be.Commit != "" && recipe.Commit != "" {
			s.BuiltOffPin = !SameCommit(be.Commit, recipe.Commit)
		}
		if checkoutHead != "" && recipe.Commit != "" {
			s.CheckoutOffPin = !SameCommit(checkoutHead, recipe.Commit)
		}
	}
	if be.Build != nil {
		if at, err := time.Parse(time.RFC3339, be.Build.At); err == nil {
			s.BuiltAt, s.BuiltAtSource = at, "recorded"
		}
	}
	if s.BuiltAt.IsZero() && !binModTime.IsZero() {
		s.BuiltAt, s.BuiltAtSource = binModTime, "binary file time"
	}
	if !s.BuiltAt.IsZero() {
		s.Age = now.Sub(s.BuiltAt)
		if s.Age < 0 {
			s.Age = 0
		}
	}
	return s
}
