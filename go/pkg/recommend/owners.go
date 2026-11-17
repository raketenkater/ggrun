package recommend

import "strings"

// knownQuantizers are the GGUF publishers the catalog and the downloader
// score highest (TRUSTED_GGUF_OWNERS in tools/models/update_recommendations.py
// and tools/download/download_any_gguf.py). Selection stays open to every
// publisher (decision D3); a row from anyone else is marked, not hidden.
var knownQuantizers = []string{"unsloth", "bartowski", "maziyarpanahi", "prithivmlmods", "lmstudio-community", "second-state"}

// RepoOwner is the Hugging Face owner of "owner/name".
func RepoOwner(repo string) string {
	owner, _, ok := strings.Cut(strings.TrimSpace(repo), "/")
	if !ok {
		return ""
	}
	return owner
}

// KnownQuantizer reports whether owner is one of the named GGUF publishers.
func KnownQuantizer(owner string) bool {
	for _, q := range knownQuantizers {
		if strings.EqualFold(owner, q) {
			return true
		}
	}
	return false
}

// OwnerLabel is the owner as shown in a recommendation list: the name, with
// "*" when it is not one of the named publishers.
func OwnerLabel(repo string) string {
	owner := RepoOwner(repo)
	if owner == "" {
		return "?"
	}
	if KnownQuantizer(owner) {
		return owner
	}
	return owner + "*"
}

// OwnerLegend explains the "*" mark.
func OwnerLegend() string {
	return "* publisher outside " + strings.Join(knownQuantizers, ", ") + "; check the repository before downloading"
}
