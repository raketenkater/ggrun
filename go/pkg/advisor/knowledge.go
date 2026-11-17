package advisor

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

//go:embed knowledge/core.md
var coreKnowledge string

//go:embed knowledge/cache.md
var cacheKnowledge string

//go:embed knowledge/backends.md
var backendKnowledge string

func BundledKnowledge(incident Incident) []Source {
	now := time.Now().UTC()
	sources := []Source{{
		Title: "ggrun controller invariants", URL: "ggrun://knowledge/core", RetrievedAt: now,
		Excerpt: coreKnowledge,
	}}
	wantsCache := strings.Contains(strings.ToLower(incident.ProfileState), "cache") ||
		strings.Contains(strings.ToLower(incident.Architecture), "deepseek")
	for _, observation := range incident.Observations {
		if strings.Contains(strings.ToLower(observation.Component), "cache") || strings.Contains(strings.ToLower(observation.Code), "cache") {
			wantsCache = true
		}
	}
	if wantsCache {
		sources = append(sources, Source{Title: "ggrun cache/checkpoint knowledge", URL: "ggrun://knowledge/cache", RetrievedAt: now, Excerpt: cacheKnowledge})
	}
	sources = append(sources, Source{Title: "ggrun backend compatibility knowledge", URL: "ggrun://knowledge/backends", RetrievedAt: now, Excerpt: backendKnowledge})
	return sources
}

type Researcher struct {
	Client *http.Client
}

// errResearchDegraded marks an online research tier that could not contribute
// new evidence but must not abort the analysis. The fallback chain (Official ->
// ModelCard -> Bundled) treats it as a degraded note: the analysis continues
// with whatever evidence it has and records a "research_degraded" Settings
// entry so the model and history know the evidence pool was incomplete.
var errResearchDegraded = errors.New("research degraded: no new online evidence")

// modelCardPin pins the authoritative model card for an architecture. Research
// reads ONLY the README of this exact repo@revision and never follows user- or
// model-supplied URLs, matching the deterministic-search rule of
// ResearchOfficial. The pin is the provenance root: a mutable HF README cannot
// redirect ggrun's research to an unreviewed repo or a model-produced value.
type modelCardPin struct {
	Repo     string
	Revision string
}

// pinnedModelCard is the registry consulted by ResearchModelCard. It is seeded
// from the official Nanbeige4.2 checkpoint, which is the one architecture ggrun
// has a reviewed, provenance-pinned relationship with today (the support-expert
// artifact manifest pins the same source). An incident may carry an explicit
// settings pin for an unseeded architecture; that is the only route outside
// this registry, and it still points at a fixed repo@revision README, never a
// model-produced value.
var pinnedModelCard = map[string]modelCardPin{
	"nanbeige": {
		Repo:     "Nanbeige/Nanbeige4.2-3B",
		Revision: "main",
	},
}

// DefaultResearchHTTPClient bounds every online research fetch. The official
// issue index and model-card READMEs are evidence, never executables, so a
// bounded client keeps a hostile or misconfigured endpoint from stalling the
// optional advisor for long.
func DefaultResearchHTTPClient() *http.Client {
	return &http.Client{Timeout: 20 * time.Second}
}

// defaultOfficialSearchBase is the live GitHub search API. Tests inject a local
// httptest endpoint via ResearchOfficial's base parameter so the network path is
// exercised deterministically without touching api.github.com.
const defaultOfficialSearchBase = "https://api.github.com"

// ResearchOfficial queries only the official ggml-org/llama.cpp issue/PR index.
// Search terms come from typed architecture and observation codes, never user
// prompts or raw logs. Results are evidence for explanation/ranking only; URLs
// or repository names returned by the model are not executable actions.
// base is the search host (scheme+host only, e.g. https://api.github.com); an
// empty base selects the live GitHub API. The full search path is recorded on
// each Source for reproducibility.
func (researcher Researcher) ResearchOfficial(ctx context.Context, incident Incident, base string) ([]Source, error) {
	terms := []string{cleanSearchTerm(incident.Architecture)}
	for _, observation := range incident.Observations {
		if term := cleanSearchTerm(observation.Code); term != "" {
			terms = append(terms, term)
		}
		if len(terms) >= 4 {
			break
		}
	}
	query := strings.TrimSpace(strings.Join(terms, " "))
	if query == "" {
		return nil, nil
	}
	if base == "" {
		base = defaultOfficialSearchBase
	}
	searchPath := base + "/search/issues?q=" + url.QueryEscape("repo:ggml-org/llama.cpp "+query) + "&sort=updated&order=desc&per_page=5"
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, searchPath, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Accept", "application/vnd.github+json")
	request.Header.Set("User-Agent", "ggrun-support-expert")
	client := researcher.Client
	if client == nil {
		client = &http.Client{Timeout: 20 * time.Second}
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("official research returned HTTP %d", response.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, 2<<20))
	if err != nil {
		return nil, err
	}
	var result struct {
		Items []struct {
			Title     string    `json:"title"`
			HTMLURL   string    `json:"html_url"`
			Body      string    `json:"body"`
			UpdatedAt time.Time `json:"updated_at"`
		} `json:"items"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}
	sources := make([]Source, 0, len(result.Items))
	for _, item := range result.Items {
		if !strings.HasPrefix(item.HTMLURL, "https://github.com/ggml-org/llama.cpp/") {
			continue
		}
		excerpt := cleanToken(item.Body, 1800)
		if excerpt == "" {
			excerpt = cleanToken(item.Title, 300)
		}
		sources = append(sources, Source{
			Title: cleanToken(item.Title, 240), URL: item.HTMLURL,
			RetrievedAt: time.Now().UTC(), Excerpt: excerpt,
			SearchPath: searchPath,
		})
	}
	return sources, nil
}

// defaultModelCardBase is the live Hugging Face host for model-card READMEs.
// Tests inject a local httptest endpoint via ResearchModelCard's base parameter,
// mirroring ResearchOfficial's injected-base seam.
const defaultModelCardBase = "https://huggingface.co"

// ResearchModelCard fetches the pinned model card (repo@revision README) for
// the incident's architecture and returns it as a single Source — the second
// tier of the deterministic evidence chain (Official -> ModelCard -> Bundled).
// base is the Hugging Face host (scheme+host only); an empty base selects the
// live site.
//
// A nil, nil return means no card is pinned for this architecture (the tier
// contributes nothing). errResearchDegraded means the tier exists but added no
// evidence (e.g. the card body is empty). Any other error is the concrete fetch
// failure (network, HTTP status) so the chain can record a specific
// "research_degraded" Settings note.
func (researcher Researcher) ResearchModelCard(ctx context.Context, incident Incident, base string) ([]Source, error) {
	pin, ok := pinnedModelCard[strings.ToLower(strings.TrimSpace(incident.Architecture))]
	if !ok {
		if custom := strings.TrimSpace(incident.Settings["model_card_pin"]); custom != "" {
			parts := strings.SplitN(custom, "@", 2)
			pin = modelCardPin{Repo: parts[0]}
			if len(parts) == 2 {
				pin.Revision = parts[1]
			}
		} else {
			return nil, nil
		}
	}
	if base == "" {
		base = defaultModelCardBase
	}
	body, cardPath, err := fetchModelCard(ctx, researcher.Client, base, pin)
	if err != nil {
		return nil, err
	}
	excerpt := cleanToken(string(body), 1800)
	if excerpt == "" {
		return nil, errResearchDegraded
	}
	return []Source{{
		Title:       fmt.Sprintf("Model card: %s", pin.Repo),
		URL:         fmt.Sprintf("https://huggingface.co/%s", pin.Repo),
		RetrievedAt: time.Now().UTC(),
		Excerpt:     excerpt,
		SHA256:      sha256Of(body),
		SearchPath:  cardPath,
	}}, nil
}

// fetchModelCard reads the pinned repo@revision README, capped at 2 MiB like
// the artifact download and the official issue search, so a misbehaving or
// oversized card cannot balloon the evidence pool or stall the optional helper.
func fetchModelCard(ctx context.Context, client *http.Client, base string, pin modelCardPin) ([]byte, string, error) {
	if strings.TrimSpace(pin.Repo) == "" {
		return nil, "", errors.New("no model card pin for this architecture")
	}
	repo := strings.Trim(pin.Repo, "/")
	revision := strings.TrimSpace(pin.Revision)
	if revision == "" {
		revision = "main"
	}
	cardPath := fmt.Sprintf("%s/%s/raw/%s/README.md", base, repo, url.PathEscape(revision))
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, cardPath, nil)
	if err != nil {
		return nil, "", err
	}
	request.Header.Set("User-Agent", "ggrun-support-expert")
	if client == nil {
		client = DefaultResearchHTTPClient()
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, "", err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("model-card research returned HTTP %d", response.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, 2<<20))
	if err != nil {
		return nil, "", err
	}
	return body, cardPath, nil
}

// sha256Of returns the uppercase SHA-256 hex digest of data. It is exposed for
// the model-card tier so the controller can record a pinned digest on the Source
// and compare it against a locally cached card's quoted checksum.
func sha256Of(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

func cleanSearchTerm(value string) string {
	value = strings.ToLower(value)
	var b strings.Builder
	for _, r := range value {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			b.WriteRune(r)
		} else {
			b.WriteByte(' ')
		}
	}
	terms := strings.Fields(b.String())
	if len(terms) > 3 {
		terms = terms[:3]
	}
	return strings.Join(terms, " ")
}
