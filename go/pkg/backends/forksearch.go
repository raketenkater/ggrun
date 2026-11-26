package backends

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
)

// ArchForkPR is an official llama.cpp pull request whose head fork claims to
// add a GGUF architecture that the installed backends do not load. It is a
// discovery result, not a reviewed recipe: install still clones, builds, and
// conformance-probes the binary before routing.
type ArchForkPR struct {
	Arch   string
	Number int
	Title  string
	URL    string
	Merged bool
	GitURL string
	Branch string
	Commit string
	// Cited is true when a Hugging Face GGUF model card named this official
	// llama.cpp PR. Discovery still clones the PR head, not the card's text.
	Cited bool
	// Author and AuthorAssociation (GitHub's OWNER, MEMBER, COLLABORATOR,
	// CONTRIBUTOR, FIRST_TIME_CONTRIBUTOR, NONE...) say who wrote the code ggrun
	// would build; CIState is the head commit's check result. The user sees
	// all three before agreeing to build an unreviewed PR.
	Author            string
	AuthorAssociation string
	CIState           string
}

// Provenance is the one-line account of who wrote a PR and how its CI stands.
func (p ArchForkPR) Provenance() string {
	author := strings.TrimSpace(p.Author)
	if author == "" {
		author = "unknown author"
	}
	assoc := strings.ToLower(strings.ReplaceAll(strings.TrimSpace(p.AuthorAssociation), "_", " "))
	if assoc == "" {
		assoc = "association unknown"
	}
	ci := strings.TrimSpace(p.CIState)
	if ci == "" {
		ci = "unknown"
	}
	return fmt.Sprintf("by %s (%s to llama.cpp), CI %s", author, assoc, ci)
}

// ArchForkSearch is the architecture-generic lookup for an unsupported GGUF.
// Arch is required. QuantizedBy/Name/Basename come from local GGUF metadata so
// the finder can read the publisher's model card without a model-specific pin.
type ArchForkSearch struct {
	Arch        string
	QuantizedBy string
	Name        string
	Basename    string
}

// ForkSearchClient injects HTTP for tests. Empty GitHubBase and HFBase select
// the live hosts. A test that sets only GitHubBase skips Hugging Face so the
// original PR-index tests stay offline.
type ForkSearchClient struct {
	HTTP       *http.Client
	GitHubBase string
	HFBase     string
}

const (
	defaultForkSearchBase  = "https://api.github.com"
	defaultHFModelCardBase = "https://huggingface.co"
	maxArchForkCandidates  = 2
	maxGitHubPRFetches     = 5
	maxHFCardFetches       = 3
)

var (
	archForkQuerySafe   = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)
	llamaCppPullPath    = regexp.MustCompile(`^https://github.com/ggml-org/llama.cpp/pull/(\d+)$`)
	llamaCppPullMention = regexp.MustCompile(`(?:https://)?github\.com/ggml-org/llama\.cpp/pull/(\d+)`)
	hfRepoPart          = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,96}$`)
	modelShardSuffix    = regexp.MustCompile(`(?i)-00001-of-[0-9]{5}$`)
	modelGGUFSuffix     = regexp.MustCompile(`(?i)\.gguf$`)
	modelQuantSuffix    = regexp.MustCompile(`(?i)[-_.]((ud[-_.])?(iq|q)\d+[a-z0-9_]*|mxfp\d+|bf16|fp16|f16|f32)$`)
	gitCommitSHA        = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// Recipe is RecipeForModel with no model name. Prefer RecipeForModel so the
// fork manager lists the backend under the model it was installed to run.
func (p ArchForkPR) Recipe() Recipe {
	return p.RecipeForModel("")
}

// RecipeForModel turns a discovered PR head into an isolated fork recipe named
// after the model being launched. RouteArch stays the GGUF architecture so later
// launches of that arch still select this backend; Tag/Name are the model so
// `ggrun backend list` and the TUI fork manager can update or remove it.
func (p ArchForkPR) RecipeForModel(modelName string) Recipe {
	arch := strings.ToLower(strings.TrimSpace(p.Arch))
	tag := BackendTagFromModel(modelName, arch, p.Number)
	title := strings.TrimSpace(p.Title)
	if title == "" {
		title = "architecture support"
	}
	desc := fmt.Sprintf("Isolated llama.cpp PR #%d fork for %s: %s", p.Number, tag, title)
	return Recipe{
		Name:        tag,
		Description: desc,
		Tag:         tag,
		GitURL:      p.GitURL,
		Branch:      p.Branch,
		Commit:      p.Commit,
		RouteArch:   arch,
	}
}

// BackendTagFromModel builds the fork-manager tag from a model identity.
// Shard and trailing quant suffixes are dropped so the tag is the model, not
// the file. Empty or reserved names fall back to the architecture, then pr-N.
func BackendTagFromModel(modelName, arch string, prNumber int) string {
	for _, candidate := range []string{modelName, arch} {
		if tag := sanitizeBackendTag(stripModelFileDecorations(candidate)); tag != "" && !reservedBackendTag(tag) {
			return tag
		}
	}
	if prNumber > 0 {
		return fmt.Sprintf("pr-%d", prNumber)
	}
	return "fork"
}

func stripModelFileDecorations(raw string) string {
	s := strings.TrimSpace(raw)
	s = strings.ReplaceAll(s, "\\", "/")
	if i := strings.LastIndexByte(s, '/'); i >= 0 {
		s = s[i+1:]
	}
	s = modelGGUFSuffix.ReplaceAllString(s, "")
	s = modelShardSuffix.ReplaceAllString(s, "")
	for i := 0; i < 2; i++ {
		next := modelQuantSuffix.ReplaceAllString(s, "")
		if next == s {
			break
		}
		s = next
	}
	return strings.TrimSpace(s)
}

func sanitizeBackendTag(raw string) string {
	var b strings.Builder
	lastDash := true
	for _, r := range strings.TrimSpace(raw) {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(unicode.ToLower(r))
			lastDash = false
		case r == '.' || r == '_' || r == '-' || r == ' ' || r == '/':
			if !lastDash {
				b.WriteByte('-')
				lastDash = true
			}
		}
	}
	s := strings.Trim(b.String(), "-")
	if len(s) > 48 {
		s = strings.Trim(s[:48], "-")
	}
	return s
}

func reservedBackendTag(tag string) bool {
	switch strings.ToLower(strings.TrimSpace(tag)) {
	case "llama", "llama.cpp", "llama-cpp", "ik", "ik_llama", "ik-llama",
		"ik_llama.cpp", "ik-llama-cpp", "mainline", "auto":
		return true
	default:
		return false
	}
}

// IsolatedForkSourceDir is the checkout path for a registered fork. It always
// lives under .src/fork-* so a PR install cannot land in .src/llama.cpp.
func IsolatedForkSourceDir(appHome, gitURL, branch string, nameHint ...string) string {
	if appHome == "" {
		appHome = AppHome()
	}
	if len(nameHint) > 0 {
		if tag := sanitizeBackendTag(nameHint[0]); tag != "" && !reservedBackendTag(tag) {
			return filepath.Join(appHome, ".src", "fork-"+tag)
		}
	}
	name := strings.TrimSpace(gitURL)
	if i := strings.LastIndexByte(name, '/'); i >= 0 {
		name = name[i+1:]
	}
	name = strings.TrimSuffix(name, ".git")
	if branch = strings.TrimSpace(branch); branch != "" {
		if i := strings.LastIndexByte(branch, '/'); i >= 0 {
			branch = branch[i+1:]
		}
		name = name + "-" + branch
	}
	name = strings.ToLower(name)
	name = strings.NewReplacer("/", "-", " ", "-", "_", "-").Replace(name)
	if name == "" {
		name = "fork"
	}
	return filepath.Join(appHome, ".src", "fork-"+name)
}

// MainlineSourceDir is the production llama.cpp checkout. Fork installs must
// never share this directory.
func MainlineSourceDir(appHome string) string {
	if appHome == "" {
		appHome = AppHome()
	}
	return filepath.Join(appHome, ".src", "llama.cpp")
}

func IsIsolatedForkSourceDir(srcDir string) bool {
	base := filepath.Base(strings.TrimSpace(srcDir))
	return strings.HasPrefix(base, "fork-") &&
		!strings.EqualFold(base, "llama.cpp") &&
		!strings.EqualFold(base, "ik_llama.cpp")
}

// SearchArchForkPRs is the GitHub-only lookup kept for tests. Live launch uses
// SearchArchForks so a GGUF publisher's Hugging Face card can name the PR.
func SearchArchForkPRs(ctx context.Context, arch, base string, client *http.Client) ([]ArchForkPR, error) {
	return SearchArchForks(ctx, ArchForkSearch{Arch: arch}, ForkSearchClient{
		HTTP:       client,
		GitHubBase: base,
	})
}

// SearchArchForks finds an isolated llama.cpp head fork for an architecture the
// installed backends cannot load. It queries the official ggml-org/llama.cpp
// pull-request index and, when Hugging Face is enabled, reads GGUF model cards
// derived from local metadata (quantized_by / name) plus an architecture search.
// Only github.com/ggml-org/llama.cpp/pull/N URLs are followed, and only an open,
// unmerged PR with a cloneable github.com head is returned. Publisher-cited PRs
// and titles that add the architecture rank ahead of later "fix" PRs that merely
// mention it. Network failures on GitHub return an error so the caller can fall
// through to the mainline-update path; a Hugging Face miss is not fatal.
func SearchArchForks(ctx context.Context, query ArchForkSearch, client ForkSearchClient) ([]ArchForkPR, error) {
	arch := strings.TrimSpace(query.Arch)
	if !archForkQuerySafe.MatchString(arch) {
		return nil, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	client = client.resolve()

	githubPRs, githubErr := searchGitHubArchForkPRs(ctx, client.HTTP, client.GitHubBase, arch)
	var cardPRs []ArchForkPR
	if client.HFBase != "" {
		cardPRs = searchHFCitedArchForkPRs(ctx, client, query)
	}
	merged := mergeArchForks(cardPRs, githubPRs)
	if len(merged) == 0 {
		return nil, githubErr
	}
	// Only the first candidate is offered, so only it costs a CI lookup.
	merged[0].CIState = fetchCIState(ctx, client.HTTP, client.GitHubBase, merged[0].Commit)
	return merged, nil
}

// fetchCIState summarises the check runs on a commit: "failing" when any
// failed, "running" while any is unfinished, "passing" when all finished
// without failure, "none" when there are none, "unknown" when GitHub did not
// answer.
func fetchCIState(ctx context.Context, client *http.Client, base, sha string) string {
	if !gitCommitSHA.MatchString(sha) {
		return "unknown"
	}
	body, err := getGitHubJSON(ctx, client, fmt.Sprintf("%s/repos/ggml-org/llama.cpp/commits/%s/check-runs?per_page=100", base, sha))
	if err != nil {
		return "unknown"
	}
	var runs struct {
		Total int `json:"total_count"`
		Runs  []struct {
			Status     string `json:"status"`
			Conclusion string `json:"conclusion"`
		} `json:"check_runs"`
	}
	if json.Unmarshal(body, &runs) != nil {
		return "unknown"
	}
	if runs.Total == 0 || len(runs.Runs) == 0 {
		return "none"
	}
	running := false
	for _, r := range runs.Runs {
		switch strings.ToLower(r.Conclusion) {
		case "failure", "timed_out", "cancelled", "action_required", "startup_failure":
			return "failing"
		}
		if !strings.EqualFold(r.Status, "completed") {
			running = true
		}
	}
	if running {
		return "running"
	}
	return "passing"
}

func (c ForkSearchClient) resolve() ForkSearchClient {
	live := strings.TrimSpace(c.GitHubBase) == "" && strings.TrimSpace(c.HFBase) == ""
	if c.HTTP == nil {
		c.HTTP = &http.Client{Timeout: 20 * time.Second}
	}
	if strings.TrimSpace(c.GitHubBase) == "" {
		c.GitHubBase = defaultForkSearchBase
	}
	if live {
		c.HFBase = defaultHFModelCardBase
	}
	return c
}

func searchGitHubArchForkPRs(ctx context.Context, client *http.Client, base, arch string) ([]ArchForkPR, error) {
	searchURL := base + "/search/issues?q=" + url.QueryEscape("repo:ggml-org/llama.cpp is:pr is:open "+arch) +
		"&sort=updated&order=desc&per_page=10"
	body, err := getGitHubJSON(ctx, client, searchURL)
	if err != nil {
		return nil, err
	}
	var index struct {
		Items []struct {
			Title   string `json:"title"`
			HTMLURL string `json:"html_url"`
			Body    string `json:"body"`
		} `json:"items"`
	}
	if err := json.Unmarshal(body, &index); err != nil {
		return nil, fmt.Errorf("decode llama.cpp PR search: %w", err)
	}

	needle := strings.ToLower(arch)
	out := make([]ArchForkPR, 0, maxGitHubPRFetches)
	fetches := 0
	for _, item := range index.Items {
		if fetches >= maxGitHubPRFetches {
			break
		}
		match := llamaCppPullPath.FindStringSubmatch(strings.TrimRight(item.HTMLURL, "/"))
		if match == nil {
			continue
		}
		haystack := strings.ToLower(item.Title + "\n" + item.Body)
		if !archTokenPresent(haystack, needle) {
			continue
		}
		number, err := strconv.Atoi(match[1])
		if err != nil || number <= 0 {
			continue
		}
		fetches++
		pr, err := fetchLlamaCppPR(ctx, client, base, arch, number)
		if err != nil || pr == nil {
			continue
		}
		out = append(out, *pr)
	}
	return out, nil
}

func searchHFCitedArchForkPRs(ctx context.Context, client ForkSearchClient, query ArchForkSearch) []ArchForkPR {
	out := citedPRsFromHFRepos(ctx, client, query, hfGGUFCardRepos(query))
	if len(out) > 0 {
		return out
	}
	return citedPRsFromHFRepos(ctx, client, query, searchHFGGUFRepos(ctx, client.HTTP, client.HFBase, query.Arch))
}

func citedPRsFromHFRepos(ctx context.Context, client ForkSearchClient, query ArchForkSearch, repos []string) []ArchForkPR {
	seenPR := map[int]bool{}
	out := make([]ArchForkPR, 0, maxArchForkCandidates)
	cardFetches := 0
	prFetches := 0
	for _, repo := range repos {
		if cardFetches >= maxHFCardFetches || prFetches >= maxGitHubPRFetches {
			break
		}
		cardFetches++
		body, err := fetchHFReadme(ctx, client.HTTP, client.HFBase, repo)
		if err != nil || len(body) == 0 {
			continue
		}
		for _, number := range llamaCppPRsCitedIn(string(body)) {
			if prFetches >= maxGitHubPRFetches {
				break
			}
			if seenPR[number] {
				continue
			}
			seenPR[number] = true
			prFetches++
			pr, err := fetchLlamaCppPR(ctx, client.HTTP, client.GitHubBase, query.Arch, number)
			if err != nil || pr == nil {
				continue
			}
			pr.Cited = true
			out = append(out, *pr)
		}
	}
	return out
}

func hfGGUFCardRepos(query ArchForkSearch) []string {
	org := sanitizeHFRepoPart(query.QuantizedBy)
	if org == "" {
		return nil
	}
	var out []string
	seen := map[string]bool{}
	add := func(name string) {
		name = sanitizeHFRepoPart(name)
		if name == "" {
			return
		}
		for _, repo := range []string{org + "/" + name + "-GGUF", org + "/" + name} {
			key := strings.ToLower(repo)
			if seen[key] {
				continue
			}
			seen[key] = true
			out = append(out, repo)
		}
	}
	for _, raw := range []string{query.Basename, query.Name} {
		add(stripModelFileDecorations(raw))
	}
	if len(out) > maxHFCardFetches {
		out = out[:maxHFCardFetches]
	}
	return out
}

func searchHFGGUFRepos(ctx context.Context, client *http.Client, base, arch string) []string {
	searchURL := strings.TrimRight(base, "/") + "/api/models?search=" + url.QueryEscape(arch) + "&filter=gguf&limit=5"
	body, err := getPlainURL(ctx, client, searchURL, false)
	if err != nil {
		return nil
	}
	var models []struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(body, &models); err != nil {
		return nil
	}
	var out []string
	seen := map[string]bool{}
	for _, model := range models {
		repo := strings.Trim(strings.TrimSpace(model.ID), "/")
		if !safeHFRepoID(repo) {
			continue
		}
		key := strings.ToLower(repo)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, repo)
		if len(out) >= maxHFCardFetches {
			break
		}
	}
	return out
}

func fetchHFReadme(ctx context.Context, client *http.Client, base, repo string) ([]byte, error) {
	if !safeHFRepoID(repo) {
		return nil, nil
	}
	rawURL := strings.TrimRight(base, "/") + "/" + repo + "/raw/main/README.md"
	return getPlainURL(ctx, client, rawURL, true)
}

func llamaCppPRsCitedIn(body string) []int {
	matches := llamaCppPullMention.FindAllStringSubmatch(body, 8)
	if len(matches) == 0 {
		return nil
	}
	seen := map[int]bool{}
	var out []int
	for _, match := range matches {
		if len(match) < 2 {
			continue
		}
		number, err := strconv.Atoi(match[1])
		if err != nil || number <= 0 || seen[number] {
			continue
		}
		seen[number] = true
		out = append(out, number)
	}
	return out
}

func mergeArchForks(groups ...[]ArchForkPR) []ArchForkPR {
	seen := map[int]int{}
	var out []ArchForkPR
	for _, group := range groups {
		for _, pr := range group {
			if pr.Number <= 0 {
				continue
			}
			if i, ok := seen[pr.Number]; ok {
				if pr.Cited {
					out[i].Cited = true
				}
				continue
			}
			seen[pr.Number] = len(out)
			out = append(out, pr)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		ri, rj := archForkRank(out[i]), archForkRank(out[j])
		if ri != rj {
			return ri < rj
		}
		return out[i].Number < out[j].Number
	})
	if len(out) > maxArchForkCandidates {
		out = out[:maxArchForkCandidates]
	}
	return out
}

func archForkRank(pr ArchForkPR) int {
	score := 100
	if pr.Cited {
		score -= 50
	}
	title := strings.ToLower(strings.TrimSpace(pr.Title))
	arch := strings.ToLower(strings.TrimSpace(pr.Arch))
	if arch != "" && archTokenPresent(title, arch) {
		score -= 20
	}
	if strings.Contains(title, "add") {
		score -= 10
	}
	if strings.HasPrefix(title, "model:") {
		score -= 5
	}
	if strings.Contains(title, "fix") && !strings.Contains(title, "add") {
		score += 8
	}
	return score
}

func sanitizeHFRepoPart(raw string) string {
	s := strings.TrimSpace(raw)
	if s == "" || strings.ContainsAny(s, "/\\") || strings.Contains(s, "..") {
		return ""
	}
	if !hfRepoPart.MatchString(s) {
		return ""
	}
	return s
}

func safeHFRepoID(repo string) bool {
	parts := strings.Split(strings.Trim(repo, "/"), "/")
	if len(parts) != 2 {
		return false
	}
	return sanitizeHFRepoPart(parts[0]) != "" && sanitizeHFRepoPart(parts[1]) != ""
}

func getPlainURL(ctx context.Context, client *http.Client, rawURL string, allowNotFound bool) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("User-Agent", "ggrun-backend-search")
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 2<<20))
	if err != nil {
		return nil, err
	}
	if allowNotFound && response.StatusCode == http.StatusNotFound {
		return nil, nil
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("huggingface %s: HTTP %d", request.URL.Path, response.StatusCode)
	}
	return body, nil
}

func fetchLlamaCppPR(ctx context.Context, client *http.Client, base, arch string, number int) (*ArchForkPR, error) {
	body, err := getGitHubJSON(ctx, client, fmt.Sprintf("%s/repos/ggml-org/llama.cpp/pulls/%d", base, number))
	if err != nil {
		return nil, err
	}
	var raw struct {
		Number int    `json:"number"`
		Title  string `json:"title"`
		Body   string `json:"body"`
		URL    string `json:"html_url"`
		State  string `json:"state"`
		Merged bool   `json:"merged"`
		User   struct {
			Login string `json:"login"`
		} `json:"user"`
		AuthorAssociation string `json:"author_association"`
		Head              struct {
			Ref  string `json:"ref"`
			SHA  string `json:"sha"`
			Repo *struct {
				CloneURL string `json:"clone_url"`
			} `json:"repo"`
		} `json:"head"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("decode llama.cpp PR %d: %w", number, err)
	}
	officialURL := fmt.Sprintf("https://github.com/ggml-org/llama.cpp/pull/%d", number)
	if raw.Number != number || strings.TrimRight(raw.URL, "/") != officialURL {
		return nil, nil
	}
	if raw.Merged || !strings.EqualFold(raw.State, "open") {
		return nil, nil
	}
	if !archTokenPresent(strings.ToLower(raw.Title+"\n"+raw.Body), strings.ToLower(strings.TrimSpace(arch))) {
		return nil, nil
	}
	if raw.Head.Repo == nil || !looksLikeGitHubCloneURL(raw.Head.Repo.CloneURL) {
		return nil, nil
	}
	commit := strings.ToLower(strings.TrimSpace(raw.Head.SHA))
	if !gitCommitSHA.MatchString(commit) {
		return nil, nil
	}
	branch := strings.TrimSpace(raw.Head.Ref)
	if branch == "" {
		return nil, nil
	}
	return &ArchForkPR{
		Arch:   strings.ToLower(strings.TrimSpace(arch)),
		Number: raw.Number,
		Title:  strings.TrimSpace(raw.Title),
		URL:    strings.TrimRight(raw.URL, "/"),
		GitURL: strings.TrimSpace(raw.Head.Repo.CloneURL),
		Branch: branch,
		Commit: commit,

		Author:            strings.TrimSpace(raw.User.Login),
		AuthorAssociation: strings.TrimSpace(raw.AuthorAssociation),
	}, nil
}

func getGitHubJSON(ctx context.Context, client *http.Client, rawURL string) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Accept", "application/vnd.github+json")
	request.Header.Set("User-Agent", "ggrun-backend-search")
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 2<<20))
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("github %s: HTTP %d", request.URL.Path, response.StatusCode)
	}
	return body, nil
}

func looksLikeGitHubCloneURL(raw string) bool {
	raw = strings.TrimSpace(raw)
	if !strings.HasPrefix(raw, "https://github.com/") || !strings.HasSuffix(raw, ".git") {
		return false
	}
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Host != "github.com" || parsed.Scheme != "https" {
		return false
	}
	return strings.Count(strings.Trim(parsed.Path, "/"), "/") == 1
}

func archTokenPresent(haystack, arch string) bool {
	if arch == "" || haystack == "" {
		return false
	}
	for _, field := range strings.FieldsFunc(haystack, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' && r != '-' && r != '.'
	}) {
		if field == arch {
			return true
		}
	}
	return strings.Contains(haystack, arch)
}
