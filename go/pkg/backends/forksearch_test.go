package backends

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestSearchArchForkPRsReturnsOpenHeadFork(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/check-runs") {
			io.WriteString(w, `{"total_count":1,"check_runs":[{"status":"completed","conclusion":"success"}]}`)
			return
		}
		switch {
		case r.URL.Path == "/search/issues":
			q := r.URL.Query().Get("q")
			if !strings.Contains(q, "repo:ggml-org/llama.cpp") || !strings.Contains(q, "is:pr") ||
				!strings.Contains(q, "is:open") || !strings.Contains(q, "qwen4exp") {
				t.Fatalf("search query = %q", q)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"items": []map[string]any{
					{
						"title":    "model: add Qwen3.8-Flash-Next (qwen4exp)",
						"html_url": "https://github.com/ggml-org/llama.cpp/pull/27742",
						"body":     "Adds GGUF architecture qwen4exp.",
					},
					{
						"title":    "Off-topic PR",
						"html_url": "https://github.com/someone-else/llama.cpp/pull/1",
						"body":     "qwen4exp mention in a foreign repo",
					},
				},
			})
		case r.URL.Path == "/repos/ggml-org/llama.cpp/pulls/27742":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"number":             27742,
				"title":              "model: add Qwen3.8-Flash-Next (qwen4exp)",
				"html_url":           "https://github.com/ggml-org/llama.cpp/pull/27742",
				"state":              "open",
				"merged":             false,
				"user":               map[string]any{"login": "pr-author"},
				"author_association": "CONTRIBUTOR",
				"head": map[string]any{
					"ref": "qwen4exp/qwen3.8-flash-next",
					"sha": "250b61446efc91e3a179c8677956f2667c8fbda0",
					"repo": map[string]any{
						"clone_url": "https://github.com/unslothai/llama.cpp.git",
					},
				},
			})
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	got, err := SearchArchForkPRs(context.Background(), "qwen4exp", server.URL, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("candidates=%d, want 1: %+v", len(got), got)
	}
	pr := got[0]
	if pr.Number != 27742 || pr.GitURL != "https://github.com/unslothai/llama.cpp.git" ||
		pr.Branch != "qwen4exp/qwen3.8-flash-next" || pr.Commit != "250b61446efc91e3a179c8677956f2667c8fbda0" {
		t.Fatalf("discovered PR = %+v", pr)
	}
	if pr.Author != "pr-author" || pr.AuthorAssociation != "CONTRIBUTOR" || pr.CIState != "passing" {
		t.Fatalf("provenance not recorded: %+v", pr)
	}
	recipe := pr.Recipe()
	if recipe.Tag != "qwen4exp" || recipe.RouteArch != "qwen4exp" || recipe.GitURL != pr.GitURL {
		t.Fatalf("recipe from PR = %+v", recipe)
	}
	named := pr.RecipeForModel("Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf")
	if named.Tag != "qwen3-8-flash-next" || named.RouteArch != "qwen4exp" || named.Name != named.Tag {
		t.Fatalf("model-named recipe = %+v", named)
	}
}

func TestSearchArchForkPRsSkipsMergedAndUnknownArch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/check-runs") {
			io.WriteString(w, `{"total_count":1,"check_runs":[{"status":"completed","conclusion":"success"}]}`)
			return
		}
		switch r.URL.Path {
		case "/search/issues":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"items": []map[string]any{{
					"title":    "model: add qwen4exp",
					"html_url": "https://github.com/ggml-org/llama.cpp/pull/1",
					"body":     "qwen4exp",
				}},
			})
		case "/repos/ggml-org/llama.cpp/pulls/1":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"number": 1, "title": "merged", "html_url": "https://github.com/ggml-org/llama.cpp/pull/1",
				"state": "closed", "merged": true,
				"head": map[string]any{
					"ref": "main", "sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
					"repo": map[string]any{"clone_url": "https://github.com/example/llama.cpp.git"},
				},
			})
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	got, err := SearchArchForkPRs(context.Background(), "qwen4exp", server.URL, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("merged PR became an install candidate: %+v", got)
	}
	if got, err := SearchArchForkPRs(context.Background(), "qwen4exp; rm -rf /", server.URL, server.Client()); err != nil || got != nil {
		t.Fatalf("unsafe arch queried github: got=%v err=%v", got, err)
	}
}

func TestDiscoveredForkRecipeNeverUsesMainlineTag(t *testing.T) {
	pr := ArchForkPR{Arch: "llama.cpp", Number: 7, Title: "x", GitURL: "https://github.com/unslothai/llama.cpp.git", Branch: "x", Commit: "cccccccccccccccccccccccccccccccccccccccc"}
	recipe := pr.Recipe()
	if recipe.Tag == "llama" || recipe.Tag == "llama.cpp" || recipe.Tag != "pr-7" {
		t.Fatalf("reserved arch became backend tag %q", recipe.Tag)
	}
	src := IsolatedForkSourceDir("/app", pr.GitURL, "qwen4exp/qwen3.8-flash-next")
	mainline := MainlineSourceDir("/app")
	if src == mainline || !IsIsolatedForkSourceDir(src) {
		t.Fatalf("fork src %q collides with mainline %q", src, mainline)
	}
	if filepath.Base(src) != "fork-llama.cpp-qwen3.8-flash-next" {
		t.Fatalf("isolated src = %q", src)
	}
	namedSrc := IsolatedForkSourceDir("/app", pr.GitURL, "x", "qwen3-8-flash-next")
	if filepath.Base(namedSrc) != "fork-qwen3-8-flash-next" {
		t.Fatalf("model-named src = %q", namedSrc)
	}
}

func TestBackendTagFromModelDropsShardAndQuant(t *testing.T) {
	got := BackendTagFromModel("Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf", "qwen4exp", 27742)
	if got != "qwen3-8-flash-next" {
		t.Fatalf("tag=%q, want qwen3-8-flash-next", got)
	}
	if got := BackendTagFromModel("", "qwen4exp", 27742); got != "qwen4exp" {
		t.Fatalf("arch fallback tag=%q", got)
	}
	if got := BackendTagFromModel("", "llama.cpp", 9); got != "pr-9" {
		t.Fatalf("reserved fallback tag=%q", got)
	}
}

func TestSearchArchForkPRsRejectsNonGitHubHead(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/check-runs") {
			io.WriteString(w, `{"total_count":1,"check_runs":[{"status":"completed","conclusion":"success"}]}`)
			return
		}
		switch r.URL.Path {
		case "/search/issues":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"items": []map[string]any{{
					"title":    "qwen4exp support",
					"html_url": "https://github.com/ggml-org/llama.cpp/pull/9",
					"body":     "qwen4exp",
				}},
			})
		case "/repos/ggml-org/llama.cpp/pulls/9":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"number": 9, "title": "qwen4exp", "html_url": "https://github.com/ggml-org/llama.cpp/pull/9",
				"state": "open", "merged": false,
				"head": map[string]any{
					"ref": "x", "sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
					"repo": map[string]any{"clone_url": "https://evil.example/llama.cpp.git"},
				},
			})
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	got, err := SearchArchForkPRs(context.Background(), "qwen4exp", server.URL, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("non-github clone URL was accepted: %+v", got)
	}
}

func TestSearchArchForkPRsPrefersAddTitleOverLaterFix(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/check-runs") {
			io.WriteString(w, `{"total_count":1,"check_runs":[{"status":"completed","conclusion":"success"}]}`)
			return
		}
		switch r.URL.Path {
		case "/search/issues":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"items": []map[string]any{
					githubSearchItem(27774, "fix: Qwen3.8-Flash-Next (qwen4exp) support + rotated-KV QSA fix", "depends on qwen4exp"),
					githubSearchItem(27742, "model: add Qwen3.8-Flash-Next (qwen4exp)", "Adds GGUF architecture qwen4exp."),
				},
			})
		case "/repos/ggml-org/llama.cpp/pulls/27774":
			writeOpenPR(w, 27774, "fix: Qwen3.8-Flash-Next (qwen4exp) support + rotated-KV QSA fix", "fix", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
		case "/repos/ggml-org/llama.cpp/pulls/27742":
			writeOpenPR(w, 27742, "model: add Qwen3.8-Flash-Next (qwen4exp)", "qwen4exp/qwen3.8-flash-next", "250b61446efc91e3a179c8677956f2667c8fbda0")
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	got, err := SearchArchForkPRs(context.Background(), "qwen4exp", server.URL, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("candidates=%d, want 2: %+v", len(got), got)
	}
	if got[0].Number != 27742 || got[1].Number != 27774 {
		t.Fatalf("rank order = #%d then #%d, want add-PR 27742 before fix-PR 27774", got[0].Number, got[1].Number)
	}
}

func TestSearchArchForksPrefersHuggingFaceCitedPR(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/check-runs") {
			io.WriteString(w, `{"total_count":1,"check_runs":[{"status":"completed","conclusion":"success"}]}`)
			return
		}
		switch {
		case r.URL.Path == "/search/issues":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"items": []map[string]any{
					githubSearchItem(27774, "fix: Qwen3.8-Flash-Next (qwen4exp) support", "qwen4exp"),
				},
			})
		case r.URL.Path == "/repos/ggml-org/llama.cpp/pulls/27774":
			writeOpenPR(w, 27774, "fix: Qwen3.8-Flash-Next (qwen4exp) support", "fix", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
		case r.URL.Path == "/repos/ggml-org/llama.cpp/pulls/27742":
			writeOpenPR(w, 27742, "model: add Qwen3.8-Flash-Next (qwen4exp)", "qwen4exp/qwen3.8-flash-next", "250b61446efc91e3a179c8677956f2667c8fbda0")
		case r.URL.Path == "/unsloth/Qwen3.8-Flash-Next-GGUF/raw/main/README.md":
			_, _ = w.Write([]byte("To run, please use our llama.cpp PR https://github.com/ggml-org/llama.cpp/pull/27742 or Unsloth Desktop.\n"))
		case strings.HasSuffix(r.URL.Path, "/raw/main/README.md"):
			http.NotFound(w, r)
		case r.URL.Path == "/api/models":
			_ = json.NewEncoder(w).Encode([]map[string]any{})
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	got, err := SearchArchForks(context.Background(), ArchForkSearch{
		Arch:        "qwen4exp",
		QuantizedBy: "unsloth",
		Name:        "Qwen3.8-Flash-Next",
		Basename:    "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
	}, ForkSearchClient{HTTP: server.Client(), GitHubBase: server.URL, HFBase: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || got[0].Number != 27742 || !got[0].Cited {
		t.Fatalf("want Hugging Face-cited PR 27742 first, got %+v", got)
	}
}

func TestSearchArchForksIgnoresNonOfficialPullLinks(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/check-runs") {
			io.WriteString(w, `{"total_count":1,"check_runs":[{"status":"completed","conclusion":"success"}]}`)
			return
		}
		switch {
		case r.URL.Path == "/search/issues":
			_ = json.NewEncoder(w).Encode(map[string]any{"items": []map[string]any{}})
		case r.URL.Path == "/unsloth/Qwen3.8-Flash-Next-GGUF/raw/main/README.md":
			_, _ = w.Write([]byte(`
https://evil.example/llama.cpp/pull/9
https://github.com/evil/llama.cpp/pull/9
https://github.com/ggml-org/llama.cpp/pull/27742
`))
		case r.URL.Path == "/repos/ggml-org/llama.cpp/pulls/27742":
			writeOpenPR(w, 27742, "model: add qwen4exp", "qwen4exp/qwen3.8-flash-next", "250b61446efc91e3a179c8677956f2667c8fbda0")
		case r.URL.Path == "/repos/ggml-org/llama.cpp/pulls/9":
			t.Fatal("followed a non-official llama.cpp PR from the model card")
		case strings.HasSuffix(r.URL.Path, "/raw/main/README.md"):
			http.NotFound(w, r)
		case r.URL.Path == "/api/models":
			_ = json.NewEncoder(w).Encode([]map[string]any{})
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	got, err := SearchArchForks(context.Background(), ArchForkSearch{
		Arch: "qwen4exp", QuantizedBy: "unsloth", Basename: "Qwen3.8-Flash-Next",
	}, ForkSearchClient{HTTP: server.Client(), GitHubBase: server.URL, HFBase: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Number != 27742 || !got[0].Cited {
		t.Fatalf("want only official cited PR 27742, got %+v", got)
	}
}

func TestSearchArchForksUsesHubSearchWhenMetadataMissing(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/check-runs") {
			io.WriteString(w, `{"total_count":1,"check_runs":[{"status":"completed","conclusion":"success"}]}`)
			return
		}
		switch {
		case r.URL.Path == "/search/issues":
			_ = json.NewEncoder(w).Encode(map[string]any{"items": []map[string]any{}})
		case r.URL.Path == "/api/models":
			if r.URL.Query().Get("search") != "qwen4exp" || r.URL.Query().Get("filter") != "gguf" {
				t.Fatalf("hub search = %s", r.URL.RawQuery)
			}
			_ = json.NewEncoder(w).Encode([]map[string]any{{"id": "unsloth/Qwen3.8-Flash-Next-GGUF"}})
		case r.URL.Path == "/unsloth/Qwen3.8-Flash-Next-GGUF/raw/main/README.md":
			_, _ = w.Write([]byte("Use https://github.com/ggml-org/llama.cpp/pull/27742\n"))
		case r.URL.Path == "/repos/ggml-org/llama.cpp/pulls/27742":
			writeOpenPR(w, 27742, "model: add qwen4exp", "qwen4exp/qwen3.8-flash-next", "250b61446efc91e3a179c8677956f2667c8fbda0")
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	got, err := SearchArchForks(context.Background(), ArchForkSearch{Arch: "qwen4exp"}, ForkSearchClient{
		HTTP: server.Client(), GitHubBase: server.URL, HFBase: server.URL,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Number != 27742 || !got[0].Cited {
		t.Fatalf("hub-search cited PR = %+v", got)
	}
}

func TestHFGGUFCardReposFromMetadata(t *testing.T) {
	got := hfGGUFCardRepos(ArchForkSearch{
		QuantizedBy: "unsloth",
		Basename:    "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
		Name:        "Qwen3.8-Flash-Next",
	})
	if len(got) == 0 || got[0] != "unsloth/Qwen3.8-Flash-Next-GGUF" {
		t.Fatalf("card repos = %v, want unsloth/Qwen3.8-Flash-Next-GGUF first", got)
	}
	if got := hfGGUFCardRepos(ArchForkSearch{QuantizedBy: "unsloth/../evil", Basename: "x"}); got != nil {
		t.Fatalf("path-like quantized_by became a card repo: %v", got)
	}
}

func githubSearchItem(number int, title, body string) map[string]any {
	return map[string]any{
		"title":    title,
		"html_url": fmt.Sprintf("https://github.com/ggml-org/llama.cpp/pull/%d", number),
		"body":     body,
	}
}

func writeOpenPR(w http.ResponseWriter, number int, title, branch, sha string) {
	_ = json.NewEncoder(w).Encode(map[string]any{
		"number": number, "title": title, "body": title,
		"html_url": fmt.Sprintf("https://github.com/ggml-org/llama.cpp/pull/%d", number),
		"state":    "open", "merged": false,
		"head": map[string]any{
			"ref": branch, "sha": sha,
			"repo": map[string]any{"clone_url": "https://github.com/unslothai/llama.cpp.git"},
		},
	})
}

func TestSearchArchForksRejectsUnrelatedCardCitedPR(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/check-runs") {
			io.WriteString(w, `{"total_count":1,"check_runs":[{"status":"completed","conclusion":"success"}]}`)
			return
		}
		switch {
		case r.URL.Path == "/search/issues":
			_ = json.NewEncoder(w).Encode(map[string]any{"items": []map[string]any{}})
		case r.URL.Path == "/publisher/model-GGUF/raw/main/README.md":
			_, _ = w.Write([]byte("Use https://github.com/ggml-org/llama.cpp/pull/77\n"))
		case strings.HasSuffix(r.URL.Path, "/raw/main/README.md"):
			http.NotFound(w, r)
		case r.URL.Path == "/repos/ggml-org/llama.cpp/pulls/77":
			writeOpenPR(w, 77, "docs: improve build instructions", "docs", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
		case r.URL.Path == "/api/models":
			_ = json.NewEncoder(w).Encode([]map[string]any{})
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	got, err := SearchArchForks(context.Background(), ArchForkSearch{
		Arch: "qwen4exp", QuantizedBy: "publisher", Basename: "model",
	}, ForkSearchClient{HTTP: server.Client(), GitHubBase: server.URL, HFBase: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("unrelated cited PR became an install candidate: %+v", got)
	}
}

func TestSearchArchForkPRsRejectsMalformedCommitSHA(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/check-runs") {
			io.WriteString(w, `{"total_count":1,"check_runs":[{"status":"completed","conclusion":"success"}]}`)
			return
		}
		switch r.URL.Path {
		case "/search/issues":
			_ = json.NewEncoder(w).Encode(map[string]any{"items": []map[string]any{
				githubSearchItem(88, "model: add qwen4exp", "qwen4exp"),
			}})
		case "/repos/ggml-org/llama.cpp/pulls/88":
			writeOpenPR(w, 88, "model: add qwen4exp", "feature", strings.Repeat("z", 40))
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	got, err := SearchArchForkPRs(context.Background(), "qwen4exp", server.URL, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("non-hex commit became an install candidate: %+v", got)
	}
}

func TestSearchArchForkPRsCapsDetailFetches(t *testing.T) {
	detailFetches := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/check-runs") {
			io.WriteString(w, `{"total_count":1,"check_runs":[{"status":"completed","conclusion":"success"}]}`)
			return
		}
		if r.URL.Path == "/search/issues" {
			items := make([]map[string]any, 10)
			for i := range items {
				items[i] = githubSearchItem(100+i, "model: add qwen4exp", "qwen4exp")
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"items": items})
			return
		}
		detailFetches++
		number, _ := strconv.Atoi(filepath.Base(r.URL.Path))
		writeOpenPR(w, number, "docs only", "docs", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	}))
	defer server.Close()

	got, err := SearchArchForkPRs(context.Background(), "qwen4exp", server.URL, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 || detailFetches != maxGitHubPRFetches {
		t.Fatalf("got=%+v detail fetches=%d, want cap %d", got, detailFetches, maxGitHubPRFetches)
	}
}

// S-04: the prompt shows who wrote the PR, their relation to llama.cpp and
// the head commit's CI result before ggrun builds it.
func TestForkPRProvenanceAndCIState(t *testing.T) {
	pr := ArchForkPR{Author: "someone", AuthorAssociation: "FIRST_TIME_CONTRIBUTOR", CIState: "failing"}
	if got := pr.Provenance(); got != "by someone (first time contributor to llama.cpp), CI failing" {
		t.Fatalf("provenance %q", got)
	}
	if got := (ArchForkPR{}).Provenance(); !strings.Contains(got, "unknown author") || !strings.Contains(got, "CI unknown") {
		t.Fatalf("empty provenance %q", got)
	}
	sha := "250b61446efc91e3a179c8677956f2667c8fbda0"
	for body, want := range map[string]string{
		`{"total_count":2,"check_runs":[{"status":"completed","conclusion":"success"},{"status":"completed","conclusion":"failure"}]}`: "failing",
		`{"total_count":2,"check_runs":[{"status":"completed","conclusion":"success"},{"status":"in_progress","conclusion":null}]}`:    "running",
		`{"total_count":2,"check_runs":[{"status":"completed","conclusion":"success"},{"status":"completed","conclusion":"skipped"}]}`: "passing",
		`{"total_count":0,"check_runs":[]}`: "none",
		`not json`:                          "unknown",
	} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, body) }))
		if got := fetchCIState(context.Background(), srv.Client(), srv.URL, sha); got != want {
			t.Errorf("%s: got %s, want %s", body, got, want)
		}
		srv.Close()
	}
	if got := fetchCIState(context.Background(), http.DefaultClient, "http://127.0.0.1:1", "not-a-sha"); got != "unknown" {
		t.Errorf("bad sha: %s", got)
	}
}
