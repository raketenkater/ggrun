package claudeauto

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

const severityStage1Body = `{"model":"local","max_tokens":64,"stop_sequences":["</severity>"],"system":[{"type":"text","text":"You are a security monitor for autonomous AI coding agents. Review it."}],"messages":[{"role":"user","content":"Respond with <severity>N</severity> ONLY."}]}`

func TestSeverityVariantRequestDetection(t *testing.T) {
	for name, tc := range map[string]struct {
		body string
		want bool
	}{
		"stage1 stop sequence": {severityStage1Body, true},
		"stage2 prompt only":   {`{"messages":[{"role":"user","content":"then respond with <severity>N</severity>, plus"}]}`, true},
		"stage2 escaped":       {`{"messages":[{"role":"user","content":"<severity>N</severity>"}]}`, true},
		"block stage1":         {`{"stop_sequences":["</block>"],"messages":[{"role":"user","content":"begin with <block>"}]}`, false},
		"block stage2":         {`{"messages":[{"role":"user","content":"respond with <block>"}]}`, false},
		"not json":             {`nope`, false},
	} {
		if got := severityVariantRequest([]byte(tc.body)); got != tc.want {
			t.Errorf("%s: got %v want %v", name, got, tc.want)
		}
	}
}

func TestValidReviewerVerdictFormats(t *testing.T) {
	cases := []struct {
		name     string
		text     string
		severity bool
		want     bool
	}{
		{"block no", "<block>no</block>", false, true},
		{"block stripped", "<block>yes", false, true},
		{"block two tags", "<block>no</block><block>yes</block>", false, false},
		{"severity ignored for block request", "<severity>5", false, false},
		{"severity stripped", "<severity>95", true, true},
		{"severity stripped zero", " <severity>0\n", true, true},
		{"severity closed", "<severity>7</severity>", true, true},
		{"severity closed with category", "<severity>72</severity><category>Data Exfiltration</category>", true, true},
		{"severity 100", "<severity>100", true, true},
		{"severity above 100", "<severity>101", true, false},
		{"severity four digits", "<severity>0050", true, false},
		{"severity prose partial", "safe: <severity>5", true, false},
		{"severity prose before closed", "Looking at it: <severity>5</severity>", true, true}, // same leniency as block: one complete tag
		{"severity two tags", "<severity>5</severity><severity>90</severity>", true, false},
		{"severity stripped then tag", "<severity>5<severity>90", true, false},
		{"severity non digits", "<severity>high", true, false},
		{"severity empty value", "<severity>", true, false},
		{"severity empty value closed", "<severity></severity>", true, false},
		{"severity negative", "<severity>-1</severity>", true, false},
		{"empty", "", true, false},
		{"thinking only", "<thinking>hmm</thinking>", true, false},
		{"block still valid in severity request", "<block>no</block>", true, true},
	}
	for _, c := range cases {
		if got := validReviewerVerdictFor(c.text, c.severity); got != c.want {
			t.Errorf("%s: validReviewerVerdictFor(%q, %v) = %v, want %v", c.name, c.text, c.severity, got, c.want)
		}
	}
}

func TestClassifierVerdictSeverity(t *testing.T) {
	js := http.Header{"Content-Type": {"application/json"}}
	sse := http.Header{"Content-Type": {"text/event-stream"}}
	mk := func(text string) []byte { return []byte(`{"content":[{"type":"text","text":"` + text + `"}]}`) }
	for _, c := range []struct {
		text     string
		severity bool
		want     string
	}{
		{"<severity>0", true, verdictAllow},
		{"<severity>49</severity>", true, verdictAllow},
		{"<severity>51", true, verdictBlock},
		{"<severity>95</severity><category>Arbitrary Code Execution</category>", true, verdictBlock},
		{"<severity>50", true, "severity:50"},
		{"<severity>95", false, verdictInvalid},
		{"<severity>x", true, verdictInvalid},
		{"<block>yes", true, verdictBlock},
	} {
		if got := classifierVerdictFor(200, js, mk(c.text), c.severity); got != c.want {
			t.Errorf("%q severity=%v: %s, want %s", c.text, c.severity, got, c.want)
		}
	}
	body := "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"<severity>\"}}\n\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"88\"}}\n\n"
	if got := classifierVerdictFor(200, sse, []byte(body), true); got != verdictBlock {
		t.Errorf("sse severity: %s", got)
	}
	if got := classifierVerdict(200, js, mk("<severity>3")); got != verdictInvalid {
		t.Errorf("legacy classifierVerdict must stay block-only: %s", got)
	}
}

func TestReviewerAcceptsSeverityVerdict(t *testing.T) {
	var mainHits, reviewerHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	reviewerBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reviewerHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"type":"message","content":[{"type":"text","text":"<severity>95"}],"stop_reason":"stop_sequence","stop_sequence":"</severity>","usage":{"output_tokens":4}}`))
	}))
	t.Cleanup(reviewerBackend.Close)

	router, path := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	resp, err := http.Post(router.URL()+"/v1/messages", "application/json", strings.NewReader(severityStage1Body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	payload, _ := io.ReadAll(resp.Body)
	if got := reviewerJSONText(payload); got != "<severity>95" {
		t.Fatalf("client verdict = %q", got)
	}
	if reviewerHits.Load() != 1 || mainHits.Load() != 0 {
		t.Fatalf("reviewer=%d main=%d; want reviewer-only", reviewerHits.Load(), mainHits.Load())
	}
	rec := waitForRecords(t, path, 1)[0]
	if rec.Route != routeReviewerStopStripped || rec.Verdict != verdictBlock {
		t.Fatalf("route=%q verdict=%q", rec.Route, rec.Verdict)
	}
}

func TestReviewerRejectsInvalidSeverityVerdict(t *testing.T) {
	var mainHits, reviewerHits atomic.Int64
	mainBackend := countingBackend(t, &mainHits)
	reviewerBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reviewerHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"type":"message","content":[{"type":"text","text":"looks fine <severity>5"}]}`))
	}))
	t.Cleanup(reviewerBackend.Close)
	router, _ := newTestRouterPair(t, mainBackend.URL, reviewerBackend.URL, 1)
	resp, err := http.Post(router.URL()+"/v1/messages", "application/json", strings.NewReader(severityStage1Body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if reviewerHits.Load() != 1 || mainHits.Load() != 1 {
		t.Fatalf("reviewer=%d main=%d; want fallback to main", reviewerHits.Load(), mainHits.Load())
	}
}
