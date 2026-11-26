package claudeauto

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// W7/W0: records carry who sent a request, what it found on arrival and, for
// Auto-mode classifier requests, which verdict came back; never content.
func TestRecordEncodingCarriesQueueAndVerdictFields(t *testing.T) {
	rec := RequestRecord{Route: routeMain, Kind: kindClassifier, Verdict: verdictBlock, Client: "claude-cli", QueuedAtArrival: 3, InServiceAtArrival: 1}
	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"kind":"classifier"`, `"verdict":"block"`, `"client":"claude-cli"`, `"queued_at_arrival":3`, `"in_service_at_arrival":1`} {
		if !bytes.Contains(b, []byte(want)) {
			t.Errorf("record JSON lacks %s: %s", want, b)
		}
	}
	plain, _ := json.Marshal(RequestRecord{Route: routeMain})
	if bytes.Contains(plain, []byte(`"verdict"`)) || bytes.Contains(plain, []byte(`"kind"`)) {
		t.Errorf("main-model record carries classifier fields: %s", plain)
	}
}

// The record type must never grow a field that could carry request or
// response content.
func TestRequestRecordHasNoContentField(t *testing.T) {
	banned := []string{"body", "prompt", "content", "text", "message", "system", "tool", "input_text", "transcript"}
	var walk func(reflect.Type, string)
	walk = func(rt reflect.Type, path string) {
		for i := 0; i < rt.NumField(); i++ {
			f := rt.Field(i)
			name := strings.ToLower(f.Name + " " + f.Tag.Get("json"))
			for _, b := range banned {
				if strings.Contains(name, b) {
					t.Errorf("%s%s may carry content (%q)", path, f.Name, b)
				}
			}
			switch f.Type.Kind() {
			case reflect.Struct:
				walk(f.Type, path+f.Name+".")
			case reflect.Slice, reflect.Map, reflect.Interface, reflect.Pointer:
				t.Errorf("%s%s is a %s; records hold scalars only", path, f.Name, f.Type.Kind())
			}
		}
	}
	walk(reflect.TypeOf(RequestRecord{}), "")
}

func TestClassifierVerdictClasses(t *testing.T) {
	sse := http.Header{"Content-Type": {"text/event-stream"}}
	js := http.Header{"Content-Type": {"application/json"}}
	cases := []struct {
		status int
		header http.Header
		body   string
		want   string
	}{
		{200, js, `{"content":[{"type":"text","text":"<block>no</block>"}]}`, verdictAllow},
		{200, js, `{"content":[{"type":"text","text":"<block>yes"}]}`, verdictBlock},
		{200, sse, "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"<block>\"}}\n\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"yes</block>\"}}\n\n", verdictBlock},
		{200, js, `{"content":[{"type":"text","text":"I think this is fine"}]}`, verdictInvalid},
		{200, js, `{"content":[{"type":"text","text":"<block>no</block><block>yes</block>"}]}`, verdictInvalid},
		{400, js, `{"error":"bad"}`, verdictNone},
		{200, js, ``, verdictNone},
	}
	for _, c := range cases {
		if got := classifierVerdict(c.status, c.header, []byte(c.body)); got != c.want {
			t.Errorf("%d %q: %s, want %s", c.status, c.body, got, c.want)
		}
	}
}

func TestClientFamily(t *testing.T) {
	for ua, want := range map[string]string{
		"claude-cli/2.1.325 (external, sdk-cli)": "claude-cli",
		"Go-http-client/1.1":                     "go-http-client",
		"":                                       "",
		"Weird<script>/1":                        "weirdscript",
		strings.Repeat("a", 50) + "/1.0":         strings.Repeat("a", 32),
		"ggrun-client/0.1.0-alpha.8 linux/amd64 x": "ggrun-client",
	} {
		if got := clientFamily(ua); got != want {
			t.Errorf("%q: %q, want %q", ua, got, want)
		}
	}
}

func TestExpectedWaitEstimate(t *testing.T) {
	for _, c := range []struct{ queued, limit, mean, want int64 }{
		{0, 1, 5000, 0},
		{3, 1, 0, 0},
		{3, 1, 5000, 15000},
		{3, 2, 5000, 7500},
		{1, 0, 4000, 4000},
	} {
		if got := expectedWaitMS(c.queued, c.limit, c.mean); got != c.want {
			t.Errorf("%+v: %d", c, got)
		}
	}
}

// End to end: a classifier request answered by the main model records its
// kind and verdict; the client family comes from the User-Agent.
func TestRouterRecordsClassifierVerdictAndClient(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"<block>yes</block>"}],"usage":{"input_tokens":10,"output_tokens":3}}`))
	}))
	defer backend.Close()
	router, path := newTestRouter(t, backend.URL, 1)
	body := `{"model":"local","system":"` + ClassifierMarker + ` decide","messages":[{"role":"user","content":"rm -rf /"}]}`
	req, _ := http.NewRequest("POST", router.URL()+"/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "claude-cli/2.1.325 (external, cli)")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	recs := readRecords(t, path)
	if len(recs) != 1 {
		t.Fatalf("records: %+v", recs)
	}
	r := recs[0]
	if r.Kind != kindClassifier || r.Verdict != verdictBlock || r.Client != "claude-cli" {
		t.Fatalf("record = %+v", r)
	}
}

func TestMeanServiceTimeCountsOnlyMainModelWork(t *testing.T) {
	sink := &metricsSink{}
	sink.record(RequestRecord{Route: routeMain, Status: 200, TotalMS: 10_000})
	sink.record(RequestRecord{Route: routeMain, Kind: kindClassifier, Status: 200, TotalMS: 400})
	sink.record(RequestRecord{Route: routeMain, Kind: kindUtility, Status: 200, TotalMS: 300})
	if got := sink.meanMainServiceMS(); got != 10_000 {
		t.Fatalf("mean service %d, want 10000", got)
	}
}
