package claudeauto

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestAdmissionStatusDoesNotPollBackend(t *testing.T) {
	var requests atomic.Int64
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer backend.Close()
	router, err := StartRouter(backend.URL, backend.URL, false, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()
	for i := 0; i < 100; i++ {
		if got := router.AdmissionStatus(); got != (RouterAdmissionStatus{Limit: 2}) {
			t.Fatal(got)
		}
	}
	if requests.Load() != 0 {
		t.Fatalf("serving display consumed %d backend requests", requests.Load())
	}
	var absent *Router
	if absent.AdmissionStatus() != (RouterAdmissionStatus{}) {
		t.Fatal("absent router invented active requests")
	}
}
