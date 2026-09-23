package controller

import (
	"os"
	"sync"
	"testing"
	"time"
)

func TestProfileRequiresCompleteVerificationBeforePromotion(t *testing.T) {
	store := Store{CacheDir: t.TempDir()}
	candidate, err := store.Begin(Profile{Scope: "scope", ModelIdentity: "model", BackendIdentity: "backend", ArgsHash: "args"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Transition("scope", candidate.ID, StateActive, "", ""); err == nil {
		t.Fatal("HTTP proposal promoted without verification")
	}
	for _, state := range []State{StateAllocationVerified, StateLoadHealthy, StateFunctionalVerified, StateCacheVerified, StatePerformanceVerified, StateActive} {
		if _, err := store.Transition("scope", candidate.ID, state, "ok", "test"); err != nil {
			t.Fatalf("transition to %s: %v", state, err)
		}
	}
	if !store.IsActive("scope", "args") {
		t.Fatal("fully verified profile was not active")
	}
}

func TestRejectedCandidatePreservesLastKnownGood(t *testing.T) {
	store := Store{CacheDir: t.TempDir()}
	active, _ := store.Begin(Profile{Scope: "scope", ArgsHash: "good"})
	for _, state := range []State{StateAllocationVerified, StateLoadHealthy, StateFunctionalVerified, StateCacheVerified, StatePerformanceVerified, StateActive} {
		if _, err := store.Transition("scope", active.ID, state, "", ""); err != nil {
			t.Fatal(err)
		}
	}
	bad, _ := store.Begin(Profile{Scope: "scope", ArgsHash: "bad"})
	if _, err := store.Transition("scope", bad.ID, StateRejected, "cache regression", "canary"); err != nil {
		t.Fatal(err)
	}
	if !store.IsActive("scope", "good") {
		t.Fatal("rejected candidate replaced last-known-good")
	}
}

func TestRuntimeFailureRevokesOnlyExactActiveProfile(t *testing.T) {
	store := Store{CacheDir: t.TempDir()}
	active, _ := store.Begin(Profile{Scope: "scope", ArgsHash: "oom-args"})
	for _, state := range []State{StateAllocationVerified, StateLoadHealthy, StateFunctionalVerified, StateCacheVerified, StatePerformanceVerified, StateActive} {
		if _, err := store.Transition("scope", active.ID, state, "", ""); err != nil {
			t.Fatal(err)
		}
	}
	if rejected, err := store.RejectActiveIfMatch("scope", "other-args", "OOM", "runtime"); err != nil || rejected {
		t.Fatalf("different argv revoked the LKG: rejected=%t err=%v", rejected, err)
	}
	if rejected, err := store.RejectActiveIfMatch("scope", "oom-args", "CUDA OOM after health", "runtime-oom"); err != nil || !rejected {
		t.Fatalf("exact runtime-failed profile was not revoked: rejected=%t err=%v", rejected, err)
	}
	record, err := store.Load("scope")
	if err != nil {
		t.Fatal(err)
	}
	if record.Active != nil || len(record.History) == 0 || record.History[len(record.History)-1].State != StateRejected {
		t.Fatalf("runtime rejection was not persisted: %#v", record)
	}
}

func TestConcurrentProfileUpdatesStayValidJSON(t *testing.T) {
	dir := t.TempDir()
	store := Store{CacheDir: dir}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = store.Begin(Profile{Scope: "scope", ArgsHash: ScopeKey(time.Now().String())})
		}()
	}
	wg.Wait()
	if _, err := store.Load("scope"); err != nil {
		t.Fatalf("concurrent writers corrupted profile: %v", err)
	}
	if _, err := os.Stat(store.path("scope") + ".lock"); !os.IsNotExist(err) {
		t.Fatalf("profile lock leaked: %v", err)
	}
}

func TestReachedFunctionalSeesDegradedButNotRejected(t *testing.T) {
	store := Store{CacheDir: t.TempDir()}
	p, _ := store.Begin(Profile{Scope: "scope", ArgsHash: "args"})
	for _, state := range []State{StateAllocationVerified, StateLoadHealthy, StateFunctionalVerified, StateDegraded} {
		if _, err := store.Transition("scope", p.ID, state, "", ""); err != nil {
			t.Fatalf("transition to %s: %v", state, err)
		}
	}
	if store.IsActive("scope", "args") || !store.ReachedFunctional("scope", "args") {
		t.Fatal("a degraded profile that answered the functional canary was not recognized")
	}
	if store.ReachedFunctional("scope", "other-args") {
		t.Fatal("a different argv inherited functional evidence")
	}
	q, _ := store.Begin(Profile{Scope: "scope", ArgsHash: "args"})
	if _, err := store.Transition("scope", q.ID, StateRejected, "failed", "canary"); err != nil {
		t.Fatal(err)
	}
	if store.ReachedFunctional("scope", "args") {
		t.Fatal("a rejected profile counted as functional")
	}
}
