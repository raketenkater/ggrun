// Package controller owns the lifecycle of measured launch profiles. A server
// becoming HTTP-healthy is only one stage; it is not permission to replace the
// last-known-good placement until functional, cache, and performance checks
// have passed.
package controller

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const SchemaVersion = 1

type State string

const (
	StateProposed            State = "proposed"
	StateAllocationVerified  State = "allocation_verified"
	StateLoadHealthy         State = "load_healthy"
	StateFunctionalVerified  State = "functional_verified"
	StateCacheVerified       State = "cache_verified"
	StatePerformanceVerified State = "performance_verified"
	StateActive              State = "active"
	StateDegraded            State = "degraded"
	StateRejected            State = "rejected"
)

type Metric struct {
	Name   string  `json:"name"`
	Value  float64 `json:"value"`
	Unit   string  `json:"unit,omitempty"`
	Source string  `json:"source,omitempty"`
}

type Event struct {
	State    State     `json:"state"`
	At       time.Time `json:"at"`
	Reason   string    `json:"reason,omitempty"`
	Evidence string    `json:"evidence,omitempty"`
}

// Profile is immutable in identity and append-only in observations. ArgsHash
// identifies effective runtime argv without persisting user paths or prompts.
type Profile struct {
	ID               string            `json:"id"`
	Scope            string            `json:"scope"`
	State            State             `json:"state"`
	ModelIdentity    string            `json:"model_identity"`
	BackendIdentity  string            `json:"backend_identity"`
	HardwareIdentity string            `json:"hardware_identity"`
	ArgsHash         string            `json:"args_hash"`
	CreatedAt        time.Time         `json:"created_at"`
	UpdatedAt        time.Time         `json:"updated_at"`
	Properties       map[string]string `json:"properties,omitempty"`
	Metrics          []Metric          `json:"metrics,omitempty"`
	Events           []Event           `json:"events"`
}

type Record struct {
	SchemaVersion int        `json:"schema_version"`
	Scope         string     `json:"scope"`
	Active        *Profile   `json:"active,omitempty"`
	Candidate     *Profile   `json:"candidate,omitempty"`
	History       []*Profile `json:"history,omitempty"`
}

type Store struct {
	CacheDir string
	Now      func() time.Time
}

func ScopeKey(parts ...string) string {
	h := sha256.New()
	for _, part := range parts {
		_, _ = h.Write([]byte(strings.TrimSpace(part)))
		_, _ = h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil)[:16])
}

func HashArgs(args []string) string { return ScopeKey(args...) }

func (s Store) now() time.Time {
	if s.Now != nil {
		return s.Now().UTC()
	}
	return time.Now().UTC()
}

func (s Store) path(scope string) string {
	return filepath.Join(s.CacheDir, "profiles", "profile-"+scope+".json")
}

func (s Store) Load(scope string) (Record, error) {
	var record Record
	if strings.TrimSpace(s.CacheDir) == "" || strings.TrimSpace(scope) == "" {
		return record, errors.New("profile store requires cache directory and scope")
	}
	data, err := os.ReadFile(s.path(scope))
	if err != nil {
		return record, err
	}
	if err := json.Unmarshal(data, &record); err != nil {
		return Record{}, err
	}
	if record.SchemaVersion != SchemaVersion || record.Scope != scope {
		return Record{}, errors.New("profile schema or scope mismatch")
	}
	return record, nil
}

func (s Store) Begin(profile Profile) (Profile, error) {
	if profile.Scope == "" {
		return Profile{}, errors.New("candidate profile requires scope")
	}
	now := s.now()
	if profile.ID == "" {
		profile.ID = ScopeKey(profile.Scope, profile.ModelIdentity, profile.BackendIdentity,
			profile.HardwareIdentity, profile.ArgsHash, now.Format(time.RFC3339Nano))
	}
	profile.State = StateProposed
	profile.CreatedAt, profile.UpdatedAt = now, now
	profile.Events = append(profile.Events, Event{State: StateProposed, At: now})
	err := s.update(profile.Scope, func(record *Record) error {
		if record.Candidate != nil && record.Candidate.ID != profile.ID {
			record.History = appendBounded(record.History, record.Candidate)
		}
		copy := cloneProfile(profile)
		record.Candidate = &copy
		return nil
	})
	return profile, err
}

func (s Store) Transition(scope, id string, to State, reason, evidence string, metrics ...Metric) (Profile, error) {
	var result Profile
	err := s.update(scope, func(record *Record) error {
		if record.Candidate == nil || record.Candidate.ID != id {
			return errors.New("candidate profile not found")
		}
		if !validTransition(record.Candidate.State, to) {
			return fmt.Errorf("invalid profile transition %s -> %s", record.Candidate.State, to)
		}
		now := s.now()
		record.Candidate.State = to
		record.Candidate.UpdatedAt = now
		record.Candidate.Events = append(record.Candidate.Events, Event{State: to, At: now, Reason: reason, Evidence: evidence})
		record.Candidate.Metrics = append(record.Candidate.Metrics, metrics...)
		result = cloneProfile(*record.Candidate)
		if to == StateActive {
			if record.Active != nil && record.Active.ID != record.Candidate.ID {
				record.History = appendBounded(record.History, record.Active)
			}
			active := cloneProfile(*record.Candidate)
			record.Active = &active
			record.Candidate = nil
		} else if to == StateRejected {
			record.History = appendBounded(record.History, record.Candidate)
			record.Candidate = nil
		}
		return nil
	})
	return result, err
}

func (s Store) IsActive(scope, argsHash string) bool {
	record, err := s.Load(scope)
	return err == nil && record.Active != nil && record.Active.State == StateActive && record.Active.ArgsHash == argsHash
}

// RejectActiveIfMatch revokes an LKG only when the exact active argv has since
// proved unsafe at runtime. Ordinary candidate failures must keep the LKG, but
// a post-health OOM is evidence about the active profile itself.
func (s Store) RejectActiveIfMatch(scope, argsHash, reason, evidence string) (bool, error) {
	rejected := false
	err := s.update(scope, func(record *Record) error {
		if record.Active == nil || record.Active.State != StateActive || record.Active.ArgsHash != argsHash {
			return nil
		}
		now := s.now()
		record.Active.State = StateRejected
		record.Active.UpdatedAt = now
		record.Active.Events = append(record.Active.Events, Event{
			State: StateRejected, At: now, Reason: reason, Evidence: evidence,
		})
		record.History = appendBounded(record.History, record.Active)
		record.Active = nil
		rejected = true
		return nil
	})
	return rejected, err
}

func validTransition(from, to State) bool {
	if to == StateRejected || to == StateDegraded {
		return from != StateActive && from != StateRejected
	}
	next := map[State]State{
		StateProposed:            StateAllocationVerified,
		StateAllocationVerified:  StateLoadHealthy,
		StateLoadHealthy:         StateFunctionalVerified,
		StateFunctionalVerified:  StateCacheVerified,
		StateCacheVerified:       StatePerformanceVerified,
		StatePerformanceVerified: StateActive,
	}
	return next[from] == to
}

func (s Store) update(scope string, mutate func(*Record) error) error {
	if strings.TrimSpace(s.CacheDir) == "" || strings.TrimSpace(scope) == "" {
		return errors.New("profile store requires cache directory and scope")
	}
	dir := filepath.Dir(s.path(scope))
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	release, err := acquireLock(s.path(scope)+".lock", 3*time.Second)
	if err != nil {
		return err
	}
	defer release()

	record := Record{SchemaVersion: SchemaVersion, Scope: scope}
	if data, readErr := os.ReadFile(s.path(scope)); readErr == nil {
		if err := json.Unmarshal(data, &record); err != nil {
			return err
		}
		if record.SchemaVersion != SchemaVersion || record.Scope != scope {
			return errors.New("profile schema or scope mismatch")
		}
	}
	if err := mutate(&record); err != nil {
		return err
	}
	record.SchemaVersion, record.Scope = SchemaVersion, scope
	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".profile-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(append(data, '\n')); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, s.path(scope))
}

func acquireLock(path string, timeout time.Duration) (func(), error) {
	deadline := time.Now().Add(timeout)
	for {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			_, _ = fmt.Fprintf(f, "%d\n", os.Getpid())
			_ = f.Close()
			return func() { _ = os.Remove(path) }, nil
		}
		if !os.IsExist(err) {
			return nil, err
		}
		if info, statErr := os.Stat(path); statErr == nil && time.Since(info.ModTime()) > 2*time.Minute {
			_ = os.Remove(path)
			continue
		}
		if time.Now().After(deadline) {
			return nil, errors.New("timed out waiting for profile lock")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func appendBounded(history []*Profile, profile *Profile) []*Profile {
	if profile == nil {
		return history
	}
	copy := cloneProfile(*profile)
	history = append(history, &copy)
	if len(history) > 12 {
		history = history[len(history)-12:]
	}
	return history
}

func cloneProfile(profile Profile) Profile {
	copy := profile
	copy.Events = append([]Event(nil), profile.Events...)
	copy.Metrics = append([]Metric(nil), profile.Metrics...)
	if profile.Properties != nil {
		copy.Properties = make(map[string]string, len(profile.Properties))
		for key, value := range profile.Properties {
			copy.Properties[key] = value
		}
	}
	return copy
}

// ReachedFunctional reports whether the profile for this exact argv answered
// the functional canary, whatever its final state. A degraded profile serves
// correctly but could not prove cache reuse; it is not active, yet a baseline
// decision measured on it is still worth keeping. A rejected profile is not.
func (s Store) ReachedFunctional(scope, argsHash string) bool {
	record, err := s.Load(scope)
	if err != nil {
		return false
	}
	for _, p := range []*Profile{record.Active, record.Candidate} {
		if p == nil || p.ArgsHash != argsHash || p.State == StateRejected {
			continue
		}
		for _, e := range p.Events {
			if e.State == StateFunctionalVerified {
				return true
			}
		}
	}
	return false
}
