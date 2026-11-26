package server

// scopeMode identifies the systemd instance that owns a process's memory scope.
// It is captured at launch and retained for statistics, limit updates and cleanup.
type scopeMode int

const (
	scopeUnsupported scopeMode = iota
	scopeUser
	scopeSystem
	// scopeNone is the explicit GGRUN_SCOPE_MODE=none opt-out for hosts without
	// systemd (containers). No scope exists to observe, resize or stop.
	scopeNone
)
