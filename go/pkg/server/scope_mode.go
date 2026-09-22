package server

// scopeMode identifies the systemd instance that owns a process's memory scope.
// It is captured at launch and retained for statistics, limit updates and cleanup.
type scopeMode int

const (
	scopeUnsupported scopeMode = iota
	scopeUser
	scopeSystem
)
