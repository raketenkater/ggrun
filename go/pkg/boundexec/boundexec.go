// Package boundexec runs external probes under a deadline. A vendor tool that
// hangs (nvidia-smi on a wedged driver, a backend binary that never exits) must
// not hang ggrun with it, and a timeout must stay distinguishable from "the
// tool reported nothing".
package boundexec

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

const (
	// ProbeTimeout bounds hardware and backend probes.
	ProbeTimeout = 10 * time.Second
	// HelperTimeout bounds the GGUF metadata helper.
	HelperTimeout = 30 * time.Second
	// waitDelay bounds how long a killed command's pipes may stay open, e.g.
	// when it left a child holding stdout.
	waitDelay = 2 * time.Second
)

// TimeoutError reports a command that did not finish within its deadline.
type TimeoutError struct {
	Command string
	Wait    time.Duration
}

func (e *TimeoutError) Error() string {
	return fmt.Sprintf("%s did not answer within %s", e.Command, e.Wait)
}

// IsTimeout reports whether err is, or wraps, a *TimeoutError.
func IsTimeout(err error) bool {
	var t *TimeoutError
	return errors.As(err, &t)
}

// Command returns a command bound to timeout and a finish function. Run the
// command, then pass its error through finish: finish releases the deadline
// and turns an expiry into a *TimeoutError.
func Command(timeout time.Duration, name string, args ...string) (*exec.Cmd, func(error) error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.WaitDelay = waitDelay
	return cmd, func(err error) error {
		expired := errors.Is(ctx.Err(), context.DeadlineExceeded)
		cancel()
		if expired {
			return &TimeoutError{Command: filepath.Base(name), Wait: timeout}
		}
		return err
	}
}

// Output is exec.Command(name, args...).Output() under timeout.
func Output(timeout time.Duration, name string, args ...string) ([]byte, error) {
	cmd, finish := Command(timeout, name, args...)
	out, err := cmd.Output()
	return out, finish(err)
}

var (
	reportMu sync.Mutex
	reported = map[string]bool{}
	// reportTo is where Report writes; tests redirect it.
	reportTo io.Writer = os.Stderr
)

// Report prints one line for a timeout, naming the command, the wait and what
// ggrun does without it. Each command is reported once per process. Other
// errors are left to the caller.
func Report(err error, consequence string) {
	var t *TimeoutError
	if !errors.As(err, &t) {
		return
	}
	reportMu.Lock()
	defer reportMu.Unlock()
	if reported[t.Command] {
		return
	}
	reported[t.Command] = true
	fmt.Fprintf(reportTo, "ggrun: %s; %s\n", t.Error(), consequence)
}
