package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// logRetainPerKind is how many of the newest files ggrun keeps for each of its
// own log kinds (backend server, reviewer, router request logs) in a log dir.
// Older files are removed only once they have also been idle for
// logRetainMinIdle, so a long-running server's open log is not deleted by a
// burst of newer launches (experiments, calibration).
const (
	logRetainPerKind = 20
	logRetainMinIdle = 7 * 24 * time.Hour
)

// ggrunLogPatterns are the only names rotation may delete.
var ggrunLogPatterns = []string{
	"ggrun-claude-server-v2-*.log",
	"ggrun-claude-reviewer-*.log",
	"ggrun-claude-requests-*.jsonl",
}

// openPrivateLog opens path for append with mode 0600, also tightening a file
// an older ggrun created world-readable. Windows ACLs are left alone.
func openPrivateLog(path string) (*os.File, error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err == nil && runtime.GOOS != "windows" {
		_ = f.Chmod(0o600)
	}
	return f, err
}

// rotateGgrunLogs keeps the newest `keep` files of each ggrun log kind in dir
// and removes older ones. Files not matching ggrunLogPatterns are never touched.
func rotateGgrunLogs(dir string, keep int) {
	if dir == "" || keep < 1 {
		return
	}
	for _, pat := range ggrunLogPatterns {
		matches, err := filepath.Glob(filepath.Join(dir, pat))
		if err != nil || len(matches) <= keep {
			continue
		}
		type entry struct {
			path string
			mod  int64
		}
		var files []entry
		for _, m := range matches {
			if info, err := os.Lstat(m); err == nil && info.Mode().IsRegular() {
				files = append(files, entry{m, info.ModTime().UnixNano()})
			}
		}
		sort.Slice(files, func(i, j int) bool {
			if files[i].mod != files[j].mod {
				return files[i].mod > files[j].mod
			}
			return files[i].path > files[j].path
		})
		idleBefore := time.Now().Add(-logRetainMinIdle).UnixNano()
		for i := keep; i < len(files); i++ {
			if files[i].mod < idleBefore {
				_ = os.Remove(files[i].path)
			}
		}
	}
}

// headlessMaxLogVerbosity is the highest backend log level headless accepts:
// at 5 and above llama.cpp-style backends log full request bodies.
const headlessMaxLogVerbosity = 4

// validateHeadlessLogVerbosity refuses a backend log verbosity that would write
// request bodies (the owner's prompts) to the backend log.
func validateHeadlessLogVerbosity(extra []string) error {
	refuse := func(src string, level int) error {
		return fmt.Errorf("--headless refuses backend log verbosity %d from %s: at level %d or higher the backend logs full request bodies; use %d or lower",
			level, src, headlessMaxLogVerbosity+1, headlessMaxLogVerbosity)
	}
	check := func(src, val string) error {
		val = strings.TrimSpace(val)
		var n int
		if _, err := fmt.Sscanf(val, "%d", &n); err != nil || fmt.Sprint(n) != strings.TrimPrefix(val, "+") {
			return fmt.Errorf("--headless cannot verify backend log verbosity %q from %s; use an integer %d or lower", val, src, headlessMaxLogVerbosity)
		}
		if n > headlessMaxLogVerbosity {
			return refuse(src, n)
		}
		return nil
	}
	if v := os.Getenv("LLAMA_ARG_LOG_VERBOSITY"); strings.TrimSpace(v) != "" {
		if err := check("env LLAMA_ARG_LOG_VERBOSITY", v); err != nil {
			return err
		}
	}
	for i := 0; i < len(extra); i++ {
		a := extra[i]
		switch a {
		case "-v", "--verbose", "--log-verbose":
			return fmt.Errorf("--headless refuses %s: it sets the maximum backend log verbosity, which logs full request bodies", a)
		}
		for _, f := range []string{"-lv", "--log-verbosity", "--verbosity"} {
			if a == f {
				if i+1 >= len(extra) {
					return fmt.Errorf("--headless cannot verify backend log verbosity: %s has no value", f)
				}
				if err := check(f, extra[i+1]); err != nil {
					return err
				}
			} else if strings.HasPrefix(a, f+"=") {
				if err := check(f, strings.TrimPrefix(a, f+"=")); err != nil {
					return err
				}
			}
		}
	}
	return nil
}
