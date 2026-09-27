package main

import (
	"os"
	"runtime"
	"strings"
)

// displayCommand renders argv for a person to copy and replay in the shell the
// project documents: POSIX sh, or PowerShell on Windows (issue #74). Never use
// it for comparison, hashing or logs that must match across platforms; that is
// formatCommand.
func displayCommand(args []string) string {
	return displayCommandFor(runtime.GOOS, os.Getenv("MSYSTEM"), nil, args)
}

// displayCommandWithEnv is displayCommand with K=V environment assignments
// that apply to the command.
func displayCommandWithEnv(env, args []string) string {
	return displayCommandFor(runtime.GOOS, os.Getenv("MSYSTEM"), env, args)
}

// displayCommandFor is the pure renderer. Off Windows, and on Windows under
// Git Bash/MSYS (MSYSTEM set), the output is byte-identical to formatCommand
// with a raw K=V prefix, so POSIX users can still replay it.
func displayCommandFor(goos, msystem string, env, args []string) string {
	if goos != "windows" || msystem != "" {
		s := formatCommand(args)
		if len(env) > 0 {
			s = strings.Join(env, " ") + " " + s
		}
		return s
	}
	var b strings.Builder
	for _, kv := range env {
		k, v, _ := strings.Cut(kv, "=")
		b.WriteString("$env:" + k + "=" + powerShellLiteral(v) + "; ")
	}
	if len(args) > 0 {
		quoted := make([]string, len(args))
		for i, a := range args {
			quoted[i] = powerShellArg(a)
		}
		b.WriteString("& " + strings.Join(quoted, " "))
	}
	return b.String()
}

// powerShellLiteral returns a single-quoted verbatim string. PowerShell also
// treats U+2018..U+201B as single quotes, so every quote character is doubled.
func powerShellLiteral(s string) string {
	var b strings.Builder
	b.WriteByte('\'')
	for _, r := range s {
		if isPowerShellSingleQuote(r) {
			b.WriteRune(r)
		}
		b.WriteRune(r)
	}
	b.WriteByte('\'')
	return b.String()
}

func isPowerShellSingleQuote(r rune) bool {
	return r == '\'' || r == '\u2018' || r == '\u2019' || r == '\u201A' || r == '\u201B'
}

// powerShellArg leaves an argument bare only when PowerShell passes it to a
// native command unchanged: ASCII letters and digits plus -_.+=:/\, and for a
// dash-led argument only letters, digits and dashes (PowerShell/PowerShell#6291
// splits -a.b and -a:b). A bare "-" or "--" is quoted too, since PowerShell's
// binder can consume "--". Everything else is single-quoted, which is harmless.
// The native binder drops any string argument whose value is "--%" (the
// stop-parsing marker), quoted or not, so that one is passed as an array.
func powerShellArg(a string) string {
	if a == "--%" {
		return "@(" + powerShellLiteral(a) + ")"
	}
	if a == "" || a == "-" || a == "--" {
		return powerShellLiteral(a)
	}
	dash := a[0] == '-'
	for _, r := range a {
		if r >= 0x80 {
			return powerShellLiteral(a)
		}
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' {
			continue
		}
		if !dash && strings.ContainsRune(`_.+=:/\`, r) {
			continue
		}
		return powerShellLiteral(a)
	}
	return a
}
