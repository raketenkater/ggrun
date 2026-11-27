package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/raketenkater/ggrun/pkg/backends"
)

// Commands that report on and clean up registered backends: `backend status`,
// `backend remove --purge` and `backend gc`. They read backends.json and the
// app home's .src tree and never select or launch a backend. The deletion rules
// live in pkg/backends/cleanup.go; this file is flag parsing, prompts and
// output.

// cleanupIO is the terminal side of a command, injectable for tests.
type cleanupIO struct {
	In       io.Reader
	Out      io.Writer
	Terminal bool
}

func defaultCleanupIO() cleanupIO {
	return cleanupIO{In: os.Stdin, Out: os.Stdout, Terminal: stdinIsTerminal()}
}

// cleanupOpts are the flags of remove and gc.
type cleanupOpts struct {
	Tag              string
	Purge            bool
	Delete           bool // gc: delete after confirming
	DryRun           bool
	Yes              bool
	ForceNotRunning  bool
	DiscardLocalWork bool
}

func parseCleanupArgs(args []string, wantTag bool) (cleanupOpts, error) {
	var o cleanupOpts
	for _, a := range args {
		switch a {
		case "--purge":
			o.Purge = true
		case "--delete":
			o.Delete = true
		case "--dry-run":
			o.DryRun = true
		case "--yes", "-y":
			o.Yes = true
		case "--force-not-running":
			o.ForceNotRunning = true
		case "--discard-local-work":
			o.DiscardLocalWork = true
		default:
			if strings.HasPrefix(a, "-") {
				return o, backendUsageError(fmt.Sprintf("unknown flag %s", a))
			}
			if !wantTag || o.Tag != "" {
				return o, backendUsageError(fmt.Sprintf("unexpected argument %q", a))
			}
			o.Tag = a
		}
	}
	if wantTag && o.Tag == "" {
		return o, backendUsageError("remove needs a tag")
	}
	return o, nil
}

func defaultCleanupEnv(o cleanupOpts) backends.CleanupEnv {
	list, err := backends.LoadStrict()
	return backends.CleanupEnv{
		AppHome:         backends.AppHome(),
		Backends:        list,
		Processes:       backends.PlatformProcessLister(),
		ForceNotRunning: o.ForceNotRunning,
		// Deleting a tree that may hold local work needs both flags; a typed "y"
		// is not enough.
		DiscardLocalWork: o.Yes && o.DiscardLocalWork,
		ManifestErr:      err,
	}
}

// confirmDeletion enforces "no deletion without a terminal or --yes".
func confirmDeletion(o cleanupOpts, io cleanupIO, question string) error {
	if o.Yes {
		return nil
	}
	if !io.Terminal {
		return errors.New("refusing to delete without a terminal; pass --yes to confirm")
	}
	fmt.Fprintf(io.Out, "%s [y/N] ", question)
	line, _ := bufio.NewReader(io.In).ReadString('\n')
	if answer := strings.ToLower(strings.TrimSpace(line)); answer != "y" && answer != "yes" {
		return errors.New("not confirmed; nothing removed")
	}
	return nil
}

func printPlan(out io.Writer, p backends.TreePlan, now time.Time) {
	fmt.Fprintf(out, "  %s\n    %s, %s in %d files", p.Path, p.Kind, cleanupBytes(p.Bytes), p.Files)
	if !p.Newest.IsZero() {
		fmt.Fprintf(out, ", last changed %s ago", cleanupAge(now.Sub(p.Newest)))
	}
	fmt.Fprintln(out)
	switch {
	case p.LocalWork != "":
		fmt.Fprintf(out, "    local work: %s\n", p.LocalWork)
	case p.Kind != "stale build directory" && p.Blocked == "":
		fmt.Fprintln(out, "    local work: none found (read from the .git files)")
	}
	if len(p.Tags) > 0 && p.Kind != "source tree" {
		fmt.Fprintf(out, "    belongs to registered backend(s): %s\n", strings.Join(p.Tags, ", "))
	}
	for _, l := range p.Links {
		fmt.Fprintf(out, "    also removes link %s\n", l)
	}
	if p.Blocked != "" {
		fmt.Fprintf(out, "    NOT removable now: %s\n", p.Blocked)
	}
}

// runBackendRemove is `backend remove <tag> [--purge] [--dry-run] [--yes]
// [--force-not-running]`. Without --purge it only unregisters, as before.
func runBackendRemove(args []string, io cleanupIO, envFor func(cleanupOpts) backends.CleanupEnv, unregister func(string) (bool, error)) error {
	o, err := parseCleanupArgs(args, true)
	if err != nil {
		return err
	}
	env := envFor(o)
	var plan backends.TreePlan
	if o.Purge {
		// Every refusal happens here, before anything is changed.
		plan, err = backends.PlanPurge(env, o.Tag)
		if err != nil {
			return fmt.Errorf("not purging %q: %w", o.Tag, err)
		}
		fmt.Fprintf(io.Out, "Would remove the registration %q and:\n", o.Tag)
		printPlan(io.Out, plan, time.Now())
	} else if o.DryRun {
		fmt.Fprintf(io.Out, "Would remove the registration %q only (add --purge to delete its source tree too).\n", o.Tag)
	}
	if o.DryRun {
		fmt.Fprintln(io.Out, "Dry run: nothing removed.")
		return nil
	}
	if o.Purge {
		if plan.LocalWork != "" && !env.DiscardLocalWork {
			return fmt.Errorf("not purging %q: %w: %s", o.Tag, backends.ErrLocalWork, plan.LocalWork)
		}
		if err := confirmDeletion(o, io, fmt.Sprintf("Delete %s and unregister %q?", cleanupBytes(plan.Bytes), o.Tag)); err != nil {
			return err
		}
		// The confirmation may have waited a long time: read the manifest and
		// check every rule again before changing anything.
		env = envFor(o)
		again, err := backends.PlanPurge(env, o.Tag)
		if err != nil {
			return fmt.Errorf("not purging %q after confirmation: %w", o.Tag, err)
		}
		if again.Path != plan.Path {
			return fmt.Errorf("not purging %q: its tree changed from %s to %s while waiting", o.Tag, plan.Path, again.Path)
		}
		plan = again
	}
	// Unregister first: a launch must never pick a backend whose tree is going.
	found, err := unregister(o.Tag)
	if err != nil {
		return fmt.Errorf("remove failed: %w", err)
	}
	if !found {
		return fmt.Errorf("no backend tagged %q", o.Tag)
	}
	fmt.Fprintf(io.Out, "Removed backend %q\n", o.Tag)
	if !o.Purge {
		return nil
	}
	removed, err := env.Delete(plan)
	for _, r := range removed {
		fmt.Fprintf(io.Out, "Removed %s\n", r)
	}
	if err != nil {
		return fmt.Errorf("the registration is gone but the tree was not fully removed: %w (list leftovers with: ggrun backend gc)", err)
	}
	return nil
}

// runBackendGC is `backend gc [--delete] [--yes] [--dry-run]
// [--force-not-running] [--discard-local-work]`: it lists unreferenced source
// trees and stale build directories, and deletes them only with --yes or a
// confirmed --delete.
func runBackendGC(args []string, io cleanupIO, envFor func(cleanupOpts) backends.CleanupEnv) error {
	o, err := parseCleanupArgs(args, false)
	if err != nil {
		return err
	}
	if o.Purge {
		return backendUsageError("gc has no --purge; it lists, and --delete or --yes removes")
	}
	env := envFor(o)
	plans, err := backends.PlanGC(env)
	if err != nil {
		return err
	}
	if len(plans) == 0 {
		fmt.Fprintln(io.Out, "No orphaned backend source trees or stale build directories.")
		return nil
	}
	now := time.Now()
	var total int64
	var deletable []backends.TreePlan
	fmt.Fprintln(io.Out, "Unreferenced under the app home's .src (no registered backend uses them):")
	for _, p := range plans {
		printPlan(io.Out, p, now)
		if p.Blocked == "" {
			total += p.Bytes
			deletable = append(deletable, p)
		}
	}
	fmt.Fprintf(io.Out, "%d of %d removable, %s.\n", len(deletable), len(plans), cleanupBytes(total))
	if o.DryRun || (!o.Delete && !o.Yes) {
		if len(deletable) > 0 {
			fmt.Fprintln(io.Out, "Nothing removed. Delete with: ggrun backend gc --delete (asks first) or --yes.")
		}
		return nil
	}
	if len(deletable) == 0 {
		return nil
	}
	if err := confirmDeletion(o, io, fmt.Sprintf("Delete these %d directories (%s)?", len(deletable), cleanupBytes(total))); err != nil {
		return err
	}
	// Re-read the manifest and re-plan after the confirmation: delete only what
	// is still unreferenced, unblocked and quiet.
	env = envFor(o)
	fresh, err := backends.PlanGC(env)
	if err != nil {
		return fmt.Errorf("not deleting: %w", err)
	}
	stillOK := map[string]backends.TreePlan{}
	for _, p := range fresh {
		if p.Blocked == "" {
			stillOK[p.Path] = p
		}
	}
	var failed []string
	for _, planned := range deletable {
		p, ok := stillOK[planned.Path]
		if !ok {
			fmt.Fprintf(io.Out, "Skipped %s: it changed while waiting for confirmation\n", planned.Path)
			failed = append(failed, planned.Path)
			continue
		}
		removed, err := env.Delete(p)
		for _, r := range removed {
			fmt.Fprintf(io.Out, "Removed %s\n", r)
		}
		if err != nil {
			fmt.Fprintf(io.Out, "Skipped %s: %v\n", p.Path, err)
			failed = append(failed, p.Path)
		}
	}
	if len(failed) > 0 {
		return fmt.Errorf("%d directories were not removed", len(failed))
	}
	return nil
}

func cmdBackendRemove(args []string) {
	err := runBackendRemove(args, defaultCleanupIO(), defaultCleanupEnv, backends.Remove)
	if _, usage := err.(backendUsageError); usage {
		fmt.Fprintln(os.Stderr, err)
		backendUsage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func cmdBackendGC(args []string) {
	err := runBackendGC(args, defaultCleanupIO(), defaultCleanupEnv)
	if _, usage := err.(backendUsageError); usage {
		fmt.Fprintln(os.Stderr, err)
		backendUsage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// ---- status ----

// statusRecipeLookup and statusCheckoutHead are seams for tests.
var (
	statusRecipeLookup = backends.RecipeByName
	statusCheckoutHead = func(srcDir string) string {
		out, err := gitOutput(srcDir, "rev-parse", "HEAD")
		if err != nil {
			return ""
		}
		return strings.TrimSpace(out)
	}
)

func cmdBackendStatus() {
	printBackendStatus(os.Stdout, backends.Load(), time.Now())
}

// printBackendStatus writes one block per registered backend: how old its
// build is, whether the catalog says upstream has replaced the recipe, and
// whether the build and checkout sit at the recipe's pin. Local data only.
func printBackendStatus(out io.Writer, list []backends.Backend, now time.Time) {
	if len(list) == 0 {
		fmt.Fprintln(out, "No custom backends registered. Add one with: ggrun backend add <git-url>")
		return
	}
	for _, be := range list {
		var modTime time.Time
		if info, err := os.Stat(be.Path); err == nil {
			modTime = info.ModTime()
		}
		var head string
		if layout, err := backendLayoutFor(be); err == nil {
			if _, err := os.Stat(layout.srcDir); err == nil {
				head = statusCheckoutHead(layout.srcDir)
			}
		}
		s := backends.AssessBackend(be, statusRecipeLookup(be.Tag), head, modTime, now)
		fmt.Fprint(out, formatStaleness(s, be))
	}
}

func formatStaleness(s backends.Staleness, be backends.Backend) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s\n", be.Tag)
	if s.BuiltAtSource != "" {
		fmt.Fprintf(&b, "  build:    %s old (%s %s)", cleanupAge(s.Age), s.BuiltAtSource, s.BuiltAt.Format("2006-01-02"))
		if s.BuiltCommit != "" {
			fmt.Fprintf(&b, ", commit %s", shortCommit(s.BuiltCommit))
		}
		b.WriteString("\n")
	} else {
		b.WriteString("  build:    age unknown (binary not found)\n")
	}
	if s.LastBuild != nil && s.LastBuild.Seconds > 0 {
		fmt.Fprintf(&b, "  last build took %s, tree %s\n", cleanupDuration(s.LastBuild.Seconds), cleanupBytes(s.LastBuild.TreeBytes))
	}
	if s.Recipe == "" {
		b.WriteString("  recipe:   none (hand-added; nothing to compare with)\n")
		return b.String()
	}
	switch {
	case s.BuiltOffPin:
		fmt.Fprintf(&b, "  recipe:   pinned at %s; the registered build is at %s (run: ggrun backend update %s)\n", shortCommit(s.PinnedCommit), shortCommit(s.BuiltCommit), be.Tag)
	default:
		fmt.Fprintf(&b, "  recipe:   build is at the recipe pin %s\n", shortCommit(s.PinnedCommit))
	}
	if s.CheckoutCommit != "" && s.CheckoutOffPin {
		fmt.Fprintf(&b, "  checkout: at %s, not the pin %s\n", shortCommit(s.CheckoutCommit), shortCommit(s.PinnedCommit))
	}
	if s.SupersededBy != "" {
		fmt.Fprintf(&b, "  SUPERSEDED UPSTREAM: %s\n", s.SupersededBy)
		if s.Kind != "" {
			fmt.Fprintf(&b, "    catalog kind: %s\n", s.Kind)
		}
		for _, e := range s.Evidence {
			fmt.Fprintf(&b, "    evidence: %s\n", e)
		}
		b.WriteString("    Check the model loads on the replacement before retiring this backend.\n")
	}
	if s.ReviewOverdue {
		fmt.Fprintf(&b, "  catalog review date %s has passed; this entry may be out of date\n", s.Expires)
	}
	return b.String()
}

// supersededNote is the one-line `backend update` note.
func supersededNote(tag string, recipe *backends.Recipe) string {
	if recipe == nil || recipe.SupersededBy == "" {
		return ""
	}
	return fmt.Sprintf("[backend] note: the catalog marks %s superseded upstream (%s); see: ggrun backend status\n", tag, recipe.SupersededBy)
}

// ---- measured build estimates ----

// buildEstimate is what a build message quotes: the last measured run of this
// tag when there is one, else the long-standing constant.
func buildEstimate(prev *backends.Backend) string {
	if prev != nil && prev.Build != nil && prev.Build.Seconds > 0 {
		s := fmt.Sprintf("last build here took %s", cleanupDuration(prev.Build.Seconds))
		if prev.Build.TreeBytes > 0 {
			s += fmt.Sprintf(" and left a %s tree", cleanupBytes(prev.Build.TreeBytes))
		}
		return s
	}
	return "this can take 30–60 min"
}

func cleanupBytes(n int64) string {
	const gb = 1 << 30
	switch {
	case n >= gb:
		return fmt.Sprintf("%.1f GB", float64(n)/gb)
	case n >= 1<<20:
		return fmt.Sprintf("%d MB", n>>20)
	default:
		return fmt.Sprintf("%d KB", n>>10)
	}
}

func cleanupAge(d time.Duration) string {
	switch {
	case d < time.Hour:
		return fmt.Sprintf("%d min", int(d.Minutes()))
	case d < 48*time.Hour:
		return fmt.Sprintf("%d h", int(d.Hours()))
	default:
		return fmt.Sprintf("%d days", int(d.Hours()/24))
	}
}

func cleanupDuration(seconds int64) string {
	if seconds < 90 {
		return fmt.Sprintf("%d s", seconds)
	}
	return fmt.Sprintf("%d min", (seconds+30)/60)
}
