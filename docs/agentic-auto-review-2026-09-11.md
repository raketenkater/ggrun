# Agentic auto review — 2026-09-11

Scope: main at 7addc20 plus a read-only review of the separate staged hot-expert
integration. Changes are isolated in investigate/agentic-auto; the staged
integration remains untouched.

## Findings and changes

1. PR #40 fixed saved PARALLEL=1 being treated as an explicit TUI choice. Explicit
   environment and per-model choices remain constraints. Install a binary that
   includes the fix before testing auto.
2. Calibration measured fewer requests on narrower configurations and multiplied
   scenario time by predicted waves. That omitted actual queueing and cache
   effects and changed mixed traffic across slot widths. This change submits the
   same requested work (up to eight lanes) on every configuration and uses actual
   elapsed time. Failed requests invalidate the measurement. Mismatched work and
   non-finite promotion scores are rejected; evidence schema becomes 24.
3. The staged hot-expert integration adds an early cache-on success return in
   calibrationCandidateBetter. It checks phase rates but bypasses required
   workflow improvement and worst-sample confirmation. Do not integrate that
   shortcut into automatic promotion. Cache availability is eligibility, not
   evidence of an improvement. Its finalist preference also needs justification
   against other measured bottlenecks.
4. The serving probe is still synthetic and goes directly to chat completions.
   It does not measure completed coding tasks, router queues, reviewer verdict
   quality, or effective worker delegation. The mixed probe uses a fixed 75 ms
   head start, not a confirmed streaming decode start. Bounded prompt lengths
   do not validate production long-context behavior.
5. Automatic demand is a bounded preset, not a learned queue controller. GPU
   busy percentage and VRAM occupancy cannot establish a fraction of attainable
   agentic performance.

## Next acceptance steps

- Review hot-expert integration against the same promotion gates, then preserve
  exact argv admission, context and quality constraints, and clean relaunch.
- Compare baseline and challenger on identical agent tasks, seeds, tools,
  concurrency and context. Record correct task completions per elapsed minute,
  queue and turn latency distributions, timeouts/cancellations, cache reuse,
  reviewer correctness and actual worker requests. Repeat and retain raw data.
- Include long-context append, compaction/replay, and overlapping prefill/decode.
  Reject material phase regressions, even if one aggregate metric improves.
- Keep the stable baseline when measurements are incomplete or inconclusive.
  Unit tests and CI validate controller behavior, not a speedup.

## Installation and CI evidence

At review time, main CI run 34587047456 passed. Install E2E run 34587074647
passed Linux CPU, Windows CPU, and Linux GPU; Windows GPU was skipped in that
run. A later release build was in progress. Latest published release was still
v3.2.8, so main's green checks did not establish that release contained its fixes.
README now explains PATH activation, installer versus release versions, workflow
coverage, and the limits of automatic performance claims.
