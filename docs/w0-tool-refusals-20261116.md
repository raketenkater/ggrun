# W0: tool calls refused in ggrun-client — attribution, 2026-11-16

Routing and timing only. Transcripts stay local and are not copied here.

## Setup

- Server: owner's running gateway `127.0.0.1:8080` (not restarted), main
  Qwen3.8-Flash-Next UD-IQ4_XS (fork `fork-qwen3-8-flash-next`, 1 slot, 143360 ctx),
  seated reviewer Qwen3.5-4B Q4_K_M on CUDA0. ggrun binary `/home/mik/go/bin/ggrun`
  (v3.2.13-32-gea69bf1 per gateway).
- Client: installed ggrun-client v0.1.0-alpha.8 (= client `main` 7fe6f15; does not contain
  `fix/native-goal-default`), Claude Code 2.1.325.
- Harness: `wp-evidence/w0/harness/run.sh` (outside the repo) — temp project copied from `harness/fixture`, isolated
  `CLAUDE_CONFIG_DIR` (no global CLAUDE.md or user hooks), inherited `CLAUDE*` variables unset,
  headless `-p`, `--permission-mode auto`, `--max-turns 25`, 40 min wall clock. Task: read two
  files, fix a bug, run a test, run one Bash command, run `curl` which a project deny rule
  (`Bash(curl:*)`) blocks. Decision sources from Claude's debug log, the client hook's output and
  the router's request records (`.logs/ggrun-claude-requests-39389.jsonl`), joined by time.

## Live runs (one run per arm)

| Run | Arm | Connection profile | Authorised calls refused | Deliberate deny honoured | Wall time |
|---|---|---|---|---|---|
| native-1 | native Auto, no extra review | stale (`default`) | 0 of 5 | yes (user rule) | 193 s |
| review-1 | `--local-controls` | stale (`default`) | **5 of 5 non-read calls** | yes | 231 s |
| review-2 | `--local-controls` | fresh (`w0`, same server) | 0 of 5 | yes (user rule; the review itself allowed it) | 354 s |
| review-3 | `--local-controls`, client fix build (client #29) | stale (`default`) | 0 of 5 | yes | 359 s |

Observations:
- Native Auto: Claude first asks for server-side review, gets `server_no_result` and falls back to
  local classifier requests for the session (works as documented). Both classifier stages went to
  the seated reviewer (`reviewer/stop-stripped-verdict`, then `reviewer`), all allow; permission
  decisions took 18.6 s and 9.4 s. Edit skipped the classifier (allowed in acceptEdits).
- review-1: every Edit/Write/Bash hook exited 1 with
  `connection model or context changed; reconnect before review` and `onFailure: block` turned it
  into a deny. No review request reached a model. Cause: the hook compares the live profile with
  the profile saved at `connect` (Nov 6: other model, 262144 ctx); launch accepts the new model.
- review-2/3: reviews run on the main model's single slot; the first (cold, 28k tokens) took
  170 s, later ones 15–19 s. A hook `allow` replaces the native classifier.

## Refusal sources with counts

| # | Decision source | Live runs (today) | Owner's ggrun-client transcripts since Nov 1 | Router records | Fix task |
|---|---|---|---|---|---|
| R1 | ggrun hook error: stale connection profile (fail-closed) | 5 (review-1) | 6 (Nov 15, one project) | — | **F1, done:** client #29 binds the check to the launched session |
| R2 | native classifier unavailable (timed out) | 0 | 2 (Nov 2) | 4 main aborts at ~60 s (Nov 6, one conversation); historical 14 | **F2:** classify these requests in the record (W7 PR: kind + verdict class), then re-measure; route stays reviewer-first |
| R3 | reviewer answer invalid → fallback to main (not a refusal, costs a main-slot turn) | 0 | — | historical 37 invalid, 20 unusable | **F3:** W7 PR verdict class `invalid`; W17's 40-case set to pick the reviewer (E-7) |
| R4 | HTTP 400 on main (context rejection or schema, #127) | 0 | — | 1 since Nov 1 (580 KB request), historical 45 | **F4:** ggrun #127 / #130 (structured verdicts); record status per kind in W7 |
| R5 | user deny rule | 3 (one per completed run, deliberate) | 0 | — | none: correct behaviour |
| R6 | server-side review unsupported (`server_no_result`) | fallback observed in every run, no refusal | — | — | none needed: Claude falls back; advertising server review stays out (#28) |
| R7 | ggrun hook review deny/ask, deadline, malformed verdict | 0 | 0 | — | none observed; client #19 already bounds retries and never approves invalid replies |

Not ggrun-client: the owner's cloud Claude Code sessions (project `ggrun-project-ggrun`) show 51
"no verdict: a hook changed this call's input after auto mode reviewed it" on Nov 10 and 10
classifier denials (Nov 4–6). The first comes from a user PreToolUse hook that rewrites Bash input
(rtk). It would also affect a ggrun-client session that inherits such a hook; not observed in the
client project.

## Not done here

- Main-only route (no separate reviewer seated): needs relaunching the owner's backend; left to the
  fix acceptance and E-7.
- Reviewer design comparison (separate small reviewer vs main reviewing itself vs capable reviewer
  worker): blocked on E-7 and W17's 40-case evaluation; not improvised.
- #28 acceptance is met here for one run per route on one model only, not as a general claim.

## Follow-up in this change

Router records now carry `kind` (classifier, utility) and, for classifier requests, `verdict` (allow, block, invalid, none), so later refusals can be attributed from `.logs/ggrun-claude-requests-*.jsonl` without transcripts. Unit-tested only; not yet run live.
