# ggrun backlog

Audited against `main` on 2026-07-12. This replaces the stale Claude Code task
statuses with only the work that remains. Source references use
`<Claude task-list>/<task-number>`.

## P0 — finish DeepSeek-V4 MTP

The generic foundation already exists: ggrun parses NextN metadata, supports the
llama.cpp/ik_llama MTP dialects, validates draft GGUFs, searches/downloads generic
draft and EAGLE models from Hugging Face, selects a draft GPU and emits draft flags.

- [ ] Determine the actual DeepSeek-V4 MTP arrangement supported by the installed
  mainline backend: built-in NextN tensors, a separate MTP GGUF, or unsupported.
- [ ] Resolve the exact compatible Hugging Face artifact by model identity,
  architecture, revision, tokenizer/vocabulary and prediction depth—not filename
  similarity alone.
- [ ] Extend the existing draft resolver/downloader only where DeepSeek-V4 MTP needs
  a separate companion; reuse resume, validation, cache and offline behavior.
- [ ] Include any separate MTP model in the exact placement/preflight ledger instead
  of the older approximate draft-GPU calculation.
- [ ] Fall back to non-speculative serving with one clear reason when no compatible
  MTP artifact exists.
- [ ] Add resolver/compatibility/offline/corruption tests and live A/B measurements:
  output correctness, accepted drafts, serial decode, parallel-4 throughput,
  VRAM/RAM and long-run stability.
- [ ] Repeat the live test on one other MTP-capable MoE to prove the path is generic.

Source: `5e91131f/24`, retargeted by the user on 2026-07-12.

## P1 — finish Claude Code integration

The launcher, native `/v1/messages`, local aliases for every Claude tier, parallel-4,
1M total context, per-slot compaction, four-hour timeouts, anti-loop sampling and
DuckDuckGo MCP wiring are implemented. Claude Auto's hidden classifier requests are
now routed to a pinned local Qwen3.5-2B reviewer while coding stays on the selected
model. The reviewer starts before placement so its measured VRAM is in the main-model
ledger. Default local launches use fail-closed Auto, never bypass mode.

- [ ] Run one complete acceptance workflow against a running ggrun model: file edits,
  commands/tests, four workflow agents, tool results, queueing, combined response and
  context compaction.
- [ ] In that workflow, verify MCP `search` plus `fetch_content`, including a failed
  lookup/retry from a subagent.
- [x] Implement and verify local Auto permissions. Claude 2.1.207 sends a distinctive
  two-stage security-monitor request (about 25k prompt tokens) to the same model ID.
  ggrun's loopback router sends only that structured system request to the local 2B
  reviewer. Safe Bash completed end to end with zero permission denials; invalid
  reviewer output was also verified to fail closed. The pinned reviewer cold-prefilled
  the captured prompt in 2.4–5.8 seconds depending on GPU and warm reviews took ~0.18s.
- [ ] Turn the repeatable parts into a Claude acceptance harness for `/v1/messages`,
  tool-use/tool-result blocks, aliases, MCP, malformed tool recovery and timeouts.
- [x] **Add live local-request progress to Claude Code launches:** queued/prefill/
  generation/completed/failed state, prompt tokens and percentage, prompt/decode
  tok/s and elapsed time across all four slots. Prefer structured slots/metrics data;
  use versioned log parsing only as fallback. Present it through an opt-in status line,
  terminal title or companion `ggrun status` view that does not corrupt Claude's
  fullscreen TUI. Parser, state-machine, status-injection and monitor lifecycle tests
  are implemented. The existing 60k parallel-4 run supplies the long-request backend
  behavior; repeat it once with the new UI before the next release.

Sources: `db3f32cc/1`, `db3f32cc/2`, `db3f32cc/3`, user request 2026-07-12.

## P1 — finish existing product foundations

- [ ] **TUI extra parameters:** add a free-text field to the model configuration
  screen, parse without shell execution, show the resulting arguments, persist it and
  preserve the CLI's explicit last-wins behavior. CLI `--` extras already work.
  Source: `c69ee13f/3`.
- [ ] **Model swapping:** productionize the existing `ggrun daemon` `/reload` path.
  Add named models, lazy unload/load policy, bounded RAM/VRAM, tests, documentation,
  TUI controls and stable Claude/OpenAI aliases. Source: `c69ee13f/15`.
- [ ] **Architecture gotchas:** turn the DeepSeek/preflight knowledge currently spread
  through code comments and `docs/compute-preflight-plan.md` into one maintained,
  AI-facing reference plus machine-readable diagnostic rules. Source: `c69ee13f/13`.
- [ ] **Residual exact-memory audit:** the main large-MoE ledger is measured and
  preflighted; audit the remaining optional prompt-cache/CRAM constants and the old
  approximate draft-model GPU estimator. Do not disturb the validated MoE plan.
  Source: `ebffa9bc/9`.

## P2 — performance and installation

- [ ] **Small-model decode ablation:** settle the historical ~184 versus current
  151.4 tok/s result by testing minimal versus generated flags, f16 versus q4_0 KV,
  runtime repack, `-mqkv`, `-khad` and defrag. Keep only reproducible quality-neutral
  wins. Source: `e8111d05/17`.
- [ ] **Compile-free Linux CUDA installation:** CPU/Vulkan/macOS Metal/Windows CPU
  bundles and Windows prebuilt CUDA exist. Ship a Linux CUDA backend bundle so normal
  installation no longer needs CUDA toolkit, CMake or a compiler; source build becomes
  an explicit fallback. Source: `5e91131f/17`.
- [ ] **Homebrew + AUR:** publish the formula/tap and AUR/optional `-git` package with
  backend wiring, checksums, automated version bumps and smoke tests.
  Sources: `5e91131f/25`, `a769b44a/2`.
- [ ] **Community tune pool:** the offline-safe client exists. Finish the format/design
  review, moderation, hardware deduplication and poisoning controls, then publish the
  seed repository after destination approval. Source: `a769b44a/5`.

## P3 — external and later work

- [ ] Decide whether reserving `ggrun` on PyPI/npm is still useful; obtain explicit
  confirmation immediately before publishing. Source: `a769b44a/18`.
- [ ] Draft the next release announcement after MTP and Claude workflow results are
  final. Source: `fb9a268c/1`.
- [ ] Scope the separate backend-agnostic personal agent based on OpenCode + ggrun;
  keep it outside this repository until its boundary is agreed. Source: `a769b44a/19`.

## Confirmed complete or obsolete Claude TODOs

- [x] DeepSeek-V4 stable first-launch placement, full-layer expert storage, OOM
  recovery and 60k parallel load test. Source: `fb9a268c/2`.
- [x] Mainline DeepSeek-V4 backend; old antirez/cchuter fork registration is obsolete.
  Sources: `5e91131f/19`, `ebffa9bc/10`.
- [x] DeepSeek-V4 recommender inclusion. Source: `5e91131f/20`.
- [x] DeepSeek-V4 1M KV decision: GPU KV + Flash Attention. Source: `ebffa9bc/8`.
- [x] RAM headroom through CLI/config/environment/TUI/recommender/placement.
  Source: `e8111d05/13`.
- [x] Loopback server bind by default. Source: `e8111d05/11`.
- [x] Windows Python/download dependency audit and CI smoke coverage.
  Source: `e8111d05/14`.
- [x] Base local web research through DuckDuckGo MCP. Source: `db3f32cc/1`.
- [x] Auto-mode model-tier routing to the local alias. Source: `db3f32cc/3`.
- [x] README and launch-performance benchmark numbers now agree on the dated retest;
  the later direction retains Ollama only as a reference column and no separate
  AI-tune result column. Sources: `e8111d05/15`, `5e91131f/22`.

## Definition of done

- Automatic behavior is capability/metadata-driven; explicit settings win.
- Offline and failure behavior is safe and clear.
- Unit/regression tests pass, plus real hardware validation where relevant.
- Performance claims record model, quant, backend, context, parallelism, hardware
  and raw results.
- External publication always gets destination-specific approval first.
