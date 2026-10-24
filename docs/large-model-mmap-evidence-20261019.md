# Large-model mmap and sparse lookup tables: evidence, 2026-10-19

Scope: `GGRUN_MMAP_GOAL.txt`, from the worker handoff
`docs/worker-large-model-mmap-handoff-20261019.md`. This records what is
implemented, what is only unit-tested, and what blocks live acceptance. It is
not a production-readiness claim.

## Identities

| Item | Identity |
| --- | --- |
| ggrun base | `origin/main` 5dbf9be, branch `feat/sparse-lookup-mmap` |
| Backend | smalinin/llama.cpp `my_build` = 425badfe8c1c286a8984a454e991f045bb3008b2 (branch head on 2026-10-19) |
| Build | CUDA 13.2, `-DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=86;89`; libggml-cuda carries 191 sm_86 + 191 sm_89 kernels; `llama-server` and `llama-fit-params` |
| Model | prometheusAIR/DeepSeek-V4.1-Flash-1017-GGUF IQ2_XS @ 06b82ad71b66313a3baa5e5a71f200f642757667, 10 shards, 393,292,445,504 B |

Reviewed in the backend source at 425badfe:

- MMQ out-of-bounds fix: 5b621e90 ports upstream #29953 (merged upstream as
  fc9ce6b); the commit records compute-sanitizer A/B on Ada and Ampere.
- Two-level candidate mask: 270c26b9. This GGUF has no
  `attention.candidate_*` keys; the loader then uses the checkpoint values
  20/8/2048 and logs a warning. Confirm that line at the first load.
- V4.1 conversation/tool protocol: be5bf0ca. The GGUF embeds the V4.1 template.
- `deepseek41` is listed in `llm_arch_is_hybrid` and `llm_arch_supports_rs_rollback`
  and excluded from tensor (row) split.

## Loader semantics (source, then measured)

`blk.N.engram_embd` is created with `TENSOR_READ_LAZY`. With `--lazy-mode on`
(or `auto` above 4 GiB) the loader:

- returns the CPU buffer type before any `-ot` override, so the table never
  reaches a GPU;
- maps every shard file when any lazy tensor exists, even with
  `--load-mode none` (`use_mmap || lazy.any()`), and allocates the tensor from
  that mapping;
- applies `MADV_RANDOM` to the lazy ranges and leaves them out of
  `MAP_POPULATE` and mlock.

With lazy mode off the table follows its layer. `--load-mode auto` turns
mapping off when any device lacks `mmap_support` (iGPU CUDA, OpenCL, Hexagon).

The fork's own `llama-fit-params` on sparse files with the real pinned headers
and the planned argv:

| `--lazy-mode` | CUDA0 model | CUDA1 model | CUDA2 model | Host model |
| --- | ---: | ---: | ---: | ---: |
| on | 10,316 MiB | 21,087 MiB | 10,336 MiB | 333,327 MiB |
| off | 109,927 MiB | 120,700 MiB | 10,336 MiB | 134,102 MiB |

The fit pass counts mapped lazy tables as host model memory
(333,327 = 134,102 + 199,225).

## Defects reproduced on main and fixed

1. `parse_gguf.py` put both Engram tables into layers 1 and 14 as ordinary
   backbone: 97.6 GiB per layer, billed to the GPU owning the layer. Schema 3
   recognises the tier only through an architecture adapter
   (`SPARSE_LOOKUP_ADAPTERS`: `deepseek41` → `<arch>.engram.layer_ids`,
   `blk.N.engram_embd.weight`). On the pinned headers the result is
   171.72 GiB of execution weights + 194.56 GiB tier, the publisher's numbers
   exactly. The per-layer split is kept (`sparse_lookup_layer_bytes`).
2. No placement tier existed (see the design below).
3. Exact admission would have refused a paged plan, because the fit pass
   reports the mapped tables as host model memory.
4. The absent mmap flag meant `auto` on `--load-mode` builds while every
   planner read it as mapped. Every `--load-mode` build now receives an
   explicit mode at process start.
5. The dry-run printed resident `--no-mmap` CPU experts as "reclaimable
   weights". Only mapped experts on a file-backed loader are labelled
   reclaimable.

## Senior review fixes (2026-10-20)

The [PR #93 review](pr93-senior-review-20261020.md) reproduced five failures
with a test overlay. Each is now a permanent invariant test:

| Finding | Fix | Tests |
| --- | --- | --- |
| P1 explicit `--load-mode none` became `... --load-mode mmap` | `server.ParseLoadMode` folds classic and native spellings in argv order (classic flags compose, a native value replaces the state, last wins). An argv that states its mode natively and has no classic flag is returned unchanged, so translation is idempotent. Passthrough `--load-mode`/`-lm` becomes the planner's `--no-mmap`/`--mmap` (+`--mlock`; `dio` passes through, planned unmapped; `auto` leaves ggrun's choice), so plan, admission, argv and cache identity agree. The memory fingerprint folds both spellings and accumulates every `-ot`. Evidence for `--load-mode` builds is re-keyed (`+load-mode-v2`) because the absent-flag default changed. | `TestLoadMode*`, `TestPassthroughLoadModeBecomesPlannerConstraint`, `TestMemoryFingerprintFoldsLoadModeSpellings` |
| P1 `--lazy-mode` help authorized reclaimable memory | Flag availability and loader semantics are separate. A file-backed capability needs a reviewed loader contract matched on the build's source commit, the table kind and the platform (`llama.cpp-tensor-read-lazy/v1`: commit 425badfe, `engram`, Linux). A help line only lets ggrun state `--lazy-mode off`. A persisted live contradiction overrides the contract. | `TestSparseLookupCapabilityNeedsReviewedContract`, `TestSyntheticSecondKindNeedsItsOwnContract` |
| P1 fit-oracle Host row reduced on loader capability alone | A separate, versioned oracle accounting contract (`host-row-includes-lazy-tables/v1`, `host-row-resident-only/v1`, unknown). Only the inclusive contract converts the row, by the tier's whole MiB (rounded down), once per row, and never below the tier. Unknown or resident-only oracles are used as reported. | `TestFitPassHostRowFollowsOracleContract` |
| P2 every table-bearing model needed lazy loading | Three policies: `layer` (ordinary tensors with their layer, charged per layer; the only policy without a contract), `resident` (lazy rows, whole tier charged to host RAM once), `paged` (lazy rows outside the resident ledgers, bounded cache). Automatic choice: resident first, paged when resident does not fit or gives less context; a plan needing automatic expert paging loses to one with resident execution weights. Explicit `--lazy-mode on/off` is a constraint; `on` without a contract is refused with the reason. Verified configs are reused only under the policy that produced them, and placement/probe caches are scoped per policy. | `TestRoomyHostHasResidentSparseLookupPlan`, `TestConstrainedHostPagesTierAndKeepsExecutionResident`, `TestExplicitLazyChoiceIsHonored`, `TestSyntheticDenseTierPolicyMatrix`, `TestLayerTierIsChargedToTheLayerDevice`, `TestResidentTierJoinsFootprintOnce`, `TestVerifiedConfigKeepsSparseLookupTier` |
| P2 constrained scope lost its reclaim band | The band is reserved first: `max = min(ceiling, non-reclaimable + tier + band)`, `high = max - band`, lookup cache `= high - non-reclaimable`. If the ceiling cannot hold the state plus a band, `high` stays at the non-reclaimable ceiling and the shortfall is logged. At start, a lazy argv gets `high = max - band` within an explicit budget. | `TestSparseLookupScopeReservesReclaimBand`, `TestSparseLookupStartLimitsKeepABand`, `TestLazyLookupArgvGetsReclaimBandAtStart` |

Rerunning the review overlay on the fixed code: the server and placement
checks pass. The three command-package checks no longer compile because the
capability probe, scope and oracle helpers changed signature; their exact
inputs (the unreviewed help string, 150,000/4,096/200,000 MiB, a 250,000 MiB
resident-only row) are cases in the permanent tests above.

## Second review fixes (2026-10-20)

The [re-review](pr93-rereview-20261020.md) of `13b1aff` reproduced three more
failures with a test overlay. Each is now a permanent test:

| Finding | Fix | Tests |
| --- | --- | --- |
| P1 sparse-policy plans could not read preflight measurements | One helper, `placement.SparseLookupScopedTag`, scopes evidence by policy for placement readers (`backendCacheTag`) and every command-side writer (`scopedProbeBackendTagForStrategy`, probe reset). A policy view no longer changes the artifact identity: `SpecTargetIdentity` and the measured-KV path use `ArtifactSizeBytes`/`ArtifactTotalSizeMB`, which add back what the view removed. Probe, runtime-growth, prompt-cache, placement-cache and KV-rate keys all go through these. | `TestSparsePolicyEvidenceRoundTrip` (preflight write → replan → verified relaunch, other policy and other build negative), `TestEvidenceTagScopesSparsePolicy` |
| P1 the full CLI broke loading-mode order | `parseLaunchArgs` no longer applies `--mmap`/`--no-mmap` on sight. All loading flags (classic, `--mlock`, native) fold once in argv order (`loadingModeFold`). A native value replaces earlier locks and retained `dio`; `--mmap=false` withdraws only its own choice. The planner fields and the passthrough come from that one state. | `TestParseLaunchArgsFoldsLoadingModesInOrder` (the four reproducers plus mixed and inline forms) |
| P2 locked execution weights read as a locked lookup table | The parser now emits each table's absolute byte range per shard (`sparse_lookup_ranges`). The smaps check keeps per-mapping file ranges and judges only mappings that map table bytes alone (one partial page allowed at each end; the loader's `MADV_RANDOM` split is page-aligned). Mappings without table bytes are not judged; mixed mappings, or missing ranges, give "unknown" and a log note, never a persisted contradiction. Coverage is checked against the ranges. | `TestSparseLookupObservationAttributesTableRanges` (clean table + locked execution weights, locked table, anonymous table, unmapped range, unsplit mixed mapping, unknown ranges), parser range test |

On the real headers, layer 1's table is shard 00008 from byte 192 to the end
(104,449,677,696 B), so that shard's mapping is attributed to the table.

## Final review follow-up (2026-10-21)

The [final review](https://github.com/raketenkater/ggrun/pull/93#issuecomment-6312504286)
of the merged `c392f89`
found one missed reader and one overstated log line:

| Finding | Fix | Tests |
| --- | --- | --- |
| P2 Claude relaunch refresh lost scoped prompt-cache measurements | `strategyAccounting` gives exported entry points that take a finished strategy the plan's policy, its scoped tag and its model view. `RefreshRuntimeCachePolicy` reads under the scoped tag and sizes CRAM on Compute's inputs (resident: tier-charged host RAM, footprint before the tier). Audit: `BuildResourceLedger`/`AnalyzeStrategy` (calibration) had the same unscoped read and ignored measured graphs; same fix, and an underived resident footprint now charges the tier. | `TestRuntimeRefreshReadsPolicyScopedPromptCache` (paged, resident, no-evidence parity, other policy and unscoped negatives, ordinary model), `TestResourceLedgerPricesLookupTierOnPolicyView`, `TestPreviousClaudePromptCacheReachesRuntimeRefresh` (served log → record → refresh; other policy; ordinary model) |
| Summary claimed clean table pages without attribution | The passing check's summary says "table pages unknown" when ranges are missing or a mixed mapping holds locks or anonymous pages. Missing data still never rejects. | `TestSparseLookupObservationSummaryStaysUnknown` |

Against the unfixed code, the refresh tests read 0 MiB from the command-side
writer, or picked up a foreign unscoped entry, and the ledger priced a 3,226 MiB
estimate instead of the measured graph. Not fixed here (pre-existing, outside
this tier): the refresh still sizes against raw detected host RAM, without
Compute's RAM policy or headroom (core backlog: RUNTIME REFRESH in
`core-standard-launch-todos.md`).

## Sparse lookup tier: design

- Tensor role: the parser adapter table. Go mirrors it
  (`gguf.sparseLookupArchitectures`) and rejects a tier for an architecture or
  kind it has no adapter for, or a per-layer split that does not sum to the
  total.
- Loader contract: `reviewedSparseLookupLoaders` in `cmd/ggrun`. Only the
  reviewed fork commit is listed. A synthetic second kind exists only in tests.
- Policy: `pkg/placement/sparselookup.go`. Placement emits `--lazy-mode on` for
  resident and paged, and `--lazy-mode off` for layer when the backend
  advertises the flag. No `-ot` is added, so expert overrides and their
  recovery parsers are untouched.
- Guarantee of the scope: one cgroup. Above `memory.high` the kernel reclaims
  the scope's clean file pages (lookup rows) and throttles growth; anonymous
  growth past `memory.max` is killed inside the scope. There are no separate
  kernel budgets for anonymous state and lookup pages. On platforms without
  cgroup scopes (Windows) there are no limits; the post-load mapping check is
  Linux-only and its absence is logged, never recorded as proof.

## Planner and admission on the real headers (no load)

Recorded 2026-10-19 with the pre-review revision (a branch build, since
removed). The paged policy reproduces this plan; it has not been rerun.

`ggrun memory-probe`, ctx 65,536, pinned fork, this host (212 GiB RAM,
4070/3090 Ti/3060):

- The first plan over-committed CUDA0 by 2,380 MiB (fit-pass compute 2,913 MiB
  vs 1,024 MiB estimated). The re-plan fits all three GPUs: 8,774 / 22,037 /
  6,877 MiB.
- Host resident 142,826 MiB, with the 199,225 MiB Engram tier excluded and
  logged separately. Argv: `--no-mmap --lazy-mode on`, 8 expert layers on
  GPUs and 32 on CPU.
- Performance is not tuned. Layer split across three cards disables the fork's
  fused mHC ops (publisher note); CUDA0 and CUDA2 are under-used after the
  re-plan.

## Small real control (live)

Qwen3.5-4B Q4_K_M through the branch binary. ggrun auto-selected production
mainline (load-mode-only). Results:

- The backend received `--load-mode mmap`; ggrun's internal argv had no flag.
- The deterministic arithmetic prompt answered 391, correctly; decode
  175.6 tok/s; 63 of 67 prompt tokens were cache hits.
- SIGINT gave a clean stop in 2 s, removed the scope and released all GPUs.

The pinned fork also served a 2B model on CUDA0 (sm_89, 191.7 tok/s) and
CUDA1 (sm_86, 246.2 tok/s) with coherent reasoning reaching the correct
answer. This is a wiring and lifecycle check only.

## Host DRAM fault (blocks correctness acceptance)

A 150 GiB userspace pattern probe ran 36 passes over six patterns
(`memflip.c`, contained scope) and found 63 bit flips, every one at byte
offset ≡ 10 (mod 16), bits 4–7, in both directions, repeating at fixed
addresses. A root run of the locator (2026-10-19 13:05 UTC, pagemap physical
addresses decoded through the Skylake-X memory-controller registers) found 58
flips, all on **DIMM_D1** (G.Skill F4-3200C16-32GVK, IMC1 channel 1 DIMM 0,
both ranks). This is a non-ECC DRAM fault (EDAC: ECC disabled), independent
of the disk and filesystem. The owner scheduled the hardware fix for
2026-10-23; no BIOS change or reboot was made.

Model and runtime correctness cannot be claimed on this host until a rerun
of the locator shows zero flips. Weights and lookup pages held in RAM can be
silently corrupted. Hashes that match, and serving that works, are artifact
and mechanics evidence only.

### Mitigation without a reboot (2026-10-23)

The DIMM was not replaced. `offline-bad-ram.sh` (root, private) soft-offlined
every 4 KiB page within 1 MiB of each recorded flip: 2,272 pages, 9,088 kB
`HardwareCorrupted`, 32 kernel-held pages skipped. A 193 GiB pattern pass then
found zero flips (receipt 14:54 UTC, CLEAN). A single-page pass the day before
was not clean, because new flips kept appearing in adjacent pages. The
offlined pages return on reboot, so this holds only until the next boot or the
stick swap. The live results below ran on this mitigated host.

## Live acceptance (2026-10-23)

Every cell ran through the PATH `ggrun v3.2.12-23-ge546a94` (sha256
`b908c7b53dc682b1…`) with `--claude-code`, one large process at a time, with no other GPU or agent workload. The backend was
`deepseek41-1017` (`425badfe`, sha256 `6b142ef1…`). The harness, sampler and
traffic scripts and the raw records are private
(`ggrun-accept-20261013-evidence/live-20261022/`).

### DeepSeek V4.1 Flash IQ2_XS, cold (paged tier)

- Plan: paged lookup tier (199,224 MiB of tables, file-backed lazy rows),
  resident host model 150,708 MiB, ctx 1,048,576 (4 × 262,144), KV q8_0 on
  GPU, `-b 128 -ub 64`, 36 CPU MoE layers. The argv carried `--lazy-mode on
  -cram 7168`, and the `--no-mmap` request was folded to `--load-mode none`.
- Live tier check: 10 mappings, 375,072 MiB mapped, no table page anonymous or
  locked. During serving the sampled table-page RSS stayed at or below
  2,653 MiB; `memory.peak` was 160,484 MiB under a 205,992 MiB cgroup limit.
- Ready after 3,003 s: two full loads of about 1,220 s each (oracle plan, then
  one backend-measured re-plan that moved one MoE layer to GPU). The optimizer
  refused the challenger: it needed about 1 h 2 min against 12 min of budget.
- Stream: first token 8.2 s. Cancel after 5 events, health ok, reconnect first
  token 3.4 s.
- Lookup traffic (`cache_prompt: false`, 8-token replies; HDD = sda sectors):

| Phase | Prompt tokens | Prompt tok/s | HDD read per request |
| --- | --- | --- | --- |
| first request, cold | 943 | 3.24 | 141 MiB |
| same prompt ×4 | 943 | 16.02 | 0 |
| five different prompts | 811–1,126 | 3.17–4.02 | 116–143 MiB |
| first prompt again ×2 | 943 | 15.9–16.0 | 0 |

  Decode in these requests was 1.6–8.2 tok/s. Page-cache residency of the
  two table shards rose from 53 MiB to 452 MiB each. A cold request is bound
  by scattered HDD reads through ntfs-3g (about 0.5 MB/s effective). A warm
  request is bound by CPU expert bandwidth at `-ub 64`.
- Claude Code task `calc`: timeout at 3,600 s. Its first request (about
  13.2k prompt tokens) reached 11,520 tokens at 3.22 tok/s and never produced a
  reply; the two remaining tasks were skipped. Correct tasks: 0.
- Stop: clean in 23.9 s, no leftover backend or scope, GPUs released. During
  this stop ggrun printed "backend died mid-session" and checked the backend
  log for an OOM. The Claude-mode health monitor is still ticking while ggrun
  shuts the backend down (main.go, `healthCtx`); no OOM was recorded.

### DeepSeek V4.1 Flash IQ2_XS, relaunch

- Ready after 1,621 s with one full load (1,221 s). ggrun planned 37 CPU MoE
  layers again, not the 36 that the cold cell had measured as fitting, and
  logged "memory plan stable at oracle-planned evidence". The faster start
  came from skipping the measurement load, not from reusing its result.
- Memory scope: non-reclaimable ceiling 170,150 MiB plus a 31,718 MiB lookup
  cache gave `memory.high` 201,868 MiB and `memory.max` 205,964 MiB. Peak was
  199,540 MiB as the table page cache filled toward the high mark.
- Stream: first token 8.5 s. Cancel and reconnect ok (3.5 s).
- Lookup traffic matched the cold cell to within 2%. The first request read
  141.5 MiB again, so the table pages cached during the cold cell did not
  survive the 155 GiB weight reload. Warm repeats ran at 15.6 tok/s, new text
  at 3.2–4.0 tok/s.
- Claude Code task `ledger` (limit 7,200 s): **correct**, all 11 tests pass,
  tests untouched, 28 turns, 6,521 s. The first request was 13,011 prompt
  tokens in 4,055 s (3.21 tok/s), which is 62% of the task. After it, prompt
  caching held: 368k prompt tokens came from the cache and 19.5k were
  processed. Later requests ran at 3.8–14.6 tok/s, depending on how much new
  text they carried.
- After the task the two table shards held 1,802 MiB of cached pages each, in
  426k separate runs spread evenly across each 99.6 GB file (about one page in
  55). Hashed n-gram rows have no locality. A session's hot set is small
  (3.6 GiB), but a blind cache of the first N GiB catches only N/199 of rows.
- Stop: 22.0 s, clean. The "backend died mid-session" shutdown message
  appeared again.

### DeepSeek experiment: prefetch the previous session's hot pages

Owner-requested. This is the same relaunch (ready 1,638 s, one load, 37 CPU
MoE layers). After readiness and the stream/cancel checks, the hook read
back the 853k page runs recorded after the `ledger` session. Runs less than
64 KiB apart were merged, then read in offset order with `pread`. It read
10,267 MiB in 1,500 s (6.8 MB/s on the HDD), leaving 11.1 GiB of table pages
cached. Then came Claude Code task `calc`, which timed out cold:

- First request: 12,970 prompt tokens in 811 s at **16.0 tok/s**. Cold, the
  same prompt took 4,055 s at 3.2 tok/s and `calc` did not finish within
  3,600 s.
- `calc` **correct**: 4 tests pass, tests untouched, 7 turns, 1,285 s.
- Later requests that brought new file text fell back to 4.2–8.1 tok/s. The
  prefetch covers only the text that earlier sessions saw.
- Peak 163,307 MiB; stop 22.1 s, clean.

A session's hot rows are a small set: 3.6 GiB of the 199 GiB tables, and
11 GiB once merged for the HDD. Keeping that set across a relaunch moves the
Claude Code system prompt from the cold rate to the warm rate. On an SSD
(7,285 random 4 KiB reads/s at queue depth 1, measured on this host's SATA
SSD) the prefetch would take seconds rather than 25 minutes.

### DeepSeek experiment: fill idle RAM with table pages

Owner-requested, on the same relaunch (ready 1,284 s). After readiness the
hook read the first 18 GiB of each table shard sequentially: 36,864 MiB in
307 s (120 MB/s). Then it sent new text it had not sent before.

- Only 16.5 GiB stayed cached (shard 09 15.5 GiB, shard 08 0.6 GiB). The
  first shard's pages were evicted while the second was read. During serving
  about 25 GiB was free and about 25 GiB was reclaimable cache: 151 GiB of
  the host's "cached" memory is the backend's shared-memory weights. The
  eviction mechanism is not established.
- New text ran at 3.14–3.96 tok/s and read 125–147 MiB per request from the
  HDD, no better than the cold cell (3.17–4.02 tok/s, 116–143 MiB). Warm
  repeats ran at 15.3–15.8 tok/s.

Hashed rows have no locality, so a blind cache of N GiB serves about N/199
of lookups. Filling free RAM is not a useful policy for this tier. Keeping
the specific hot pages is (previous section).

### DeepSeek experiment: the optimizer's unmeasured finalist

Owner-requested. In the cold cell, auto calibration had refused to measure
`-b 256 -ub 128` (predicted relative 1.182) because the experiment needed
about 62 minutes against its 20-minute budget. This cell launched it
explicitly: `-b 256 -ub 128 --calibrate off`, on a copy of the DeepSeek
cache. ggrun kept ctx 1,048,576 with 4 slots and 37 CPU MoE layers. Ready in
1,273 s with one load; GPU peak 11,389 / 20,246 / 11,277 MiB; host peak
201,351 MiB; stop 21.9 s, clean.

| Lookup traffic | auto `-b 128 -ub 64` | `-b 256 -ub 128` |
| --- | --- | --- |
| warm repeat, prompt tok/s | 15.6–16.0 | **20.1–20.7 (+29%)** |
| new text, prompt tok/s | 3.2–4.0 | 3.4–4.4 |
| decode tok/s (8-token replies) | 5.3–8.2 | 6.9–8.0 |

The finalist beats the served baseline by more than the optimizer predicted.
The elapsed budget, not the evidence, kept it out. On a model whose full
load takes about 20 minutes, the auto budget cannot measure any challenger.

### DeepSeek experiment: smaller context, larger microbatch

Owner-requested: `-b 512 -ub 512 -ctx 262144 --parallel 1 --calibrate off`.
ggrun placed the KV cache on the CPU (`--no-kv-offload`) and moved 4 more MoE
layers to GPU (33 CPU MoE layers instead of 37). Ready in 1,268 s with one
load; GPU peak 9,727 / 20,754 / 8,873 MiB; host peak 163,954 MiB; stream
first token 8.5 s, cancel and reconnect ok; stop 21.4 s, clean.

| Lookup traffic | auto (1M ctx, 4 slots, `-ub 64`) | `-ub 128` | 262k ctx, 1 slot, `-ub 512` |
| --- | --- | --- | --- |
| warm repeat, prompt tok/s | 15.6–16.0 | 20.1–20.7 | **46.9–47.0 (3.0×)** |
| new text, prompt tok/s | 3.2–4.0 | 3.4–4.4 | 3.7–4.9 |
| decode tok/s (8-token replies) | 5.3–8.2 | 6.9–8.0 | 5.1–7.7 |

On this host the automatic plan fills context first: 1,048,576 tokens over 4
slots (the model/policy cap). The compute buffers this needs force `-ub 64`,
and warm prefill drops to a third. For one Claude Code agent, a 262k slot
holds the whole session (peak observed context in earlier live runs:
128,256). The trade is fewer concurrent slots. Cold new text stays
HDD-bound in every shape.

### DeepSeek V4.1 Flash IQ2_XS through the TUI

`ggrun tui` (PATH binary above), started from a terminal, with `LLM_MODEL_DIR`
set to the DeepSeek folder and a copy of the DeepSeek cache. Model picked
from the list, Claude Code turned on (`x`), Launch, Confirm. The dry run
shows the same command as the CLI cells
(`--ctx-size fit --kv-placement auto --support-expert auto --claude-code`,
port 8081).

- Plan as in the CLI cells: paged lookup tier, ctx 1,048,576 (4 slots),
  `-b 128 -ub 64`. Full SWA was turned off at launch, although the settings
  screen showed it on.
- Ready in 20m23s with one load. The live tier check showed 10 mappings,
  375,072 MiB mapped and no table page anonymous or locked. Memory scope:
  high 201,856 MiB, max 205,952 MiB.
- Gateway `127.0.0.1:41303`, Anthropic streaming request: HTTP 200, first
  token 9.5 s, 69 events in 42.9 s. Cancel after 5 events, backend health
  ok; reconnect first token 0.35 s (prefix reused), 20.4 s total. Decode
  2.9 tok/s on a cold page cache.
- The TUI then started Claude Code in its working directory. Declining the
  folder-trust prompt ended the session: ggrun exited 1 with no message.
  The backend stopped within 8 s; there was no leftover process or scope,
  the GPUs were back to 1 MiB, and 205 GiB was available.
- Harness note: piping the TUI's output (`| tee`) leaves it on "Loading..."
  because it never receives a terminal size. Use a real terminal.

### GLM-5.3 UD-IQ2_M without `--mmap` (refusal)

The first GLM cell (no extra flags) planned for 101 s and then refused:
"placement requires file-backed mmap; rerun with --mmap to approve it
explicitly". The model is 223 GB on a 205 GiB host. No weights were loaded,
no backend was left behind, and the error names the approving flag. The
planner had moved between 72 and 76 CPU MoE layers and a context of 346k to
381k tokens before it settled on the last proven plan. The GLM cells were
re-queued with `--mmap`.

### GLM-5.3 UD-IQ2_M cold with `--mmap` (memory.high collapse)

- Model: 6 shards, 238,577,585,701 bytes, on the HDD. Args
  `--claude-code --mmap`. Plan: ctx 233,472 (3 slots), `-b 2048 -ub 256`,
  KV q8_0 on GPU, 201,504 MiB of mapped CPU experts. Two loads (one
  recovery); ready at 3,869 s. GPU peak 11,047 / 22,822 / 10,521 MiB.
- Before ready, the short prefill pilot and the calibration baseline both hit
  their client timeouts, so ggrun served the default placement. The verifier
  measured 0.1 tok/s prefill and checked the completion endpoint only.
- At ready the live pageability check measured 838 MiB non-reclaimable.
  `resizeScopeToMeasuredFootprint` then lowered `memory.high` from 205,963 to
  24,183 MiB; `memory.max` (210,305 MiB) was unchanged. Within 12 minutes the
  backend RSS fell from 199,930 to 22,678 MiB and stayed at 22–24 GiB for the
  rest of the run. `MemAvailable` stayed near 207 GiB throughout.
- Serving after the resize (2.5 h): 4.26 M major faults and 1.16 TiB read
  from the HDD (about 132 MiB/s, its sequential ceiling). IO pressure
  reached 46% full.
- Pilot request: 298 tokens at 0.44 tok/s. Stream: first token after 736 s,
  65 chunks, 3,232 s in total. Cancel: 5 events before the abort, health ok;
  reconnect first token after 319 s. Task `calc` timed out at 3,600 s (0/1);
  `textstats` and `inventory` were skipped. Stop: 3.6 s, exit 0, no leftover
  backend. The shutdown "backend died mid-session" message appeared again.
- The relaunch cell was not run, since it would hit the same throttle. Fix
  (PR #99): `memory.high` = the non-reclaimable ceiling plus the mapped
  experts, capped one headroom below `memory.max`, with a warning when the
  experts do not fit. GLM cold and relaunch are rerun after review and merge.

## Download

Receipts: `.cache/mmap-goal-20261019/download-receipts.jsonl` and
`ds-range-repair.jsonl` (private). Status at 2026-10-21 16:30 UTC: all ten
shards match the pin on buffered and `O_DIRECT` reads.

| Shard | State |
| --- | --- |
| 00002, 00003, 00004, 00006, 00007, 00010 | pinned SHA-256 on buffered and `O_DIRECT` reads |
| 00001 | both reads wrong after transfer; block repair rewrote 1 byte (offset 7,654,585,592, mod16 8, XOR 0x40); now verified on both reads |
| 00005 | buffered right, `O_DIRECT` wrong; block repair rewrote the 2 recorded bytes (17,779,795,818 and 23,447,032,778, mod16 10, XOR 0x10); now verified on both reads |
| 00008 | both reads wrong after transfer; block repair rewrote 2 bytes in one block (92,877,494,250 and 92,877,504,362, mod16 10); now verified on both reads |
| 00009 | wrong after transfer; block repair rewrote 4 bytes in two blocks (three at mod16 10 from 57,313,309,674 to 57,313,335,146, XOR 0x10; one at 63,123,792,334, mod16 14, XOR 0x40, not the DIMM_D1 pattern); now verified on both reads |

On this `fuseblk` (ntfs-3g) mount `O_DIRECT` is served from the block-device
cache, so the two reads are independent RAM copies, not a platter re-read.
Block repair (`range_repair.py`) compares 32 MiB blocks with HTTP range reads
of the pinned revision. A block is rewritten only when two upstream reads
agree and both local reads differ. The original bytes are kept, every rewrite
is read back, the streamed upstream bytes must hash to the pin, and the file
must then match on both reads. The owned failed GLM staging file and the old
Q2_K shards were removed with the owner's approval. Further transfers use
ggrun's production downloader (180 s no-progress stall bound).

## Matrix

Live = through a real backend process on this host. Synthetic = unit
fixtures only; no claim of model support.

| Case | Unit / fixture | Backend fit pass | Live |
| --- | --- | --- | --- |
| Ordinary resident model, no tier | existing suite + `TestOrdinaryModelIgnoresSparseLookupInputs` | — | Qwen3.5-4B control (2026-10-19, branch binary) |
| Explicit mapping on `--load-mode` builds; native/legacy fold, idempotence | `TestLoadMode*`, passthrough and fingerprint tests | — | `--load-mode mmap` observed (2026-10-19) |
| Ordinary oversized mmap (CPU experts) | existing MMAP-1..4 tests; `TestMMapReclaimHighKeepsMappedExperts` (PR #99) | — | GLM-5.3 (2026-10-24): refusal without `--mmap`; with it, `memory.high` collapse and thrash. Rerun after PR #99 |
| Engram, unreviewed loader → layer tier | placement tests | lazy off puts tables on GPU | not run |
| Engram, reviewed loader, roomy host → resident tier | placement test (900 GB) | — | not possible on this host |
| Engram, reviewed loader, this host → paged tier | placement, scope, smaps, oracle tests | real headers, lazy on/off | DeepSeek V4.1 Flash IQ2_XS (2026-10-23, host with faulty pages offlined): cold, relaunch and 4 experiment cells; `ledger` and `calc` correct, one cold timeout |
| Automatic expert paging beside a paged tier refused | placement test | — | not run |
| Explicit `--lazy-mode on` without contract refused | placement test | — | not run |
| Synthetic second architecture (dense, `synthetic-rowtable`) | parser + placement + contract tests, CPU-only / single / multi GPU | — | none (synthetic) |
| Synthetic dense and MoE × 2 GiB and 96 GiB tables × CPU-only / single / multi GPU × unknown / reviewed loader (64 GB host) | `TestSyntheticTierSizeTopologyMatrix`: small → resident (reviewed) or layer; large → paged with execution weights resident (reviewed) or a refusal naming the loader contract | — | none (synthetic) |
| Negative fixtures (no metadata, wrong prefix, unlisted layer, other arch, "ngram" name) | parser tests | — | — |
| Windows | lookup contract is Linux-only (unknown on Windows → layer tier); cross-compile and CI | — | not run (no Windows GPU run) |

## Gates

After the review fixes (2026-10-20): uncached `scripts/verify-core-engine.sh`
passed; `go test -count=1 ./...` passed; `GOOS=windows go vet` of the changed
packages passed; `python3 tests/test_parse_gguf.py` passed. CI results are on
the PR.

After the final-review follow-up (#97) and the size/topology matrix, combined
on main after the #97 merge (2026-10-21): uncached `scripts/verify-core-engine.sh`
passed.

PR #99 (mmap `memory.high`, 2026-10-24): uncached
`scripts/verify-core-engine.sh` passed; `GOOS=windows go vet ./cmd/ggrun
./pkg/server` passed. CI: every Linux and Windows job passed; the macOS
release smoke failed (deprecated, non-blocking).

## Not done

- GLM-5.3 cold and relaunch with `--mmap` after PR #99 is reviewed, merged
  and installed as the PATH binary.
- The DeepSeek cells ran on a host with the faulty DRAM pages offlined, not
  on repaired hardware. The offlining is lost on reboot, and the stick
  swap is still pending.
- The paged tier on an SSD. Only the device was measured: 7,285 random 4 KiB
  reads/s at queue depth 1 and 76,235 at depth 16, against about 150 on the
  HDD. No model run was made, because the SSD has 4.4 GB free.
- Open defects (backlog in `docs/core-standard-launch-todos.md`): the
  shutdown health-monitor message; the auto calibration budget rules out
  every challenger on 20-minute loads; context is filled before ubatch; the
  relaunch does not reuse the measured CPU MoE count; table page cache is
  lost on relaunch.
- Review of PR #99 and of this record.
