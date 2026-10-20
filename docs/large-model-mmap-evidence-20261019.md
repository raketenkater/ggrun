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

## Download

Receipts: `.cache/mmap-goal-20261019/download-receipts.jsonl` and
`ds-range-repair.jsonl` (private). Status at 2026-10-20 15:30 UTC:

| Shard | State |
| --- | --- |
| 00002, 00003, 00004, 00006, 00007, 00010 | pinned SHA-256 on buffered and `O_DIRECT` reads |
| 00001 | both reads wrong after transfer; block repair rewrote 1 byte (offset 7,654,585,592, mod16 8, XOR 0x40); now verified on both reads |
| 00005 | buffered right, `O_DIRECT` wrong; block repair rewrote the 2 recorded bytes (17,779,795,818 and 23,447,032,778, mod16 10, XOR 0x10); now verified on both reads |
| 00008 | both reads wrong after transfer; block repair queued after the GLM-5.3 run so it does not share the HDD with those measurements |
| 00009 | transferring |

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
| Ordinary oversized mmap (CPU experts) | existing MMAP-1..4 tests | — | not run |
| Engram, unreviewed loader → layer tier | placement tests | lazy off puts tables on GPU | not run |
| Engram, reviewed loader, roomy host → resident tier | placement test (900 GB) | — | not possible on this host |
| Engram, reviewed loader, this host → paged tier | placement, scope, smaps, oracle tests | real headers, lazy on/off | blocked: shard 00008/00009, DRAM fault |
| Automatic expert paging beside a paged tier refused | placement test | — | not run |
| Explicit `--lazy-mode on` without contract refused | placement test | — | not run |
| Synthetic second architecture (dense, `synthetic-rowtable`) | parser + placement + contract tests, CPU-only / single / multi GPU | — | none (synthetic) |
| Negative fixtures (no metadata, wrong prefix, unlisted layer, other arch, "ngram" name) | parser tests | — | — |
| Windows | lookup contract is Linux-only (unknown on Windows → layer tier); cross-compile and CI | — | not run (no Windows GPU run) |

## Gates

After the review fixes (2026-10-20): uncached `scripts/verify-core-engine.sh`
passed; `go test -count=1 ./...` passed; `GOOS=windows go vet` of the changed
packages passed; `python3 tests/test_parse_gguf.py` passed. CI results are on
the PR.

## Not done

Not done: the live 366 GiB load; memory tiers, major faults and HDD reads
under cold/warm and repeated/spread lookup traffic; Claude gateway coding and
tool tasks; cancel, reconnect and relaunch. These need the verified download
and a clean DRAM result. An HDD random-row lookup rate has not been measured;
the expected order is about 48 row gathers per token, mostly separate 4 KiB
pages on a cold cache.
