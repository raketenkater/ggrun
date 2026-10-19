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
   recognises the tier only from `<arch>.engram.layer_ids` under the model's
   own architecture prefix. On the pinned headers the result is 171.72 GiB of
   execution weights + 194.56 GiB tier, the publisher's numbers exactly.
2. No placement tier existed. Placement now refuses a model with this tier on
   a backend without `--lazy-mode`. Otherwise it keeps the tier out of GPU and
   resident host ledgers and emits `--lazy-mode on` explicitly.
3. Exact admission would have refused the plan, because the fit pass reports
   the mapped tables as host model. The Host row is reduced by exactly the
   tier, only under lazy mode on a classified backend.
4. The absent mmap flag meant `auto` on `--load-mode` builds while every
   planner read it as mapped. Every `--load-mode` build, including forks that
   keep deprecated `--mmap/--no-mmap` aliases, now receives `--load-mode mmap`
   or `none` at process start. The planner argv and argv-keyed evidence are
   unchanged.
5. The dry-run printed resident `--no-mmap` CPU experts as "reclaimable
   weights". Only mapped experts on a file-backed loader are labelled
   reclaimable.

New fail-closed rules:

- Automatic expert paging beside a paged lookup tier is refused. Only an
  explicit `--mmap` accepts it as last resort.
- After the canary, `memory.high` = non-reclaimable ceiling + lookup-cache
  allowance (at most the tier, at most the user ceiling) and `memory.max` =
  high + headroom. Anonymous growth squeezes the lookup cache first.
- After load, `/proc/<pid>/smaps` must show the model files mapped with no
  anonymous or locked pages, and memory.stat must not show a copied tier.
  Contradictions are persisted per model/backend. Missing observations are
  logged and never recorded as proof.
- `deepseek41` gets recurrent/hybrid checkpoint semantics from the backend's
  arch list. KV pricing stays generic; exact admission measures it.

## Planner and admission on the real headers (no load)

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
(`memflip.c`, contained scope). It found 63 bit flips:

- every flip is at byte offset ≡ 10 (mod 16), bits 4–7, in both directions;
- the flips repeat at fixed addresses;
- the signature matches the earlier GLM shard (6 flips) and the DeepSeek
  Q2_K page-cache flip.

This is a non-ECC DRAM fault (EDAC: ECC disabled) consistent with one x4 DRAM
device or its data lane. It is independent of the disk and filesystem. The
physical DIMM is not identified; that needs root pagemap/EDAC address decode.
Firmware reports the 2666-rated Team Group A2 DIMM running at 3200 MT/s among
mixed kits. No BIOS change or reboot was made.

Model and runtime correctness cannot be claimed on this host until the fault
is removed. Weights and lookup pages held in RAM can be silently corrupted.

## Download

Receipts: `.cache/mmap-goal-20261019/download-receipts.jsonl` (private).
Shards 00010, 00007, 00002, 00003 and 00004 matched the pinned SHA-256 on the
buffered read and on the `O_DIRECT` read. Shard 00005 matched on the buffered
read but not on the `O_DIRECT` read, and the queue stopped as designed. Two
repeated `O_DIRECT` reads gave the same wrong hash. A byte comparison found two
differing bytes, at offsets 17,779,795,818 and 23,447,032,778. Both are
≡ 10 (mod 16) and differ by XOR `0x10`, the DIMM_D1 signature. The cached
block-device copy is corrupt. Whether the platter copy is clean is unknown
until caches are dropped on clean RAM. Shards 00001, 00006, 00008 and 00009 are
not downloaded. On this `fuseblk` (ntfs-3g) mount `O_DIRECT` is served
from the block-device cache, so the two reads are independent RAM copies, not
a platter re-read. HDD free space (~349 GiB) is below the 366.3 GiB set; the
last 104 GB shard needs the owned failed GLM staging file
(41,914,650,304 B, inode 212143) removed. That removal was not performed.

## Matrix

| Case | Unit/fixture | Backend fit pass | Live |
| --- | --- | --- | --- |
| Ordinary resident (dense) | existing suite | — | Qwen3.5-4B control, mainline |
| Explicit mapping on `--load-mode` builds | translation tests | — | `--load-mode mmap` observed |
| Ordinary oversized mmap (CPU experts) | existing MMAP-1..4 tests | — | not run |
| Sparse-table mmap (Engram) | parser, placement, scope, smaps, fit-row tests | real headers, lazy on/off | blocked: DRAM fault, download |
| Unsupported/anonymous lookup loader refused | placement + contradiction tests | lazy off puts tables on GPU | not run |
| Automatic expert paging beside the tier refused | placement test | — | not run |
| Windows | cross-compile, vet, test build | — | not run (no Windows GPU run) |

Gates run on this branch: `scripts/verify-core-engine.sh` (uncached) passed;
`go test ./...`, `-race` on changed packages, parser tests, shell estimator,
safety and MoE suites, and selected pytest suites all pass.

## Not done

Not done: the live 366 GiB load; memory tiers, major faults and HDD reads
under cold/warm and repeated/spread lookup traffic; Claude gateway coding and
tool tasks; cancel, reconnect and relaunch. These need the verified download
and a clean DRAM result. An HDD random-row lookup rate has not been measured;
the expected order is about 48 row gathers per token, mostly separate 4 KiB
pages on a cold cache.
