# Speculative Decoding

Speculative decoding is useful only when the backend accepts enough proposed
tokens to pay for the extra draft work. ggrun therefore treats speculation
as a validated runtime mode, not a universal default.

## Modes

```bash
ggrun model.gguf --spec off
ggrun model.gguf --spec auto
ggrun model.gguf --spec mtp
ggrun model.gguf --spec dflash
ggrun model.gguf --spec eagle3
ggrun model.gguf --spec draft
ggrun model.gguf --spec ngram
ggrun model.gguf --spec ngram-mod
ggrun model.gguf --spec ngram-k4v
```

## Auto Policy

`--spec auto` uses this priority order:

1. Embedded MTP if the target GGUF has NextN metadata, or a compatible
   MTP-only companion when mainline llama.cpp advertises `draft-mtp`, but only
   when an exact matching `ggrun spec-test` profile proves it is faster.
2. A target-specific DFlash/DSpark companion when the backend advertises
   `draft-dflash`, its own loader accepts the companion, and an exact performance
   profile proves the configuration.
3. EAGLE-3 if the backend advertises EAGLE-3 and a matching speculator is
   available.
4. A compatible draft GGUF found in the model directory or downloaded from
   Hugging Face.
5. Off.

Auto mode does not fall back to ngram speculation. Ngram modes are explicit
because they are prompt-shape dependent.

## Draft Model Search

For `--spec draft`, ggrun searches local files first. If no compatible file
is found, it searches Hugging Face for a small same-family GGUF drafter and
validates the candidate before use.

Validation rejects:

- non-GGUF files
- projector, vision, tokenizer, calibration, and imatrix artifacts
- mismatched vocab, embedding width, tokenizer model/preprocessor, or architecture
- candidates that are too large relative to the target model
- backend-specific unsupported draft architectures
- GGUF architectures or tensor types the selected backend cannot actually load

The selected draft is passed through normal llama.cpp / ik_llama.cpp draft flags,
for example `--model-draft`, `--ctx-size-draft`, `--cache-type-k-draft`, and
`--draft-max` where supported.

Large interrupted downloads retain their `.tmp` file and resume with an HTTP
Range request on the next run. Known target-specific companions may pin an
immutable Hugging Face revision; every completed file is parsed again locally
and passed through the selected backend's no-allocation loader when available.

## MTP and DFlash

MTP is target-model dependent. ggrun accepts either NextN layers embedded in the
served GGUF or a small same-architecture GGUF whose metadata explicitly contains
NextN prediction layers. It rejects a full replacement model and cross-family
files even when their repository names contain the target name.

MTP uses a two-token draft ceiling when explicitly selected. That matches Qwen's
official Qwen3.5 serving recipe and avoids reusing the much larger ceiling intended for
an independent tiny draft model. The best value is still backend, model,
hardware, sampling, context and concurrency dependent. Auto therefore uses only
the ceiling from a matching cached A/B profile; without one, MTP remains off.

DFlash is a different speculative architecture, not an MTP alias. DeepSeek V4
Flash currently has no NextN/MTP tensors. The apparent
`deepseek4-dflash-draft` artifact is not enabled: mainline llama.cpp rejects its
private GGML tensor type, while the public DS4/Lucebox runtimes that know that
type do not expose a llama-server backend capable of loading this target/draft
pair. Auto therefore serves DeepSeek V4 without speculation until a reproducible
compatible pair exists.

Before launch, a separate companion is run through the selected backend's
no-allocation memory oracle. Its measured per-GPU model, context, and compute
bytes are added to the target preflight so MoE placement is corrected before a
real load. If the companion fails that check, ggrun disables speculation and
recomputes the target-only placement instead of gambling on startup.

For embedded mainline MTP, the target no-allocation ledger does not include its
second context. ggrun derives the prediction-layer KV allocation from GGUF
metadata, adds a conservative compute reserve on every active GPU, and disables
speculation if that bound cannot fit or cannot be established.

## Benchmarking

Run the implemented core matrix with the same launch flags that production will
use. For the Claude Code target profile:

```bash
ggrun spec-test model.gguf --claude-code --ctx 1048576 --parallel 4 --rounds 2
```

The command launches a target-only baseline and MTP ceilings 1 through 4, with a
warmup before each configuration. It runs nine prompt classes twice, checks a
unique output marker, includes a 60k prompt when per-slot capacity allows it,
writes raw JSON under the ggrun cache, and saves an exact scoped performance
profile. Saving a profile does not imply activation: Auto consumes it only when
decode and end-to-end wall time both improve by at least 2%, prompt processing
regresses by no more than 5%, mean output length stays within 10%, and all
required stability/load checks pass. The final post-tuning server arguments are
fingerprinted too; a changed placement or tune cache disables speculation and
recomputes target-only serving instead of reusing stale evidence.

A useful speculative benchmark must compare the exact same target GGUF with
speculation off and on, after warmup, and record:

- backend commit, model hash, hardware and all sampling parameters
- prompt-eval speed/TTFT separately from decode speed
- generation tok/s
- draft tokens generated
- draft tokens accepted
- acceptance rate
- mean accepted length and end-to-end wall time
- serial latency and parallel throughput as separate results
- short and 60k-context requests, peak VRAM/RAM and stability
- output sanity result
- prompt profile

The workload must include code, explanation, summary, factual QA, translation,
creative text, math and a long agent/code-review prompt. Sweep small MTP ceilings
(1-4) under both deterministic and the model's recommended sampling. llama.cpp's
merged MTP implementation explicitly notes slower prompt processing and that
parallel decoding works but is not fully optimized, so a serial win cannot be
promoted to Claude Code's parallel-4 profile without its own test.

The initial local smoke test (Qwen3.5-4B Q4_K_M, CUDA) showed why this matters:

- a generic ceiling of 16 regressed from about 182 tok/s to 110-134 tok/s
- a ceiling of 2 reached about 219-234 tok/s on three greedy code prompts
- that is evidence for the safer default, not a generic performance result

Auto caches a winner by artifact/backend/hardware/GPU-set/context/sampling and
parallel scope and otherwise remains conservative. The remaining extended matrix
(sampling and thinking variants, TTFT, resource peaks, and soak testing) is tracked
in `TODO.md` and must be completed before broad performance claims.
