# Speculative Decoding

Speculative decoding is useful only when the backend accepts enough proposed
tokens to pay for the extra draft work. llm-server therefore treats speculation
as a validated runtime mode, not a universal default.

## Modes

```bash
llm-server model.gguf --spec off
llm-server model.gguf --spec auto
llm-server model.gguf --spec mtp
llm-server model.gguf --spec eagle3
llm-server model.gguf --spec draft
llm-server model.gguf --spec ngram
llm-server model.gguf --spec ngram-mod
llm-server model.gguf --spec ngram-k4v
```

## Auto Policy

`--spec auto` uses this priority order:

1. MTP if the target GGUF has NextN/MTP metadata and the selected backend
   supports the required flags.
2. EAGLE-3 if the backend advertises EAGLE-3 and a matching speculator is
   available.
3. A compatible draft GGUF found in the model directory or downloaded from
   Hugging Face.
4. Off.

Auto mode does not fall back to ngram speculation. Ngram modes are explicit
because they are prompt-shape dependent.

## Draft Model Search

For `--spec draft`, llm-server searches local files first. If no compatible file
is found, it searches Hugging Face for a small same-family GGUF drafter and
validates the candidate before use.

Validation rejects:

- non-GGUF files
- projector, vision, tokenizer, calibration, and imatrix artifacts
- mismatched vocab or architecture when metadata is available
- candidates that are too large relative to the target model
- backend-specific unsupported draft architectures

The selected draft is passed through normal llama.cpp / ik_llama.cpp draft flags,
for example `--model-draft`, `--ctx-size-draft`, `--cache-type-k-draft`, and
`--draft-max` where supported.

## MTP

MTP is target-model dependent. A separate small model is not enough for MTP mode;
the served GGUF must expose NextN/MTP prediction-layer metadata or the backend
must explicitly support the selected MTP dialect. If the target lacks that
metadata, llm-server skips MTP and reports why.

## Benchmarking

A useful speculative benchmark must record:

- generation tok/s
- draft tokens generated
- draft tokens accepted
- acceptance rate
- output sanity result
- prompt profile

Recent local testing showed:

- structured continuation: 100 percent acceptance and large speedup
- code continuation: low acceptance and slower output

That is the basis for the current conservative policy.
