# Fork backends for new model architectures

New GGUF architectures often work in a llama.cpp or ik_llama.cpp fork before
they reach upstream. ggrun keeps those builds isolated from mainline and routes
only the matching GGUF architecture to them.

## Reviewed recipes

```bash
ggrun backend recipes
ggrun backend install hy3
ggrun backend install minimax-m3
ggrun backend install laguna
ggrun backend install hot-experts
```

A recipe contains a Git repository, branch, immutable commit, backend tag and
optionally a GGUF `general.architecture` route or an exact capability contract.
ggrun clones it under `.src/fork-*`, builds only `llama-server`, registers the
resulting binary and records the exact commit in `.config/backends.json`.
Architecture recipes route matching models automatically; capability-only
performance recipes remain explicit backend choices.

The experimental `hot-experts` implementation is also available as a reviewed
source feature. A standalone recipe pins `csantiago78/llama.cpp` commit
`bccbacdb8945680f1cfc7e6bffd1e59014705750`; for a newer architecture fork,
ggrun can instead apply that exact reviewed change on top of the fork's exact
pinned source revision:

```bash
ggrun backend feature install hot-experts --base glm5next
```

The result is a separate `glm5next-hot-experts` checkout, build, and manifest
entry. The original `glm5next` checkout and binary remain untouched. ggrun
applies the base recipe's reviewed patches first and then the feature patch,
requires the exact `--moe-expert-cache` and `--moe-expert-cache-inserts` help
surface, and registers the composite only after architecture, accelerator, and
server conformance pass. Repeating the command reuses a conformant build or
uses the staged update/rollback path.

This is deterministic source composition, not a promise that a patch can merge
into literally every repository called a llama.cpp fork. The base must have a
recorded Git URL and immutable commit. A compatible llama.cpp-derived graph
usually applies directly; a divergent source family such as ik_llama.cpp needs
a separately reviewed adapter patch. A patch conflict fails before compilation
and leaves the base usable. See
[the eligibility and fallback contract](usage.md#experimental-hot-expert-cache).

The HY3 recipe currently maps `hy_v3` to the reviewed `noonr48/ik_llama-hy3`
`hy3-support` revision. Because this is an IK fork, ggrun preserves the IK flag
dialect behind the friendly `hy3` selector. That server currently limits every
speculative stage chain, including its built-in MTP path, to `--parallel 1`.
For multi-slot launches ggrun leaves speculation off instead of passing flags
that the fork would ignore or reject. Mainline llama.cpp's parallel MTP support
is unaffected by this fork-specific limit.

CUDA defaults to the native GPU architecture. A portable multi-card build can
name architectures explicitly:

```bash
ggrun backend install hy3 --cuda-arch "86;89"
```

The MiniMax-M3 recipe pins the open preliminary-support branch revision that
includes its dedicated structured tool-call parser. Its architecture route is
`minimax-m3`. Sparse attention is not implemented in that revision, so it uses
the mathematically correct dense fallback and remains an experimental backend
until the 60k agent/workflow farm passes on the target machine.

The Laguna recipe pins Poolside's open upstream `llama.cpp` PR revision and
routes GGUFs with `general.architecture=laguna` to that isolated build. Mainline
does not support Laguna until that PR lands. DFlash GGUFs are companion
speculators rather than standalone target models and are not offered in the TUI.

If no reviewed recipe exists, a launch that cannot load the GGUF architecture
searches open `ggml-org/llama.cpp` pull requests for that architecture name and
reads Hugging Face GGUF model cards derived from the file's `quantized_by` /
name metadata (falling back to an architecture search on the Hub). Only
official `ggml-org/llama.cpp/pull/N` links are followed. A publisher-cited PR
ranks ahead of GitHub hits, and a title that adds the architecture ranks ahead
of a later fix PR that only mentions it. An open PR with a cloneable head fork
is offered as `ggrun backend add` (same clone/build/register path as a recipe).
A miss or a declined prompt still offers a mainline llama.cpp update, in case
support has already landed upstream.

Discovery results are unreviewed code, not reviewed recipes. Before a result is
offered, ggrun re-reads the official PR API response and requires the requested
PR number and canonical `ggml-org/llama.cpp` URL, an open/unmerged state, an
architecture mention in the current title or body, an HTTPS `github.com` head
repository, a non-empty branch, and a hexadecimal 40-character commit. Search,
model-card, and PR-detail fetches are bounded. Installation pins that immutable
head commit in a separate `.src/fork-*` checkout and still runs the normal
build/conformance path; it never modifies `.src/llama.cpp`.

## Any other fork

The same workflow accepts an arbitrary fork without adding model-specific code:

```bash
ggrun backend add https://github.com/example/llama.cpp \
  --branch feature/new-model \
  --commit <40-character-commit> \
  --tag new-model \
  --route-arch new_model_arch \
  --accel cuda \
  --cuda-arch "86;89"
```

`--commit` is strongly recommended for reproducibility. Without it, re-running
the command fetches the latest requested branch. ggrun refuses to refresh a fork
checkout with local changes, so experimental edits are never silently erased.

Once a source-built fork has a commit pin, reviewed features can be composed on
top without changing its architecture route:

```bash
ggrun backend feature install hot-experts --base new-model
```

Backends registered only from a binary have no source tree that ggrun can
reproduce, so they cannot be feature bases. Re-add the fork from Git with
`--commit` if source composition is required.

An already-built binary can be registered without cloning:

```bash
ggrun backend register --tag new-model --path /path/to/llama-server \
  --route-arch new_model_arch
```

Use `ggrun backend list` to see the binary, source revision and route. A CLI or
config `--backend` selection remains authoritative and disables automatic
architecture routing for that launch.
