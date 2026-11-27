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
```

A recipe contains a Git repository, branch, immutable commit, backend tag and
GGUF `general.architecture`. ggrun clones it under `.src/fork-*`, builds only
`llama-server`, registers the resulting binary and records the exact commit in
`.config/backends.json`. Later launches of a matching model route automatically.

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

An already-built binary can be registered without cloning:

```bash
ggrun backend register --tag new-model --path /path/to/llama-server \
  --route-arch new_model_arch
```

Use `ggrun backend list` to see the binary, source revision and route. A CLI or
config `--backend` selection remains authoritative and disables automatic
architecture routing for that launch.

## Status and cleanup

`ggrun backend status` reports, per backend and without network access: the
build's age, whether its build or checkout is off the recipe's pinned commit,
and whether the catalog marks the recipe `superseded_by` something upstream
(with the catalog's evidence). It never changes which backend a launch picks.
`ggrun backend update` prints a one-line note for a superseded entry.

`ggrun backend remove <tag>` unregisters a backend. With `--purge` it also
deletes the backend's `.src/fork-*` source tree and builds. It first lists what
goes, with sizes, and asks (or takes `--yes`; with no terminal and no `--yes`
it refuses). `--dry-run` only lists. It refuses a tree outside `.src`, a
symlinked tree, a tree another registration uses, the mainline and ik
checkouts, and any tree a running process executes from or has as its working
directory; it never kills anything. Where ggrun cannot list processes (Windows,
macOS) it refuses unless `--force-not-running` says you checked.

`ggrun backend gc` lists `.src/fork-*` trees no registration uses, plus
leftover `build-*-prev-*`/`-candidate-*` directories of failed or replaced
updates. It deletes nothing by default; `--delete` asks first, `--yes` does not.
Same refusals, and trees changed in the last 30 minutes are skipped.

Each build records its duration and resulting tree size in `backends.json`
(`build`); the next build of that tag quotes them instead of "30-60 min".
