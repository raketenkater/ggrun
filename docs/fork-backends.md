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
