# MiMo V2.6 recommendation repair — 2026-10-06

The October 4 catalog offered MTP and DFlash companions as small main-model
quantizations of MiMo V2.6 Pro and Flash. Their leaderboard scores then applied
to files that cannot serve the main model. Flash's real Q2_K artifact was also
missing because the old quant regex did not recognize bare K quants.

The fix starts with PR #80's draft filtering and bare-K recognition, and adds
complete artifact validation in the catalog generator and downloader. Draft-only
repositories cannot supply a scored main-model recommendation; deliberately
downloading a draft-only repository remains supported. Incomplete shards,
unknown sizes, ambiguous model variants, projectors, adapters, and importance
matrices cannot become deceptively small catalog quants. The final quant label
wins when a model name itself contains BF16. Legacy bare/dynamic quant aliases
remain compatible.

Only the two MiMo catalog rows were refreshed. Other model rows, intelligence
scores, ranking policy, backend selection, and protected serving code are
unchanged. Header range reads confirmed the existing MiMo architecture fields.

## Upstream evidence

These are weight-file totals in the selected repositories, excluding companions
and runtime overhead; they are not measured serving-memory requirements.

| Model | Repository revision | Main quants and total bytes |
| --- | --- | --- |
| [Pro](https://huggingface.co/pcuenq/MiMo-V2.6-Pro-RL-GGUF) | `94312aea7b760effb1e90e944d08828b1c3fefa6` | MXFP4: 554,211,765,568 bytes (516.15 GiB) |
| [Flash](https://huggingface.co/ggml-org/MiMo-V2.6-Flash-RL-GGUF) | `a5d1269fdcf9c3346934411f700038e9cd9d5102` | Q2_K: 126,214,593,856 bytes (117.55 GiB); MXFP4: 167,369,465,152 bytes (155.87 GiB) |

The frozen public API manifests, old catalog quants, and header comparison are
in `tests/fixtures/mimo_v26_hf_artifacts.json`. Regression tests check exact
main-weight totals and agreement between catalog offers and download selection.
Both regular Python CI and release verification include these tests.

## Verification

- `python3 -m pytest -q tests/test_update_recommendations.py tests/test_downloader_quant.py tests/test_mimo_catalog.py tests/test_parse_gguf.py`: **55 passed**.
- `go test -p 2 -count=1 ./pkg/recommend`: **passed**.
- Simulated 12 GiB GPU / 32 GiB RAM: neither Pro nor Flash is offered.
- Simulated 48 GiB GPU / 256 GiB RAM: Flash remains eligible; Pro is rejected.
- `git diff --check`: passed.

These checks validate recommendation and artifact selection. No model weights
were downloaded and no serving-performance or launch claim is made. The active
worker's model runs and working tree were left alone.

## Integration

Prepared on `fix/catalog-artifact-validation`, based on main `98748078`, with
PR #80 cherry-picked as `e359a19`. This is a local repair, not a public release
or replacement of the installed binary. Integrate PR #80 and this follow-up
together so daily catalog regeneration cannot reintroduce the bad entries.
The repaired catalog can then reach existing clients through their catalog
refresh; newly built binaries embed it. No remote merge or release is claimed.
