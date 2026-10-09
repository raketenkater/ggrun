# Model Recommendations

The TUI and `ggrun recommend` filter models by hardware capacity, choose a
practical fitting quant for each model, and offer three categories:

- **Best overall:** catalog intelligence first among usable models. A model
  predicted below 6 tokens/s decode ranks after every usable one; extra speed
  above that threshold does not override intelligence.
- **Smartest:** highest catalog intelligence that fits, including slower models.
  It uses the same practical quant selection as Best overall, so an unnecessarily
  slow BF16 variant does not hide a model with a usable Q4/Q5 variant.
- **Fastest:** highest estimated speed among models with at least 40% of the
  best fitting model's catalog intelligence, using Q4-class or smaller quants.

Quantization heuristics compare variants of the same model; they do not multiply
another model's benchmark score down. Within a model, a usable 3-bit-or-higher
variant is preferred over 1–2-bit alternatives. Lower-bit variants remain
available when memory or speed requires them.

The `Intel` column shows base-model catalog intelligence from Artificial Analysis;
`~` marks a fallback estimate. It does not measure the accuracy of the downloaded
quant or its ability to complete an agent workflow. The legacy JSON fields
`AdjustedIntelligence` and `QualityRetained` are within-model heuristics, not
measured accuracy. Speed and fit are also planning estimates; launch performs
its own admission checks.

To plan for one GPU and a RAM ceiling:

```bash
ggrun recommend --gpus 0 --ram-budget 32G
ggrun recommend --gpus 0 --ram-budget 32G --json
```

GPU numbers are physical indices from the detected inventory. RAM budgets,
RAM percentages, and RAM/VRAM headroom default to the saved configuration;
command-line options override them. An explicit RAM budget overrides the RAM
percentage, then RAM headroom is subtracted. Recommendations use installed
capacity, so another running model does not reduce the RAM planning budget.
The TUI applies the same configured RAM budget and headroom.

These options restrict recommendations only. Repeat the GPU and memory limits
when launching. They do not emulate a different CPU or memory bandwidth, and
estimated token rates are not benchmark results. `--first` prints only the top
balanced repository; `--json` includes the planning inventory and all categories.

The checked-in catalog lives at `go/pkg/recommend/catalog.json` and is embedded
into the Go binary, so users get recommendations offline. When a user downloads
a recommended model, the downloader searches Hugging Face for matching GGUF
quantization repos and prefers trusted quantizers such as Unsloth and Bartowski
before choosing the best fitting quant.

## Artificial Analysis Refresh

Artificial Analysis data can refresh the catalog through GitHub Actions. Store
your key as the repository secret `ARTIFICIAL_ANALYSIS_API_KEY`; the workflow
also accepts the existing `ARTIFICIALANALYSISAPIKEY` spelling.

The scheduled workflow `.github/workflows/update-recommendations.yml` runs every
three days and can also be started manually. Installed clients refresh their
copy of the published catalog at most once per 24 hours and keep the last valid
catalog when a refresh fails. The workflow calls:

```bash
python3 tools/models/update_recommendations.py
```

The key is read only from the workflow environment and is never written to the
repo. The workflow commits `catalog.json` back to `main` when the API refresh
changes the catalog and the recommendation tests pass on the result.

A quant is offered only when it is a complete main-model artifact: every shard
of one variant, with known sizes. Draft heads (MTP, DFlash), projectors,
adapters and importance matrices are excluded, and two different variants that
share a quant label are not added together.

The catalog marks architectures against upstream llama.cpp, which can be ahead
of the backend you installed. `ggrun recommend` and the TUI probe the installed
backends; a model none of them loads is marked `+` and listed after the models
they load. Its first launch offers a 20-40 minute backend build. The same
applies on an NVIDIA host when a default launch would serve the model on the
Vulkan build (an architecture only mainline loads, or a large MoE whose
file-backed experts win the backend choice): it runs, but far below the CUDA
estimate until the offered CUDA build exists.

Attribution is required when using Artificial Analysis data; the catalog and GUI
include attribution to `https://artificialanalysis.ai/`.
