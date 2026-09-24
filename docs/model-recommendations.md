# Model Recommendations

The TUI and `ggrun recommend` offer balanced, smartest, and fastest categories.
They filter the catalog using hardware capacity and rank candidates using catalog
intelligence, estimated local speed, quantization quality, and fit. Speed and fit
are planning estimates; launch performs its own admission checks.

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

The scheduled workflow `.github/workflows/update-recommendations.yml` runs weekly
and can also be started manually. It calls:

```bash
python3 tools/models/update_recommendations.py
```

The key is read only from the workflow environment and is never written to the
repo. The workflow commits `catalog.json` back to `main` when the API refresh
changes the catalog.

Attribution is required when using Artificial Analysis data; the catalog and GUI
include attribution to `https://artificialanalysis.ai/`.
