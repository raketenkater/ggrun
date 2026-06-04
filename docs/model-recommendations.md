# Model Recommendations

The GUI shows a `Recommended downloads` fast path for first-run users. It first
filters known GGUF repositories against detected RAM/VRAM and the selected
backend path, including the Linux Vulkan fallback path, then ranks viable models
by the intelligence signal in the cached catalog. Speed stays visible as
metadata, but it is not part of the ranking score.

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
