# Model Recommendations

The GUI shows a `Recommended downloads` fast path for first-run users. It ranks
known GGUF repositories against detected RAM/VRAM and the selected backend path,
including the Linux Vulkan fallback path.

The checked-in catalog lives at `go/pkg/recommend/catalog.json` and is embedded
into the Go binary, so users get recommendations offline.

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
