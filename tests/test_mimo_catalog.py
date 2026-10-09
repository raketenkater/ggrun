"""Frozen upstream manifests reproduce the MiMo V2.6 sidecar regression offline."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = json.loads((ROOT / "tests/fixtures/mimo_v26_hf_artifacts.json").read_text())
EXPECTED = {
    "pcuenq/MiMo-V2.6-Pro-RL-GGUF": {"MXFP4": 554211765568},
    "ggml-org/MiMo-V2.6-Flash-RL-GGUF": {"Q2_K": 126214593856, "MXFP4": 167369465152},
}


def load_tool(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda m: m["repo"])
def test_mimo_catalog_and_download_select_complete_main_model(manifest, monkeypatch):
    updater = load_tool("tools/models/update_recommendations.py")
    downloader = load_tool("tools/download/download_any_gguf.py")
    repo = manifest["repo"]
    siblings = manifest["siblings"]
    names = [s["rfilename"] for s in siblings]
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda _: {"siblings": siblings})
    monkeypatch.setattr(downloader, "HfApi", lambda: SimpleNamespace(
        model_info=lambda *a, **kw: SimpleNamespace(siblings=[SimpleNamespace(**s) for s in siblings])))
    monkeypatch.setattr(downloader, "list_repo_files", lambda _: names)

    assert {q["name"]: q["size_bytes"] for q in updater.fetch_hf_quants(repo)} == EXPECTED[repo]
    assert dict(downloader.list_available_quantizations(repo)) == EXPECTED[repo]
    for quant, size in EXPECTED[repo].items():
        selected = downloader.get_model_files(repo, quant)
        assert selected
        assert all(not updater.is_draft_head_gguf(p) and "mmproj" not in p for p in selected)
        assert sum(s["lfs"]["size"] for s in siblings if s["rfilename"] in selected) == size
    for companion_quant in ("BF16", "Q4_0", "Q8_0"):
        assert downloader.get_model_files(repo, companion_quant) == []


def test_embedded_mimo_catalog_offers_only_complete_main_models():
    """The embedded catalog is regenerated on a schedule and may resolve another
    repository for the same model, so check what must hold for any correct
    regeneration rather than pinning today's upstream bytes: the MiMo rows offer
    no companion-sized quant, and every offered quant is a full main-model total."""
    catalog = json.loads((ROOT / "go/pkg/recommend/catalog.json").read_text())
    rows = {r["name"]: r for r in catalog["candidates"]}
    for name in ("Xiaomi MiMo-V2.6-Pro", "Xiaomi MiMo-V2.6-Flash"):
        if name not in rows:
            continue  # the source leaderboard may drop a model; absence is not a wrong offer
        quants = rows[name]["quants"]
        assert quants, name
        assert all(q["size_bytes"] >= 100 * 1024**3 for q in quants), (name, quants)
    for row in catalog["candidates"]:
        sizes = [q["size_bytes"] for q in row["quants"] if q.get("size_bytes")]
        if sizes:
            assert min(sizes) * 25 >= max(sizes), (row["name"], row["repo"], row["quants"])
