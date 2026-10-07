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


def test_embedded_mimo_catalog_matches_verified_manifests():
    catalog = json.loads((ROOT / "go/pkg/recommend/catalog.json").read_text())
    rows = {r["repo"]: r for r in catalog["candidates"]}
    for repo, expected in EXPECTED.items():
        assert {q["name"]: q["size_bytes"] for q in rows[repo]["quants"]} == expected
