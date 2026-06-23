import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_downloader():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools" / "download" / "download_any_gguf.py"
    spec = importlib.util.spec_from_file_location("download_any_gguf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gib(value):
    return int(value * 1024**3)


def test_moe_quant_prefers_vram_resident_fit_over_larger_ram_spill():
    downloader = load_downloader()
    quant_list = [
        ("Q4_K_M", gib(20.6)),
        ("BF16", gib(64.6)),
    ]

    quant, reason = downloader.recommend_quant(
        quant_list,
        vram_mb=24 * 1024,
        ram_mb=64 * 1024,
        repo="unsloth/Qwen3.6-35B-A3B-GGUF",
    )

    assert quant == "Q4_K_M"
    assert "entirely in VRAM" in reason


def test_quant_uses_total_memory_when_no_vram_fit_exists():
    downloader = load_downloader()
    quant_list = [
        ("IQ2_XXS", gib(10.0)),
        ("Q4_K_M", gib(20.6)),
    ]

    quant, reason = downloader.recommend_quant(
        quant_list,
        vram_mb=12 * 1024,
        ram_mb=64 * 1024,
        repo="unsloth/Qwen3.6-35B-A3B-GGUF",
    )

    assert quant == "Q4_K_M"
    assert "offload" in reason


class _Sibling:
    def __init__(self, name, size):
        self.rfilename = name
        self.size = size


class _Info:
    def __init__(self, siblings):
        self.siblings = siblings


def test_quant_pattern_lists_bare_k_and_legacy_one_quants(monkeypatch):
    downloader = load_downloader()

    class FakeHfApi:
        def model_info(self, repo, files_metadata=True):
            return _Info([
                _Sibling("model-Q6_K.gguf", 6),
                _Sibling("model-Q2_K.gguf", 2),
                _Sibling("model-Q4_1.gguf", 4),
                _Sibling("model-Q5_1.gguf", 5),
                _Sibling("model-Q4_K_M.gguf", 44),
                _Sibling("model-Q8_0.gguf", 8),
            ])

    monkeypatch.setattr(downloader, "HfApi", FakeHfApi)
    quants = {name for name, _ in downloader.list_available_quantizations("owner/repo")}

    assert {"Q6_K", "Q2_K", "Q4_1", "Q5_1", "Q4_K_M", "Q8_0"} <= quants


def test_main_exits_nonzero_and_skips_index_on_partial_download(monkeypatch, tmp_path):
    downloader = load_downloader()
    index_calls = []

    monkeypatch.setattr(downloader, "get_args", lambda: SimpleNamespace(
        repo="owner/model",
        dir=str(tmp_path),
        vram=0,
        ram=0,
        cache_dir=None,
        quant="",
        no_repo_search=True,
    ))
    monkeypatch.setattr(downloader, "clear_screen", lambda: None)
    monkeypatch.setattr(downloader, "print_header", lambda: None)
    monkeypatch.setattr(downloader, "resolve_best_gguf_repo", lambda repo, *args: repo)
    monkeypatch.setattr(downloader, "select_quantization", lambda *args: "Q4_K_M")
    monkeypatch.setattr(downloader, "get_model_files", lambda *args: [
        "model-00001-of-00002.gguf",
        "model-00002-of-00002.gguf",
    ])
    monkeypatch.setattr(downloader, "get_download_directory", lambda default_path: tmp_path)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    monkeypatch.setattr(downloader, "download_files", lambda *args: (
        [("model-00001-of-00002.gguf", tmp_path / "model-00001-of-00002.gguf")],
        ["model-00002-of-00002.gguf"],
    ))
    monkeypatch.setattr(downloader, "update_model_index", lambda *args: index_calls.append(args))
    monkeypatch.setattr(downloader, "print_usage_instructions", lambda *args: None)

    with pytest.raises(SystemExit) as exc:
        downloader.main()

    assert exc.value.code == 1
    assert index_calls == []


def test_main_exits_nonzero_when_no_files_downloaded(monkeypatch, tmp_path):
    downloader = load_downloader()

    monkeypatch.setattr(downloader, "get_args", lambda: SimpleNamespace(
        repo="owner/model",
        dir=str(tmp_path),
        vram=0,
        ram=0,
        cache_dir=None,
        quant="",
        no_repo_search=True,
    ))
    monkeypatch.setattr(downloader, "clear_screen", lambda: None)
    monkeypatch.setattr(downloader, "print_header", lambda: None)
    monkeypatch.setattr(downloader, "resolve_best_gguf_repo", lambda repo, *args: repo)
    monkeypatch.setattr(downloader, "select_quantization", lambda *args: "Q4_K_M")
    monkeypatch.setattr(downloader, "get_model_files", lambda *args: ["model.gguf"])
    monkeypatch.setattr(downloader, "get_download_directory", lambda default_path: tmp_path)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    monkeypatch.setattr(downloader, "download_files", lambda *args: ([], []))

    with pytest.raises(SystemExit) as exc:
        downloader.main()

    assert exc.value.code == 1


def test_select_quantization_recommends_with_ram_only_budget(monkeypatch):
    downloader = load_downloader()
    monkeypatch.setattr(downloader, "list_repo_files", lambda repo: ["model-Q4_K_M.gguf"])
    monkeypatch.setattr(downloader, "list_available_quantizations", lambda repo: [
        ("Q2_K", gib(10.0)),
        ("Q4_K_M", gib(20.0)),
    ])
    monkeypatch.setattr(builtins, "input", lambda prompt="": "")

    selected = downloader.select_quantization(
        "owner/model-GGUF",
        vram_mb=0,
        ram_mb=64 * 1024,
    )

    assert selected == "Q4_K_M"


def test_select_quantization_accepts_unsloth_dynamic_catalog_alias(monkeypatch):
    downloader = load_downloader()
    monkeypatch.setattr(downloader, "list_repo_files", lambda repo: ["model-Q4_K_XL.gguf"])
    monkeypatch.setattr(downloader, "list_available_quantizations", lambda repo: [
        ("Q4_K_XL", gib(2.7)),
        ("Q5_K_M", gib(2.9)),
    ])
    monkeypatch.setattr(builtins, "input", lambda prompt="": pytest.fail("should not prompt"))

    selected = downloader.select_quantization(
        "unsloth/Qwen3.5-4B-GGUF",
        requested_quant="UD-Q4_K_XL",
    )

    assert selected == "Q4_K_XL"


def test_get_model_files_accepts_unsloth_dynamic_catalog_alias(monkeypatch):
    downloader = load_downloader()
    monkeypatch.setattr(downloader, "list_repo_files", lambda repo: [
        "model-Q4_K_XL.gguf",
        "model-Q5_K_M.gguf",
    ])

    files = downloader.get_model_files("unsloth/Qwen3.5-4B-GGUF", "UD-Q4_K_XL")

    assert files == ["model-Q4_K_XL.gguf"]
