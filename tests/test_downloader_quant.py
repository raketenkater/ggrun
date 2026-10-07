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


def test_draft_heads_are_neither_listed_nor_downloaded_as_the_model(monkeypatch):
    downloader = load_downloader()
    names = [
        "Model-Q8_0-00001-of-00002.gguf", "Model-Q8_0-00002-of-00002.gguf", "Model-Q6_K.gguf",
        "mtp-Model-BF16.gguf", "mtp-Model-Q8_0.gguf", "dflash-Model-BF16.gguf",
        "MTP/mtp-Model-shared-Q8_0.gguf", "mmproj-Model-Q8_0.gguf",
    ]

    class FakeHfApi:
        def model_info(self, repo, files_metadata=True):
            return _Info([_Sibling(name, 1) for name in names])

    monkeypatch.setattr(downloader, "HfApi", FakeHfApi)
    monkeypatch.setattr(downloader, "list_repo_files", lambda repo: names)
    assert {name for name, _ in downloader.list_available_quantizations("owner/repo")} == {"Q8_0", "Q6_K"}
    assert downloader.get_model_files("owner/repo", "Q8_0") == [
        "Model-Q8_0-00001-of-00002.gguf", "Model-Q8_0-00002-of-00002.gguf", "mmproj-Model-Q8_0.gguf"]
    assert downloader.get_model_files("owner/repo", "BF16") == []


def test_a_draft_only_repo_still_downloads(monkeypatch):
    downloader = load_downloader()
    monkeypatch.setattr(downloader, "list_repo_files", lambda repo: ["dflash-Model-BF16.gguf", "README.md"])
    assert downloader.get_model_files("owner/Model-DFlash", "BF16") == ["dflash-Model-BF16.gguf"]


def test_listing_and_download_use_the_final_quant_and_complete_shards(monkeypatch):
    downloader = load_downloader()
    names = ["Model-BF16-Q4_K_M-00001-of-00002.gguf",  # incomplete: no offer/download
             "Model-BF16-Q8_0-00001-of-00002.gguf",
             "Model-BF16-Q8_0-00002-of-00002.gguf",
             "Model-BF16-UD-Q6_K.gguf"]
    class FakeHfApi:
        def model_info(self, repo, files_metadata=True):
            return _Info([_Sibling(n, 1024**3) for n in names])
    monkeypatch.setattr(downloader, "HfApi", FakeHfApi)
    monkeypatch.setattr(downloader, "list_repo_files", lambda repo: names)
    assert dict(downloader.list_available_quantizations("owner/model")) == {
        "Q8_0": 2 * 1024**3, "UD-Q6_K": 1024**3}
    assert downloader.get_model_files("owner/model", "BF16") == []
    assert downloader.get_model_files("owner/model", "Q4_K_M") == []
    assert downloader.get_model_files("owner/model", "Q8_0") == names[1:3]
    assert downloader.get_model_files("owner/model", "UD-Q6_K") == names[3:]


def test_missing_shard_size_does_not_underestimate_download(monkeypatch):
    downloader = load_downloader()
    class FakeHfApi:
        def model_info(self, repo, files_metadata=True):
            return _Info([_Sibling("m-Q4_K_M-00001-of-00002.gguf", 1024**3),
                          _Sibling("m-Q4_K_M-00002-of-00002.gguf", None)])
    monkeypatch.setattr(downloader, "HfApi", FakeHfApi)
    assert downloader.list_available_quantizations("owner/model") == []


def test_repo_search_does_not_substitute_a_draft_for_main_model():
    downloader = load_downloader()
    assert not downloader._repo_relevant("unsloth/Qwen3.8-Flash-Next-MTP-GGUF", "Qwen/Qwen3.8-Flash-Next")
    assert downloader._repo_relevant("unsloth/Qwen3.8-Flash-Next-GGUF", "Qwen/Qwen3.8-Flash-Next")


def test_ambiguous_models_are_neither_combined_nor_downloaded(monkeypatch):
    downloader = load_downloader()
    names = ["Model-2B-Q4_K_M.gguf", "Model-20B-Q4_K_M.gguf"]
    monkeypatch.setattr(downloader, "list_repo_files", lambda repo: names)
    assert downloader.model_quant_groups(names) == {}
    assert downloader.get_model_files("owner/model", "Q4_K_M") == []


def test_calibration_ggufs_are_never_downloaded_as_models(monkeypatch):
    downloader = load_downloader()
    names = ["Model-BF16-imatrix.gguf", "Model-BF16-Q4_K_M.gguf"]
    monkeypatch.setattr(downloader, "list_repo_files", lambda repo: names)
    assert downloader.get_model_files("owner/model", "BF16") == []
    assert downloader.get_model_files("owner/model", None) == names[1:]
    assert downloader.without_draft_heads(names[:1]) == []


def test_bare_quant_request_still_accepts_dynamic_files(monkeypatch):
    downloader = load_downloader()
    names = ["Model-UD-Q4_K_XL.gguf"]
    monkeypatch.setattr(downloader, "list_repo_files", lambda repo: names)
    monkeypatch.setattr(downloader, "list_available_quantizations", lambda repo: [("UD-Q4_K_XL", gib(4))])
    assert downloader.select_quantization("owner/model", requested_quant="Q4_K_XL") == "UD-Q4_K_XL"
    assert downloader.get_model_files("owner/model", "Q4_K_XL") == names
    names.append("Model-Q4_K_XL.gguf")
    assert downloader.get_model_files("owner/model", "Q4_K_XL") == names[1:]
    assert downloader.get_model_files("owner/model", "UD-Q4_K_XL") == names[:1]
