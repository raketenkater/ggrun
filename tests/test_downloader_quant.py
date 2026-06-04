import importlib.util
from pathlib import Path


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
