#!/usr/bin/env python3
"""Model index for llm-server model management.

Maintains <model-dir>/.llm-server/models.json from local GGUF files, optional
download metadata, local mmproj files, and AI Tune cache files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from parse_gguf import parse as parse_gguf
except Exception:  # pragma: no cover - exercised by shell fallback behavior
    parse_gguf = None


INDEX_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def index_path(model_dir: Path) -> Path:
    return model_dir / ".llm-server" / "models.json"


def load_index(model_dir: Path) -> dict[str, Any]:
    path = index_path(model_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"version": INDEX_VERSION, "models": []}


def save_index(model_dir: Path, data: dict[str, Any]) -> None:
    path = index_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def is_runnable_gguf(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".gguf"):
        return False
    if "mmproj" in name or "projector" in name:
        return False
    match = re.search(r"-(\d+)-of-(\d+)\.gguf$", name)
    if match and match.group(1) != "00001":
        return False
    return True


def split_pattern(path: Path) -> str:
    return re.sub(r"-\d+-of-\d+\.gguf$", "*.gguf", path.name)


def total_size(path: Path) -> int:
    pattern = split_pattern(path)
    if "*" not in pattern:
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.parent.glob(pattern) if p.is_file())


def detect_quant(name: str) -> str:
    match = re.search(
        r"(IQ[2-8]_(?:XXS|XS|NL|S|M|L)|Q[2-9]_(?:K_?(?:XL|XL_?M|L|S|M)|0|[1-9]_?[KS])|MXFP4|MXP4|BF16|F16|F32|F8|I4)",
        name,
        re.IGNORECASE,
    )
    return match.group(1).upper() if match else "unknown"


def parse_metadata(path: Path) -> dict[str, Any]:
    if parse_gguf is None:
        return {}
    try:
        meta = parse_gguf(str(path))
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def mmproj_candidates(model_path: Path, model_dir: Path) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    for root in (model_path.parent, model_dir):
        if not root.exists():
            continue
        for path in root.glob("*.gguf"):
            low = path.name.lower()
            if "mmproj" in low or "projector" in low:
                candidates.append(path)
    seen: set[Path] = set()
    rows: list[dict[str, Any]] = []
    for path in candidates:
        real = path.resolve()
        if real in seen:
            continue
        seen.add(real)
        meta = parse_metadata(path)
        rows.append(
            {
                "path": str(path),
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "name": meta.get("name", ""),
                "basename": meta.get("basename", ""),
            }
        )
    return sorted(rows, key=lambda row: row["file"])


def tune_configs(cache_dir: Path, model_file: str) -> list[dict[str, Any]]:
    if not cache_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in cache_dir.glob(f"tune_{model_file}_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("model") != model_file:
            continue
        best = data.get("best_config") or {}
        flags = best.get("flags") or {}
        if not isinstance(flags, dict):
            flags = {}
        gen = best.get("gen_tps") or 0
        base = data.get("baseline_gen_tps") or 0
        try:
            gen_f = float(gen)
            base_f = float(base)
        except Exception:
            gen_f = 0.0
            base_f = 0.0
        rows.append(
            {
                "path": str(path),
                "file": path.name,
                "backend": "ik_llama" if path.name.endswith("_ik.json") else "llama",
                "vision": "_v_" in path.name,
                "placement": "_unlimited" in path.name
                or any(k in flags for k in ("--tensor-split", "--split-mode", "--device", "-mg", "--ngl")),
                "gen_tps": gen_f,
                "gain_pct": ((gen_f - base_f) / base_f * 100.0) if base_f > 0 else 0.0,
                "tuned_at": data.get("tuned_at", ""),
            }
        )
    return sorted(rows, key=lambda row: row.get("gen_tps", 0), reverse=True)


def previous_downloads(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for model in index.get("models", []):
        if isinstance(model, dict) and model.get("path"):
            rows[model["path"]] = model.get("download", {}) if isinstance(model.get("download"), dict) else {}
    return rows


def scan_models(model_dir: Path, cache_dir: Path) -> dict[str, Any]:
    old = load_index(model_dir)
    downloads = previous_downloads(old)
    models: list[dict[str, Any]] = []
    for path in sorted(model_dir.rglob("*.gguf")):
        if not is_runnable_gguf(path):
            continue
        meta = parse_metadata(path)
        size_bytes = total_size(path)
        model: dict[str, Any] = {
            "path": str(path),
            "relative_path": str(path.relative_to(model_dir)) if path.is_relative_to(model_dir) else path.name,
            "file": path.name,
            "size_bytes": size_bytes,
            "size_gb": round(size_bytes / 1073741824, 2),
            "mtime": int(path.stat().st_mtime),
            "quant": detect_quant(path.name),
            "architecture": meta.get("arch", "unknown"),
            "name": meta.get("name", ""),
            "basename": meta.get("basename", ""),
            "quantized_by": meta.get("quantized_by", ""),
            "layers": int(meta.get("layers", 0) or 0),
            "experts": int(meta.get("experts", 0) or 0),
            "ctx_train": int(meta.get("ctx_train", 0) or 0),
            "moe": bool(int(meta.get("experts", 0) or 0) > 0 or re.search(r"A\d+B", path.name)),
            "mmproj": mmproj_candidates(path, model_dir),
            "tune_configs": tune_configs(cache_dir, path.name),
        }
        if downloads.get(str(path)):
            model["download"] = downloads[str(path)]
        models.append(model)
    return {"version": INDEX_VERSION, "generated_at": utc_now(), "model_dir": str(model_dir), "models": models}


def update_download(model_dir: Path, cache_dir: Path, repo: str, quant: str, files: list[str]) -> dict[str, Any]:
    data = scan_models(model_dir, cache_dir)
    files_by_name = {Path(f).name for f in files}
    for model in data["models"]:
        if model["file"] in files_by_name or model["relative_path"] in files:
            model["download"] = {
                "repo": repo,
                "quant": quant or "all",
                "downloaded_at": utc_now(),
                "files": files,
            }
    save_index(model_dir, data)
    return data


def gui_rows(data: dict[str, Any]) -> None:
    for model in data.get("models", []):
        arch = "MoE" if model.get("moe") else (model.get("architecture") or "dense")
        parts = [f"{model.get('size_gb', 0):.1f}GB", arch]
        quant = model.get("quant")
        if quant and quant != "unknown":
            parts.append(quant)
        if model.get("mmproj"):
            parts.append("vision")
        tune_count = len(model.get("tune_configs") or [])
        if tune_count:
            parts.append(f"{tune_count} cfg")
        repo = (model.get("download") or {}).get("repo")
        if repo:
            parts.append(repo)
        print(f"{model['path']}\t{model['file']} ({', '.join(parts)})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and inspect llm-server model index")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "llm-server"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--format", choices=("json", "gui"), default="json")
    upd = sub.add_parser("update-download")
    upd.add_argument("--repo", required=True)
    upd.add_argument("--quant", default="")
    upd.add_argument("--file", action="append", default=[])
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "scan":
        data = scan_models(model_dir, cache_dir)
        save_index(model_dir, data)
        if args.format == "gui":
            gui_rows(data)
        else:
            print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    if args.cmd == "update-download":
        data = update_download(model_dir, cache_dir, args.repo, args.quant, args.file)
        print(json.dumps({"index": str(index_path(model_dir)), "models": len(data.get("models", []))}, sort_keys=True))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
