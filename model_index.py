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
                "selectable": bool(flags) and gen_f > 0,
                "baseline_wins": bool(data.get("baseline_wins")),
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
            "tokenizer_model": meta.get("tokenizer_model", ""),
            "tokenizer_pre": meta.get("tokenizer_pre", ""),
            "vocab_size": int(meta.get("vocab_size", 0) or 0),
            "layers": int(meta.get("layers", 0) or 0),
            "experts": int(meta.get("experts", 0) or 0),
            "ctx_train": int(meta.get("ctx_train", 0) or 0),
            "moe": bool(int(meta.get("experts", 0) or 0) > 0 or re.search(r"A\d+B", path.name)),
            "mmproj": mmproj_candidates(path, model_dir),
            "tune_configs": tune_configs(cache_dir, path.name),
        }
        model["role"] = "draft" if draft_backend_requirements(model) else "model"
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


def draft_backend_requirements(model: dict[str, Any]) -> dict[str, Any]:
    arch = str(model.get("architecture") or "").lower()
    name = f"{model.get('file') or ''} {model.get('name') or ''} {model.get('basename') or ''}".lower()
    if arch == "dflash-draft" or "dflash" in name:
        return {
            "backend": "buun-llama-cpp",
            "capability": "dflash",
            "flags": ["--spec-type", "dflash"],
            "reason": "requires buun-llama-cpp DFlash backend",
        }
    return {}


def backend_satisfies_requirements(requirements: dict[str, Any], backend: str, supports_dflash: bool) -> bool:
    if not requirements:
        return True
    capability = requirements.get("capability")
    required_backend = requirements.get("backend")
    if capability == "dflash" and (supports_dflash or backend == required_backend):
        return True
    return False


def suggest_drafts(
    data: dict[str, Any],
    target: dict[str, Any],
    *,
    backend: str = "generic",
    supports_dflash: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_size = int(target.get("size_bytes", 0) or 0)
    target_tok = (target.get("tokenizer_model") or "", target.get("tokenizer_pre") or "", target.get("vocab_size") or 0)
    for model in data.get("models", []):
        if model.get("path") == target.get("path"):
            continue
        size = int(model.get("size_bytes", 0) or 0)
        reasons: list[str] = []
        score = 0
        compatible = True
        auto_enable = True

        if target_size and size >= target_size:
            compatible = False
            auto_enable = False
            reasons.append("not smaller than target")
        elif target_size and size <= target_size * 0.25:
            score += 30
            reasons.append("small enough for draft")
        else:
            score += 5
            reasons.append("smaller than target")

        draft_tok = (model.get("tokenizer_model") or "", model.get("tokenizer_pre") or "", model.get("vocab_size") or 0)
        if target_tok == draft_tok:
            score += 60
            reasons.append("exact tokenizer match")
        else:
            compatible = False
            auto_enable = False
            reasons.append(
                "tokenizer mismatch "
                f"target={target_tok[0]}/{target_tok[1]}/{target_tok[2]} "
                f"draft={draft_tok[0]}/{draft_tok[1]}/{draft_tok[2]}"
            )

        if model.get("moe"):
            score -= 10
            reasons.append("draft is MoE")
        else:
            score += 5
            reasons.append("draft is dense")

        requirements = draft_backend_requirements(model)
        requirements_met = backend_satisfies_requirements(requirements, backend, supports_dflash)
        if requirements:
            reasons.append(requirements["reason"])
            flags = " ".join(requirements.get("flags") or [])
            if flags:
                reasons.append(f"requires {flags}")
        if compatible and not requirements_met:
            auto_enable = False
            reasons.append("not auto-enabled for this backend")

        rows.append(
            {
                "path": model["path"],
                "file": model["file"],
                "score": score,
                "safe": bool(auto_enable),
                "compatible": bool(compatible),
                "auto_enable": bool(auto_enable),
                "requires_backend": requirements.get("backend", ""),
                "requires_capability": requirements.get("capability", ""),
                "required_flags": requirements.get("flags", []),
                "reasons": reasons,
            }
        )
    return sorted(rows, key=lambda row: (row["auto_enable"], row["compatible"], row["score"]), reverse=True)


def gui_rows(data: dict[str, Any]) -> None:
    for model in data.get("models", []):
        if model.get("role") == "draft":
            continue
        arch = "MoE" if model.get("moe") else (model.get("architecture") or "dense")
        parts = [f"{model.get('size_gb', 0):.1f}GB", arch]
        quant = model.get("quant")
        if quant and quant != "unknown":
            parts.append(quant)
        if model.get("mmproj"):
            parts.append("vision")
        tune_count = len(model.get("tune_configs") or [])
        if tune_count:
            parts.append(f"{tune_count} tune cache")
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
    sug = sub.add_parser("suggest-drafts")
    sug.add_argument("--target", required=True)
    sug.add_argument("--backend", choices=("generic", "llama", "ik_llama", "buun-llama-cpp"), default="generic")
    sug.add_argument("--supports-dflash", action="store_true")
    sug.add_argument("--format", choices=("json", "text"), default="text")
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

    if args.cmd == "suggest-drafts":
        data = scan_models(model_dir, cache_dir)
        save_index(model_dir, data)
        target_path = str(Path(args.target).expanduser().resolve())
        target = next((m for m in data.get("models", []) if str(Path(m["path"]).resolve()) == target_path), None)
        if target is None:
            print(f"target not found in model index: {args.target}", file=sys.stderr)
            return 1
        rows = suggest_drafts(data, target, backend=args.backend, supports_dflash=args.supports_dflash)
        if args.format == "json":
            print(json.dumps(rows, indent=2, sort_keys=True))
        elif not rows:
            print("No local draft candidates found.")
        else:
            for row in rows:
                if row["safe"]:
                    status = "safe"
                elif row.get("compatible") and row.get("requires_backend"):
                    status = "requires-backend"
                else:
                    status = "blocked"
                print(f"{status}\t{row['score']}\t{row['path']}\t{'; '.join(row['reasons'])}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
