#!/usr/bin/env python3
"""Refresh llm-server's local recommendation catalog.

The catalog maps known local-friendly GGUF repos to public model-ranking signals.
Artificial Analysis API access requires a key in ARTIFICIAL_ANALYSIS_API_KEY.
The key is used only by this updater and is never written to the catalog.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_URL = "https://artificialanalysis.ai/api/v2/language/models"
FAMILY_PREFIXES = {
    "llama",
    "qwen",
    "mistral",
    "mixtral",
    "phi",
    "gemma",
    "deepseek",
    "kimi",
    "minimax",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def tokens(text: str) -> set[str]:
    stop = {"instruct", "chat", "reasoning", "non", "max", "effort", "preview"}
    return {t for t in norm(text).split() if t and t not in stop}


def family_match(query_tokens: set[str], hay_tokens: set[str]) -> bool:
    for family in FAMILY_PREFIXES:
        if any(t.startswith(family) for t in query_tokens) and any(t.startswith(family) for t in hay_tokens):
            return True
    return False


def model_size_tokens(query_tokens: set[str]) -> set[str]:
    return {t for t in query_tokens if re.fullmatch(r"a?\d+(?:b|m)", t)}


def version_tuples(text: str) -> set[tuple[str, ...]]:
    versions: set[tuple[str, ...]] = set()
    for match in re.finditer(r"(?<!\d)(\d+)[.-](\d+)(?:[.-](\d+))?", text.lower()):
        versions.add(tuple(part for part in match.groups() if part is not None))
    return versions


def clear_aa_fields(candidate: dict[str, Any]) -> None:
    for key in list(candidate):
        if key.startswith("aa_"):
            candidate.pop(key, None)


def fetch_json(api_key: str, url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"x-api-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Artificial Analysis API request failed ({exc.code}) for {url}: {exc.reason}") from exc
    if not isinstance(data, dict):
        raise SystemExit("Artificial Analysis API response was not a JSON object")
    return data


def fetch_models(api_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"page": page, "page_size": 200})
        data = fetch_json(api_key, f"{API_URL}?{query}")
        page_rows = data.get("data")
        if not isinstance(page_rows, list):
            raise SystemExit("Artificial Analysis API response did not contain a data array")
        rows.extend(row for row in page_rows if isinstance(row, dict))
        pagination = data.get("pagination")
        if not isinstance(pagination, dict) or not pagination.get("has_more"):
            break
        page += 1
    return rows


def intelligence(row: dict[str, Any]) -> float:
    ev = row.get("evaluations")
    if isinstance(ev, dict):
        for key in (
            "artificial_analysis_intelligence_index",
            "artificial_analysis_coding_index",
            "mmlu_pro",
        ):
            val = ev.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    return 0.0


def output_tps(row: dict[str, Any]) -> float:
    perf = row.get("performance")
    if isinstance(perf, dict):
        val = perf.get("median_output_tokens_per_second")
        if isinstance(val, (int, float)):
            return float(val)
    val = row.get("median_output_tokens_per_second")
    return float(val) if isinstance(val, (int, float)) else 0.0


def open_weights(row: dict[str, Any]) -> bool:
    licensing = row.get("licensing")
    if isinstance(licensing, dict) and isinstance(licensing.get("is_open_weights"), bool):
        return bool(licensing.get("is_open_weights"))
    if isinstance(row.get("open_weights"), bool):
        return bool(row.get("open_weights"))
    # Older/free API shapes may omit licensing. A weights URL is the best safe fallback.
    return bool(row.get("huggingface_url"))


def parameters_summary(row: dict[str, Any]) -> str:
    params = row.get("parameters")
    if not isinstance(params, dict):
        return ""
    total = params.get("total")
    active = params.get("active")
    if isinstance(total, (int, float)) and isinstance(active, (int, float)) and active != total:
        return f"{total:g}B/{active:g}B"
    if isinstance(total, (int, float)):
        return f"{total:g}B"
    return ""


def context_summary(row: dict[str, Any]) -> str:
    ctx = row.get("context_window_tokens")
    return str(int(ctx)) if isinstance(ctx, (int, float)) and ctx > 0 else ""


def huggingface_repo(row: dict[str, Any]) -> str:
    url = str(row.get("huggingface_url") or "")
    prefix = "https://huggingface.co/"
    return url[len(prefix):] if url.startswith(prefix) else url


def quality_score(index: float, previous: int) -> int:
    if index <= 0:
        return previous
    # Artificial Analysis Intelligence Index is roughly 0-60+ for current LLMs.
    return max(1, min(100, round(index * 1.65)))


def speed_score(tps: float, previous: int) -> int:
    if tps <= 0:
        return previous
    # Provider output speed is display metadata only; recommendations rank by intelligence.
    return max(1, min(100, round(tps)))


def row_name(row: dict[str, Any]) -> str:
    creator = row.get("model_creator")
    creator_name = ""
    if isinstance(creator, dict):
        creator_name = str(creator.get("name") or "")
    return " ".join(str(row.get(k) or "") for k in ("name", "slug")) + " " + creator_name


def display_name(row: dict[str, Any]) -> str:
    name = str(row.get("name") or "").strip()
    creator = row.get("model_creator")
    creator_name = ""
    if isinstance(creator, dict):
        creator_name = str(creator.get("name") or "").strip()
    if creator_name and creator_name.lower() not in name.lower():
        return f"{creator_name} {name}".strip()
    return name or str(row.get("slug") or "")


def print_top_models(rows: list[dict[str, Any]], limit: int, *, open_weights_only: bool = False) -> None:
    filtered = [row for row in rows if not open_weights_only or open_weights(row)]
    ranked = sorted(filtered, key=intelligence, reverse=True)[:limit]
    label = "open-weight " if open_weights_only else ""
    print(f"Top {len(ranked)} Artificial Analysis {label}models by Intelligence Index")
    print("| rank | model | slug | params | intelligence | output tps | ctx | hf/model weights |")
    print("|---:|---|---|---:|---:|---:|---:|---|")
    for i, row in enumerate(ranked, 1):
        name = display_name(row).replace("|", "/")
        slug = str(row.get("slug") or "").replace("|", "/")
        hf = huggingface_repo(row).replace("|", "/")
        print(
            f"| {i} | {name} | {slug} | {parameters_summary(row)} | "
            f"{intelligence(row):.3f} | {output_tps(row):.3f} | {context_summary(row)} | {hf} |"
        )


def match_row(candidate: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    query = str(candidate.get("aa_query") or candidate.get("name") or "")
    query_tokens = tokens(query)
    query_versions = version_tuples(query)
    if not query_tokens:
        return None

    best: tuple[int, float, dict[str, Any] | None] = (-1, 0.0, None)
    for row in rows:
        hay = row_name(row)
        if query_versions and not (query_versions & version_tuples(hay)):
            continue
        hay_tokens = tokens(hay)
        if not family_match(query_tokens, hay_tokens):
            continue
        required_size = model_size_tokens(query_tokens)
        if required_size and not (required_size & hay_tokens):
            continue
        overlap = len(query_tokens & hay_tokens)
        if overlap == 0:
            continue
        score = overlap * 100 + intelligence(row)
        if score > best[0] + best[1]:
            best = (overlap * 100, intelligence(row), row)
    return best[2]


def refresh_catalog(catalog: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed = 0
    for cand in catalog.get("candidates", []):
        if not isinstance(cand, dict):
            continue
        row = match_row(cand, rows)
        if not row:
            clear_aa_fields(cand)
            continue
        idx = intelligence(row)
        tps = output_tps(row)
        cand["quality"] = quality_score(idx, int(cand.get("quality") or 0))
        cand["speed"] = speed_score(tps, int(cand.get("speed") or 0))
        cand["aa_slug"] = row.get("slug") or ""
        cand["aa_id"] = row.get("id") or ""
        cand["aa_intelligence_index"] = round(idx, 3) if idx else 0
        cand["aa_output_tps"] = round(tps, 3) if tps else 0
        cand["aa_updated_at"] = utc_now()
        changed += 1
    catalog["generated_at"] = utc_now()
    catalog["source"] = f"Artificial Analysis API refresh; matched {changed} known GGUF recommendation rows"
    catalog["attribution"] = "Artificial Analysis intelligence data cached from https://artificialanalysis.ai/ and filtered by llm-server hardware fit"
    catalog["candidates"] = sorted(
        [c for c in catalog.get("candidates", []) if isinstance(c, dict)],
        key=lambda c: (str(c.get("family") or ""), float(c.get("size_gb") or 0), str(c.get("name") or "")),
    )
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh llm-server recommendation catalog")
    parser.add_argument("--catalog", default="go/pkg/recommend/catalog.json")
    parser.add_argument("--api-key-env", default="ARTIFICIAL_ANALYSIS_API_KEY")
    parser.add_argument("--allow-missing-key", action="store_true", help="exit 0 without changes when the API key is missing")
    parser.add_argument("--print-top", type=int, default=0, help="print top N API models by intelligence index")
    parser.add_argument("--print-open-weights-top", type=int, default=0, help="print top N open-weight API models by intelligence index")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        msg = f"{args.api_key_env} is not set"
        if args.allow_missing_key:
            print(msg + "; leaving recommendation catalog unchanged")
            return 0
        raise SystemExit(msg)

    rows = fetch_models(api_key)
    if args.print_top > 0:
        print_top_models(rows, args.print_top)
    if args.print_open_weights_top > 0:
        print_top_models(rows, args.print_open_weights_top, open_weights_only=True)
    catalog = refresh_catalog(catalog, rows)
    tmp = catalog_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(catalog_path)
    print(f"updated {catalog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
