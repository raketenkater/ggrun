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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def tokens(text: str) -> set[str]:
    stop = {"instruct", "chat", "reasoning", "non", "max", "effort", "preview"}
    return {t for t in norm(text).split() if t and t not in stop}


def fetch_models(api_key: str) -> list[dict[str, Any]]:
    req = urllib.request.Request(API_URL, headers={"x-api-key": api_key})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise SystemExit("Artificial Analysis API response did not contain a data array")
    return [row for row in rows if isinstance(row, dict)]


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
    val = row.get("median_output_tokens_per_second")
    return float(val) if isinstance(val, (int, float)) else 0.0


def quality_score(index: float, previous: int) -> int:
    if index <= 0:
        return previous
    # Artificial Analysis Intelligence Index is roughly 0-60+ for current LLMs.
    return max(1, min(100, round(index * 1.65)))


def speed_score(tps: float, previous: int) -> int:
    if tps <= 0:
        return previous
    # Provider output speed is only a ranking signal for local recommendations.
    return max(1, min(100, round(tps)))


def row_name(row: dict[str, Any]) -> str:
    creator = row.get("model_creator")
    creator_name = ""
    if isinstance(creator, dict):
        creator_name = str(creator.get("name") or "")
    return " ".join(str(row.get(k) or "") for k in ("name", "slug")) + " " + creator_name


def match_row(candidate: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    query = str(candidate.get("aa_query") or candidate.get("name") or "")
    query_tokens = tokens(query)
    if not query_tokens:
        return None

    best: tuple[int, float, dict[str, Any] | None] = (-1, 0.0, None)
    for row in rows:
        hay = row_name(row)
        hay_tokens = tokens(hay)
        overlap = len(query_tokens & hay_tokens)
        if overlap == 0:
            continue
        # Require family/name overlap and at least one size-ish token when present.
        size_tokens = {t for t in query_tokens if re.fullmatch(r"\d+b|a\d+b|\d+", t)}
        if size_tokens and not (size_tokens & hay_tokens):
            continue
        score = overlap * 10 + intelligence(row)
        if score > best[0] + best[1]:
            best = (overlap * 10, intelligence(row), row)
    return best[2]


def refresh_catalog(catalog: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed = 0
    for cand in catalog.get("candidates", []):
        if not isinstance(cand, dict):
            continue
        row = match_row(cand, rows)
        if not row:
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
    catalog["attribution"] = "Artificial Analysis leaderboard data cached from https://artificialanalysis.ai/ and filtered by llm-server hardware fit"
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
    catalog = refresh_catalog(catalog, rows)
    tmp = catalog_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(catalog_path)
    print(f"updated {catalog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
