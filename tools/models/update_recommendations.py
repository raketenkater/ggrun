#!/usr/bin/env python3
"""Refresh ggrun's local recommendation catalog.

The catalog is generated from Artificial Analysis open-weight model rankings and
resolved to Hugging Face GGUF repos with downloadable quant sizes. API access is
used only by this updater; keys are never written to the catalog or required at
ggrun runtime.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

API_URL = "https://artificialanalysis.ai/api/v2/language/models"
LEGACY_API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
OPEN_WEIGHTS_PAGE_URL = "https://artificialanalysis.ai/leaderboards/models?weights=open"
HF_MODEL_API_URL = "https://huggingface.co/api/models"
DEFAULT_CATALOG_LIMIT = 100
QUANT_PATTERN = re.compile(
    r"(TQ[12]_0|IQ[1-8]_(?:XXS|XS|NL(?:_XL)?|S|M|L)|Q[1-9]_(?:K(?:_?(?:XL(?:_?M)?|L|S|M))?|[01]|[1-9]_?[KS])|MXFP4(?:_MOE)?|MXP4(?:_MOE)?|BF16|F16|F32|F8|I4)",
    re.IGNORECASE,
)

# Speculative-decoding heads published beside the weights under the same quant
# names (mtp-X-Q8_0.gguf, MTP/..., dflash-X-BF16.gguf). Counting them made a
# 2 GB MTP head a "Q4_0" of a 1T model, and on 2026-10-25 an 8 GB
# dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf the Q8_0 of a 508 GB model.
# Shared with tools/download/download_any_gguf.py (test_companion_rule_matches_downloader).
DRAFT_HEAD_NAMES = ("mtp", "dflash", "dspark", "draft", "eagle", "eagle3", "medusa")
# "mtp" may also name a main model that carries its MTP layers, and "eagle" a
# main model (RWKV Eagle); the others are heads wherever they appear in a name.
DRAFT_HEAD_WORDS = frozenset(DRAFT_HEAD_NAMES) - {"mtp", "eagle"}


def is_calibration_or_adapter(path):
    # Importance matrices can themselves be GGUFs, even hundreds of MiB,
    # and inherit a source precision in their name. They are not weights.
    parts = path.lower().replace("_", "-").split("/")
    return (parts[-1].endswith(("-imatrix.gguf", "-adapter.gguf", "-lora.gguf"))
            or parts[-1].startswith(("imatrix-", "adapter-", "lora-", "indexer-"))
            or any(p in {"imatrix", "adapters", "adapter", "lora", "sparse-indexer"} for p in parts[:-1]))


def is_draft_head_gguf(path: str) -> bool:
    parts = path.lower().split("/")
    return (any(parts[-1].startswith(name + "-") for name in DRAFT_HEAD_NAMES)
            or any(part in DRAFT_HEAD_NAMES for part in parts[:-1])
            or bool(set(re.split(r"[-_.]+", parts[-1])) & DRAFT_HEAD_WORDS))


# Bytes per parameter by quant label, mirroring bytesPerWeight in
# go/pkg/recommend/speed.go (first match wins, same order).
BYTES_PER_WEIGHT = (
    ("F32", 4.0), ("BF16", 2.0), ("F16", 2.0), ("F8", 1.0), ("Q8", 1.06),
    ("Q6", 0.82), ("Q5", 0.69), ("MXFP4", 0.53), ("MXP4", 0.53), ("IQ4", 0.51),
    ("Q4", 0.56), ("I4", 0.56), ("IQ3", 0.41), ("Q3", 0.44), ("IQ2", 0.28),
    ("Q2", 0.33), ("Q1", 0.20),
)
COMPANION_PARAM_RATIO = 15


def drop_companion_sized(variants: list[tuple[str, int | None, Any]],
                         label: str | None = None) -> list[tuple[str, int | None, Any]]:
    """Drop (quant, size_bytes, item) artifacts implying under 1/15 of the
    largest artifact's parameter count: a draft head or projector that escaped
    the name rules (dspark-X-Q8_0.gguf, 2 GB beside a 300 GB model), never a
    quant of the model. Unknown sizes are kept; with a label, drops are
    reported on stderr. Mirrors tools/download/download_any_gguf.py and
    dropCompanionSizedQuants in go/pkg/recommend/recommend.go."""
    def implied(quant: str, size: int | None) -> float:
        q = (quant or "").upper()
        bpw = next((b for tag, b in BYTES_PER_WEIGHT if tag in q), 0.55)
        return size / bpw if size else 0
    params = [implied(quant, size) for quant, size, _ in variants]
    most = max(params, default=0)
    kept = [v for v, p in zip(variants, params) if not p or p * COMPANION_PARAM_RATIO >= most]
    if label and len(kept) != len(variants):
        dropped = [str(v[0]) for v, p in zip(variants, params) if p and p * COMPANION_PARAM_RATIO < most]
        print(f"warning: {label}: dropped companion-sized quants {', '.join(dropped)} "
              f"(under 1/{COMPANION_PARAM_RATIO} of the largest parameter count)", file=sys.stderr)
    return kept


def model_weight_siblings(siblings: list[Any]) -> list[dict[str, Any]]:
    """Only main-model weights may inherit a leaderboard score, even in draft-only repos."""
    ggufs = [item for item in siblings if isinstance(item, dict)
             and str(item.get("rfilename") or "").lower().endswith(".gguf")
             and "mmproj" not in str(item.get("rfilename") or "").lower()
             and not is_calibration_or_adapter(str(item.get("rfilename") or ""))]
    weights = [item for item in ggufs if not is_draft_head_gguf(str(item.get("rfilename") or ""))]
    return weights

TRUSTED_GGUF_OWNERS = {
    "unsloth": 140,
    "bartowski": 95,
    "maziyarpanahi": 80,
    "prithivmlmods": 70,
    "lmstudio-community": 60,
    "second-state": 45,
}

FAMILY_PREFIXES = {
    "aya",
    "baichuan",
    "codestral",
    "command",
    "deepseek",
    "ernie",
    "exaone",
    "falcon",
    "gemma",
    "glm",
    "gpt",
    "granite",
    "hunyuan",
    "internlm",
    "kimi",
    "llama",
    "magistral",
    "minicpm",
    "minimax",
    "mistral",
    "mimo",
    "mixtral",
    "mpt",
    "nemotron",
    "olmo",
    "phi",
    "qwen",
    "smollm",
    "starcoder",
    "step",
    "yi",
}

HF_MODEL_CACHE: dict[str, dict[str, Any] | None] = {}
HF_QUANT_CACHE: dict[str, list[dict[str, Any]]] = {}
HF_SEARCH_CACHE: dict[tuple[str, int], list[dict[str, Any]]] = {}
HF_MIN_DELAY_SECONDS = 0.0
HF_429_BACKOFF_SECONDS = 0.0
HF_LAST_REQUEST_AT = 0.0
HF_WARNED_429 = False


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def tokens(text: str) -> set[str]:
    stop = {
        "base",
        "chat",
        "effort",
        "gguf",
        "high",
        "instruct",
        "low",
        "max",
        "model",
        "models",
        "non",
        "preview",
        "reasoning",
        "thinking",
        "turbo",
    }
    return {t for t in norm(text).split() if t and t not in stop}


def family_tokens(row_tokens: set[str]) -> set[str]:
    return {family for family in FAMILY_PREFIXES if any(t.startswith(family) for t in row_tokens)}


def family_match(query_tokens: set[str], hay_tokens: set[str]) -> bool:
    left = family_tokens(query_tokens)
    right = family_tokens(hay_tokens)
    return not left or not right or bool(left & right)


def model_size_tokens(query_tokens: set[str]) -> set[str]:
    return {t for t in query_tokens if re.fullmatch(r"a?\d+(?:b|m)", t)}


def version_tuples(text: str) -> set[tuple[str, ...]]:
    versions: set[tuple[str, ...]] = set()
    for match in re.finditer(r"(?<!\d)(\d+)[.-](\d+)(?:[.-](\d+))?", text.lower()):
        versions.add(tuple(part for part in match.groups() if part is not None))
    return versions


def versions_match(target: set[tuple[str, ...]], candidate: set[tuple[str, ...]]) -> bool:
    """True when some candidate version starts with a target version.

    The parser also reads a hyphenated size as a third component, so
    "Qwen3.6-27B" yields (3, 6, 27); requiring equality with the row's (3, 6)
    rejected every such repo.
    """
    return any(c[: len(t)] == t for t in target for c in candidate)


def uniq(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


class APIRequestError(RuntimeError):
    def __init__(self, code: int, url: str, reason: str):
        super().__init__(f"request failed ({code}) for {url}: {reason}")
        self.code = code
        self.url = url
        self.reason = reason


def is_hf_url(url: str) -> bool:
    return urllib.parse.urlparse(url).netloc.lower().endswith("huggingface.co")


def hf_auth_headers(url: str) -> dict[str, str]:
    if not is_hf_url(url):
        return {}
    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def throttle_hf(url: str) -> None:
    global HF_LAST_REQUEST_AT
    if not is_hf_url(url) or HF_MIN_DELAY_SECONDS <= 0:
        return
    now = time.monotonic()
    wait = HF_MIN_DELAY_SECONDS - (now - HF_LAST_REQUEST_AT)
    if wait > 0:
        time.sleep(wait)
    HF_LAST_REQUEST_AT = time.monotonic()


def fetch_text(url: str, *, headers: dict[str, str] | None = None, timeout: int = 60) -> str:
    req_headers = {"User-Agent": "ggrun-catalog-refresh"}
    req_headers.update(hf_auth_headers(url))
    if headers:
        req_headers.update(headers)
    attempts = 3 if is_hf_url(url) and HF_429_BACKOFF_SECONDS > 0 else 1
    for attempt in range(attempts):
        throttle_hf(url)
        req = urllib.request.Request(url, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt + 1 < attempts:
                delay = HF_429_BACKOFF_SECONDS * (attempt + 1)
                print(f"Hugging Face rate limited; waiting {delay:.0f}s before retry", file=sys.stderr)
                time.sleep(delay)
                continue
            raise APIRequestError(exc.code, url, exc.reason) from exc
    raise RuntimeError(f"request failed for {url}")


def fetch_json(api_key: str, url: str) -> dict[str, Any]:
    text = fetch_text(url, headers={"x-api-key": api_key})
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Artificial Analysis API response was not a JSON object")
    return data


def fetch_public_data(url: str) -> Any:
    return json.loads(fetch_text(url))


def fetch_public_json(url: str) -> dict[str, Any]:
    data = fetch_public_data(url)
    if not isinstance(data, dict):
        raise ValueError(f"Public API response for {url} was not a JSON object")
    return data


def warn_hf_error(prefix: str, exc: Exception) -> None:
    global HF_WARNED_429
    if isinstance(exc, APIRequestError):
        if exc.code in (401, 404):
            return
        if exc.code == 429:
            if HF_WARNED_429:
                return
            HF_WARNED_429 = True
            print(f"{prefix}: Hugging Face rate limit reached; unresolved rows will be skipped unless backoff/token is configured", file=sys.stderr)
            return
    print(f"{prefix}: {exc}", file=sys.stderr)


def json_array_at(text: str, start: int) -> str:
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("unterminated JSON array")


def parse_open_weight_page_models(page: str) -> list[dict[str, Any]]:
    # Decode the transport before reading its JSON. Replacing escaped quotes
    # globally corrupts quoted names, and table JSON can span multiple chunks.
    decoder = json.JSONDecoder()
    chunks = []
    for match in re.finditer(r"self\.__next_f\.push\(\s*", page):
        try:
            payload, _ = decoder.raw_decode(page, match.end())
        except ValueError:
            continue
        if (isinstance(payload, list) and len(payload) > 1
                and payload[0] == 1 and isinstance(payload[1], str)):
            chunks.append(payload[1])
    rows: list[dict[str, Any]] = []
    # October 2026: initialModels is the scored table; models is now a
    # navigation summary without licensing or score fields. Support the older
    # table key too, but never infer openness from the page's query string.
    for text in (page, "".join(chunks)):
        for match in re.finditer(r'"(?:initialModels|models)"\s*:\s*(\[)', text):
            try:
                parsed, _ = decoder.raw_decode(text, match.start(1))
            except ValueError:
                continue
            rows.extend(row for row in parsed if isinstance(row, dict)
                        and has_open_weight_metadata(row) and open_weights(row))
    return dedupe_rows(rows)


def fetch_public_open_weight_rows() -> list[dict[str, Any]]:
    page = fetch_text(OPEN_WEIGHTS_PAGE_URL, timeout=90)
    rows = parse_open_weight_page_models(page)
    if not rows:
        raise SystemExit("could not extract open-weight rows from Artificial Analysis leaderboard page")
    return rows


def fetch_models_from_url(api_key: str, base_url: str, *, paginated: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        if paginated:
            query = urllib.parse.urlencode({"page": page, "page_size": 200})
            url = f"{base_url}?{query}"
        else:
            url = base_url
        data = fetch_json(api_key, url)
        page_rows = data.get("data")
        if not isinstance(page_rows, list):
            raise ValueError("Artificial Analysis API response did not contain a data array")
        rows.extend(row for row in page_rows if isinstance(row, dict))
        pagination = data.get("pagination")
        if not paginated or not isinstance(pagination, dict) or not pagination.get("has_more"):
            break
        page += 1
    return rows


def fetch_models(api_key: str) -> list[dict[str, Any]]:
    try:
        return fetch_models_from_url(api_key, API_URL, paginated=True)
    except APIRequestError as exc:
        if exc.code not in (401, 403):
            raise
        # The v2 models endpoint is subscription-only; a free key gets 403.
        print(f"warning: Artificial Analysis v2 API returned {exc.code}; using the legacy endpoint", file=sys.stderr)
    return fetch_models_from_url(api_key, LEGACY_API_URL, paginated=False)


def load_open_weight_rows(api_key: str) -> tuple[list[dict[str, Any]], str]:
    if api_key:
        # Any API failure falls back to the public page, which needs no key.
        try:
            rows = fetch_models(api_key)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"warning: Artificial Analysis API unavailable ({exc})", file=sys.stderr)
            rows = []
        if rows and any(has_open_weight_metadata(row) for row in rows):
            open_rows = [row for row in rows if open_weights(row)]
            if open_rows:
                return dedupe_rows(open_rows), "Artificial Analysis API open-weight rows"
        print("Artificial Analysis API response lacks open-weight metadata; using public open-weight leaderboard page", file=sys.stderr)
    rows = fetch_public_open_weight_rows()
    return rows, "Artificial Analysis open-weight leaderboard page"


def value_from(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        val = row.get(key)
        if val is not None and val != "$undefined":
            return val
    return None


def intelligence(row: dict[str, Any]) -> float:
    val = value_from(row, "intelligenceIndex", "intelligence_index", "artificial_analysis_intelligence_index")
    if isinstance(val, (int, float)):
        return float(val)
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
    val = value_from(row, "medianOutputTokensPerSecond", "median_output_tokens_per_second")
    if isinstance(val, (int, float)):
        return float(val)
    perf = row.get("performance")
    if isinstance(perf, dict):
        val = perf.get("median_output_tokens_per_second")
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


def model_creator_name(row: dict[str, Any]) -> str:
    val = value_from(row, "modelCreatorName", "model_creator_name")
    if isinstance(val, str) and val.strip():
        return val.strip()
    creator = row.get("model_creator") or row.get("creator")
    if isinstance(creator, dict):
        val = creator.get("name")
        if isinstance(val, str):
            return val.strip()
    return ""


def has_open_weight_metadata(row: dict[str, Any]) -> bool:
    if isinstance(row.get("isOpenWeights"), bool):
        return True
    licensing = row.get("licensing")
    if isinstance(licensing, dict) and isinstance(licensing.get("is_open_weights"), bool):
        return True
    return isinstance(row.get("open_weights"), bool)


def open_weights(row: dict[str, Any]) -> bool:
    if isinstance(row.get("isOpenWeights"), bool):
        return bool(row.get("isOpenWeights"))
    licensing = row.get("licensing")
    if isinstance(licensing, dict) and isinstance(licensing.get("is_open_weights"), bool):
        return bool(licensing.get("is_open_weights"))
    if isinstance(row.get("open_weights"), bool):
        return bool(row.get("open_weights"))
    return bool(huggingface_repo(row))


def parameters_value(row: dict[str, Any], key: str) -> float:
    direct = "totalParameters" if key == "total" else "activeParameters"
    val = row.get(direct)
    if isinstance(val, (int, float)) and val > 0:
        return float(val)
    params = row.get("parameters")
    if isinstance(params, dict):
        val = params.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    return 0.0


def parameters_summary(row: dict[str, Any]) -> str:
    total = parameters_value(row, "total")
    active = parameters_value(row, "active")
    if total > 0 and active > 0 and active != total:
        return f"{total:g}B/{active:g}B"
    if total > 0:
        return f"{total:g}B"
    return ""


def context_summary(row: dict[str, Any]) -> str:
    ctx = value_from(row, "contextWindowTokens", "context_window_tokens")
    return str(int(ctx)) if isinstance(ctx, (int, float)) and ctx > 0 else ""


def huggingface_repo(row: dict[str, Any]) -> str:
    url = str(value_from(row, "huggingfaceUrl", "huggingface_url") or "").strip()
    prefix = "https://huggingface.co/"
    if url.startswith(prefix):
        repo = url[len(prefix) :]
    else:
        repo = url
    repo = repo.split("/tree/", 1)[0].split("/blob/", 1)[0].strip("/")
    return repo if "/" in repo else ""


def row_slug(row: dict[str, Any]) -> str:
    return str(value_from(row, "slug", "id") or "")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or "")


def quality_score(index: float, previous: int = 0) -> int:
    if index <= 0:
        return previous
    # Artificial Analysis Intelligence Index is roughly 0-60+ for current LLMs.
    return max(1, min(100, round(index * 1.65)))


def speed_score(tps: float, previous: int = 0) -> int:
    if tps <= 0:
        return previous
    # Provider output speed is display metadata only; recommendations rank by intelligence.
    return max(1, min(100, round(tps)))


def display_name(row: dict[str, Any]) -> str:
    name = str(value_from(row, "name", "shortName", "slug") or "").strip()
    creator_name = model_creator_name(row)
    if creator_name and creator_name.lower() not in name.lower():
        return f"{creator_name} {name}".strip()
    return name or row_slug(row)


def bare_model_name(row: dict[str, Any]) -> str:
    """The model's own name, without the creator prefix display_name() adds.

    Hugging Face repos are named after the model alone ("GLM-5.3-Flash" ->
    unsloth/GLM-5.3-Flash-GGUF), so the creator-prefixed spelling names no real
    repo for any creator whose name is not already part of the model name.
    """
    return str(value_from(row, "name", "shortName", "slug") or "").strip()


def row_name(row: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(value_from(row, "name", "shortName") or ""),
            row_slug(row),
            model_creator_name(row),
            huggingface_repo(row),
        )
        if part
    )


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row_id(row) or row_slug(row) or display_name(row).lower()
        if not key:
            continue
        prev = best.get(key)
        if prev is None or intelligence(row) > intelligence(prev):
            best[key] = row
    return sorted(best.values(), key=intelligence, reverse=True)


def print_top_models(rows: list[dict[str, Any]], limit: int, *, open_weights_only: bool = False) -> None:
    filtered = [row for row in rows if not open_weights_only or open_weights(row)]
    ranked = sorted(filtered, key=intelligence, reverse=True)[:limit]
    label = "open-weight " if open_weights_only else ""
    print(f"Top {len(ranked)} Artificial Analysis {label}models by Intelligence Index")
    print("| rank | model | slug | params | intelligence | output tps | ctx | hf/model weights |")
    print("|---:|---|---|---:|---:|---:|---:|---|")
    for i, row in enumerate(ranked, 1):
        name = display_name(row).replace("|", "/")
        slug = row_slug(row).replace("|", "/")
        hf = huggingface_repo(row).replace("|", "/")
        print(
            f"| {i} | {name} | {slug} | {parameters_summary(row)} | "
            f"{intelligence(row):.3f} | {output_tps(row):.3f} | {context_summary(row)} | {hf} |"
        )


def normalize_quant(name: str) -> str:
    return name.upper().replace("-", "_")


def fetch_hf_model_info(repo: str) -> dict[str, Any] | None:
    if not repo or "/" not in repo:
        return None
    key = repo.lower()
    if key in HF_MODEL_CACHE:
        return HF_MODEL_CACHE[key]
    encoded = urllib.parse.quote(repo, safe="/")
    url = f"{HF_MODEL_API_URL}/{encoded}?blobs=true"
    try:
        data = fetch_public_json(url)
    except (OSError, ValueError, RuntimeError) as exc:
        warn_hf_error(f"Could not inspect Hugging Face repo {repo}", exc)
        HF_MODEL_CACHE[key] = None
        return None
    HF_MODEL_CACHE[key] = data
    return data


SHARD_PATTERN = re.compile(r"^(.*)-(\d+)-of-(\d+)\.gguf$", re.IGNORECASE)


def artifact_quant(path: str) -> str:
    """Use the final quant label; model names can themselves contain BF16."""
    basename = path.rsplit("/", 1)[-1]
    matches = list(QUANT_PATTERN.finditer(basename))
    if not matches:
        matches = list(QUANT_PATTERN.finditer(path.rsplit("/", 1)[0])) if "/" in path else []
        source = path.rsplit("/", 1)[0]
    else:
        source = basename
    if not matches:
        return ""
    match = matches[-1]
    quant = normalize_quant(match.group())
    if source[max(0, match.start() - 3):match.start()].upper() == "UD-":
        quant = "UD-" + quant
    return quant


def complete_shards(paths: list[str]) -> bool:
    """An artifact is one standalone file or every shard, exactly once."""
    matches = [SHARD_PATTERN.fullmatch(p) for p in paths]
    if not any(matches):
        return len(paths) == 1
    if not all(matches):
        return False
    totals = {int(m.group(3)) for m in matches}
    if len(totals) != 1:
        return False
    total = totals.pop()
    indices = {int(m.group(2)) for m in matches}
    return total > 0 and len(paths) == total and indices == set(range(1, total + 1))


def drop_companion_quants(quants: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    """drop_companion_sized over catalog quant rows."""
    def size(q: dict[str, Any]) -> int | None:
        b = q.get("size_bytes")
        if isinstance(b, (int, float)) and not isinstance(b, bool) and b > 0:
            return int(b)
        gb = q.get("size_gb")
        if isinstance(gb, (int, float)) and not isinstance(gb, bool) and gb > 0:
            return int(gb * 1073741824)
        return None
    return [q for _, _, q in drop_companion_sized([(str(q.get("name") or ""), size(q), q) for q in quants], label)]


def fetch_hf_quants(repo: str) -> list[dict[str, Any]]:
    key = repo.lower()
    if key in HF_QUANT_CACHE:
        return HF_QUANT_CACHE[key]
    data = fetch_hf_model_info(repo)
    if not data:
        HF_QUANT_CACHE[key] = []
        return []
    siblings = data.get("siblings")
    if not isinstance(siblings, list):
        HF_QUANT_CACHE[key] = []
        return []
    # Group complete artifacts before adding sizes. Different files with the
    # same quant are not necessarily shards of the same model.
    groups: dict[str, dict[str, list[tuple[str, int | None]]]] = {}
    for item in model_weight_siblings(siblings):
        fname = str(item.get("rfilename") or "")
        quant = artifact_quant(fname)
        if not quant:
            continue
        size = item.get("size")
        if isinstance(item.get("lfs"), dict):
            size = item["lfs"].get("size", size)
        shard = SHARD_PATTERN.fullmatch(fname)
        stem = shard.group(1) if shard else fname
        groups.setdefault(quant, {}).setdefault(stem, []).append((fname, size))
    variants = []
    for quant, stems in groups.items():
        for files in stems.values():
            if not complete_shards([name for name, _ in files]):
                continue
            known = all(isinstance(size, int) and not isinstance(size, bool) and size > 0 for _, size in files)
            variants.append((quant, sum(size for _, size in files) if known else None, files))
    # Drop companion-sized artifacts before the ambiguity check, so a draft
    # head sharing a label with the model cannot hide the model's own quant.
    complete: dict[str, list[int | None]] = {}
    for quant, size, _ in drop_companion_sized(variants, repo):
        complete.setdefault(quant, []).append(size)
    quants = []
    for quant, found in complete.items():
        if len(found) != 1:
            continue  # ambiguous model variants must not be added together
        size = found[0]
        if size is None:
            continue  # missing size must not make a shard set appear smaller
        quants.append({"name": quant, "size_gb": round(size / 1073741824, 2),
                       "size_bytes": size, "dynamic": quant.startswith("UD-")})
    quants.sort(key=lambda q: q["size_bytes"])
    HF_QUANT_CACHE[key] = quants
    return quants


# GGUF architecture geometry, read from the binary header so the recommender
# can compute launch overhead with the placement engine's exact formula
# (go/pkg/placement/placement.go) instead of estimating it. Hugging Face's
# /api/models endpoint exposes architecture/context_length/total but NOT the
# layer/expert/KV geometry (embd, exp_ff, hkv, kl, vl, layers, experts,
# leading_dense, kv_lora) — those live only inside the GGUF KV metadata block.
# The geometry is identical across every quant of a model, so one range request
# per repo (on any one GGUF file) is enough; no per-quant fetch.
# 16 MiB covers KV metadata + tokenizer for most models; a large vocabulary
# (Sarvam 105B) needs the second, larger range.
GGUF_HEADER_FETCH_STEPS = (16 * 1024 * 1024, 64 * 1024 * 1024)
HF_GGUF_ARCH_CACHE: dict[str, dict[str, Any] | None] = {}


def _gguf_resolve_url(repo: str, fname: str) -> str:
    repo_enc = urllib.parse.quote(repo, safe="/")
    file_enc = urllib.parse.quote(fname, safe="/")
    return f"https://huggingface.co/{repo_enc}/resolve/main/{file_enc}"


def _representative_gguf_file(siblings: list[dict[str, Any]]) -> str | None:
    # Prefer the first shard of a split model (KV metadata is duplicated across
    # shards, so shard 1 always carries it). Fall back to any non-mmproj GGUF.
    groups: dict[str, list[str]] = {}
    for item in model_weight_siblings(siblings):
        fname = str(item.get("rfilename") or "")
        shard = SHARD_PATTERN.fullmatch(fname)
        groups.setdefault(shard.group(1) if shard else fname, []).append(fname)
    candidates = [(name, 0) for paths in groups.values() if complete_shards(paths) for name in paths]
    if not candidates:
        return None
    # "00001-of" or "-of-00001" (single file) sorts first.
    candidates.sort(key=lambda fs: (0 if "00001-of" in fs[0] or "-of-0000" not in fs[0] else 1, fs[0]))
    return candidates[0][0]


def _fetch_gguf_range(url: str, nbytes: int) -> bytes:
    req_headers = {"Range": f"bytes=0-{nbytes - 1}", "User-Agent": "ggrun-catalog-refresh"}
    req_headers.update(hf_auth_headers(url))
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read(nbytes)


def parse_gguf_header(raw: bytes) -> tuple[dict[str, Any] | None, bool]:
    """Parse GGUF KV metadata: (meta, complete), or (None, False) when not GGUF.

    complete is False when the bytes ran out mid-metadata; meta then holds the
    keys read before the cut.
    """
    # parse_gguf._read_kv reads from a file-like object; BytesIO works and lets
    # us reuse the exact tested parser instead of duplicating the KV walk.
    import io
    import struct
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf"))
    from parse_gguf import _read_kv

    buf = io.BytesIO(raw)
    if len(raw) < 24 or buf.read(4) != b"GGUF":
        return None, False
    buf.read(4)  # gguf version
    struct.unpack("<Q", buf.read(8))[0]  # tensor_count (0 on shard 1 of split models)
    kv_count = struct.unpack("<Q", buf.read(8))[0]
    meta: dict[str, Any] = {"fused": 0, "expert_bytes": 0, "non_expert_bytes": 0}
    try:
        _read_kv(buf, meta, kv_count)
    except Exception:  # noqa: BLE001 - any parse failure is reported by the caller
        return meta, False
    return meta, True


def fetch_gguf_arch(repo: str) -> dict[str, Any] | None:
    """Read GGUF KV metadata geometry from one HF range request.

    Returns a dict with the fields the placement engine's overhead formula
    needs (arch, layers, experts, exp_used, exp_ff, exp_shared_ff, embd, ff,
    hkv, kl, vl, kv_lora, q_lora, leading_dense, ctx_train), or None when the
    repo has no GGUF or the header could not be parsed. Metadata larger than
    the first range is retried with a larger one. Best-effort: any failure
    leaves the caller to fall back to the size-only overhead estimate.
    """
    if not repo or "/" not in repo:
        return None
    key = repo.lower()
    if key in HF_GGUF_ARCH_CACHE:
        return HF_GGUF_ARCH_CACHE[key]
    data = fetch_hf_model_info(repo)
    if not data:
        HF_GGUF_ARCH_CACHE[key] = None
        return None
    siblings = data.get("siblings")
    if not isinstance(siblings, list):
        HF_GGUF_ARCH_CACHE[key] = None
        return None
    fname = _representative_gguf_file(siblings)
    if not fname:
        HF_GGUF_ARCH_CACHE[key] = None
        return None
    url = _gguf_resolve_url(repo, fname)
    meta: dict[str, Any] | None = None
    complete = truncated = False
    for nbytes in GGUF_HEADER_FETCH_STEPS:
        try:
            raw = _fetch_gguf_range(url, nbytes)
        except Exception as exc:
            warn_hf_error(f"Could not read GGUF header for {repo}/{fname}", exc)
            HF_GGUF_ARCH_CACHE[key] = None
            return None
        meta, complete = parse_gguf_header(raw)
        # A short read is the whole file: a parse error there is not truncation.
        truncated = meta is not None and not complete and len(raw) >= nbytes
        if not truncated:
            break
    if meta is None or not (complete or truncated):
        print(f"warning: GGUF header unreadable for {repo}/{fname}; geometry unknown", file=sys.stderr)
        HF_GGUF_ARCH_CACHE[key] = None
        return None
    if truncated:
        # The metadata outgrew every range. Sarvam 105B's 262k-entry tokenizer
        # overran 16 MiB, and dropping the whole header catalogued that MoE as
        # dense. Keys before the cut are exact, but later geometry may be
        # missing, so keep only what classifies the model and leave the rest
        # unknown; the recommender then uses its size-only overhead estimate.
        print(f"warning: GGUF metadata for {repo}/{fname} exceeds {GGUF_HEADER_FETCH_STEPS[-1] >> 20} MiB; "
              "keeping only architecture and expert counts", file=sys.stderr)
        meta = {k: meta[k] for k in ("arch", "experts", "exp_used") if k in meta}
    # Project to the fields the recommender/overhead formula consumes.
    out: dict[str, Any] = {}
    for field in (
        "arch", "layers", "experts", "exp_used", "exp_ff", "exp_shared_ff",
        "embd", "ff", "hkv", "kl", "vl", "kv_lora", "q_lora",
        "leading_dense", "ctx_train", "ssm", "nextn_predict_layers",
    ):
        val = meta.get(field)
        if val is not None:
            out[field] = val
    HF_GGUF_ARCH_CACHE[key] = out or None
    return out or None


def search_hf_models(query: str, limit: int) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        return []
    key = (query.lower(), limit)
    if key in HF_SEARCH_CACHE:
        return HF_SEARCH_CACHE[key]
    url = f"{HF_MODEL_API_URL}?{urllib.parse.urlencode({'search': query, 'limit': limit})}"
    try:
        data = fetch_public_data(url)
    except (OSError, ValueError, RuntimeError) as exc:
        warn_hf_error(f"Hugging Face search unavailable for {query!r}", exc)
        HF_SEARCH_CACHE[key] = []
        return []
    if not isinstance(data, list):
        HF_SEARCH_CACHE[key] = []
        return []
    rows = [row for row in data if isinstance(row, dict)]
    HF_SEARCH_CACHE[key] = rows
    return rows


def repo_name(repo: str) -> str:
    return repo.split("/", 1)[-1] if "/" in repo else repo


def repo_owner(repo: str) -> str:
    return repo.split("/", 1)[0].lower() if "/" in repo else ""


def clean_repo_model_name(repo_or_name: str) -> str:
    name = repo_name(repo_or_name)
    name = re.sub(r"(?i)(?:-?mtp)?-?gguf.*$", "", name)
    name = re.sub(r"(?i)-(?:ud-)?(?:iq|q)\d.*$", "", name)
    name = re.sub(r"\([^)]*\)", " ", name)
    return name.replace("_", " ").replace("-", " ").strip()


def model_query(row: dict[str, Any]) -> str:
    base = huggingface_repo(row)
    if base:
        return clean_repo_model_name(base)
    return clean_repo_model_name(display_name(row))


# Words that name a different model of the same family and generation. A repo
# carrying one the scored model lacks (or lacking one it has) is another model:
# "Devstral 2" is not "Devstral Small 2", and an instruct row must not inherit a
# "-Base-" repo. "base" is a stop word for search, so these are compared on the
# raw name.
VARIANT_QUALIFIERS = {
    "air", "base", "coder", "distill", "flash", "large", "lite", "medium", "mini",
    "nano", "next", "omni", "small", "tiny", "vl",
    # Modified derivatives are different models too: an "Uncensored" or
    # "Heretic" retune does not carry the original's benchmark score.
    "abliterated", "ablated", "decensored", "heretic", "lorablated", "merge",
    "merged", "nsfw", "roleplay", "uncensored",
    # Domain fine-tunes carry their own leaderboard rows: Ling 3.0 Flash was
    # resolved to the finance retune Ling-3.0-flash-Fin, and the duplicate repo
    # then dropped the base model from the catalog.
    "fin",
    # Command A+ is not Command A; norm() drops "+", so it is spelled out.
    "plus",
}
# Inflections a qualifier still names: "Distilled" is a distill.
QUALIFIER_SUFFIXES = ("", "s", "ed", "ing", "ation")


def raw_name_tokens(text: str) -> set[str]:
    return set(norm(text).split())


def name_qualifiers(text: str) -> set[str]:
    words = raw_name_tokens(text.replace("+", " plus "))
    return {q for q in VARIANT_QUALIFIERS for w in words
            if w.startswith(q) and w[len(q):] in QUALIFIER_SUFFIXES}


def spelled_versions(name: str) -> str:
    """Read "_" between digits as a version dot: Llama-3_3, v1_5."""
    return re.sub(r"(?<=\d)_(?=\d)", ".", name)


def identity_tokens(text: str) -> set[str]:
    """Name tokens without parenthetical notes; "v1" and "1" are one version."""
    return {re.sub(r"^v(?=\d)", "", t) for t in tokens(re.sub(r"\([^)]*\)", " ", text))}


def fused_versions(text: str) -> set[str]:
    """Words fusing a family and its version: lfm2, lfm2.5, k2, m3.1, v4.1."""
    words = re.split(r"[\s_\-/()]+", text.lower())
    return {w for w in words if re.fullmatch(r"[a-z]+\d+(?:\.\d+)*", w)}


def fused_versions_match(target: str, candidate: str) -> bool:
    """LFM2 is not LFM2.5 and MiniMax M3 is not M3.1: a candidate word that
    spells the target's family with another version names another model."""
    wanted = fused_versions(target)
    for word in fused_versions(candidate):
        family = re.match(r"[a-z]+", word).group()
        same_family = {w for w in wanted if re.match(r"[a-z]+", w).group() == family}
        if same_family and word not in same_family:
            return False
    return True


def creator_prefix_only(candidate_name: str, target_tokens: set[str], row: dict[str, Any]) -> bool:
    """Words before the model's own name may only be its creator's name.

    Quantizers prefix the publisher ("mistralai_Devstral", "LiquidAI_LFM2.5");
    anything else there names a retune ("MANGO-Qwen3-Omni", "Huihui-...").
    """
    words = norm(candidate_name).split()
    families = family_tokens(target_tokens)
    start = next((i for i, w in enumerate(words)
                  if w in target_tokens or any(w.startswith(f) for f in families)), None)
    if not start:
        return True
    prefix = "".join(words[:start])
    creator = norm(model_creator_name(row)).replace(" ", "")
    return bool(creator) and (prefix.startswith(creator) or creator.startswith(prefix))


def auxiliary_repo(repo: str) -> bool:
    return bool(raw_name_tokens(repo_name(repo)) & ({"mtp", "mmproj", "adapter", "lora"} | DRAFT_HEAD_WORDS))


def candidate_relevant(candidate_repo: str, row: dict[str, Any]) -> bool:
    if auxiliary_repo(candidate_repo):
        return False
    target_query = model_query(row)
    target_tokens = tokens(target_query)
    # Judge the repository's model name, not its owner: "DevQuasar-4" once
    # supplied the "4" of "Mistral Small 4" for a Small-24B-Base-2501 repo.
    candidate_name = spelled_versions(repo_name(candidate_repo))
    # Quant and GGUF suffixes are not part of the name: "Q4_0_4_8" once
    # supplied the "4" of "Mistral Small 4" for a Small-Instruct-2409 repo.
    candidate_tokens = tokens(clean_repo_model_name(candidate_name))
    if not target_tokens or not candidate_tokens:
        return False
    # Every word of the model's own name must appear: one shared word let
    # "A.X-K2" resolve to a 3.7B distill and "Command A" to Command A+.
    own_name = re.sub(r"\([^)]*\)", " ", bare_model_name(row))
    if not identity_tokens(own_name) <= identity_tokens(clean_repo_model_name(candidate_name)):
        return False
    if not fused_versions_match(own_name, clean_repo_model_name(candidate_name)):
        return False
    if not family_match(target_tokens, candidate_tokens):
        return False
    needed_sizes = model_size_tokens(target_tokens)
    if needed_sizes and not (needed_sizes & candidate_tokens):
        return False
    target_versions = version_tuples(target_query)
    if target_versions and not versions_match(target_versions, version_tuples(candidate_name)):
        return False
    # A bare generation number ("Small 4", "Devstral 2") identifies the model as
    # much as a dotted version does.
    generations = {t for t in target_tokens if t.isdigit()}
    if generations - candidate_tokens:
        return False
    # Parenthetical notes ("based on GLM-4.5-Air") describe lineage, not the
    # model's own name.
    target_qualifiers = name_qualifiers(" ".join([target_query, re.sub(r"\([^)]*\)", " ", row_name(row))]))
    if target_qualifiers != name_qualifiers(clean_repo_model_name(candidate_name)):
        return False
    if not creator_prefix_only(clean_repo_model_name(candidate_name), target_tokens, row):
        return False
    overlap = len(target_tokens & candidate_tokens)
    if overlap == 0 and not versions_match(target_versions, version_tuples(candidate_name)):
        return False
    return True


def direct_repo_candidates(row: dict[str, Any]) -> list[str]:
    base = huggingface_repo(row)
    repos: list[str] = []
    if base:
        repos.append(base)
        owner = base.split("/", 1)[0] if "/" in base else ""
        base_name = repo_name(base)
        if not base_name.lower().endswith("gguf"):
            if owner:
                repos.append(f"{owner}/{base_name}-GGUF")
            repos.append(f"unsloth/{base_name}-GGUF")
            repos.append(f"bartowski/{base_name}-GGUF")
    # Artificial Analysis stopped publishing the HF repo link in September 2026,
    # so with no base repo the name has to be rebuilt from the row. model_query()
    # falls back to display_name(), which prefixes the creator ("Z AI
    # GLM-5.3-Flash") and names no real repo; the bare model name does
    # (unsloth/GLM-5.3-Flash-GGUF). Try the bare spelling first and keep the
    # prefixed one for creators that really are part of the repo name.
    for name in (bare_model_name(row), display_name(row)):
        slug = re.sub(r"\s+", "-", clean_repo_model_name(name))
        if slug:
            repos.append(f"unsloth/{slug}-GGUF")
            repos.append(f"bartowski/{slug}-GGUF")
    return uniq(repos)


def search_queries(row: dict[str, Any]) -> list[str]:
    query = model_query(row)
    display = display_name(row)
    base = huggingface_repo(row)
    parts = [f"{query} GGUF", f"{bare_model_name(row)} GGUF", f"{display} GGUF"]
    if base:
        parts.append(f"{repo_name(base)} GGUF")
    return uniq(parts)


def search_model_id(item: dict[str, Any]) -> str:
    return str(item.get("modelId") or item.get("id") or "")


def base_model_tag_matches(item: dict[str, Any], row: dict[str, Any]) -> bool:
    base = huggingface_repo(row).lower()
    if not base:
        return False
    tags = item.get("tags")
    if not isinstance(tags, list):
        return False
    needles = {f"base_model:{base}", f"base_model:quantized:{base}"}
    return any(isinstance(tag, str) and tag.lower() in needles for tag in tags)


def quick_hf_score(repo: str, row: dict[str, Any], item: dict[str, Any] | None = None) -> int:
    score = TRUSTED_GGUF_OWNERS.get(repo_owner(repo), 20)
    target_key = norm(model_query(row))
    candidate_key = norm(clean_repo_model_name(repo))
    if huggingface_repo(row).lower() == repo.lower():
        score += 220
    if target_key and target_key == candidate_key:
        score += 160
    score += 18 * len(tokens(target_key) & tokens(candidate_key))
    if candidate_relevant(repo, row):
        score += 80
    if "gguf" in repo.lower():
        score += 40
    else:
        score -= 80
    target_has_mtp = "mtp" in row_name(row).lower()
    if "mtp" in repo.lower() and not target_has_mtp:
        score -= 120
    if item:
        downloads = item.get("downloads")
        if isinstance(downloads, int) and downloads > 0:
            score += min(60, len(str(downloads)) * 8)
        if base_model_tag_matches(item, row):
            score += 260
        tags = item.get("tags")
        if isinstance(tags, list) and any(isinstance(tag, str) and tag.lower() == "gguf" for tag in tags):
            score += 40
    return score


def hf_repo_score(repo: str, row: dict[str, Any], quants: list[dict[str, Any]], item: dict[str, Any] | None = None) -> int:
    score = quick_hf_score(repo, row, item)
    # Capped low: a well-stocked repo of the wrong model (a distill with eight
    # quants) must not outscore the right model's sparser repo.
    score += 10 * min(4, len(quants))
    names = {str(q.get("name") or "").upper() for q in quants}
    if any(q.startswith(("Q4", "IQ4", "MXFP4", "MXP4")) for q in names):
        score += 80
    if any(q.startswith(("Q5", "Q6", "Q8", "F16", "BF16")) for q in names):
        score += 40
    return score


# A model's representative quant may move between refreshes (another repo,
# another quant ladder), but not by 5x: that is another model.
MAX_SIZE_JUMP = 5


def size_jump(previous_gb: float, quants: list[dict[str, Any]]) -> float:
    size = representative_size_gb(quants)
    if previous_gb <= 0 or size <= 0:
        return 1.0
    return max(size / previous_gb, previous_gb / size)


def resolve_gguf_repo(row: dict[str, Any], search_limit: int,
                      previous: dict[str, Any] | None = None) -> tuple[str, list[dict[str, Any]], str] | None:
    ranked: list[tuple[int, str, list[dict[str, Any]], str]] = []
    inspected: set[str] = set()

    base = huggingface_repo(row)
    if base and "gguf" in base.lower() and not auxiliary_repo(base):
        quants = fetch_hf_quants(base)
        inspected.add(base.lower())
        if quants:
            ranked.append((hf_repo_score(base, row, quants), base, quants, "direct"))

    search_items: dict[str, dict[str, Any]] = {}
    for query in search_queries(row):
        for item in search_hf_models(query, search_limit):
            repo = search_model_id(item)
            if not repo or "gguf" not in repo.lower() or auxiliary_repo(repo):
                continue
            if not candidate_relevant(repo, row) and not base_model_tag_matches(item, row):
                continue
            search_items.setdefault(repo, item)
    scored = sorted(
        ((quick_hf_score(repo, row, item), repo, item) for repo, item in search_items.items()),
        reverse=True,
    )
    for _, repo, item in scored[:4]:
        if repo.lower() in inspected:
            continue
        inspected.add(repo.lower())
        quants = fetch_hf_quants(repo)
        if quants:
            ranked.append((hf_repo_score(repo, row, quants, item), repo, quants, "hf-search"))

    if not ranked:
        for repo in direct_repo_candidates(row)[:4]:
            if auxiliary_repo(repo):
                continue
            if repo.lower() in inspected:
                continue
            inspected.add(repo.lower())
            quants = fetch_hf_quants(repo)
            if quants:
                ranked.append((hf_repo_score(repo, row, quants), repo, quants, "direct"))

    previous_gb = float((previous or {}).get("size_gb") or 0)
    for entry in list(ranked):
        jump = size_jump(previous_gb, entry[2])
        if jump > MAX_SIZE_JUMP:
            print(f"warning: {display_name(row)}: rejected {entry[1]} ({representative_size_gb(entry[2]):g} GB), "
                  f"{jump:.0f}x off the previous {previous_gb:g} GB", file=sys.stderr)
            ranked.remove(entry)
    if not ranked:
        return None
    ranked.sort(reverse=True, key=lambda entry: entry[0])
    _, repo, quants, method = ranked[0]
    return repo, quants, method


def representative_size_gb(quants: list[dict[str, Any]]) -> float:
    if not quants:
        return 0.0
    by_name = {str(q.get("name") or "").upper(): float(q.get("size_gb") or 0) for q in quants}
    for preferred in ("Q4_K_M", "Q4_K_S", "Q4_K_XL", "IQ4_XS", "IQ4_NL", "MXFP4_MOE", "MXFP4", "Q5_K_M", "Q3_K_M", "Q3_K_XL", "Q8_0"):
        # Check both UD- (Unsloth dynamic) and plain variants — prefer UD- when present.
        for name in (f"UD-{preferred}", preferred):
            if by_name.get(name, 0) > 0:
                return by_name[name]
    sizes = sorted(float(q.get("size_gb") or 0) for q in quants if float(q.get("size_gb") or 0) > 0)
    return sizes[len(sizes) // 2] if sizes else 0.0


def candidate_family(row: dict[str, Any]) -> str:
    creator = model_creator_name(row)
    if creator:
        return creator
    for tok in tokens(display_name(row)):
        for family in FAMILY_PREFIXES:
            if tok.startswith(family):
                return family.capitalize()
    return "Open weights"


def candidate_notes(row: dict[str, Any], method: str) -> str:
    bits = ["Artificial Analysis open-weight row"]
    params = parameters_summary(row)
    if params:
        bits.append(f"{params} params")
    license_name = str(value_from(row, "licenseName", "license_name") or "").strip()
    if license_name:
        bits.append(f"license: {license_name}")
    bits.append(f"GGUF resolved by {method}")
    return "; ".join(bits)


def candidate_model_shape(row: dict[str, Any], arch: dict[str, Any], hf_info: dict[str, Any], quants: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep artifact geometry when the leaderboard omits parameter counts."""
    total = parameters_value(row, "total")
    active = parameters_value(row, "active")
    gguf = hf_info.get("gguf")
    artifact_total = gguf.get("total") if isinstance(gguf, dict) else None
    artifact_size = gguf.get("totalFileSize") if isinstance(gguf, dict) else None
    artifact_arch = str(gguf.get("architecture") or "") if isinstance(gguf, dict) else ""
    main_artifact = (isinstance(artifact_size, int) and not isinstance(artifact_size, bool)
                     and artifact_size > 0
                     and any(q.get("size_bytes") == artifact_size for q in quants))
    same_arch = not artifact_arch or not arch.get("arch") or artifact_arch == arch["arch"]
    if (not total and isinstance(artifact_total, (int, float))
            and not isinstance(artifact_total, bool)
            and math.isfinite(artifact_total) and artifact_total > 0
            and main_artifact and same_arch):
        # HF's GGUF total counts individual parameters; AA reports billions.
        # Its summary can describe a DFlash head or projector in the same repo.
        # Accept it only when its size belongs to a complete main artifact.
        total = artifact_total / 1e9
    experts = arch.get("experts", 0)
    moe = ((isinstance(experts, int) and experts > 1)
           or (total > 0 and active > 0 and active < total * 0.85))
    # Expert count alone does not tell us active parameter count: attention,
    # embeddings and shared experts also participate. Leave it unknown.
    return {"total_params_b": round(total, 3) if total else 0,
            "active_params_b": round(active, 3) if active else 0,
            "moe": moe}


def candidate_from_row(row: dict[str, Any], repo: str, quants: list[dict[str, Any]], method: str) -> dict[str, Any]:
    idx = intelligence(row)
    tps = output_tps(row)
    # Read exact GGUF geometry from one HF range request so the recommender can
    # compute launch overhead with the placement engine's formula, not an
    # estimate. Best-effort: missing geometry leaves the recommender's fallback.
    arch = fetch_gguf_arch(repo) or {}
    shape = candidate_model_shape(row, arch, fetch_hf_model_info(repo) or {}, quants)
    cand = {
        "aa_id": row_id(row),
        "aa_intelligence_index": round(idx, 3) if idx else 0,
        "aa_output_tps": round(tps, 3) if tps else 0,
        "aa_query": model_query(row),
        "aa_slug": row_slug(row),
        "aa_updated_at": utc_now(),
        "family": candidate_family(row),
        "hf_quant_updated_at": utc_now(),
        "name": display_name(row),
        "notes": candidate_notes(row, method),
        "quality": quality_score(idx),
        "quants": quants,
        "repo": repo,
        "size_gb": round(representative_size_gb(quants), 2),
        "speed": speed_score(tps),
        **shape,
    }
    if arch:
        # Embed only the geometry fields the overhead formula consumes; keep
        # the catalog lean and the schema explicit.
        for field in (
            "arch", "layers", "experts", "exp_used", "exp_ff", "exp_shared_ff",
            "embd", "ff", "hkv", "kl", "vl", "kv_lora", "q_lora",
            "leading_dense", "ctx_train",
        ):
            if field in arch:
                cand[field] = arch[field]
    return cand


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        repo = str(cand.get("repo") or "").lower()
        if not repo:
            continue
        prev = best.get(repo)
        prev_score = float(prev.get("aa_intelligence_index") or 0) if prev else -1
        score = float(cand.get("aa_intelligence_index") or 0)
        if prev is None or score > prev_score:
            best[repo] = cand
    return sorted(best.values(), key=lambda row: float(row.get("aa_intelligence_index") or 0), reverse=True)


# A row missing from one leaderboard scrape (Kimi K3, October 2026) is more
# often a scrape hiccup than a withdrawn model. Its last resolved entry is kept
# this long after it was last seen, so one bad scrape cannot delete it.
CARRY_FORWARD_DAYS = 14
# Losing one of the previous catalog's top rows earns a louder warning.
TOP_ROWS_WATCHED = 10

# Open-weight models the leaderboard does not score, listed by review. They
# carry no Artificial Analysis index; quality is a "~" estimate one point under
# the scored row named by estimate_from, and the recommender ranks every
# estimated row after the scored ones. Facts were checked against the Hugging
# Face model cards and config.json on 2026-10-27 (MiniMax-M3.1-Flash: ~428B
# total, 23B active, 60 layers, 128 experts, 4 per token; Nanbeige4.3-33B:
# LlamaForCausalLM, 52 layers); the GGUF header wins where they disagree.
REVIEWED_UNSCORED_MODELS: tuple[dict[str, Any], ...] = (
    {"name": "MiniMax-M3.1-Flash", "family": "MiniMax", "repo": "unsloth/MiniMax-M3.1-Flash-GGUF",
     "estimate_from": "MiniMax-M3",
     "why": "Artificial Analysis lists MiniMax as proprietary, but MiniMaxAI/MiniMax-M3.1-Flash publishes weights",
     "facts": {"arch": "minimax-m3", "layers": 60, "experts": 128, "exp_used": 4,
               "total_params_b": 428, "active_params_b": 23, "moe": True}},
    {"name": "Nanbeige4.3-33B", "family": "Nanbeige", "repo": "bartowski/Nanbeige_Nanbeige4.3-33B-GGUF",
     "estimate_from": "Nanbeige4.1-3B",
     "why": "Artificial Analysis has no page for it; Nanbeige/Nanbeige4.3-33B publishes weights",
     "facts": {"arch": "llama", "layers": 52, "total_params_b": 33.119, "moe": False}},
)


def parse_utc(text: Any) -> datetime | None:
    try:
        when = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def previous_row(prev: dict[str, Any]) -> dict[str, Any]:
    """The leaderboard row a previous catalog entry was built from, as far as it says."""
    name, family = str(prev.get("name") or ""), str(prev.get("family") or "")
    if family and name.lower().startswith(family.lower() + " "):
        name = name[len(family) + 1:]  # display_name() prefixed the creator
    return {"name": name, "slug": prev.get("aa_slug") or "", "creator": {"name": family}}


def previous_matches(prev: dict[str, Any], row: dict[str, Any]) -> bool:
    """The previous entry still names this model under today's matcher.

    A repo resolved by its constructed name ("direct") was never searched for,
    so only a searched one can be an old mismatch (Command A -> Command A+).
    """
    return ("resolved by direct" in str(prev.get("notes") or "")
            or candidate_relevant(str(prev.get("repo") or ""), row))


def carry_forward(prev: dict[str, Any], row: dict[str, Any], now: datetime, why: str) -> dict[str, Any] | None:
    """The previous entry, kept while fresh and still the same model, else None."""
    name = prev.get("name")
    seen = parse_utc(prev.get("aa_updated_at"))
    if seen is None or now - seen > timedelta(days=CARRY_FORWARD_DAYS):
        print(f"warning: {name}: {why}; dropped (last seen {prev.get('aa_updated_at') or 'never'})", file=sys.stderr)
        return None
    if not previous_matches(prev, row):
        print(f"warning: {name}: {why}; dropped, {prev.get('repo')} no longer matches the model", file=sys.stderr)
        return None
    cand = dict(prev)
    cand["quants"] = drop_companion_quants(list(prev.get("quants") or []), str(prev.get("repo")))
    if not cand["quants"]:
        return None
    note = f"last seen on the leaderboard {seen.date().isoformat()}"
    if note not in str(cand.get("notes") or ""):
        cand["notes"] = f"{cand.get('notes') or ''}; {note}".lstrip("; ")
    expires = (seen + timedelta(days=CARRY_FORWARD_DAYS)).date().isoformat()
    print(f"warning: {name}: {why}; kept the previous entry until {expires}", file=sys.stderr)
    return cand


def reviewed_candidates(candidates: list[dict[str, Any]], previous: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Catalog rows for REVIEWED_UNSCORED_MODELS, ranked below their anchor."""
    by_name = {str(c.get("name") or "").lower(): c for c in candidates}
    repos = {str(c.get("repo") or "").lower() for c in candidates}
    out = []
    for spec in REVIEWED_UNSCORED_MODELS:
        name, repo = spec["name"], spec["repo"]
        if name.lower() in by_name or repo.lower() in repos:
            continue  # the leaderboard scores it now
        anchor = by_name.get(spec["estimate_from"].lower())
        if not anchor or catalog_intelligence(anchor) <= 0 or int(anchor.get("quality") or 0) <= 1:
            print(f"warning: {name}: no scored {spec['estimate_from']} row to estimate from; not listed", file=sys.stderr)
            continue
        quants = fetch_hf_quants(repo)
        if not quants:
            old = next((c for c in previous if str(c.get("repo") or "").lower() == repo.lower()), None)
            quants = drop_companion_quants(list((old or {}).get("quants") or []), repo)
        if not quants:
            print(f"warning: {name}: no GGUF quants in {repo}; not listed", file=sys.stderr)
            continue
        row = {"name": name, "creator": {"name": spec["family"]}}
        try:
            cand = candidate_from_row(row, repo, quants, "reviewed override")
        except Exception as exc:  # noqa: BLE001 - one bad repo never fails the refresh
            print(f"warning: {name}: {exc}; not listed", file=sys.stderr)
            continue
        for field, want in spec["facts"].items():
            have = cand.get(field)
            if field in ("total_params_b", "active_params_b") and not have:
                cand[field] = want
            elif field == "moe":
                cand["moe"] = bool(have) or want
            elif have is None:
                cand[field] = want
            elif have != want:
                print(f"warning: {name}: GGUF {field}={have}, reviewed {want}; keeping the GGUF value", file=sys.stderr)
        cand.update({
            "aa_query": "", "aa_slug": "", "aa_updated_at": "",
            "family": spec["family"],
            "quality": int(anchor["quality"]) - 1,
            "notes": (f"Unscored: {spec['why']}; ~intelligence is an estimate one point under "
                      f"{anchor['name']}'s score; listed by review"),
        })
        out.append(cand)
    return out


def validate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair or drop rows the Go catalog tests reject; never fail the refresh."""
    out = []
    for cand in candidates:
        label = f"{cand.get('name')} ({cand.get('repo')})"
        quants = drop_companion_quants(list(cand.get("quants") or []), label)
        if not quants:
            print(f"warning: {label}: no main-model quants; dropped", file=sys.stderr)
            continue
        if len(quants) != len(cand.get("quants") or []):
            cand["quants"] = quants
            cand["size_gb"] = round(representative_size_gb(quants), 2)
        experts = cand.get("experts")
        if isinstance(experts, int) and experts > 1 and not cand.get("moe"):
            print(f"warning: {label}: {experts} experts but classified dense; marked MoE", file=sys.stderr)
            cand["moe"] = True
        out.append(cand)
    return out


def build_catalog(rows: list[dict[str, Any]], source_label: str, limit: int, search_limit: int,
                  previous: list[dict[str, Any]] | None = None, now: datetime | None = None) -> dict[str, Any]:
    previous = [c for c in previous or [] if isinstance(c, dict) and c.get("repo")]
    now = now or datetime.now(timezone.utc)
    scored_before = [c for c in previous if catalog_intelligence(c) > 0]
    by_slug = {str(c.get("aa_slug")): c for c in scored_before if c.get("aa_slug")}
    by_name = {str(c.get("name") or "").lower(): c for c in scored_before}

    def previous_for(row: dict[str, Any]) -> dict[str, Any] | None:
        return by_slug.get(row_slug(row)) or by_name.get(display_name(row).lower())

    ranked_rows = [row for row in sorted(dedupe_rows(rows), key=intelligence, reverse=True) if intelligence(row) > 0]
    candidates: list[dict[str, Any]] = []
    carried: list[dict[str, Any]] = []
    skipped = 0
    seen_repos: set[str] = set()
    for row in ranked_rows:
        if len(candidates) >= limit:
            break
        if row.get("deprecated") is True:
            continue
        prev = previous_for(row)
        if prev and not previous_matches(prev, row):
            prev = None  # an earlier mismatch must not anchor the size guard
        cand = None
        try:
            resolved = resolve_gguf_repo(row, search_limit, prev)
            if resolved and resolved[0].lower() in seen_repos:
                continue
            if resolved:
                cand = candidate_from_row(row, *resolved)
        except Exception as exc:  # noqa: BLE001 - one bad repo never fails the refresh
            print(f"warning: {display_name(row)}: {exc}", file=sys.stderr)
        if cand is None:
            kept = carry_forward(prev, row, now, "unresolved this run") if prev else None
            if kept:
                carried.append(kept)
            else:
                skipped += 1
            continue
        seen_repos.add(cand["repo"].lower())
        candidates.append(cand)

    present = ({row_slug(row) for row in rows} | {display_name(row).lower() for row in rows}) - {""}
    top = {id(c) for c in sorted(scored_before, key=catalog_intelligence, reverse=True)[:TOP_ROWS_WATCHED]}
    for prev in scored_before:
        if {prev.get("aa_slug"), str(prev.get("name") or "").lower()} & present:
            continue
        if id(prev) in top:
            print(f"warning: top-{TOP_ROWS_WATCHED} model {prev.get('name')} is missing from the leaderboard scrape", file=sys.stderr)
        kept = carry_forward(prev, previous_row(prev), now, "missing from the leaderboard scrape")
        if kept:
            carried.append(kept)
    carried = [c for c in carried if str(c["repo"]).lower() not in seen_repos]
    candidates = dedupe_candidates(candidates + carried)[:limit]
    kept_count = sum(1 for c in carried if c in candidates)
    candidates = validate_candidates(candidates + reviewed_candidates(candidates, previous))
    return {
        "attribution": "Artificial Analysis intelligence data cached from https://artificialanalysis.ai/ and filtered by ggrun hardware fit; Hugging Face GGUF metadata cached from public model APIs",
        "candidates": candidates,
        "generated_at": utc_now(),
        "source": f"{source_label}; resolved {len(candidates) - kept_count} GGUF repos with quant sizes; kept {kept_count} previous rows; skipped {skipped} open-weight rows without matching GGUF quants",
        "version": 2,
    }


# The architecture tables of the backends ggrun builds. A catalog row whose
# GGUF architecture neither registers needs a fork backend on first launch,
# which ggrun builds automatically; the recommender labels such rows.
UPSTREAM_ARCH_TABLES = (
    ("llama.cpp", "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp"),
    ("ik_llama.cpp", "https://raw.githubusercontent.com/ikawrakow/ik_llama.cpp/main/src/llama-arch.cpp"),
)


def parse_arch_names(source: str) -> set[str]:
    """Architecture names registered in a llama-arch.cpp LLM_ARCH_NAMES table."""
    start = source.find("LLM_ARCH_NAMES")
    if start < 0:
        return set()
    body = source[start:]
    end = body.find("};")
    if end < 0:
        return set()
    return {name.lower() for name in re.findall(r'\{\s*LLM_ARCH_\w+\s*,\s*"([^"]+)"', body[:end])}


def fetch_upstream_arch_tables() -> dict[str, set[str]] | None:
    """Architecture names per upstream table, or None when any table is unreadable.

    A partial union would mark rows the missing backend loads as unrunnable, so
    an incomplete fetch stamps nothing and the recommender falls back to its
    explicit blocklist.
    """
    tables: dict[str, set[str]] = {}
    for label, url in UPSTREAM_ARCH_TABLES:
        try:
            parsed = parse_arch_names(fetch_text(url, timeout=30))
        except Exception as exc:  # noqa: BLE001 - any failure leaves rows unstamped
            print(f"warning: {label} architecture table unavailable ({exc}); runnability not stamped", file=sys.stderr)
            return None
        if not parsed:
            print(f"warning: {label} architecture table had no entries; runnability not stamped", file=sys.stderr)
            return None
        tables[label] = parsed
    return tables


def fetch_upstream_arches() -> set[str] | None:
    """Union of upstream architecture names, or None when any table is unreadable."""
    tables = fetch_upstream_arch_tables()
    return set().union(*tables.values()) if tables else None


def stamp_runnable(candidates: list[dict[str, Any]], arches: set[str] | None, mainline: set[str] | None = None) -> None:
    """Mark each row with a known GGUF architecture as loadable upstream or not.

    The stamp labels fork-only rows; it never removes them from recommendations.
    runnable_mainline records the mainline table alone: the launcher's automatic
    builds are mainline (or a reviewed recipe), so an architecture only ik_llama
    registers has no build route.
    """
    if not arches:
        return
    for cand in candidates:
        arch = str(cand.get("arch") or "").lower()
        if arch:
            cand["runnable"] = arch in arches
            if mainline:
                cand["runnable_mainline"] = arch in mainline


def stamp_upstream(candidates: list[dict[str, Any]], tables: dict[str, set[str]] | None) -> None:
    """Stamp runnable and runnable_mainline from fetched upstream tables."""
    if not tables:
        return
    stamp_runnable(candidates, set().union(*tables.values()), tables.get("llama.cpp"))


def catalog_intelligence(row: dict[str, Any]) -> float:
    val = row.get("aa_intelligence_index")
    return float(val) if isinstance(val, (int, float)) else 0.0


def print_catalog_models(catalog: dict[str, Any], limit: int) -> None:
    rows = [row for row in catalog.get("candidates", []) if isinstance(row, dict)]
    ranked = sorted(rows, key=lambda row: (catalog_intelligence(row), int(row.get("quality") or 0)), reverse=True)[:limit]
    print(f"Top {len(ranked)} ggrun GGUF candidates by cached Artificial Analysis intelligence")
    print("| rank | model | repo | q4-ish GB | quants | moe | intelligence | output tps |")
    print("|---:|---|---|---:|---:|---|---:|---:|")
    for i, row in enumerate(ranked, 1):
        name = str(row.get("name") or "").replace("|", "/")
        repo = str(row.get("repo") or "").replace("|", "/")
        size = float(row.get("size_gb") or 0)
        moe = "yes" if row.get("moe") else "no"
        tps = float(row.get("aa_output_tps") or 0)
        quants = row.get("quants") if isinstance(row.get("quants"), list) else []
        print(f"| {i} | {name} | {repo} | {size:.1f} | {len(quants)} | {moe} | {catalog_intelligence(row):.3f} | {tps:.3f} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh ggrun recommendation catalog")
    parser.add_argument("--catalog", default="go/pkg/recommend/catalog.json")
    parser.add_argument("--api-key-env", default="ARTIFICIAL_ANALYSIS_API_KEY")
    parser.add_argument("--allow-missing-key", action="store_true", help="use public leaderboard fallback when the API key is missing")
    parser.add_argument("--catalog-limit", type=int, default=DEFAULT_CATALOG_LIMIT, help="max GGUF candidates to keep")
    parser.add_argument("--hf-search-limit", type=int, default=20, help="Hugging Face search rows to inspect per query")
    parser.add_argument("--hf-min-delay", type=float, default=0.0, help="minimum seconds between Hugging Face API requests")
    parser.add_argument("--hf-429-backoff", type=float, default=0.0, help="seconds to wait and retry after Hugging Face HTTP 429")
    parser.add_argument("--print-top", type=int, default=0, help="print top N source models by intelligence index")
    parser.add_argument("--print-open-weights-top", type=int, default=0, help="print top N open-weight source models by intelligence index")
    parser.add_argument("--print-catalog-top", type=int, default=0, help="print top N checked-in GGUF recommendation rows by cached intelligence")
    parser.add_argument("--stamp-runnable-only", action="store_true", help="only re-stamp upstream runnability on the existing catalog")
    args = parser.parse_args()

    global HF_MIN_DELAY_SECONDS, HF_429_BACKOFF_SECONDS
    HF_MIN_DELAY_SECONDS = max(0.0, args.hf_min_delay)
    HF_429_BACKOFF_SECONDS = max(0.0, args.hf_429_backoff)

    catalog_path = Path(args.catalog)
    if args.stamp_runnable_only:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        tables = fetch_upstream_arch_tables()
        if not tables:
            raise SystemExit("upstream architecture tables unavailable")
        stamp_upstream(catalog.get("candidates", []), tables)
        tmp = catalog_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(catalog_path)
        print(f"stamped runnability in {catalog_path}")
        return 0
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and not args.allow_missing_key:
        raise SystemExit(f"{args.api_key_env} is not set")

    try:
        rows, source_label = load_open_weight_rows(api_key)
        if args.print_top > 0:
            print_top_models(rows, args.print_top)
        if args.print_open_weights_top > 0:
            print_top_models(rows, args.print_open_weights_top, open_weights_only=True)
        try:
            previous = json.loads(catalog_path.read_text(encoding="utf-8")).get("candidates") or []
        except (OSError, ValueError, AttributeError) as exc:
            print(f"warning: previous catalog unreadable ({exc}); nothing carried forward", file=sys.stderr)
            previous = []
        catalog = build_catalog(rows, source_label, max(1, args.catalog_limit), max(1, args.hf_search_limit), previous)
    except Exception as exc:
        if args.allow_missing_key:
            print(f"recommendation catalog unchanged: {exc}", file=sys.stderr)
            return 0
        raise

    if not catalog.get("candidates"):
        if args.allow_missing_key:
            print("recommendation catalog unchanged: no GGUF candidates resolved", file=sys.stderr)
            return 0
        raise SystemExit("no GGUF candidates resolved")

    stamp_upstream(catalog["candidates"], fetch_upstream_arch_tables())
    if args.print_catalog_top > 0:
        print_catalog_models(catalog, args.print_catalog_top)
    tmp = catalog_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(catalog_path)
    print(f"updated {catalog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
