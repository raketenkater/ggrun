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
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_URL = "https://artificialanalysis.ai/api/v2/language/models"
LEGACY_API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
OPEN_WEIGHTS_PAGE_URL = "https://artificialanalysis.ai/leaderboards/models?weights=open"
HF_MODEL_API_URL = "https://huggingface.co/api/models"
DEFAULT_CATALOG_LIMIT = 100
QUANT_PATTERN = re.compile(
    r"(IQ[1-8]_(?:XXS|XS|NL|S|M|L)|Q[1-9]_(?:K_?(?:XL|XL_?M|L|S|M)|0|[1-9]_?[KS])|MXFP4(?:_MOE)?|MXP4(?:_MOE)?|BF16|F16|F32|F8|I4)",
    re.IGNORECASE,
)

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
        raise SystemExit("Artificial Analysis API response was not a JSON object")
    return data


def fetch_public_data(url: str) -> Any:
    return json.loads(fetch_text(url))


def fetch_public_json(url: str) -> dict[str, Any]:
    data = fetch_public_data(url)
    if not isinstance(data, dict):
        raise SystemExit(f"Public API response for {url} was not a JSON object")
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
    # Next.js stores the model table as JSON inside escaped script payloads.
    # Unescaping " is enough to recover a valid JSON array for the table rows.
    normalized = html.unescape(page).replace('\\"', '"')
    rows: list[dict[str, Any]] = []
    pos = 0
    while True:
        idx = normalized.find('"models":[', pos)
        if idx < 0:
            break
        start = normalized.find("[", idx)
        pos = idx + 1
        try:
            parsed = json.loads(json_array_at(normalized, start))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(parsed, list):
            continue
        dict_rows = [row for row in parsed if isinstance(row, dict)]
        if any("isOpenWeights" in row for row in dict_rows):
            rows.extend(row for row in dict_rows if open_weights(row))
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
            raise SystemExit("Artificial Analysis API response did not contain a data array")
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
        if exc.code != 403:
            raise SystemExit(f"Artificial Analysis API {exc}") from exc
        print(f"Artificial Analysis API {exc}; falling back to legacy endpoint", file=sys.stderr)
    try:
        return fetch_models_from_url(api_key, LEGACY_API_URL, paginated=False)
    except APIRequestError as exc:
        raise SystemExit(f"Artificial Analysis API {exc}") from exc


def load_open_weight_rows(api_key: str) -> tuple[list[dict[str, Any]], str]:
    if api_key:
        rows = fetch_models(api_key)
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
    except (APIRequestError, json.JSONDecodeError) as exc:
        warn_hf_error(f"Could not inspect Hugging Face repo {repo}", exc)
        HF_MODEL_CACHE[key] = None
        return None
    HF_MODEL_CACHE[key] = data
    return data


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
    quant_sizes: dict[str, dict[str, int | bool]] = {}
    for item in siblings:
        if not isinstance(item, dict):
            continue
        fname = str(item.get("rfilename") or "")
        if not fname.lower().endswith(".gguf") or "mmproj" in fname.lower():
            continue
        # For LFS-tracked files (all real GGUF weights) item["size"] is the tiny
        # pointer-file size; the actual blob size is in item["lfs"]["size"].
        # Reading the pointer size is what produced phantom quants like
        # "F16 = 0.9GB" for a 30B model. Prefer the LFS size.
        size = None
        lfs = item.get("lfs")
        if isinstance(lfs, dict) and isinstance(lfs.get("size"), int):
            size = lfs["size"]
        elif isinstance(item.get("size"), int):
            size = item.get("size")
        if not isinstance(size, int) or size <= 0:
            continue
        basename = fname.rsplit("/", 1)[-1]
        # Unsloth dynamic quants carry a "UD-" prefix (e.g. UD-IQ4_XS).
        # Preserve it so the recommender knows this is an optimized quant
        # (loss * 0.7) rather than a generic quant of the same type.
        is_dynamic = "-UD-" in basename or basename.upper().startswith("UD-")
        matches = QUANT_PATTERN.findall(basename)
        if not matches and "/" in fname:
            matches = QUANT_PATTERN.findall(fname.split("/", 1)[0])
        for match in matches:
            quant = normalize_quant(match)
            if is_dynamic:
                quant = "UD-" + quant
            entry = quant_sizes.setdefault(quant, {"size": 0, "dynamic": is_dynamic})
            entry["size"] = entry["size"] + size
            entry["dynamic"] = is_dynamic or entry["dynamic"]
    quants = [
        {"name": name, "size_gb": round(info["size"] / 1073741824, 2), "size_bytes": info["size"], "dynamic": info["dynamic"]}
        for name, info in sorted(quant_sizes.items(), key=lambda row: row[1]["size"])
    ]
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
GGUF_HEADER_FETCH_BYTES = 16 * 1024 * 1024  # 16 MiB covers KV metadata + tokenizer for current models
HF_GGUF_ARCH_CACHE: dict[str, dict[str, Any] | None] = {}


def _gguf_resolve_url(repo: str, fname: str) -> str:
    repo_enc = urllib.parse.quote(repo, safe="/")
    file_enc = urllib.parse.quote(fname, safe="/")
    return f"https://huggingface.co/{repo_enc}/resolve/main/{file_enc}"


def _representative_gguf_file(siblings: list[dict[str, Any]]) -> str | None:
    # Prefer the first shard of a split model (KV metadata is duplicated across
    # shards, so shard 1 always carries it). Fall back to any non-mmproj GGUF.
    candidates: list[tuple[str, int]] = []
    for item in siblings:
        if not isinstance(item, dict):
            continue
        fname = str(item.get("rfilename") or "")
        if not fname.lower().endswith(".gguf") or "mmproj" in fname.lower():
            continue
        size = 0
        lfs = item.get("lfs")
        if isinstance(lfs, dict) and isinstance(lfs.get("size"), int):
            size = lfs["size"]
        candidates.append((fname, size))
    if not candidates:
        return None
    # "00001-of" or "-of-00001" (single file) sorts first.
    candidates.sort(key=lambda fs: (0 if "00001-of" in fs[0] or "-of-0000" not in fs[0] else 1, fs[0]))
    return candidates[0][0]


def fetch_gguf_arch(repo: str) -> dict[str, Any] | None:
    """Read GGUF KV metadata geometry from one HF range request.

    Returns a dict with the fields the placement engine's overhead formula
    needs (arch, layers, experts, exp_used, exp_ff, exp_shared_ff, embd, ff,
    hkv, kl, vl, kv_lora, q_lora, leading_dense, ctx_train), or None when the
    repo has no GGUF or the header could not be parsed. Best-effort: any
    failure leaves the caller to fall back to the size-only overhead estimate.
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
    # parse_gguf._read_kv reads from a file-like object; BytesIO works and lets
    # us reuse the exact tested parser instead of duplicating the KV walk.
    import io
    import struct
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf"))
    from parse_gguf import _read_kv

    req_headers = {"Range": f"bytes=0-{GGUF_HEADER_FETCH_BYTES - 1}", "User-Agent": "ggrun-catalog-refresh"}
    req_headers.update(hf_auth_headers(url))
    try:
        req = urllib.request.Request(url, headers=req_headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read(GGUF_HEADER_FETCH_BYTES)
    except Exception as exc:
        warn_hf_error(f"Could not read GGUF header for {repo}/{fname}", exc)
        HF_GGUF_ARCH_CACHE[key] = None
        return None
    buf = io.BytesIO(raw)
    try:
        if buf.read(4) != b"GGUF":
            HF_GGUF_ARCH_CACHE[key] = None
            return None
        buf.read(4)  # gguf version
        struct.unpack("<Q", buf.read(8))[0]  # tensor_count (0 on shard 1 of split models)
        kv_count = struct.unpack("<Q", buf.read(8))[0]
        meta: dict[str, Any] = {"fused": 0, "expert_bytes": 0, "non_expert_bytes": 0}
        _read_kv(buf, meta, kv_count)
    except Exception as exc:
        warn_hf_error(f"GGUF header parse failed for {repo}/{fname}", exc)
        HF_GGUF_ARCH_CACHE[key] = None
        return None
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
    except (APIRequestError, json.JSONDecodeError) as exc:
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


def candidate_relevant(candidate_repo: str, row: dict[str, Any]) -> bool:
    target_query = model_query(row)
    target_tokens = tokens(target_query)
    candidate_tokens = tokens(candidate_repo)
    if not target_tokens or not candidate_tokens:
        return False
    if not family_match(target_tokens, candidate_tokens):
        return False
    needed_sizes = model_size_tokens(target_tokens)
    if needed_sizes and not (needed_sizes & candidate_tokens):
        return False
    target_versions = version_tuples(target_query)
    if target_versions and not (target_versions & version_tuples(candidate_repo)):
        return False
    overlap = len(target_tokens & candidate_tokens)
    if overlap == 0 and not (version_tuples(target_query) & version_tuples(candidate_repo)):
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
    query = re.sub(r"\s+", "-", model_query(row))
    if query:
        repos.append(f"unsloth/{query}-GGUF")
        repos.append(f"bartowski/{query}-GGUF")
    return uniq(repos)


def search_queries(row: dict[str, Any]) -> list[str]:
    query = model_query(row)
    display = display_name(row)
    base = huggingface_repo(row)
    parts = [f"{query} GGUF", f"{display} GGUF"]
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
    score += 25 * min(8, len(quants))
    names = {str(q.get("name") or "").upper() for q in quants}
    if any(q.startswith(("Q4", "IQ4", "MXFP4", "MXP4")) for q in names):
        score += 80
    if any(q.startswith(("Q5", "Q6", "Q8", "F16", "BF16")) for q in names):
        score += 40
    return score


def resolve_gguf_repo(row: dict[str, Any], search_limit: int) -> tuple[str, list[dict[str, Any]], str] | None:
    ranked: list[tuple[int, str, list[dict[str, Any]], str]] = []
    inspected: set[str] = set()

    base = huggingface_repo(row)
    if base and "gguf" in base.lower():
        quants = fetch_hf_quants(base)
        inspected.add(base.lower())
        if quants:
            ranked.append((hf_repo_score(base, row, quants), base, quants, "direct"))

    search_items: dict[str, dict[str, Any]] = {}
    for query in search_queries(row):
        for item in search_hf_models(query, search_limit):
            repo = search_model_id(item)
            if not repo or "gguf" not in repo.lower():
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
            if repo.lower() in inspected:
                continue
            inspected.add(repo.lower())
            quants = fetch_hf_quants(repo)
            if quants:
                ranked.append((hf_repo_score(repo, row, quants), repo, quants, "direct"))

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


def candidate_from_row(row: dict[str, Any], repo: str, quants: list[dict[str, Any]], method: str) -> dict[str, Any]:
    total = parameters_value(row, "total")
    active = parameters_value(row, "active")
    moe = total > 0 and active > 0 and active < total * 0.85
    idx = intelligence(row)
    tps = output_tps(row)
    # Read exact GGUF geometry from one HF range request so the recommender can
    # compute launch overhead with the placement engine's formula, not an
    # estimate. Best-effort: missing geometry leaves the recommender's fallback.
    arch = fetch_gguf_arch(repo) or {}
    cand = {
        "aa_id": row_id(row),
        "aa_intelligence_index": round(idx, 3) if idx else 0,
        "aa_output_tps": round(tps, 3) if tps else 0,
        "aa_query": model_query(row),
        "aa_slug": row_slug(row),
        "aa_updated_at": utc_now(),
        "active_params_b": round(active, 3) if active else 0,
        "family": candidate_family(row),
        "hf_quant_updated_at": utc_now(),
        "moe": moe,
        "name": display_name(row),
        "notes": candidate_notes(row, method),
        "quality": quality_score(idx),
        "quants": quants,
        "repo": repo,
        "size_gb": round(representative_size_gb(quants), 2),
        "speed": speed_score(tps),
        "total_params_b": round(total, 3) if total else 0,
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


def build_catalog(rows: list[dict[str, Any]], source_label: str, limit: int, search_limit: int) -> dict[str, Any]:
    ranked_rows = [row for row in sorted(dedupe_rows(rows), key=intelligence, reverse=True) if intelligence(row) > 0]
    candidates: list[dict[str, Any]] = []
    skipped = 0
    seen_repos: set[str] = set()
    for row in ranked_rows:
        if len(candidates) >= limit:
            break
        if row.get("deprecated") is True:
            continue
        resolved = resolve_gguf_repo(row, search_limit)
        if not resolved:
            skipped += 1
            continue
        repo, quants, method = resolved
        if repo.lower() in seen_repos:
            continue
        seen_repos.add(repo.lower())
        candidates.append(candidate_from_row(row, repo, quants, method))
    candidates = dedupe_candidates(candidates)[:limit]
    return {
        "attribution": "Artificial Analysis intelligence data cached from https://artificialanalysis.ai/ and filtered by ggrun hardware fit; Hugging Face GGUF metadata cached from public model APIs",
        "candidates": candidates,
        "generated_at": utc_now(),
        "source": f"{source_label}; resolved {len(candidates)} GGUF repos with quant sizes; skipped {skipped} open-weight rows without matching GGUF quants",
        "version": 2,
    }


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
    args = parser.parse_args()

    global HF_MIN_DELAY_SECONDS, HF_429_BACKOFF_SECONDS
    HF_MIN_DELAY_SECONDS = max(0.0, args.hf_min_delay)
    HF_429_BACKOFF_SECONDS = max(0.0, args.hf_429_backoff)

    catalog_path = Path(args.catalog)
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and not args.allow_missing_key:
        raise SystemExit(f"{args.api_key_env} is not set")

    try:
        rows, source_label = load_open_weight_rows(api_key)
        if args.print_top > 0:
            print_top_models(rows, args.print_top)
        if args.print_open_weights_top > 0:
            print_top_models(rows, args.print_open_weights_top, open_weights_only=True)
        catalog = build_catalog(rows, source_label, max(1, args.catalog_limit), max(1, args.hf_search_limit))
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

    if args.print_catalog_top > 0:
        print_catalog_models(catalog, args.print_catalog_top)
    tmp = catalog_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(catalog_path)
    print(f"updated {catalog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
