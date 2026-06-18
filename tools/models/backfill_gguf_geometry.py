#!/usr/bin/env python3
"""Backfill GGUF geometry into an existing recommendation catalog.

Reads go/pkg/recommend/catalog.json, fetches the GGUF KV metadata header (one
HF range request per repo, ~8 MB) for candidates that lack the geometry fields,
and writes the geometry back. This is the offline equivalent of what
tools/models/update_recommendations.py now does at catalog-build time, so a
catalog generated before the geometry support landed can be upgraded without
re-running the full Artificial Analysis pipeline.

Best-effort: repos whose header can't be read are left unchanged (the Go
recommender falls back to the size-based estimate for them).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from update_recommendations import fetch_gguf_arch  # noqa: E402

GEOMETRY_FIELDS = (
    "arch", "layers", "experts", "exp_used", "exp_ff", "exp_shared_ff",
    "embd", "ff", "hkv", "kl", "vl", "kv_lora", "q_lora",
    "leading_dense", "ctx_train",
)


def has_geometry(c: dict) -> bool:
    return all(c.get(f) for f in ("layers", "hkv", "kl", "vl"))


def main() -> int:
    catalog_path = Path(__file__).resolve().parents[2] / "go" / "pkg" / "recommend" / "catalog.json"
    d = json.loads(catalog_path.read_text())
    candidates = d.get("candidates", [])
    updated = 0
    skipped = 0
    failed = 0
    for i, c in enumerate(candidates, 1):
        if has_geometry(c):
            skipped += 1
            continue
        repo = c.get("repo", "")
        if not repo:
            failed += 1
            continue
        print(f"[{i}/{len(candidates)}] {repo} ...", flush=True)
        arch = fetch_gguf_arch(repo)
        if not arch:
            print(f"  FAILED (no geometry)", flush=True)
            failed += 1
            # Throttle even on failure to avoid hammering HF on 404s.
            time.sleep(0.4)
            continue
        for f in GEOMETRY_FIELDS:
            if f in arch:
                c[f] = arch[f]
        updated += 1
        print(f"  ok: arch={arch.get('arch')} L={arch.get('layers')} E={arch.get('experts')} hkv={arch.get('hkv')} kl={arch.get('kl')}", flush=True)
        time.sleep(0.4)
    tmp = catalog_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    tmp.replace(catalog_path)
    print(f"\nupdated={updated} skipped(already had geometry)={skipped} failed={failed}")
    print(f"wrote {catalog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
