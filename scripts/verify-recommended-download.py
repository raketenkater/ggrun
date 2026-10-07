#!/usr/bin/env python3
"""Follow ggrun's own recommendation for this machine: recommend -> download -> verify.

Records the recommender's top choices, checks the offered quant is a complete
main-model artifact (not a draft head or a partial shard set), downloads it with
the documented `ggrun download`, and verifies the bytes on disk equal the
catalog's size. A top pick too large for the runner's disk/bandwidth limit is
recorded as such and the first recommendation within the limit is used instead;
the substitution is visible in the evidence, never silent.

Prints MODEL=<path> on success (append it to $GITHUB_ENV in CI).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--launcher", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-download-gb", type=float, default=12.0)
    ap.add_argument("--category", default="Balanced")
    ap.add_argument("--recommend-args", default="", help="extra recommend arguments, e.g. '--cpu --ram-budget 16G'")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"passed": False, "category": a.category, "max_download_gb": a.max_download_gb}
    try:
        argv = [a.launcher, "recommend", "--json", "-n", "5"] + a.recommend_args.split()
        r = run(argv, 300)
        if r.returncode != 0:
            raise RuntimeError(f"recommend failed: {r.stderr[-800:]}")
        doc = json.loads(r.stdout)
        (out / "recommend.json").write_text(json.dumps(doc, indent=1))
        rec["planning_hardware"] = {k: doc["planning_hardware"].get(k) for k in ("os", "arch", "ram", "gpus")}
        rows = doc["categories"].get(a.category) or []
        rec["top"] = [{k: row.get(k) for k in ("name", "repo", "QuantName", "QuantSizeGB", "MemoryNeedGB", "Fit", "Reason")}
                      for row in rows[:5]]
        if not rows:
            raise RuntimeError(f"no {a.category} recommendation for this hardware")
        chosen = None
        for row in rows:
            sizes = [q["size_bytes"] for q in row.get("quants", []) if q.get("size_bytes")]
            quant = next((q for q in row.get("quants", []) if q["name"] == row["QuantName"]), None)
            if quant is None or not quant.get("size_bytes"):
                raise RuntimeError(f"recommended quant {row['QuantName']} has no catalog size: {row['repo']}")
            # A complete quant is never ~1/25 of the model's largest one; a draft
            # head or partial shard set is. This is the defect that offered a
            # 1.2 GB MTP head as a 156 GB model's Q4_0.
            if quant["size_bytes"] * 25 < max(sizes):
                raise RuntimeError(f"recommended {row['repo']} {quant['name']} ({quant['size_bytes']} B) is a companion-sized artifact")
            if quant["size_bytes"] / 1e9 <= a.max_download_gb:
                chosen = (row, quant)
                break
            rec.setdefault("skipped_for_runner_limits", []).append(
                {"repo": row["repo"], "quant": quant["name"], "size_gb": round(quant["size_bytes"] / 1e9, 2)})
        if chosen is None:
            raise RuntimeError("no recommendation fits the runner download limit")
        row, quant = chosen
        rec["chosen"] = {"repo": row["repo"], "quant": quant["name"], "catalog_bytes": quant["size_bytes"],
                         "rank": rows.index(row) + 1}
        models_dir = Path(a.models_dir)
        before = {p for p in models_dir.rglob("*.gguf")} if models_dir.exists() else set()
        t0 = time.monotonic()
        d = subprocess.run([a.launcher, "download", row["repo"], "--quant", quant["name"]], timeout=a.timeout)
        rec["download_s"] = round(time.monotonic() - t0, 1)
        if d.returncode != 0:
            raise RuntimeError(f"ggrun download exited {d.returncode}")
        new = sorted(p for p in models_dir.rglob("*.gguf") if p not in before)
        if not new:
            # a re-run with the file already present is still a valid selection
            new = sorted(p for p in models_dir.rglob("*.gguf") if quant["name"].lower() in p.name.lower())
        on_disk = sum(p.stat().st_size for p in new)
        rec["downloaded"] = [{"path": str(p), "bytes": p.stat().st_size} for p in new]
        rec["bytes_on_disk"] = on_disk
        if on_disk != quant["size_bytes"]:
            raise RuntimeError(f"downloaded {on_disk} bytes, catalog says {quant['size_bytes']}")
        first = next((p for p in new if "-00001-of-" in p.name), new[0])
        rec["model"] = str(first)
        rec["passed"] = True
        print(f"MODEL={first}")
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
    finally:
        (out / "recommended-download.json").write_text(json.dumps(rec, indent=2))
    if not rec["passed"]:
        print(json.dumps(rec, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
