#!/usr/bin/env python3
"""Compare a GPU acceptance run with a known-good run of the same workload.

Only matched runs are compared: same model file (sha256 when both recorded,
else path and size), same effective backend argv, same serving context. A
mismatch is reported as "not comparable", never as a pass or a regression.
There is deliberately no VRAM-fill target: fuller memory is not faster work.

usage: compare-gpu-acceptance.py <baseline result.json> <candidate result.json>
       [--ready-tolerance 0.25] [--decode-tolerance 0.10]
Exit 0: comparable, no regression. 1: regression. 2: not comparable.
"""
import argparse
import json
import os
import re
import sys

# Flags whose value is a file inside each run's own cache directory. Two runs
# with separate caches write the same generated file to different paths, so
# only the file name is part of the identity.
PER_CACHE_PATH_FLAGS = ("--chat-template-file",)


def normalized_argv(argv):
    if not argv:
        return argv
    for flag in PER_CACHE_PATH_FLAGS:
        argv = re.sub(rf"({re.escape(flag)} )(\S+)", lambda m: m.group(1) + os.path.basename(m.group(2)), argv)
    return argv


def ident(r):
    model = (r.get("identity") or {}).get("model") or {}
    backend = (r.get("identity") or {}).get("backend") or {}
    served = r.get("served") or {}
    return {"model": model.get("sha256") or (model.get("path"), model.get("bytes")),
            "argv": normalized_argv(backend.get("argv")), "context": served.get("context")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--ready-tolerance", type=float, default=0.25)
    ap.add_argument("--decode-tolerance", type=float, default=0.10)
    a = ap.parse_args()
    base, cand = json.load(open(a.baseline)), json.load(open(a.candidate))
    report = {"comparable": True, "regressions": [], "checks": {}}
    ib, ic = ident(base), ident(cand)
    for key in ib:
        if ib[key] != ic[key]:
            report["comparable"] = False
            report.setdefault("mismatch", {})[key] = {"baseline": ib[key], "candidate": ic[key]}
    if not report["comparable"]:
        print(json.dumps(report, indent=2))
        sys.exit(2)
    if not cand.get("passed"):
        report["regressions"].append("candidate lifecycle did not pass")
    rb, rc = (base.get("phases_s") or {}).get("ready"), (cand.get("phases_s") or {}).get("ready")
    report["checks"]["ready_s"] = {"baseline": rb, "candidate": rc}
    if rb and rc and rc > rb * (1 + a.ready_tolerance) + 5:
        report["regressions"].append(f"ready {rc}s vs {rb}s")
    db = (base.get("generation_timings") or {}).get("predicted_per_second")
    dc = (cand.get("generation_timings") or {}).get("predicted_per_second")
    report["checks"]["decode_tps"] = {"baseline": db, "candidate": dc}
    if db and dc and dc < db * (1 - a.decode_tolerance):
        report["regressions"].append(f"decode {dc:.2f} vs {db:.2f} tok/s")
    released = cand.get("memory_released") or {}
    report["checks"]["memory_released"] = released
    if any(v is False for v in released.values()):
        report["regressions"].append("GPU memory not released after stop")
    ab, ac = base.get("device_allocations_mib"), cand.get("device_allocations_mib")
    report["checks"]["device_allocations_changed"] = ab != ac
    print(json.dumps(report, indent=2))
    sys.exit(1 if report["regressions"] else 0)


if __name__ == "__main__":
    main()
