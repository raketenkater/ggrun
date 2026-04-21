#!/usr/bin/env python3
"""
Benchmark --ai-tune across multiple models.
Runs each model with heuristic baseline, then ai-tune, compares results.

Usage:
    python3 benchmark-ai-tune.py                    # All .gguf in ~/ai_models
    python3 benchmark-ai-tune.py model1.gguf model2.gguf
    python3 benchmark-ai-tune.py --rounds 4         # Fewer rounds (faster)
    python3 benchmark-ai-tune.py --skip mmproj      # Skip files matching pattern
"""

import subprocess
import json
import sys
import os
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).parent
LLM_SERVER = SCRIPT_DIR / "llm-server"
CACHE_DIR = Path.home() / ".cache" / "llm-server"
HISTORY_FILE = CACHE_DIR / "tune_history.jsonl"
RESULTS_FILE = SCRIPT_DIR / "benchmark-results.json"
PORT = 8081


def kill_port(port):
    """Kill anything on the port."""
    try:
        pids = subprocess.check_output(
            ["lsof", "-t", f"-i:{port}"], stderr=subprocess.DEVNULL
        ).decode().strip()
        if pids:
            for pid in pids.split("\n"):
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
            time.sleep(3)
    except subprocess.CalledProcessError:
        pass


def get_heuristic_baseline(model_path):
    """Run model with heuristic config, benchmark, return gen/pp tok/s."""
    kill_port(PORT)

    # Delete any existing tune cache for this model so we get pure heuristic
    for f in CACHE_DIR.glob("tune_*.json"):
        model_name = Path(model_path).name
        if model_name in f.name:
            backup = f.with_suffix(".json.bak")
            f.rename(backup)
            print(f"  Backed up tune cache: {f.name}")

    # Start server with heuristic config + benchmark mode
    print(f"  Starting heuristic baseline...")
    proc = subprocess.Popen(
        ["bash", str(LLM_SERVER), str(model_path), "--benchmark"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    gen_tps = 0.0
    pp_tps = 0.0
    try:
        stdout, _ = proc.communicate(timeout=600)
        for line in stdout.split("\n"):
            if "gen" in line and "tok/s" in line and "pp" in line:
                # Parse: "Benchmark: gen=25.94 tok/s  pp=150.54 tok/s"
                parts = line.split()
                for p in parts:
                    if p.startswith("gen="):
                        try:
                            gen_tps = float(p.split("=")[1])
                        except ValueError:
                            pass
                    elif p.startswith("pp="):
                        try:
                            pp_tps = float(p.split("=")[1])
                        except ValueError:
                            pass
    except subprocess.TimeoutExpired:
        proc.kill()
        print("  WARNING: Baseline benchmark timed out")

    kill_port(PORT)
    return gen_tps, pp_tps


def run_ai_tune(model_path, rounds=8):
    """Run --ai-tune --retune, return best gen/pp tok/s."""
    kill_port(PORT)

    # Remove tune cache to force fresh tune from heuristic
    for f in CACHE_DIR.glob("tune_*.json"):
        model_name = Path(model_path).name
        if model_name in f.name:
            f.unlink()

    print(f"  Running AI tune ({rounds} rounds)...")
    env = os.environ.copy()

    # Temporarily patch rounds if needed
    proc = subprocess.Popen(
        ["bash", str(LLM_SERVER), str(model_path), "--ai-tune", "--retune"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    baseline_gen = 0.0
    baseline_pp = 0.0
    best_gen = 0.0
    best_pp = 0.0
    best_name = "baseline"
    rounds_completed = 0

    try:
        stdout, _ = proc.communicate(timeout=7200)  # 2h max
        for line in stdout.split("\n"):
            if "Baseline:" in line:
                for p in line.split():
                    if p.startswith("gen="):
                        try:
                            baseline_gen = float(p.split("=")[1])
                        except ValueError:
                            pass
                    elif p.startswith("pp="):
                        try:
                            baseline_pp = float(p.split("=")[1])
                        except ValueError:
                            pass
            if "NEW BEST:" in line or "Result:" in line:
                rounds_completed += 1
                for p in line.split():
                    if p.startswith("gen="):
                        try:
                            g = float(p.split("=")[1])
                            if g > best_gen:
                                best_gen = g
                        except ValueError:
                            pass
                    elif p.startswith("pp="):
                        try:
                            pp = float(p.split("=")[1])
                            if best_gen == g:
                                best_pp = pp
                        except ValueError:
                            pass
            if "CRASHED" in line:
                rounds_completed += 1
            if "wins!" in line:
                # Extract winner name: "AI Tune complete: <name> wins!"
                parts = line.split(":")
                if len(parts) > 1:
                    best_name = parts[-1].replace("wins!", "").strip()
            if "baseline wins" in line:
                best_name = "baseline"
                best_gen = baseline_gen
                best_pp = baseline_pp

    except subprocess.TimeoutExpired:
        proc.kill()
        print("  WARNING: AI tune timed out after 2 hours")

    kill_port(PORT)

    # If best is still 0 but baseline was measured, baseline won
    if best_gen == 0 and baseline_gen > 0:
        best_gen = baseline_gen
        best_pp = baseline_pp
        best_name = "baseline"

    return {
        "baseline_gen": baseline_gen,
        "baseline_pp": baseline_pp,
        "tuned_gen": best_gen,
        "tuned_pp": best_pp,
        "best_name": best_name,
        "rounds": rounds_completed,
    }


def restore_caches():
    """Restore any backed-up tune caches."""
    for f in CACHE_DIR.glob("tune_*.json.bak"):
        orig = f.with_suffix("")  # remove .bak
        if not orig.exists():
            f.rename(orig)
            print(f"Restored cache: {orig.name}")
        else:
            f.unlink()


def main():
    parser = argparse.ArgumentParser(description="Benchmark --ai-tune across models")
    parser.add_argument("models", nargs="*", help="Model paths (default: all in ~/ai_models)")
    parser.add_argument("--rounds", type=int, default=10, help="Tuning rounds (default: 10)")
    parser.add_argument("--skip", nargs="*", default=["mmproj"], help="Skip files matching patterns")
    parser.add_argument("--model-dir", default=str(Path.home() / "ai_models"), help="Model directory")
    args = parser.parse_args()

    if args.models:
        models = [Path(m) for m in args.models]
    else:
        model_dir = Path(args.model_dir)
        models = sorted(model_dir.glob("*.gguf"))

    # Filter skips
    models = [m for m in models if not any(s in m.name for s in args.skip)]

    if not models:
        print("No models found.")
        sys.exit(1)

    print(f"Benchmarking {len(models)} models with --ai-tune ({args.rounds} rounds each)")
    print(f"Models: {', '.join(m.name for m in models)}")
    print()

    results = []

    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {model.name}")
        print(f"  Size: {model.stat().st_size / 1024**3:.1f} GB")

        start = time.time()
        result = run_ai_tune(model, args.rounds)
        elapsed = time.time() - start

        gain_gen = 0
        gain_pp = 0
        if result["baseline_gen"] > 0:
            gain_gen = (result["tuned_gen"] - result["baseline_gen"]) / result["baseline_gen"] * 100
        if result["baseline_pp"] > 0:
            gain_pp = (result["tuned_pp"] - result["baseline_pp"]) / result["baseline_pp"] * 100

        result["model"] = model.name
        result["gain_gen_pct"] = round(gain_gen, 1)
        result["gain_pp_pct"] = round(gain_pp, 1)
        result["elapsed_min"] = round(elapsed / 60, 1)
        result["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        results.append(result)

        print(f"  Baseline:  gen={result['baseline_gen']:.2f} tok/s  pp={result['baseline_pp']:.2f} tok/s")
        print(f"  AI-Tuned:  gen={result['tuned_gen']:.2f} tok/s  pp={result['tuned_pp']:.2f} tok/s")
        print(f"  Gain:      gen={gain_gen:+.1f}%  pp={gain_pp:+.1f}%")
        print(f"  Winner:    {result['best_name']}")
        print(f"  Rounds:    {result['rounds']}  Time: {result['elapsed_min']} min")
        print()

    # Restore backed up caches
    restore_caches()

    # Print summary table
    print("=" * 80)
    print(f"{'Model':<45} {'Baseline':>10} {'Tuned':>10} {'Gain':>8} {'Time':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<45} {r['baseline_gen']:>8.1f} → {r['tuned_gen']:>7.1f}  {r['gain_gen_pct']:>+6.1f}%  {r['elapsed_min']:>5.1f}m")
    print("=" * 80)

    avg_gain = sum(r["gain_gen_pct"] for r in results) / len(results) if results else 0
    print(f"Average generation gain: {avg_gain:+.1f}%")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
