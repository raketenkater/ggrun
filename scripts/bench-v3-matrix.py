#!/usr/bin/env python3
"""Run the local llm-server benchmark matrix with durable artifacts.

The script is intentionally self-contained: it runs each model/column, appends a
JSONL result immediately, and rewrites a Markdown summary after every row. That
keeps long GPU runs inspectable without streaming logs through an assistant.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterable


HOME = Path.home()
DEFAULT_MODELS = [
    ("4B", HOME / "ai_models/Qwen3.5-4B-Q4_K_M.gguf"),
    ("27B", HOME / "ai_models/Qwen3.6-27B-Q5_K_M.gguf"),
    ("122B", HOME / "ai_models/UD-IQ4_XS/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf"),
    ("MiniMax-M3", HOME / "ai_models/UD-IQ3_XXS/MiniMax-M3-UD-IQ3_XXS-00001-of-00005.gguf"),
]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def parse_model_arg(raw: str) -> ModelSpec:
    if "=" not in raw:
        path = expand(raw)
        return ModelSpec(path.stem, path)
    label, path = raw.split("=", 1)
    return ModelSpec(label.strip(), expand(path))


def existing_default_models() -> list[ModelSpec]:
    return [ModelSpec(label, path) for label, path in DEFAULT_MODELS if path.exists()]


def is_ik_only_model(model: ModelSpec) -> bool:
    name = f"{model.label} {model.path.name}".lower()
    return "minimax-m3" in name or "minimax-m3" in str(model.path).lower()


def backend_for_model(args: argparse.Namespace, model: ModelSpec) -> str:
    if is_ik_only_model(model):
        return "ik_llama"
    return args.backend


def server_bin_for_model(args: argparse.Namespace, model: ModelSpec) -> Path:
    if is_ik_only_model(model):
        return args.ik_raw_bin
    return args.raw_bin


def kill_matching_processes(patterns: Iterable[str]) -> None:
    me = os.getpid()
    killed: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if pid == me or pid == os.getppid():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if not cmdline:
            continue
        if any(p in cmdline for p in patterns):
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
    if killed:
        time.sleep(3)
    for pid in killed:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def wait_health(port: int, proc: subprocess.Popen | None, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    urls = [f"http://127.0.0.1:{port}/health", f"http://127.0.0.1:{port}/v1/models"]
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if 200 <= resp.status < 300:
                        return True
            except (OSError, urllib.error.URLError):
                pass
        time.sleep(1)
    return False


def wait_ollama(timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    return False


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=10)


def extract_json_objects(text: str) -> list[dict]:
    objects: list[dict] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("{"):
            continue
        buf: list[str] = []
        depth = 0
        for line2 in lines[i:]:
            buf.append(line2)
            depth += line2.count("{") - line2.count("}")
            if depth <= 0:
                break
        try:
            obj = json.loads("\n".join(buf))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objects.append(obj)
    return objects


def parse_v3_benchmark_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for obj in reversed(extract_json_objects(text)):
        if "gen_tps" in obj:
            return {
                "status": "ok",
                "gen_tps": float(obj.get("gen_tps") or 0),
                "prompt_tps": float(obj.get("prompt_tps") or 0),
                "gen_tokens": int(obj.get("gen_tokens") or 0),
                "prompt_tokens": int(obj.get("prompt_tokens") or 0),
            }
    return {"status": "failed", "error": tail_error(text)}


def parse_tune_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Best result:\s*([0-9.]+)\s*tok/s", text)
    if not matches:
        matches = re.findall(r"Best config:\s*([0-9.]+)\s*tok/s", text)
    if matches:
        return {"status": "ok", "gen_tps": float(matches[-1])}
    return {"status": "failed", "error": tail_error(text)}


def tail_error(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "no output"
    interesting = [line for line in lines if "error" in line.lower() or "failed" in line.lower() or "oom" in line.lower()]
    return (interesting[-1] if interesting else lines[-1])[:500]


def raw_completion(port: int, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "prompt": "Write a practical local LLM inference runbook for an engineer tuning llama.cpp serving.",
            "n_predict": max_tokens,
            "temperature": 0.2,
            "cache_prompt": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    timings = data.get("timings") or {}
    return {
        "status": "ok",
        "gen_tps": float(timings.get("predicted_per_second") or 0),
        "prompt_tps": float(timings.get("prompt_per_second") or 0),
        "gen_tokens": int(timings.get("predicted_n") or 0),
        "prompt_tokens": int(timings.get("prompt_n") or 0),
    }


def ollama_generate(model_name: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model_name,
            "prompt": "Write a practical local LLM inference runbook for an engineer tuning llama.cpp serving.",
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.2,
                "num_ctx": 32768,
            },
        }
    ).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    prompt_eval_count = int(data.get("prompt_eval_count") or 0)
    prompt_eval_duration = int(data.get("prompt_eval_duration") or 0)
    eval_count = int(data.get("eval_count") or 0)
    eval_duration = int(data.get("eval_duration") or 0)
    prompt_tps = prompt_eval_count / (prompt_eval_duration / 1e9) if prompt_eval_duration > 0 else 0
    gen_tps = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0
    return {
        "status": "ok",
        "gen_tps": gen_tps,
        "prompt_tps": prompt_tps,
        "gen_tokens": eval_count,
        "prompt_tokens": prompt_eval_count,
    }


def ollama_known_model_name(model: ModelSpec) -> str | None:
    # Claude's completed Ollama rows used these pre-created names, then only
    # measured /api/generate. Reusing them avoids a fresh blob-store import.
    if model.label == "4B":
        return "qwen35-4b-local"
    if model.label == "27B":
        return "qwen36-27b-local"
    return None


def ollama_model_exists(model_name: str) -> bool:
    try:
        body = json.dumps({"name": model_name}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/show",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def append_result(path: Path, result: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, sort_keys=True) + "\n")
        f.flush()


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def write_summary(out_dir: Path, results_path: Path) -> None:
    rows = load_results(results_path)
    by_key: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_key.setdefault((row["model"], row["column"]), []).append(row)

    summary = out_dir / "summary.md"
    with summary.open("w", encoding="utf-8") as f:
        f.write("# llm-server benchmark matrix\n\n")
        f.write(f"Updated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n")
        f.write("| Model | Column | Runs | Median decode tok/s | Last status | Last log |\n")
        f.write("|---|---|---:|---:|---|---|\n")
        for (model, column), group in sorted(by_key.items()):
            ok_tps = [float(r["gen_tps"]) for r in group if r.get("status") == "ok" and r.get("gen_tps")]
            med = f"{median(ok_tps):.2f}" if ok_tps else "failed"
            last = group[-1]
            log = Path(last.get("log", "")).name
            f.write(f"| {model} | {column} | {len(group)} | {med} | {last.get('status', '?')} | {log} |\n")


def run_default(args: argparse.Namespace, model: ModelSpec, round_idx: int, port: int, out_dir: Path) -> dict:
    log_path = out_dir / f"{model.label}-v3-default-r{round_idx}.log"
    server_bin = server_bin_for_model(args, model)
    backend = backend_for_model(args, model)
    cmd = [
        str(args.llm_bin),
        str(model.path),
        "--ctx-size",
        str(args.ctx_size),
        "--benchmark",
        "--port",
        str(port),
        "--backend",
        backend,
        "--server-bin",
        str(server_bin),
    ]
    env = bench_env(args, server_bin)
    with log_path.open("w", encoding="utf-8") as log:
        started = time.time()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=args.command_timeout)
    parsed = parse_v3_benchmark_log(log_path)
    parsed.update(common_result(model, "v3-default", round_idx, log_path, started, proc.returncode))
    parsed["backend"] = backend
    parsed["server_bin"] = str(server_bin)
    return parsed


def run_tune(args: argparse.Namespace, model: ModelSpec, round_idx: int, port: int, out_dir: Path) -> dict:
    log_path = out_dir / f"{model.label}-v3-ai-tune-r{round_idx}.log"
    server_bin = server_bin_for_model(args, model)
    backend = backend_for_model(args, model)
    cmd = [
        str(args.llm_bin),
        str(model.path),
        "--ctx-size",
        str(args.ctx_size),
        "--ai-tune",
        "--rounds",
        str(args.ai_tune_rounds),
        "--retune",
        "--port",
        str(port),
        "--backend",
        backend,
        "--server-bin",
        str(server_bin),
    ]
    env = bench_env(args, server_bin)
    with log_path.open("w", encoding="utf-8") as log:
        started = time.time()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=args.command_timeout)
    parsed = parse_tune_log(log_path)
    parsed.update(common_result(model, "v3-ai-tune", round_idx, log_path, started, proc.returncode))
    parsed["backend"] = backend
    parsed["server_bin"] = str(server_bin)
    return parsed


def run_raw_fit(args: argparse.Namespace, model: ModelSpec, round_idx: int, port: int, out_dir: Path) -> dict:
    log_path = out_dir / f"{model.label}-raw-fit-r{round_idx}.log"
    server_bin = server_bin_for_model(args, model)
    cmd = [
        str(server_bin),
        "-m",
        str(model.path),
        "-c",
        str(args.ctx_size),
        "-fa",
        "on",
        "--fit",
        "on",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    env = bench_env(args, server_bin)
    started = time.time()
    proc: subprocess.Popen | None = None
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            if not wait_health(port, proc, args.load_timeout):
                return {
                    **common_result(model, "raw-fit", round_idx, log_path, started, proc.poll()),
                    "status": "failed",
                    "error": tail_error(log_path.read_text(encoding="utf-8", errors="replace")),
                }
            parsed = raw_completion(port, args.max_tokens)
            parsed.update(common_result(model, "raw-fit", round_idx, log_path, started, proc.poll()))
            return parsed
        except Exception as exc:
            return {
                **common_result(model, "raw-fit", round_idx, log_path, started, proc.poll() if proc else None),
                "status": "failed",
                "error": str(exc)[:500],
            }
        finally:
            stop_process(proc)


def run_ollama(args: argparse.Namespace, model: ModelSpec, round_idx: int, port: int, out_dir: Path) -> dict:
    del port
    log_path = out_dir / f"{model.label}-ollama-r{round_idx}.log"
    model_name = ollama_known_model_name(model) or f"llm-bench-{model.label.lower().replace('_', '-').replace('.', '-')}"
    modelfile = out_dir / f"Modelfile.{model.label}"
    modelfile.write_text(
        f"FROM {model.path}\n"
        f"PARAMETER num_ctx {args.ctx_size}\n"
        f"PARAMETER temperature 0.2\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    if args.ollama_models:
        env["OLLAMA_MODELS"] = str(args.ollama_models)
    serve_proc: subprocess.Popen | None = None
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        try:
            if not wait_ollama(2):
                serve_proc = subprocess.Popen(
                    [str(args.ollama_bin), "serve"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                if not wait_ollama(args.load_timeout):
                    return {
                        **common_result(model, "ollama", round_idx, log_path, started, serve_proc.poll()),
                        "status": "failed",
                        "error": "ollama daemon did not become ready",
                    }
            if not ollama_model_exists(model_name):
                create = subprocess.run(
                    [str(args.ollama_bin), "create", model_name, "-f", str(modelfile)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=args.command_timeout,
                )
                if create.returncode != 0:
                    return {
                        **common_result(model, "ollama", round_idx, log_path, started, create.returncode),
                        "status": "failed",
                        "error": tail_error(log_path.read_text(encoding="utf-8", errors="replace")),
                    }
            parsed = ollama_generate(model_name, args.max_tokens)
            parsed.update(common_result(model, "ollama", round_idx, log_path, started, 0))
            parsed["ollama_model"] = model_name
            return parsed
        except Exception as exc:
            return {
                **common_result(model, "ollama", round_idx, log_path, started, None),
                "status": "failed",
                "error": str(exc)[:500],
            }
        finally:
            subprocess.run([str(args.ollama_bin), "stop", model_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stop_process(serve_proc)


def common_result(model: ModelSpec, column: str, round_idx: int, log_path: Path, started: float, returncode: int | None) -> dict:
    return {
        "model": model.label,
        "model_path": str(model.path),
        "column": column,
        "round": round_idx,
        "log": str(log_path),
        "elapsed_s": round(time.time() - started, 3),
        "returncode": returncode,
        "timestamp": int(time.time()),
    }


def bench_env(args: argparse.Namespace, server_bin: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("LLM_APP_HOME", str(HOME / "llm-server"))
    env.setdefault("LLM_SERVER_NO_UPDATE_CHECK", "1")
    env["LLAMA_SERVER"] = str(server_bin)
    return env


def run_matrix(args: argparse.Namespace, models: list[ModelSpec], out_dir: Path) -> None:
    results_path = out_dir / "results.jsonl"
    columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    runners = {
        "v3-default": run_default,
        "raw-fit": run_raw_fit,
        "v3-ai-tune": run_tune,
        "ollama": run_ollama,
    }
    unknown = [c for c in columns if c not in runners]
    if unknown:
        raise SystemExit(f"unknown columns: {', '.join(unknown)}")

    port = args.port_base
    for model in models:
        for column in columns:
            for round_idx in range(1, args.rounds + 1):
                if not args.no_clear_gpu:
                    kill_matching_processes(["llama-server-cuda", "ik_llama-server", "llama-server --", "llama-server -m", "ollama_llama_server"])
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {model.label} {column} round {round_idx}", flush=True)
                try:
                    row = runners[column](args, model, round_idx, port, out_dir)
                except subprocess.TimeoutExpired as exc:
                    log_path = out_dir / f"{model.label}-{column}-r{round_idx}.log"
                    row = {
                        **common_result(model, column, round_idx, log_path, time.time() - args.command_timeout, None),
                        "status": "failed",
                        "error": f"timeout after {exc.timeout}s",
                    }
                append_result(results_path, row)
                write_summary(out_dir, results_path)
                port += 1
                time.sleep(args.cooldown)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run llm-server v3 benchmark matrix")
    parser.add_argument("--model", action="append", default=[], help="LABEL=/path/model.gguf; repeat to override defaults")
    parser.add_argument("--llm-bin", type=expand, default=HOME / "llm-server/.bin/llm-server")
    parser.add_argument("--raw-bin", type=expand, default=HOME / "llm-server/.bin/llama-server-cuda")
    parser.add_argument("--ik-raw-bin", type=expand, default=HOME / "llm-server/.bin/ik_llama-server-cuda")
    parser.add_argument("--ollama-bin", type=expand, default=HOME / ".local/bin/ollama")
    parser.add_argument("--ollama-models", type=expand, default=None, help="Optional OLLAMA_MODELS directory, e.g. /dev/shm/ollama-models")
    parser.add_argument("--out-dir", type=expand, default=HOME / f"llm-server/.benchmarks/matrix-{utc_stamp()}")
    parser.add_argument("--ctx-size", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--columns", default="v3-default,raw-fit,ollama,v3-ai-tune")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--ai-tune-rounds", type=int, default=8)
    parser.add_argument("--port-base", type=int, default=18100)
    parser.add_argument("--load-timeout", type=int, default=1200)
    parser.add_argument("--command-timeout", type=int, default=7200)
    parser.add_argument("--cooldown", type=float, default=5)
    parser.add_argument("--no-clear-gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = [parse_model_arg(raw) for raw in args.model] if args.model else existing_default_models()
    if not models:
        print("no benchmark models found; pass --model LABEL=/path/model.gguf", file=sys.stderr)
        return 2
    missing = [m for m in models if not m.path.exists()]
    if missing:
        for m in missing:
            print(f"missing model {m.label}: {m.path}", file=sys.stderr)
        return 2
    if not args.llm_bin.exists() or not os.access(args.llm_bin, os.X_OK):
        print(f"llm-server binary is not executable: {args.llm_bin}", file=sys.stderr)
        return 2
    if not args.raw_bin.exists() or not os.access(args.raw_bin, os.X_OK):
        print(f"raw llama-server binary is not executable: {args.raw_bin}", file=sys.stderr)
        return 2
    if not args.ik_raw_bin.exists() or not os.access(args.ik_raw_bin, os.X_OK):
        print(f"ik llama-server binary is not executable: {args.ik_raw_bin}", file=sys.stderr)
        return 2
    if not args.ollama_bin.exists() or not os.access(args.ollama_bin, os.X_OK):
        print(f"ollama binary is not executable: {args.ollama_bin}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ctx_size": args.ctx_size,
        "columns": args.columns,
        "rounds": args.rounds,
        "ai_tune_rounds": args.ai_tune_rounds,
        "llm_bin": str(args.llm_bin),
        "raw_bin": str(args.raw_bin),
        "ik_raw_bin": str(args.ik_raw_bin),
        "ollama_bin": str(args.ollama_bin),
        "ollama_models": str(args.ollama_models) if args.ollama_models else "",
        "models": [{"label": m.label, "path": str(m.path)} for m in models],
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    run_matrix(args, models, args.out_dir)
    print(args.out_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
