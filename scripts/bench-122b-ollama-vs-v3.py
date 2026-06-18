#!/usr/bin/env python3
"""Both-run 122B speed comparison: Ollama vs llm-server (v3/ik_llama).

Ollama can't pull sharded GGUFs (#5245), so we use the largest single-file quant
(UD-IQ3_S, ~47G) which Ollama pulls into ONE blob. We then point llm-server at
that same blob, so a single copy serves both tools (no double copy, no M3 delete).

Runs autonomously and prints one compact result line. Writes JSON for the table.
"""
import glob, json, os, re, subprocess, time, urllib.request

OLLAMA = "/home/mik/.local/bin/ollama"
OLLAMA_MODELS = "/home/mik/.ollama/models"
LLM = "/home/mik/llm-server/.bin/llm-server"
REPO = "hf.co/unsloth/Qwen3.5-122B-A10B-GGUF"
QUANT = "UD-IQ3_S"
MODEL = f"{REPO}:{QUANT}"
CTX = 32768
MAXTOK = 256
SYMLINK = "/home/mik/ai_models/Qwen3.5-122B-A10B-UD-IQ3_S.gguf"
OUT = "/home/mik/llm-server/.benchmarks/122b-iq3s-ollama-vs-v3.json"
os.environ["OLLAMA_MODELS"] = OLLAMA_MODELS


def log(m):
    print(f"[bench] {time.strftime('%H:%M:%S')} {m}", flush=True)


def ensure_daemon():
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=3)
        return
    except Exception:
        log("starting ollama daemon")
        subprocess.Popen([OLLAMA, "serve"], stdout=open("/tmp/ollama-serve.log", "ab"),
                         stderr=subprocess.STDOUT)
        for _ in range(30):
            time.sleep(2)
            try:
                urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=3)
                return
            except Exception:
                pass
        raise SystemExit("ollama daemon never came up")


def pull():
    log(f"pulling {MODEL} (single-file ~47G; slow at ollama pull speed)")
    # ollama pull is resumable; retry through transient 504/registry timeouts.
    for attempt in range(1, 11):
        if subprocess.run([OLLAMA, "pull", MODEL]).returncode == 0:
            return
        log(f"pull attempt {attempt} failed (likely transient); retry in 30s")
        time.sleep(30)
    raise SystemExit("ollama pull failed after 10 attempts")


def bench_ollama():
    log("benchmarking Ollama (256-tok gen @ 32k ctx)")
    body = json.dumps({
        "model": MODEL,
        "prompt": "Write a long, detailed essay about the history of computing.",
        "stream": False,
        "options": {"num_ctx": CTX, "num_predict": MAXTOK},
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=2400) as resp:
        d = json.load(resp)
    ec, ed = d.get("eval_count", 0), d.get("eval_duration", 0)
    tps = ec / (ed / 1e9) if ed else 0.0
    log(f"Ollama: {tps:.2f} tok/s ({ec} tokens)")
    # free VRAM before llm-server loads the same model
    subprocess.run([OLLAMA, "stop", MODEL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)
    return round(tps, 2)


def model_blob():
    blobs = [b for b in glob.glob(f"{OLLAMA_MODELS}/blobs/sha256-*") if "partial" not in b]
    return max(blobs, key=os.path.getsize) if blobs else None


def bench_v3(blob):
    if os.path.islink(SYMLINK) or os.path.exists(SYMLINK):
        os.remove(SYMLINK)
    os.symlink(blob, SYMLINK)  # give the blob a .gguf name for the launcher
    log("benchmarking llm-server v3 (ik_llama) on the same blob")
    p = subprocess.run([LLM, SYMLINK, "--benchmark", "--ctx-size", str(CTX)],
                       capture_output=True, text=True, timeout=2400)
    m = re.search(r'"gen_tps":\s*([0-9.]+)', p.stdout + p.stderr)
    tps = float(m.group(1)) if m else 0.0
    log(f"llm-server v3: {tps:.2f} tok/s")
    return round(tps, 2)


def main():
    ensure_daemon()
    pull()
    o = bench_ollama()
    blob = model_blob()
    v = bench_v3(blob) if blob else 0.0
    adv = round((v / o - 1) * 100, 1) if o else None
    res = {"model": "Qwen3.5-122B-A10B", "quant": QUANT, "ctx": CTX,
           "ollama_tps": o, "v3_tps": v, "v3_vs_ollama_pct": adv}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    log("RESULT " + json.dumps(res))
    print(f"\n=== 122B {QUANT} | Ollama {o} | llm-server v3 {v} | +{adv}% v3 vs Ollama ===")


if __name__ == "__main__":
    main()
