#!/usr/bin/env python3
"""Both-run MoE speed: Ollama vs llm-server (v3/ik_llama), across 122B quants.

Ollama can't import sharded GGUFs (#5245) and its registry pull is flaky (504),
so for each SINGLE-FILE quant we:
  1. download the single .gguf with `hf` (fast, no ollama registry),
  2. `ollama create` from the LOCAL file (no registry handshake),
  3. bench Ollama, `ollama stop` to free VRAM,
  4. bench llm-server on the same file,
  5. delete the ollama copy + source before the next quant (disk stays bounded).

Runs autonomously; prints one line per quant and writes JSON for the table.
"""
import glob, json, os, re, subprocess, time, urllib.request

HF = "hf"
OLLAMA = "/home/mik/.local/bin/ollama"
OLLAMA_MODELS = "/home/mik/.ollama/models"
LLM = "/home/mik/llm-server/.bin/llm-server"
REPO = "unsloth/Qwen3.5-122B-A10B-GGUF"
QUANTS = ["UD-IQ2_M", "UD-IQ3_XXS"]  # single-file; 2x fits the free disk
DLDIR = "/home/mik/ai_models/_ollama_bench"
CTX = 32768
MAXTOK = 256
OUT = "/home/mik/llm-server/.benchmarks/122b-ollama-vs-v3-quants.json"
os.environ["OLLAMA_MODELS"] = OLLAMA_MODELS
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
try:
    os.environ.setdefault("HF_TOKEN", open(os.path.expanduser("~/.cache/huggingface/token")).read().strip())
except OSError:
    pass


def log(m):
    print(f"[bench] {time.strftime('%H:%M:%S')} {m}", flush=True)


def free_gb():
    s = os.statvfs("/")
    return s.f_bavail * s.f_frsize / 1e9


def ensure_daemon():
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=3)
        return
    except Exception:
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


def hf_download(quant):
    fn = f"Qwen3.5-122B-A10B-{quant}.gguf"
    os.makedirs(DLDIR, exist_ok=True)
    log(f"downloading {fn} via hf")
    if subprocess.run([HF, "download", REPO, fn, "--local-dir", DLDIR]).returncode != 0:
        raise SystemExit(f"hf download failed for {quant}")
    hits = glob.glob(os.path.join(DLDIR, "**", fn), recursive=True)
    if not hits:
        raise SystemExit(f"downloaded file not found for {quant}")
    return hits[0]


def ollama_create(name, gguf):
    mf = f"/tmp/Modelfile.{name}"
    with open(mf, "w") as f:
        f.write(f"FROM {gguf}\nPARAMETER num_ctx {CTX}\n")
    log(f"ollama create {name} from local file (no registry)")
    if subprocess.run([OLLAMA, "create", name, "-f", mf]).returncode != 0:
        raise SystemExit(f"ollama create failed for {name}")


def bench_ollama(name):
    body = json.dumps({
        "model": name,
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
    log(f"  Ollama {name}: {tps:.2f} tok/s ({ec} tok)")
    subprocess.run([OLLAMA, "stop", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(10)
    return round(tps, 2)


def bench_v3(gguf):
    p = subprocess.run([LLM, gguf, "--benchmark", "--ctx-size", str(CTX)],
                       capture_output=True, text=True, timeout=2400)
    m = re.search(r'"gen_tps":\s*([0-9.]+)', p.stdout + p.stderr)
    tps = float(m.group(1)) if m else 0.0
    log(f"  llm-server v3: {tps:.2f} tok/s")
    return round(tps, 2)


def cleanup(name, gguf):
    subprocess.run([OLLAMA, "rm", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        os.remove(gguf)
    except OSError:
        pass


def main():
    ensure_daemon()
    results = []
    for q in QUANTS:
        name = "q122-" + q.lower().replace("_", "-")
        log(f"=== {q}  (free {free_gb():.0f} G) ===")
        gguf = hf_download(q)
        sz = os.path.getsize(gguf) / 1e9
        if free_gb() < sz + 8:
            log(f"  SKIP {q}: not enough disk for the ollama copy ({free_gb():.0f}G free, need ~{sz+8:.0f}G)")
            os.remove(gguf)
            continue
        ollama_create(name, gguf)
        o = bench_ollama(name)
        v = bench_v3(gguf)
        adv = round((v / o - 1) * 100, 1) if o else None
        results.append({"quant": q, "size_gb": round(sz, 1), "ollama_tps": o,
                        "v3_tps": v, "v3_vs_ollama_pct": adv})
        cleanup(name, gguf)
        log(f"  -> {q}: Ollama {o} | v3 {v} | +{adv}%")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    log("RESULT " + json.dumps(results))
    print()
    for r in results:
        print(f"122B {r['quant']} ({r['size_gb']}G) | Ollama {r['ollama_tps']} | "
              f"llm-server v3 {r['v3_tps']} | +{r['v3_vs_ollama_pct']}% v3")


if __name__ == "__main__":
    main()
