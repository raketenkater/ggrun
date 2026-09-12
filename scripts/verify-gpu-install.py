#!/usr/bin/env python3
"""Run the shared GPU serving check with an existing model or an isolated download."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def select_model(directory):
    candidates = []
    for path in Path(directory).rglob("*.gguf"):
        if "mmproj" in path.name.lower():
            continue
        split = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path.name)
        if split:
            if int(split[1]) != 1:
                continue
            for shard in range(1, int(split[2]) + 1):
                name = path.name[:split.start()] + f"-{shard:05d}-of-{int(split[2]):05d}.gguf"
                if not path.with_name(name).is_file():
                    raise ValueError(f"missing model shard: {name}")
        candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"expected one model in isolated download, found {len(candidates)}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Existing local GGUF; does not test downloading")
    source.add_argument("--repo", help="Hugging Face model repository")
    parser.add_argument("--quant", default="Q4_0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--min-weight-devices", type=int, default=1)
    parser.add_argument("--port", type=int, default=18845)
    args = parser.parse_args()
    if args.ctx < 0 or min(args.timeout, args.request_timeout) <= 0 or args.min_weight_devices < 0:
        parser.error("timeouts must be positive; context/device count must be nonnegative (context 0 means auto)")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("use an empty output directory to preserve evidence and isolate downloads")
    launcher = str(Path(args.launcher).resolve())
    source_record = vars(args).copy()
    (output / "input.json").write_text(json.dumps(source_record, indent=2))
    if args.repo:
        models = output / "downloaded-models"
        models.mkdir(exist_ok=True)
        with (output / "download.log").open("w") as log:
            subprocess.run([launcher, "download", args.repo, "--quant", args.quant],
                           env=dict(os.environ, LLM_MODEL_DIR=str(models)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        model = select_model(models)
    else:
        model = Path(args.model).resolve(strict=True)
    (output / "selected-model.json").write_text(json.dumps({"model": str(model)}, indent=2))
    subprocess.run([sys.executable, str(Path(__file__).with_name("verify-installed-serving.py")),
                    "--launcher", launcher, "--model", str(model), "--output", str(output),
                    "--ctx", str(args.ctx), "--timeout", str(args.timeout),
                    "--request-timeout", str(args.request_timeout),
                    "--min-weight-devices", str(args.min_weight_devices), "--port", str(args.port)], check=True)


if __name__ == "__main__":
    main()
