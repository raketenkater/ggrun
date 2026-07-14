#!/usr/bin/env python3
"""Four-slot long-context smoke/load test for a running llama-server."""

import argparse
import concurrent.futures
import json
import time
import urllib.request


def load_prompt(tag: str, words: int) -> str:
    payload = " ".join(f"w{i % 997}" for i in range(words))
    return f"Load test {tag}. Read this data: {payload}\nReply with done."


def request_messages(
    base_url: str, tag: str, messages: list[dict], output_tokens: int, timeout: int
):
    body = json.dumps(
        {
            "model": "local",
            "max_tokens": output_tokens,
            "temperature": 0,
            "messages": messages,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read())
    elapsed = time.monotonic() - started
    usage = data.get("usage", {})
    message = (data.get("choices") or [{}])[0].get("message", {})
    text = (message.get("content") or "") + (message.get("reasoning_content") or "")
    return {
        "tag": tag,
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "nonempty": bool(text.strip()),
        "_response_text": text,
    }


def request(base_url: str, tag: str, words: int, output_tokens: int, timeout: int):
    return request_messages(
        base_url,
        tag,
        [{"role": "user", "content": load_prompt(tag, words)}],
        output_tokens,
        timeout,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--main-words", type=int, default=30000,
                        help="wNNN words; approximately two model tokens each")
    parser.add_argument("--worker-words", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--min-main-tokens", type=int, default=60000,
                        help="fail unless the main request really reaches this token count")
    parser.add_argument("--append-check", action="store_true",
                        help="after the load, append to the 60k conversation and verify it")
    parser.add_argument("--append-only", action="store_true",
                        help="probe append/LCP reuse after an earlier identical 60k run")
    args = parser.parse_args()

    if args.append_only:
        messages = [
            {"role": "user", "content": load_prompt("main-60k", args.main_words)},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "Append-only checkpoint probe. Reply with ok."},
        ]
        row = request_messages(
            args.url, "append-only", messages, args.output_tokens, args.timeout
        )
        row.pop("_response_text", None)
        print(json.dumps(row, indent=2))
        if not row["nonempty"] or row["prompt_tokens"] < args.min_main_tokens:
            raise SystemExit("append-only request did not preserve the 60k prefix")
        return

    jobs = [("main-60k", args.main_words)]
    jobs.extend((f"worker-{i}", args.worker_words + i * 64) for i in range(3))
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(request, args.url, tag, words, args.output_tokens, args.timeout)
            for tag, words in jobs
        ]
        rows = [future.result() for future in futures]
    wall = time.monotonic() - started
    append_row = None
    if args.append_check:
        messages = [
            {"role": "user", "content": load_prompt("main-60k", args.main_words)},
            {"role": "assistant", "content": rows[0]["_response_text"] or "done"},
            {"role": "user", "content": "Append-only checkpoint probe. Reply with ok."},
        ]
        append_row = request_messages(
            args.url, "append-check", messages, args.output_tokens, args.timeout
        )
    for row in rows:
        row.pop("_response_text", None)
    if append_row is not None:
        append_row.pop("_response_text", None)
    result = {"wall_s": round(wall, 3), "requests": rows}
    if append_row is not None:
        result["append_request"] = append_row
    print(json.dumps(result, indent=2))
    if not all(row["nonempty"] and row["completion_tokens"] > 0 for row in rows):
        raise SystemExit("one or more responses were empty or had no completion tokens")
    if rows[0]["prompt_tokens"] < args.min_main_tokens:
        raise SystemExit(
            f"main request had {rows[0]['prompt_tokens']} prompt tokens; "
            f"expected at least {args.min_main_tokens}"
        )
    if append_row is not None and (
        not append_row["nonempty"]
        or append_row["prompt_tokens"] < args.min_main_tokens
    ):
        raise SystemExit("append request did not preserve the 60k conversation")


if __name__ == "__main__":
    main()
