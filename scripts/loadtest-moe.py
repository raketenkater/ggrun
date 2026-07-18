#!/usr/bin/env python3
"""Four-slot long-context smoke/load test for a running llama-server."""

import argparse
import concurrent.futures
import json
import time

from perf_artifact import (
    ArtifactWriter,
    append_reuse_summary,
    llama_accounting,
    marker_matches,
    request_chat,
    response_content,
    response_text,
    write_json,
)


def load_prompt(tag: str, words: int, marker: str = "done") -> str:
    payload = " ".join(f"w{i % 997}" for i in range(words))
    return (
        f"Load test {tag}. Read this data: {payload}\n"
        f"Reply with exactly this marker and nothing else: {marker}"
    )


def request_messages(
    base_url: str,
    tag: str,
    messages: list[dict],
    output_tokens: int,
    timeout: int,
    marker: str = "done",
    *,
    cache_prompt: bool = True,
    slot: int | None = None,
    stream: bool = False,
):
    body = {
        "model": "local",
        "max_tokens": output_tokens,
        "temperature": 0,
        "seed": 42,
        "messages": messages,
        "cache_prompt": cache_prompt,
    }
    if slot is not None:
        body["id_slot"] = slot
    data, elapsed, stream_info = request_chat(base_url, body, timeout, stream=stream)
    row = llama_accounting(data, elapsed, stream_info)
    content = response_content(data)
    row.update({
        "tag": tag,
        "marker": marker,
        "marker_response": content.strip(),
        "marker_pass": marker_matches(data, marker),
        "nonempty": bool(response_text(data).strip()),
        "slot": slot,
        "_response_content": content,
    })
    return row


def request(
    base_url: str,
    tag: str,
    words: int,
    output_tokens: int,
    timeout: int,
    marker: str = "done",
    *,
    cache_prompt: bool = True,
    slot: int | None = None,
    stream: bool = False,
):
    return request_messages(
        base_url,
        tag,
        [{"role": "user", "content": load_prompt(tag, words, marker)}],
        output_tokens,
        timeout,
        marker,
        cache_prompt=cache_prompt,
        slot=slot,
        stream=stream,
    )


def _worker_slots(value: str) -> list[int | None]:
    if not value.strip():
        return [None, None, None]
    slots = [int(raw.strip()) for raw in value.split(",") if raw.strip()]
    if len(slots) > 3:
        raise ValueError("--worker-slots accepts at most three comma-separated slots")
    return slots + [None] * (3 - len(slots))


def _public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _assert_cache_reuse(summary: dict, minimum_percent: float) -> str | None:
    if not summary["evidence_available"]:
        return "append request did not return usable raw cache/timing evidence"
    hit_percent = summary["append_cache_hit_percent"]
    if hit_percent is None or hit_percent < minimum_percent:
        return (
            "append request cache hit was "
            f"{hit_percent!r}%; expected at least {minimum_percent}%"
        )
    if summary["append_prefill_smaller_than_base"] is not True:
        return "append request did not prefill fewer tokens than its reusable prefix"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--main-words", type=int, default=30000,
                        help="wNNN words; approximately two model tokens each")
    parser.add_argument("--worker-words", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--marker", default="done",
                        help="Exact marker required from normal load requests.")
    parser.add_argument("--append-marker", default="ok",
                        help="Exact marker required from the append probe.")
    parser.add_argument("--stream", action="store_true",
                        help="Use streaming responses and record TTFT where the server supports it.")
    parser.add_argument("--no-strict-marker", dest="strict_marker", action="store_false",
                        help="Record marker failures without returning non-zero for them.")
    parser.set_defaults(strict_marker=True)
    parser.add_argument("--no-cache-prompt", action="store_true",
                        help="Disable request prompt caching (useful for cold-load control runs).")
    parser.add_argument("--main-slot", type=int,
                        help="Pin main and append requests to this llama-server slot.")
    parser.add_argument("--worker-slots", default="",
                        help="Optional comma-separated slots for worker-0,worker-1,worker-2.")
    parser.add_argument("--min-cache-hit-percent", type=float, default=50.0,
                        help="Required cache-hit percentage for --append-check/--append-only.")
    parser.add_argument("--artifact-dir",
                        help="Lab root; writes append-only artifacts below <artifact-dir>/<run-id>/.")
    parser.add_argument("--run-id", help="Stable identifier shared by related lab commands.")
    parser.add_argument("--out", help="Write the complete result JSON to this path.")
    parser.add_argument("--min-main-tokens", type=int, default=60000,
                        help="fail unless the main request really reaches this token count")
    parser.add_argument("--append-check", action="store_true",
                        help="after the load, append to the 60k conversation and verify it")
    parser.add_argument("--append-only", action="store_true",
                        help=(
                            "diagnostic append probe after an earlier run; without paired base accounting "
                            "it is explicitly non-promotable cache evidence"
                        ))
    parser.add_argument(
        "--base-assistant-content",
        help=(
            "actual visible assistant content from the paired base request; required for "
            "--append-only so the probe never invents a prior assistant turn"
        ),
    )
    args = parser.parse_args()
    if args.min_cache_hit_percent < 0 or args.min_cache_hit_percent > 100:
        raise SystemExit("--min-cache-hit-percent must be between 0 and 100")
    try:
        worker_slots = _worker_slots(args.worker_slots)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    cache_prompt = not args.no_cache_prompt
    # A reuse probe is only valid when it deterministically uses the same slot.
    main_slot = args.main_slot
    if (args.append_check or args.append_only) and main_slot is None:
        main_slot = 0
    writer = ArtifactWriter(args.artifact_dir, args.run_id, "loadtest-moe", vars(args))

    if args.append_only:
        if not args.base_assistant_content:
            raise SystemExit(
                "--append-only requires --base-assistant-content from the paired base request"
            )
        messages = [
            {"role": "user", "content": load_prompt("main-60k", args.main_words, args.marker)},
            {"role": "assistant", "content": args.base_assistant_content},
            {
                "role": "user",
                "content": (
                    "Append-only checkpoint probe. Reply with exactly this marker "
                    f"and nothing else: {args.append_marker}"
                ),
            },
        ]
        row = request_messages(
            args.url,
            "append-only",
            messages,
            args.output_tokens,
            args.timeout,
            args.append_marker,
            cache_prompt=cache_prompt,
            slot=main_slot,
            stream=args.stream,
        )
        reuse = append_reuse_summary(None, row)
        result = writer.enrich({
            "mode": "append-only",
            "main_slot": main_slot,
            "requires_prior_base_request": True,
            "validation": {
                "status": "diagnostic-nonpromotable",
                "validated_cache_reuse": False,
                "reason": (
                    "The earlier base request is not present in this invocation, so its exact "
                    "token count, slot residency, and prefix relationship cannot be proven."
                ),
            },
            "append_request": _public_row(row),
            "append_reuse": reuse,
        })
        writer.append(result)
        writer.write_summary(result)
        if args.out:
            write_json(args.out, result)
        print(json.dumps(result, indent=2))
        if not row["nonempty"] or row["prompt_tokens"] < args.min_main_tokens:
            raise SystemExit("append-only request did not preserve the 60k prefix")
        if args.strict_marker and not row["marker_pass"]:
            raise SystemExit("append-only request did not return the exact required marker")
        return

    jobs = [("main-60k", args.main_words, main_slot)]
    jobs.extend(
        (f"worker-{i}", args.worker_words + i * 64, worker_slots[i]) for i in range(3)
    )
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                request,
                args.url,
                tag,
                words,
                args.output_tokens,
                args.timeout,
                args.marker,
                cache_prompt=cache_prompt,
                slot=slot,
                stream=args.stream,
            )
            for tag, words, slot in jobs
        ]
        rows = [future.result() for future in futures]
    wall = time.monotonic() - started
    append_row = None
    if args.append_check:
        main_assistant_content = rows[0]["_response_content"]
        if not isinstance(main_assistant_content, str) or not main_assistant_content.strip():
            raise SystemExit(
                "append-check main request had no visible assistant content; refusing to invent one"
            )
        messages = [
            {"role": "user", "content": load_prompt("main-60k", args.main_words, args.marker)},
            {"role": "assistant", "content": main_assistant_content},
            {
                "role": "user",
                "content": (
                    "Append-only checkpoint probe. Reply with exactly this marker "
                    f"and nothing else: {args.append_marker}"
                ),
            },
        ]
        append_row = request_messages(
            args.url,
            "append-check",
            messages,
            args.output_tokens,
            args.timeout,
            args.append_marker,
            cache_prompt=cache_prompt,
            slot=main_slot,
            stream=args.stream,
        )
    public_rows = [_public_row(row) for row in rows]
    public_append_row = _public_row(append_row) if append_row is not None else None
    result = {
        "mode": "concurrent-load",
        "wall_s": round(wall, 3),
        "main_slot": main_slot,
        "cache_prompt": cache_prompt,
        "requests": public_rows,
    }
    if append_row is not None:
        result["append_request"] = public_append_row
        result["append_reuse"] = append_reuse_summary(rows[0], append_row)
    result = writer.enrich(result)
    writer.append(result)
    writer.write_summary(result)
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result, indent=2))
    if not all(row["nonempty"] and row["completion_tokens"] > 0 for row in rows):
        raise SystemExit("one or more responses were empty or had no completion tokens")
    if args.strict_marker and not all(row["marker_pass"] for row in rows):
        raise SystemExit("one or more responses did not return the exact required marker")
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
    if append_row is not None and args.strict_marker and not append_row["marker_pass"]:
        raise SystemExit("append request did not return the exact required marker")
    if append_row is not None:
        cache_error = _assert_cache_reuse(result["append_reuse"], args.min_cache_hit_percent)
        if cache_error:
            raise SystemExit(cache_error)


if __name__ == "__main__":
    main()
