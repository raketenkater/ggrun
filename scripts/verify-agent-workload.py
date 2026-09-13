#!/usr/bin/env python3
"""Bounded tool-using repair tasks with a trusted oracle; never execute model code."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import operator
from pathlib import Path
import statistics
import time
import urllib.request

TASKS = [
    {"name": "ceiling", "request": "Repair solve(n, d) to return the ceiling of n/d for nonnegative integer n and positive integer d.",
     "source": "def solve(n, d):\n    return n // d\n", "params": ["n", "d"],
     "cases": [[[0, 3], 0], [[1, 3], 1], [[6, 3], 2], [[7, 3], 3], [[19, 4], 5]]},
    {"name": "clamp", "request": "Repair solve(x, low, high) to clamp x to the inclusive interval [low, high]. Assume low <= high.",
     "source": "def solve(x, low, high):\n    return min(x, low)\n", "params": ["x", "low", "high"],
     "cases": [[[-8, -3, 4], -3], [[0, -3, 4], 0], [[9, -3, 4], 4], [[4, 4, 4], 4]]},
    {"name": "interval", "request": "Repair solve(a, b, c, d) to return the overlap length of half-open intervals [a,b) and [c,d). Assume a <= b and c <= d.",
     "source": "def solve(a, b, c, d):\n    return min(b, d) - min(a, c)\n", "params": ["a", "b", "c", "d"],
     "cases": [[[0, 4, 2, 6], 2], [[0, 2, 4, 6], 0], [[-4, 2, -2, 0], 2], [[1, 1, 0, 2], 0]]},
]
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
       ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}
FUNCS = {"min": min, "max": max, "abs": abs}


def expression(node, variables):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
    elif isinstance(node, ast.Name) and node.id in variables:
        value = variables[node.id]
    elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
        value = OPS[type(node.op)](expression(node.left, variables), expression(node.right, variables))
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = expression(node.operand, variables) * (-1 if isinstance(node.op, ast.USub) else 1)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS and not node.keywords:
        value = FUNCS[node.func.id](*[expression(a, variables) for a in node.args])
    elif isinstance(node, ast.IfExp):
        value = expression(node.body if expression(node.test, variables) else node.orelse, variables)
    elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMP:
        value = CMP[type(node.ops[0])](expression(node.left, variables), expression(node.comparators[0], variables))
    else:
        raise ValueError("unsupported expression: use arithmetic, comparisons, min/max/abs or a conditional expression")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 10**12:
        raise ValueError("numeric result outside fixture bounds")
    return value


def oracle(source, task):
    try:
        if len(source) > 10000:
            raise ValueError("source too large")
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > 200 or len(tree.body) != 1:
            raise ValueError("expected one small function")
        fn = tree.body[0]
        if not isinstance(fn, ast.FunctionDef) or fn.name != "solve" or fn.decorator_list:
            raise ValueError("expected solve function without decorators")
        args = fn.args
        if [a.arg for a in args.args] != task["params"] or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults:
            raise ValueError("function parameters changed")
        if len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
            raise ValueError("use a single return expression")
        results = []
        for values, expected in task["cases"]:
            actual = expression(fn.body[0].value, dict(zip(task["params"], values)))
            results.append({"input": values, "expected": expected, "actual": actual, "passed": actual == expected})
        return {"passed": all(r["passed"] for r in results), "cases": results}
    except (ValueError, SyntaxError, TypeError, ArithmeticError, RecursionError) as exc:
        return {"passed": False, "error": str(exc)}


def tool(name, properties, required):
    return {"type": "function", "function": {"name": name,
            "description": {"read_file": "Read the fixture file.", "write_file": "Replace the fixture file.", "run_tests": "Run the immutable test oracle."}[name],
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}

TOOLS = [tool("read_file", {"path": {"type": "string"}}, ["path"]),
         tool("write_file", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
         tool("run_tests", {}, [])]


def http_json(url, payload=None, timeout=120):
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def run_task(args, task, index, output):
    directory = output / f"{index:03d}-{task['name']}"
    directory.mkdir()
    source = task["source"]
    (directory / "initial.py").write_text(source)
    messages = [{"role": "system", "content": "You are repairing a tiny Python fixture. Use read_file, write_file and run_tests. Only solution.py is writable. Keep the exact solve signature and use one return expression containing numeric arithmetic, comparisons, conditional expressions, min/max/abs. Do not import anything. Run tests after editing. Finish after they pass."},
                {"role": "user", "content": task["request"]}]
    start = time.monotonic()
    events, reads, writes, tests = [], 0, 0, 0
    tested_source = None
    result = {"task": task["name"], "index": index, "passed": False}
    try:
        for turn in range(args.max_turns):
            before = time.monotonic()
            reply = http_json(args.url.rstrip('/') + '/v1/chat/completions',
                {"model": args.model, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
                 "temperature": 0, "seed": args.seed + index, "max_tokens": args.max_tokens,
                 "chat_template_kwargs": {"enable_thinking": False}}, args.timeout)
            message = reply['choices'][0]['message']
            events.append({"turn": turn, "response_s": time.monotonic() - before,
                           "usage": reply.get("usage"), "timings": reply.get("timings"), "message": message})
            messages.append(message)
            calls = message.get('tool_calls') or []
            if not calls:
                break
            for call in calls:
                try:
                    name = call['function']['name']
                    data = json.loads(call['function']['arguments'])
                    if name == 'read_file' and data.get('path') == 'solution.py':
                        response = {"content": source}; reads += 1
                    elif name == 'write_file' and data.get('path') == 'solution.py' and isinstance(data.get('content'), str) and len(data['content']) <= 10000:
                        source = data['content']; writes += 1; response = {"written": True}
                    elif name == 'run_tests':
                        response = oracle(source, task); tests += 1; tested_source = source
                    else:
                        response = {"error": "Only read/write solution.py and run_tests are available"}
                except (KeyError, TypeError, ValueError) as exc:
                    response = {"error": str(exc)}
                messages.append({"role": "tool", "tool_call_id": call['id'], "content": json.dumps(response)})
            if reads and writes and tested_source == source and oracle(source, task)['passed']:
                result['passed'] = True
                break
    except Exception as exc:
        result['error'] = str(exc)
    result.update(elapsed_s=time.monotonic() - start, reads=reads, writes=writes, tests=tests,
                  oracle=oracle(source, task), turns=len(events))
    (directory / 'solution.py').write_text(source)
    (directory / 'events.json').write_text(json.dumps(events, indent=2))
    (directory / 'messages.json').write_text(json.dumps(messages, indent=2))
    (directory / 'result.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='local')
    parser.add_argument('--lanes', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--max-turns', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    if not 1 <= args.lanes <= 8 or min(args.repeats, args.max_turns, args.max_tokens, args.timeout) < 1 or args.warmups < 0:
        parser.error('positive bounded workload required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()): parser.error('use an empty evidence directory')
    contract = {"tasks": TASKS, "lanes": args.lanes, "repeats": args.repeats,
                "warmups": args.warmups, "max_turns": args.max_turns, "max_tokens": args.max_tokens, "seed": args.seed}
    manifest = {"arguments": vars(args), "suite_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "contract": contract,
                "scope": "bounded arithmetic repair with real tool calls; not full repository coding throughput"}
    try:
        props = http_json(args.url.rstrip('/') + '/props', timeout=5)
        manifest['server'] = {k: props[k] for k in ('model_path', 'total_slots', 'default_generation_settings') if k in props}
    except Exception as exc:
        manifest['server_probe_error'] = str(exc)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    warmup = output / 'warmup'; warmup.mkdir()
    for index in range(args.warmups): run_task(args, TASKS[index % len(TASKS)], index, warmup)
    started = time.monotonic()
    jobs = [(task, index) for index, task in enumerate(TASKS * args.repeats)]
    with ThreadPoolExecutor(max_workers=args.lanes) as pool:
        results = list(pool.map(lambda job: run_task(args, job[0], job[1], output), jobs))
    elapsed = time.monotonic() - started
    passed = sum(r['passed'] for r in results)
    summary = {"passed": passed == len(results), "completed_tasks": passed, "total_tasks": len(results),
               "elapsed_s": elapsed, "correct_tasks_per_minute": passed * 60 / elapsed,
               "task_latency_median_s": statistics.median(r['elapsed_s'] for r in results),
               "task_latency_max_s": max(r['elapsed_s'] for r in results),
               "suite_sha256": manifest['suite_sha256'], "results": results}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}))
    raise SystemExit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
