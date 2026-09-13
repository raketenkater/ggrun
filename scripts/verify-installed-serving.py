#!/usr/bin/env python3
"""Exercise an installed launcher; retain logs and stop only our process group."""
import argparse
import errno
import json
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def request(port, route, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def read_stream(response):
    events = []
    generated = False
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            if not generated:
                raise RuntimeError("stream completed without generated text")
            return events
        event = json.loads(data)
        events.append(event)
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            text = delta.get("content") or delta.get("reasoning_content")
            generated = generated or (isinstance(text, str) and bool(text.strip()))
    raise RuntimeError("stream closed without [DONE]")


def streaming_request(port, timeout):
    payload = {"messages": [{"role": "user", "content": "Name one colour."}],
               "max_tokens": 32, "temperature": 0, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return read_stream(response)


def check_port_available(port):
    # Refuse a live listener without imposing our socket reuse policy on the
    # backend. ik uses SO_REUSEPORT on Linux; mainline uses SO_REUSEADDR.
    # A bind probe with the other policy rejects harmless TIME_WAIT sockets.
    # The launched process must still prove readiness, generation and cleanup.
    with socket.socket() as sock:
        sock.settimeout(0.5)
        error = sock.connect_ex(("127.0.0.1", port))
    if error == 0:
        raise OSError(errno.EADDRINUSE, f"port {port} has an active listener")
    if error not in {errno.ECONNREFUSED, 10061}:  # Winsock WSAECONNREFUSED
        raise OSError(error, f"could not establish that port {port} has no listener")


def weight_devices(log):
    # Ignore allocations from abandoned admissions before the final launch.
    launches = list(re.finditer(
        r"(?m)^\[launch\] .* -m |^.*(?:llm_)?load_tensors: loading model tensors", log))
    if launches:
        log = log[launches[-1].start():]
    devices = set()
    for line in log.splitlines():
        if re.search(r"(?:KV|compute|output) buffer", line, re.I):
            continue
        match = re.search(r"(CUDA\d+|Vulkan\d+|Metal\d*)[^\n]*?buffer size\s*=\s*([0-9.]+) MiB", line)
        if match and float(match[2]) > 0:
            devices.add(match[1])
    return sorted(devices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=18843)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--parallel", type=int, default=0, help="0 preserves automatic slot selection")
    parser.add_argument("--agent-lanes", type=int, default=0, help="Run bounded tool-using repair tasks before shutdown")
    parser.add_argument("--agent-repeats", type=int, default=3)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--min-weight-devices", type=int, default=0,
                        help="Require weight allocations on this many devices in the final launch")
    args = parser.parse_args()
    if args.ctx < 0 or min(args.timeout, args.request_timeout) <= 0 or args.min_weight_devices < 0:
        parser.error("timeouts must be positive; context/device minimum nonnegative (context 0 means auto)")
    if args.cpu and args.min_weight_devices:
        parser.error("--cpu cannot require GPU allocations")
    if args.parallel < 0 or not 0 <= args.agent_lanes <= 8 or args.agent_repeats < 1:
        parser.error("invalid parallel or agent workload bounds")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    command = [str(Path(args.launcher).resolve()), str(Path(args.model).resolve()),
               "--allow-live-memory-probe", "--host", "127.0.0.1", "--port", str(args.port)]
    if args.ctx:
        command += ["--ctx", str(args.ctx)]
    if args.cpu:
        command.append("--cpu")
    if args.parallel:
        command += ["--parallel", str(args.parallel)]
    windows = os.name == "nt"
    if windows and command[0].lower().endswith((".cmd", ".bat")):
        command = 'cmd.exe /d /s /c "' + subprocess.list2cmdline(command) + '"'
    result = {"command": command, "passed": False}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    env = dict(os.environ, LLM_COMMUNITY_TUNES="off")
    proc = None
    try:
        check_port_available(args.port)
        with (output / "serve.log").open("wb") as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, env=env,
                                    start_new_session=not windows,
                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if windows else 0)
            result["pid"] = proc.pid
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"launcher exited before ready: {proc.returncode}")
                try:
                    # Backend health can precede ggrun's admission/canary and
                    # shutdown handler. Wait for the installed launcher itself.
                    launcher_ready = b"[launch] Press Ctrl+C to stop" in (output / "serve.log").read_bytes()
                    if launcher_ready and request(args.port, "/health").get("status") == "ok":
                        result["launcher_ready"] = True
                        break
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError("timed out waiting for launcher readiness and health")
            result["weight_devices"] = weight_devices((output / "serve.log").read_text(errors="replace"))
            result["min_weight_devices"] = args.min_weight_devices
            if len(result["weight_devices"]) < args.min_weight_devices:
                raise RuntimeError(f"required {args.min_weight_devices} weight devices, observed {result['weight_devices']}")
            reply = request(args.port, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "Name one colour."}],
                "max_tokens": 32, "temperature": 0, "stream": False,
            }, timeout=args.request_timeout)
            (output / "reply.json").write_text(json.dumps(reply, indent=2))
            choices = reply.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            # Some reasoning models spend the entire short budget in reasoning.
            generated = message.get("content") or message.get("reasoning_content")
            if not isinstance(generated, str) or not generated.strip():
                raise RuntimeError("completion contained no generated text")
            if proc.poll() is not None:
                raise RuntimeError("launcher exited during generation")
            result["generation"] = True
            events = streaming_request(args.port, args.request_timeout)
            (output / "stream.json").write_text(json.dumps(events, indent=2))
            if proc.poll() is not None:
                raise RuntimeError("launcher exited during streaming")
            result["streaming"] = True
            if args.agent_lanes:
                subprocess.run([sys.executable, str(Path(__file__).with_name("verify-agent-workload.py")),
                                "--url", f"http://127.0.0.1:{args.port}", "--output", str(output / "agent-workload"),
                                "--lanes", str(args.agent_lanes), "--repeats", str(args.agent_repeats),
                                "--timeout", str(args.request_timeout)], check=True)
                result["agent_workload"] = True
    except BaseException as exc:
        result["error"] = str(exc)
        raise
    finally:
        if proc is not None:
            try:
                if windows:
                    if proc.poll() is None:
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    # Also catches a backend left behind by an exited launcher.
                    os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
                result["forced_cleanup"] = False
            except ProcessLookupError:
                result["forced_cleanup"] = False
            except (OSError, subprocess.TimeoutExpired):
                result["forced_cleanup"] = True
                if windows:
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                else:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                proc.wait(timeout=10)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                with socket.socket() as sock:
                    sock.settimeout(0.5)
                    if sock.connect_ex(("127.0.0.1", args.port)) != 0:
                        result["port_released"] = True
                        break
                time.sleep(0.2)
            result["passed"] = bool(result.get("generation") and result.get("streaming") and result.get("port_released")
                                    and not result.get("forced_cleanup") and not result.get("error"))
        (output / "result.json").write_text(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("installed serving lifecycle did not pass; see result.json")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
