#!/usr/bin/env python3
"""Exercise an installed launcher; retain logs and stop only our process group."""
import argparse
import errno
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
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


def prefix_reuse(port, timeout):
    """Ask two questions behind one long shared prefix and report what the
    second had to re-evaluate.

    A real agent replays a large, stable project prefix on every turn, so this
    is the difference between a responsive session and one that pays for the
    whole context each time. llama.cpp reports the prompt tokens it actually
    evaluated in `timings.prompt_n`; a reused prefix makes the second number a
    small fraction of the first. This returns the measurement rather than
    asserting a ratio: prompt caching can be legitimately off, and the point
    here is recorded evidence, not an invented threshold.
    """
    prefix = ("The following is a project file listing that does not change "
              "between questions.\n") + "\n".join(
        f"src/module_{i:03d}.go defines helper{i:03d} and its tests" for i in range(400))
    measured = []
    for question in ("Which file defines helper007?", "Which file defines helper011?"):
        reply = request(port, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": f"{prefix}\n\n{question}"}],
            "max_tokens": 16, "temperature": 0, "stream": False,
        }, timeout=timeout)
        timings = reply.get("timings") or {}
        measured.append({"prompt_n": timings.get("prompt_n"), "prompt_ms": timings.get("prompt_ms")})
    first, second = measured
    reuse = {"first": first, "second": second}
    if isinstance(first.get("prompt_n"), int) and isinstance(second.get("prompt_n"), int) and first["prompt_n"] > 0:
        reuse["reevaluated_fraction"] = round(second["prompt_n"] / first["prompt_n"], 4)
    return reuse


def cancel_and_reconnect(port, timeout):
    """Abandon a stream mid-generation, then prove the server still serves.

    Cancellation is the common case in agent use — the user interrupts, or the
    client drops — and a slot that is never released turns the next request into
    a hang. Closing the response without draining it is what an interrupted
    client actually does.
    """
    payload = {"messages": [{"role": "user", "content": "Count slowly from one to five hundred."}],
               "max_tokens": 512, "temperature": 0, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            if raw.decode("utf-8", "replace").strip().startswith("data:"):
                chunks += 1
                if chunks >= 3:
                    break  # leaving the context manager aborts the connection
    if chunks < 3:
        raise RuntimeError("stream ended before it could be cancelled")
    # The slot must come back without a restart, and promptly.
    started = time.monotonic()
    reply = request(port, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Name one colour."}],
        "max_tokens": 16, "temperature": 0, "stream": False,
    }, timeout=timeout)
    choices = reply.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    text = message.get("content") or message.get("reasoning_content")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("the request after a cancellation produced no text")
    return {"cancelled_after_chunks": chunks, "recovery_s": round(time.monotonic() - started, 3)}


def check_port_available(port):
    if os.name == "nt":
        # Preserve the native Windows probe: a short loopback connect can
        # time out as WSAEWOULDBLOCK even when no listener exists. Do not set
        # SO_REUSEADDR on Windows, where it can share an occupied port.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
        return
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


def gpu_memory():
    # Keyed by NVML index, the physical enumeration nvidia-smi reports. A
    # backend's CUDA<n> is that index only when CUDA_DEVICE_ORDER=PCI_BUS_ID and
    # CUDA_VISIBLE_DEVICES is unset; device_map() records what applies instead of
    # assuming. Absent tooling is unknown, not zero: report None rather than 0 MiB.
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if probe.returncode != 0:
        return None
    devices = {}
    for line in probe.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        try:
            util = int(fields[5]) if fields[5].isdigit() else None
            devices["nvml%d" % int(fields[0])] = {"uuid": fields[1], "pci_bus_id": fields[2],
                                                  "used_mib": int(fields[3]), "total_mib": int(fields[4]),
                                                  "utilization_pct": util}
        except ValueError:
            continue
    return devices or None


def device_map(devices):
    """How backend device names relate to the NVML devices measured above."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    order = os.environ.get("CUDA_DEVICE_ORDER")
    mapping = None
    if devices and visible is None and order == "PCI_BUS_ID":
        mapping = {name.replace("nvml", "CUDA"): name for name in devices}
    return {"cuda_visible_devices": visible, "cuda_device_order": order,
            "backend_to_nvml": mapping,
            "note": None if mapping else "backend CUDA<n> to NVML mapping not established; compare by UUID"}


def utilization(baseline, loaded):
    """What this launch put on the GPUs, separated from what was already there.

    A device without a baseline reading is unknown, not zero: subtracting a
    defaulted 0 would attribute another process's memory to this launch.
    """
    if not loaded:
        return None
    baseline = baseline or {}
    served, capacity, per_device, unknown = 0, 0, {}, []
    for name, current in sorted(loaded.items()):
        before = baseline.get(name, {}).get("used_mib")
        launch = current["used_mib"] - before if before is not None else None
        per_device[name] = {"uuid": current.get("uuid"), "before_mib": before, "after_mib": current["used_mib"],
                            "launch_mib": launch, "total_mib": current["total_mib"],
                            "utilization_pct": current.get("utilization_pct")}
        if launch is None:
            unknown.append(name)
        else:
            served += launch
        capacity += current["total_mib"]
    return {"devices": per_device, "launch_mib": None if unknown else served, "capacity_mib": capacity,
            "unknown_baseline": unknown,
            "fraction_of_vram": round(served / capacity, 4) if capacity and not unknown else None}


class GPUSampler:
    """Samples every NVML device once per interval for the whole run.

    Raw samples go to telemetry.jsonl; summary() reports per-device peak
    memory and mean/peak utilization. No nvidia-smi means no samples, and the
    summary says so instead of reporting zeros.
    """

    def __init__(self, path, interval=1.0):
        self.path, self.interval = path, interval
        self.samples = 0
        self.stats = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        with open(self.path, "w") as fh:
            while not self._stop.is_set():
                snap = gpu_memory()
                if snap:
                    t = time.time()
                    fh.write(json.dumps({"t": round(t, 2), "devices": snap}) + "\n")
                    fh.flush()
                    self.samples += 1
                    for name, d in snap.items():
                        st = self.stats.setdefault(name, {"uuid": d.get("uuid"), "peak_used_mib": 0,
                                                          "util_sum": 0, "util_n": 0, "peak_util_pct": None})
                        st["peak_used_mib"] = max(st["peak_used_mib"], d["used_mib"])
                        if d.get("utilization_pct") is not None:
                            st["util_sum"] += d["utilization_pct"]
                            st["util_n"] += 1
                            st["peak_util_pct"] = max(st["peak_util_pct"] or 0, d["utilization_pct"])
                self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self):
        if not self.samples:
            return {"samples": 0, "devices": None, "note": "no GPU telemetry available"}
        out = {}
        for name, st in self.stats.items():
            out[name] = {"uuid": st["uuid"], "peak_used_mib": st["peak_used_mib"],
                         "mean_util_pct": round(st["util_sum"] / st["util_n"], 1) if st["util_n"] else None,
                         "peak_util_pct": st["peak_util_pct"]}
        return {"samples": self.samples, "interval_s": self.interval, "devices": out}


def device_allocations(log_text):
    """Model/KV/compute MiB per device for the final launch, from the backend log."""
    launches = list(re.finditer(r"(?m)^\[launch\] .* -m |^.*(?:llm_)?load_tensors: loading model tensors", log_text))
    if launches:
        log_text = log_text[launches[-1].start():]
    alloc = {}
    for line in log_text.splitlines():
        m = re.search(r"(CUDA\d+|Vulkan\d+|Metal\d*|CUDA_Host|CPU)[^\n]*?(model|KV|compute|RS|output) buffer size\s*=\s*([0-9.]+) MiB", line)
        if not m:
            m2 = re.search(r"(CUDA\d+|Vulkan\d+|CUDA_Host|CPU) buffer size\s*=\s*([0-9.]+) MiB", line)
            if not m2:
                continue
            dev, kind, mib = m2[1], "model", float(m2[2])
        else:
            dev, kind, mib = m[1], m[2].lower(), float(m[3])
        d = alloc.setdefault(dev, {})
        d[kind] = round(d.get(kind, 0) + mib, 2)
    return alloc or None


def released_after_stop(baseline, after, tolerance_mib=256):
    """Per device: did memory return to the pre-launch level after shutdown?"""
    if not baseline or not after:
        return None
    out = {}
    for name, current in after.items():
        before = baseline.get(name, {}).get("used_mib")
        out[name] = None if before is None else current["used_mib"] - before <= tolerance_mib
    return out


def file_identity(path):
    """Size and sha256 of a file; None for anything unreadable."""
    try:
        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                digest.update(block)
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    except OSError:
        return None


def launcher_identity(launcher):
    """The launcher's own build identity plus the hash of the executable run."""
    exe = Path(launcher).resolve()
    if exe.suffix.lower() in (".cmd", ".bat"):
        candidate = exe.parent / ".bin" / "ggrun.exe"
        if candidate.exists():
            exe = candidate
    ident = {"file": file_identity(exe)}
    try:
        out = subprocess.run([str(exe), "version", "--json"], capture_output=True, text=True, timeout=30)
        ident["build"] = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 and out.stdout.strip() else None
        if ident["build"] is None:
            plain = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=30)
            ident["version_text"] = plain.stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        ident["build"] = None
    return ident


def backend_identity(log_text):
    """The backend binary and effective argv ggrun actually launched last."""
    launches = re.findall(r"(?m)^\[launch\] (\S+llama-server\S*) (-m .*)$", log_text)
    if not launches:
        return None
    binary, argv = launches[-1]
    ident = {"binary": file_identity(binary), "argv": argv}
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30,
                             stdin=subprocess.DEVNULL)
        ident["version_text"] = (out.stdout + out.stderr).strip()[-400:]
    except (OSError, subprocess.SubprocessError):
        ident["version_text"] = None
    return ident


def revision_matches(ident, expected):
    """The installed launcher was built from the expected commit."""
    build = ident.get("build") or {}
    revision = build.get("revision") or ""
    stamped = build.get("version") or ident.get("version_text") or ""
    return bool(expected) and (revision.startswith(expected) or expected.startswith(revision or "\0")
                               or re.search(r"-g" + re.escape(expected[:7]), stamped) is not None)


def claude_free_path(path):
    """PATH without directories that provide a claude executable."""
    names = ("claude", "claude.exe", "claude.cmd")
    return os.pathsep.join(d for d in path.split(os.pathsep)
                           if d and not any(Path(d, n).exists() for n in names))


def check_ubatch_raise(log_text):
    """A staged-prefill ubatch raise must be followed by exact admission of
    the raised argv. Any model: the raise is optional, its admission is not."""
    raised = "[launch] staged expert prefill: raising ubatch" in log_text
    if raised and "[launch] raised microbatch passed exact preflight" not in log_text:
        raise RuntimeError("ubatch raise was not admitted by exact preflight")
    return {"raised": raised}


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
    parser.add_argument("--prefix-reuse", action="store_true",
                        help="Measure prompt re-evaluation behind a long shared prefix "
                             "(needs a context large enough to hold it)")
    parser.add_argument("--claude-code", action="store_true",
                        help="Launch in Claude Code mode (reviewer + agent policy); the claude client is kept off PATH so ggrun serves")
    parser.add_argument("--min-weight-devices", type=int, default=0,
                        help="Require weight allocations on this many devices in the final launch")
    parser.add_argument("--expect-revision", default="",
                        help="Fail unless the installed launcher was built from this commit")
    parser.add_argument("--hash-model", action="store_true",
                        help="Record the model file's sha256 (reads the whole file)")
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
    if args.claude_code:
        command.append("--claude-code")
    windows = os.name == "nt"
    if windows and command[0].lower().endswith((".cmd", ".bat")):
        command = 'cmd.exe /d /s /c "' + subprocess.list2cmdline(command) + '"'
    result = {"command": command, "passed": False}
    t_start = time.monotonic()
    phases = {}

    def mark(name):
        phases[name] = round(time.monotonic() - t_start, 2)
        result["phases_s"] = phases
    result["identity"] = {"launcher": launcher_identity(args.launcher)}
    model_path = Path(args.model).resolve()
    result["identity"]["model"] = (file_identity(model_path) if args.hash_model
                                   else {"path": str(model_path), "bytes": model_path.stat().st_size, "sha256": None})
    if args.expect_revision and not revision_matches(result["identity"]["launcher"], args.expect_revision):
        result["error"] = f"installed launcher is not revision {args.expect_revision}: {result['identity']['launcher']}"
        (output / "result.json").write_text(json.dumps(result, indent=2))
        raise RuntimeError(result["error"])
    (output / "result.json").write_text(json.dumps(result, indent=2))
    env = dict(os.environ, LLM_COMMUNITY_TUNES="off")
    if args.claude_code:
        # With a claude client on PATH ggrun hands it the terminal; this check
        # drives the served endpoint itself, as the acceptance harness does.
        env["PATH"] = claude_free_path(env.get("PATH", ""))
    proc = None
    baseline_memory = None
    try:
        check_port_available(args.port)
        baseline_memory = gpu_memory()
        result["device_map"] = device_map(baseline_memory)
        sampler = GPUSampler(output / "telemetry.jsonl").start() if baseline_memory else None
        mark("launch")
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
                        mark("ready")
                        break
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError("timed out waiting for launcher readiness and health")
            log_text = (output / "serve.log").read_text(errors="replace")
            result["identity"]["backend"] = backend_identity(log_text)
            result["weight_devices"] = weight_devices(log_text)
            result["device_allocations_mib"] = device_allocations(log_text)
            result["min_weight_devices"] = args.min_weight_devices
            # Record what the launch actually consumed before any assertion can
            # abort the run: a placement that underuses the hardware is the
            # thing we are hunting, so its evidence has to survive a failure.
            result["utilization"] = utilization(baseline_memory, gpu_memory())
            try:
                props = request(args.port, "/props", timeout=args.request_timeout)
                settings = props.get("default_generation_settings") or {}
                result["served"] = {"context": settings.get("n_ctx"), "slots": props.get("total_slots")}
            except (OSError, ValueError, urllib.error.URLError) as exc:
                result["served"] = {"error": str(exc)}
            if len(result["weight_devices"]) < args.min_weight_devices:
                raise RuntimeError(f"required {args.min_weight_devices} weight devices, observed {result['weight_devices']}")
            reply = request(args.port, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "Name one colour."}],
                "max_tokens": 32, "temperature": 0, "stream": False,
            }, timeout=args.request_timeout)
            (output / "reply.json").write_text(json.dumps(reply, indent=2))
            result["generation_timings"] = reply.get("timings")
            choices = reply.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            # Some reasoning models spend the entire short budget in reasoning.
            generated = message.get("content") or message.get("reasoning_content")
            if not isinstance(generated, str) or not generated.strip():
                raise RuntimeError("completion contained no generated text")
            if proc.poll() is not None:
                raise RuntimeError("launcher exited during generation")
            result["generation"] = True
            mark("generated")
            events = streaming_request(args.port, args.request_timeout)
            (output / "stream.json").write_text(json.dumps(events, indent=2))
            if proc.poll() is not None:
                raise RuntimeError("launcher exited during streaming")
            result["streaming"] = True
            mark("streamed")
            result["ubatch_raise"] = check_ubatch_raise((output / "serve.log").read_text(errors="replace"))
            # Cancellation and prefix reuse are the two acceptance items the
            # health/generate/stream sequence above cannot see. Both are the
            # ordinary agent path, not a stress test.
            result["cancel_reconnect"] = cancel_and_reconnect(args.port, args.request_timeout)
            mark("cancel_recovered")
            if proc.poll() is not None:
                raise RuntimeError("launcher exited during cancellation recovery")
            if args.prefix_reuse:
                result["prefix_reuse"] = prefix_reuse(args.port, args.request_timeout)
                mark("prefix_reuse")
                if proc.poll() is not None:
                    raise RuntimeError("launcher exited during the prefix-reuse check")
            if args.agent_lanes:
                subprocess.run([sys.executable, str(Path(__file__).with_name("verify-agent-workload.py")),
                                "--url", f"http://127.0.0.1:{args.port}", "--output", str(output / "agent-workload"),
                                "--lanes", str(args.agent_lanes), "--repeats", str(args.agent_repeats),
                                "--timeout", str(args.request_timeout)], check=True)
                result["agent_workload"] = True
                mark("agent_workload")
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
            mark("stopped")
            if baseline_memory:
                time.sleep(2)
                result["memory_released"] = released_after_stop(baseline_memory, gpu_memory())
            if "sampler" in locals() and sampler is not None:
                sampler.stop()
                result["gpu_telemetry"] = sampler.summary()
            result["passed"] = bool(result.get("generation") and result.get("streaming") and result.get("port_released")
                                    and not result.get("forced_cleanup") and not result.get("error")
                                    and all(v is not False for v in (result.get("memory_released") or {}).values()))
        (output / "result.json").write_text(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("installed serving lifecycle did not pass; see result.json")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
