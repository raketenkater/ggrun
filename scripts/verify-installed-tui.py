#!/usr/bin/env python3
"""Drive the installed `ggrun tui` in a real terminal, the way a user does.

Journey: first screen -> Recommended Downloads -> back -> Settings -> back ->
resize small and back -> filter the model list -> Configure -> Pre-launch ->
default launch (no hand-tuned flags) -> generate -> Ctrl+C -> port released and
no backend left behind -> relaunch through "latest launch" -> generate -> stop.

Linux uses pexpect on a pty; Windows uses ConPTY through pywinpty. Prompts are
answered as a user would: the contained live memory probe gets "y", any other
[y/N] keeps its default. Writes <output>/tui-result.json and the raw transcript.
"""
import argparse
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[()][A-Z0-9]|\r")
WINDOWS = os.name == "nt"


class Term:
    """Minimal expect over a real terminal on either platform."""

    def __init__(self, argv, env, cwd, rows, cols, transcript):
        self.buf = ""
        self.log = open(transcript, "a", encoding="utf-8", errors="replace")
        if WINDOWS:
            from winpty import PtyProcess  # pywinpty
            self.proc = PtyProcess.spawn(subprocess.list2cmdline(argv), cwd=cwd, env=env, dimensions=(rows, cols))
            # PtyProcess.read blocks until output arrives, so a quiet screen
            # hung the journey past every timeout. Read on a thread instead.
            self.chunks = queue.Queue()
            threading.Thread(target=self._pump, daemon=True).start()
        else:
            import pexpect
            self.proc = pexpect.spawn(argv[0], argv[1:], env=env, cwd=cwd, encoding="utf-8",
                                      codec_errors="replace", dimensions=(rows, cols), timeout=5)

    def _pump(self):
        while True:
            try:
                data = self.proc.read(65536)
            except Exception:
                data = None
            self.chunks.put(data)
            if data is None:
                return

    def _read(self, timeout):
        try:
            if WINDOWS:
                try:
                    data = self.chunks.get(timeout=timeout)
                except queue.Empty:
                    return ""
                if data is None:
                    return None
            else:
                data = self.proc.read_nonblocking(65536, timeout=timeout)
        except Exception as exc:  # EOF and timeouts differ per platform
            name = type(exc).__name__
            if "EOF" in name:
                return None
            if "TIMEOUT" in name.upper():
                return ""
            return None
        if data:
            self.log.write(data)
            self.log.flush()
            self.buf += data
            self.buf = self.buf[-200000:]
        return data

    def expect(self, patterns, timeout):
        """Index of the first pattern seen in new output; -1 on EOF; raises on timeout."""
        regs = [re.compile(p) for p in patterns]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            plain = ANSI.sub("", self.buf)
            for i, r in enumerate(regs):
                m = r.search(plain)
                if m:
                    # consume through the match in the plain view
                    self.buf = plain[m.end():]
                    return i
            got = self._read(1)
            if got is None:
                plain = ANSI.sub("", self.buf)
                for i, r in enumerate(regs):
                    if r.search(plain):
                        self.buf = ""
                        return i
                return -1
        raise TimeoutError(f"none of {patterns} within {timeout}s; screen tail: {ANSI.sub('', self.buf)[-600:]!r}")

    def send(self, keys):
        self.proc.write(keys) if WINDOWS else self.proc.send(keys)

    def resize(self, rows, cols):
        self.proc.setwinsize(rows, cols)

    def alive(self):
        return self.proc.isalive()

    def drain(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._read(0.5) is None:
                return

    def close(self):
        try:
            if self.alive():
                self.proc.terminate(force=True)
        except Exception:
            pass
        self.log.close()


def http_json(port, route, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def port_free(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def backends_running(app_home):
    """llama-server processes whose executable lives under this install."""
    home = str(Path(app_home).resolve()).lower()
    try:
        if WINDOWS:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "Get-Process -Name 'llama-server*' -ErrorAction SilentlyContinue | "
                                  "ForEach-Object { $_.Path }"], capture_output=True, text=True, timeout=30).stdout
        else:
            out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [line for line in out.splitlines() if "llama-server" in line and home in line.lower()]


def launch_and_serve(term, port, steps, t0, args, record, label):
    """From Pre-launch: start, answer prompts, prove serving, stop with Ctrl+C."""
    term.send("\r")
    while True:
        i = term.expect([r"\[launch\] Press Ctrl\+C to stop",
                         r"contained live memory probe\?.*\[y/N\]",
                         r"\[y/N\]",
                         r"Error starting server|launch failed|Error:"], timeout=args.launch_timeout)
        if i == 0:
            break
        if i == 1:
            steps.append((f"{label}: live-probe consent y", round(time.monotonic() - t0, 1)))
            term.send("y\r")
        elif i == 2:
            steps.append((f"{label}: other prompt kept default N", round(time.monotonic() - t0, 1)))
            term.send("\r")
        else:
            raise RuntimeError(f"{label}: launch failed (pattern {i}); tail {ANSI.sub('', term.buf)[-800:]!r}")
    steps.append((f"{label}: ready", round(time.monotonic() - t0, 1)))
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            if http_json(port, "/health").get("status") == "ok":
                break
        except (OSError, ValueError, urllib.error.URLError):
            pass
        term.drain(1)
    else:
        raise RuntimeError(f"{label}: health never ok on port {port}")
    props = http_json(port, "/props", timeout=60)
    reply = http_json(port, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Name one colour."}], "max_tokens": 24, "temperature": 0},
        timeout=args.request_timeout)
    msg = (reply.get("choices") or [{}])[0].get("message", {})
    text = msg.get("content") or msg.get("reasoning_content") or ""
    if not text.strip():
        raise RuntimeError(f"{label}: no generated text")
    settings = props.get("default_generation_settings") or {}
    record[label] = {"served_context": settings.get("n_ctx"), "slots": props.get("total_slots"),
                     "reply_chars": len(text)}
    steps.append((f"{label}: generated", round(time.monotonic() - t0, 1)))
    term.send("\x03")
    stop_t = time.monotonic()
    while time.monotonic() - stop_t < 90 and term.alive():
        term.drain(1)
    record[label]["exited"] = not term.alive()
    record[label]["stop_s"] = round(time.monotonic() - stop_t, 1)
    for _ in range(50):
        if port_free(port):
            break
        time.sleep(0.2)
    record[label]["port_released"] = port_free(port)
    record[label]["orphan_backends"] = backends_running(args.app_home)
    steps.append((f"{label}: stopped", round(time.monotonic() - t0, 1)))
    if not (record[label]["exited"] and record[label]["port_released"]) or record[label]["orphan_backends"]:
        raise RuntimeError(f"{label}: unclean stop {record[label]}")


def spawn(args, transcript, rows=45, cols=160):
    env = dict(os.environ, TERM="xterm-256color", LLM_PORT=str(args.port), LLM_SERVER_NO_UPDATE_CHECK="1",
               # the directory holding the model as given; resolving would follow a
               # symlinked model into its target's folder and list unrelated models
               LLM_MODEL_DIR=os.path.dirname(os.path.abspath(args.model)), LLM_COMMUNITY_TUNES="off")
    if args.app_home:
        # What the installed wrappers set (Windows ggrun.cmd: LLM_APP_HOME and .bin on PATH).
        env["LLM_APP_HOME"] = args.app_home
        env["PATH"] = str(Path(args.app_home) / ".bin") + os.pathsep + env.get("PATH", "")
    return Term([args.launcher, "tui"], env, args.app_home or os.getcwd(), rows, cols, transcript)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--launcher", required=True)
    ap.add_argument("--model", required=True, help="an already downloaded model; its directory becomes the model dir")
    ap.add_argument("--app-home", default="")
    ap.add_argument("--port", type=int, default=18850)
    ap.add_argument("--output", required=True)
    ap.add_argument("--launch-timeout", type=int, default=900)
    ap.add_argument("--request-timeout", type=int, default=180)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    transcript = out / "tui-transcript.log"
    record = {"launcher": args.launcher, "model": str(Path(args.model).resolve()), "platform": sys.platform,
              "passed": False}
    steps = []
    t0 = time.monotonic()
    term = None
    try:
        if not port_free(args.port):
            raise RuntimeError(f"port {args.port} is already in use")
        name = Path(args.model).stem
        term = spawn(args, transcript)
        # The title is also drawn on the "Starting up" screen, where keys are
        # not handled yet: wait for the menu itself.
        term.expect([r"Recommended downloads|ggrun First Run"], timeout=180)
        term.expect([re.escape(name[:18])], timeout=120)
        steps.append(("main screen lists model", round(time.monotonic() - t0, 1)))

        term.send("r")
        term.expect([r"Recommended Downloads"], timeout=120)
        term.drain(3)
        screen = ANSI.sub("", term.buf)
        record["recommendations_screen"] = {
            "none_fit": "No safe recommendation fits" in screen,
            "hardware_line": next((l.strip() for l in screen.splitlines() if "Hardware:" in l), None)}
        steps.append(("recommended downloads", round(time.monotonic() - t0, 1)))
        term.send("\x1b")
        term.expect([r"═══ ggrun ═══"], timeout=30)

        term.send("s")
        term.expect([r"═══ Settings ═══"], timeout=30)
        steps.append(("settings", round(time.monotonic() - t0, 1)))
        term.send("\x1b")
        term.expect([r"═══ ggrun ═══"], timeout=30)

        term.resize(24, 80)
        term.drain(2)
        term.resize(45, 160)
        term.expect([r"═══ ggrun ═══"], timeout=30)
        if not term.alive():
            raise RuntimeError("TUI exited on resize")
        steps.append(("resized", round(time.monotonic() - t0, 1)))

        term.send("/")
        time.sleep(0.5)
        term.send(name[:10])
        time.sleep(0.5)
        term.send("\r")      # apply the filter
        term.drain(1)
        term.send("\r")      # open the selected model
        term.expect([r"Configure"], timeout=60)
        steps.append(("filtered and opened Configure", round(time.monotonic() - t0, 1)))
        term.send("l")
        term.expect([r"Pre-launch"], timeout=60)
        steps.append(("pre-launch", round(time.monotonic() - t0, 1)))
        launch_and_serve(term, args.port, steps, t0, args, record, "first_launch")
        term.close()

        # Relaunch: a fresh TUI replays the latest launch from the main screen.
        term = spawn(args, transcript)
        term.expect([r"Run latest configuration"], timeout=180)
        term.drain(1)
        term.send("l")
        i = term.expect([r"Pre-launch", r"Configure"], timeout=60)
        if i == 1:
            term.send("l")
            term.expect([r"Pre-launch"], timeout=60)
        steps.append(("relaunch pre-launch", round(time.monotonic() - t0, 1)))
        launch_and_serve(term, args.port, steps, t0, args, record, "relaunch")
        record["passed"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
    finally:
        if term is not None:
            term.close()
        record["steps"] = steps
        (out / "tui-result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ("passed", "steps") if k in record} | ({"error": record["error"]} if "error" in record else {})))
    if not record["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
