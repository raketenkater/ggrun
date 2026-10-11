"""Offline guards for evidence integrity in the large-model GPU test."""
import importlib.util
from pathlib import Path
import tempfile
import socket
import os
import json
import io
import subprocess
import sys
import unittest
from unittest.mock import patch
import errno
import hashlib

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

serving = load("verify-installed-serving")
gpu = load("verify-gpu-install")
compare = load("compare-gpu-acceptance")

class GPUCheckTests(unittest.TestCase):
    def test_ubatch_raise_requires_exact_admission(self):
        admitted = ("[launch] staged expert prefill: raising ubatch 64 -> 512 at the same context\n"
                    "[launch] preflight: placement fits (...)\n"
                    "[launch] raised microbatch passed exact preflight; keeping that argv\n")
        self.assertEqual(serving.check_ubatch_raise(admitted), {"raised": True})
        self.assertEqual(serving.check_ubatch_raise("[launch] preflight: placement fits\n"), {"raised": False})
        with self.assertRaisesRegex(RuntimeError, "not admitted"):
            serving.check_ubatch_raise("[launch] staged expert prefill: raising ubatch 64 -> 512\n")

    def test_claude_code_mode_serves_without_a_client_on_path(self):
        with tempfile.TemporaryDirectory() as temp:
            with_claude, without = Path(temp, "a"), Path(temp, "b")
            with_claude.mkdir()
            without.mkdir()
            (with_claude / "claude").write_text("")
            path = serving.claude_free_path(os.pathsep.join([str(with_claude), str(without)]))
            self.assertEqual(path, str(without))

    def test_auto_context_reaches_launcher_without_an_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "exiting_launcher.py"
            model.write_text("raise SystemExit(1)\n")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/verify-installed-serving.py"),
                                     "--launcher", sys.executable, "--model", str(model),
                                     "--output", str(root / "evidence"), "--ctx", "0"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            evidence = json.loads((root / "evidence/result.json").read_text())
            self.assertNotIn("--ctx", evidence["command"])
            self.assertFalse(evidence["passed"])
            self.assertIn("launcher exited", evidence["error"])

    def test_stream_requires_text_and_completion_marker(self):
        event = b'data: {"choices":[{"delta":{"content":"blue"}}]}\n\n'
        self.assertEqual(len(serving.read_stream(io.BytesIO(event + b'data: [DONE]\n'))), 1)
        with self.assertRaisesRegex(RuntimeError, "without \\[DONE\\]"):
            serving.read_stream(io.BytesIO(event))
        with self.assertRaisesRegex(RuntimeError, "without generated text"):
            serving.read_stream(io.BytesIO(b'data: {"choices":[]}\ndata: [DONE]\n'))

    def test_live_listener_is_rejected(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            with self.assertRaises(OSError):
                serving.check_port_available(server.getsockname()[1])

    @unittest.skipIf(os.name == "nt", "POSIX connection probe")
    def test_unknown_port_probe_error_fails_closed(self):
        with patch.object(serving.socket, 'socket') as factory:
            factory.return_value.__enter__.return_value.connect_ex.return_value = errno.ETIMEDOUT
            with self.assertRaises(OSError):
                serving.check_port_available(18843)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux SO_REUSEPORT")
    def test_live_reuseport_listener_is_rejected(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            with self.assertRaises(OSError):
                serving.check_port_available(server.getsockname()[1])

    @unittest.skipIf(os.name == "nt", "POSIX TIME_WAIT rebinding")
    def test_clean_stop_allows_immediate_relaunch(self):
        self.assert_timewait_accepted(socket.SO_REUSEADDR)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux ik socket policy")
    def test_ik_reuseport_clean_stop_allows_immediate_relaunch(self):
        self.assert_timewait_accepted(socket.SO_REUSEPORT)

    def assert_timewait_accepted(self, reuse_option):
        with socket.socket() as server, socket.socket() as client:
            server.setsockopt(socket.SOL_SOCKET, reuse_option, 1)
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
            server.listen(1)
            client.connect(("127.0.0.1", port))
            connection, _ = server.accept()
            connection.close()
            self.assertEqual(client.recv(1), b"")
        serving.check_port_available(port)

    def test_final_launch_weights_only(self):
        log = """[launch] /bin/server -m model
CUDA0 model buffer size = 12000 MiB
CUDA1 model buffer size = 12000 MiB
[launch] /bin/server -m model --cpu
CUDA0 KV buffer size = 100 MiB
CUDA1 compute buffer size = 10 MiB
CUDA2 output buffer size = 10 MiB
CUDA3 model buffer size = 0.00 MiB
"""
        self.assertEqual(serving.weight_devices(log), [])
        log += "llm_load_tensors: CUDA1 buffer size = 526.50 MiB\nCUDA2 model buffer size = 12000 MiB\n"
        self.assertEqual(serving.weight_devices(log), ["CUDA1", "CUDA2"])

    def test_backend_restart_without_argv_does_not_reuse_old_allocations(self):
        log = """[launch] /bin/server -m model
load_tensors: loading model tensors
CUDA0 model buffer size = 100 MiB
CUDA1 model buffer size = 100 MiB
llm_load_tensors: loading model tensors
CPU model buffer size = 200 MiB
CUDA0 compute buffer size = 10 MiB
"""
        self.assertEqual(serving.weight_devices(log), [])

    def test_split_download_requires_complete_unique_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "model-Q3-00001-of-00002.gguf"
            first.touch()
            with self.assertRaisesRegex(ValueError, "missing model shard"):
                gpu.select_model(root)
            (root / "model-Q3-00002-of-00002.gguf").touch()
            (root / "mmproj.gguf").touch()
            self.assertEqual(gpu.select_model(root), first)
            (root / "another.gguf").touch()
            with self.assertRaisesRegex(ValueError, "expected one model"):
                gpu.select_model(root)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def keys_of(value):
    """Every dict key anywhere in a JSON-shaped value."""
    if isinstance(value, dict):
        return set(value) | set().union(*(keys_of(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(keys_of(v) for v in value)) if value else set()
    return set()


FAKE_GGRUN = """#!{python}
import json, os, sys
if sys.argv[1:] == ["version", "--json"]:
    me = os.path.realpath(__file__)
    print(json.dumps({{"version": "{version}", "executable": __file__, "executable_resolved": me}}))
"""


class EvidenceIdentityTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "POSIX shell wrapper")
    def test_wrapper_hash_does_not_stand_for_the_executable(self):
        # Two candidate builds behind the installer's byte-identical wrapper:
        # only the executable that answered version --json tells them apart.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".bin").mkdir()
            wrapper = root / "ggrun"
            wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/.bin/ggrun" "$@"\n')
            wrapper.chmod(0o755)
            seen = []
            for version in ("v1-ga", "v1-gb"):
                exe = root / ".bin" / "ggrun"
                exe.write_text(FAKE_GGRUN.format(python=sys.executable, version=version))
                exe.chmod(0o755)
                ident = serving.launcher_identity(str(wrapper))
                self.assertEqual(ident["entry"]["kind"], "wrapper")
                self.assertEqual(ident["entry"]["path"], str(wrapper))
                self.assertEqual(ident["build"]["version"], version)
                self.assertEqual(ident["executable"]["source"], "version --json")
                self.assertEqual(ident["executable"]["path"], str(exe.resolve()))
                self.assertEqual(ident["executable"]["sha256"], sha256(exe.read_bytes()))
                seen.append(ident)
            self.assertEqual(seen[0]["entry"]["sha256"], seen[1]["entry"]["sha256"])
            self.assertNotEqual(seen[0]["executable"]["sha256"], seen[1]["executable"]["sha256"])

    @unittest.skipIf(os.name == "nt", "POSIX shell wrapper")
    def test_result_records_wrapper_executable_and_every_shard(self):
        # The evidence is written before the launch; a launcher that exits at
        # once still leaves the identity a rerun is matched against.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".bin").mkdir()
            exe = root / ".bin" / "ggrun"
            exe.write_text(FAKE_GGRUN.format(python=sys.executable, version="v1-gd") + "else:\n    sys.exit(3)\n")
            exe.chmod(0o755)
            wrapper = root / "ggrun"
            wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/.bin/ggrun" "$@"\n')
            wrapper.chmod(0o755)
            shards = [root / f"M-{n:05d}-of-00002.gguf" for n in (1, 2)]
            for shard in shards:
                shard.write_bytes(shard.name.encode() * 64)
            result = subprocess.run([sys.executable, str(ROOT / "scripts/verify-installed-serving.py"),
                                     "--launcher", str(wrapper), "--model", str(shards[0]),
                                     "--output", str(root / "evidence"), "--hash-model"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            identity = json.loads((root / "evidence/result.json").read_text())["identity"]
            self.assertEqual(identity["launcher"]["entry"]["kind"], "wrapper")
            self.assertEqual(identity["launcher"]["executable"]["sha256"], sha256(exe.read_bytes()))
            self.assertNotIn("file", identity["launcher"])
            self.assertEqual([s["sha256"] for s in identity["model"]["shards"]],
                             [sha256(s.read_bytes()) for s in shards])

    def test_native_entry_without_version_json_is_its_own_executable(self):
        # The Python interpreter stands in for a launcher that predates
        # version --json: the entry point is then the only executable known.
        ident = serving.launcher_identity(sys.executable)
        self.assertEqual(ident["entry"]["kind"], "executable")
        self.assertIsNone(ident["build"])
        self.assertIn("Python", ident["version_text"])
        self.assertEqual(ident["executable"]["source"], "entry")
        self.assertEqual(ident["executable"]["path"], str(Path(sys.executable).resolve()))

    def test_windows_cmd_wrapper_reports_the_exe_it_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".bin").mkdir()
            wrapper = root / "ggrun.cmd"
            wrapper.write_text('@echo off\r\n"%~dp0.bin\\ggrun.exe" %*\r\n')
            exe = root / ".bin" / "ggrun.exe"
            exe.write_bytes(b"MZ\x90\x00 candidate build")
            build = {"version": "v1-gc", "executable": str(exe)}
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, json.dumps(build) + "\n", "")
            with patch.object(serving.subprocess, "run", fake_run):
                ident = serving.launcher_identity(str(wrapper))
            self.assertEqual(serving.entry_kind(wrapper), "wrapper")
            self.assertEqual(ident["entry"]["kind"], "wrapper")
            self.assertEqual(ident["entry"]["sha256"], sha256(wrapper.read_bytes()))
            self.assertEqual(ident["executable"]["path"], str(exe.resolve()))
            self.assertEqual(ident["executable"]["sha256"], sha256(exe.read_bytes()))
            self.assertEqual(ident["build"], build)
            self.assertEqual(len(calls), 1)
        # On Windows the wrapper runs through cmd.exe, exactly as the launch does.
        command = serving.entry_command(Path("C:/Users/u/ggrun/ggrun.cmd"), ["version", "--json"], windows=True)
        self.assertTrue(command.startswith('cmd.exe /d /s /c "'))
        self.assertIn("ggrun.cmd version --json", command)
        self.assertEqual(serving.entry_command(Path("/opt/ggrun"), ["version"], windows=False), ["/opt/ggrun", "version"])

    def test_split_model_identity_covers_every_shard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shards = [root / f"Big-Q2_K-{n:05d}-of-00003.gguf" for n in (1, 2, 3)]
            for n, shard in enumerate(shards):
                shard.write_bytes(bytes([n]) * (1000 + n))
            (root / "mmproj-Big-f16.gguf").write_bytes(b"projector")
            (root / "Big-draft-Q8_0.gguf").write_bytes(b"draft")
            for given in (shards[0], shards[1]):
                ident = serving.model_identity(given, "sha256")
                self.assertTrue(ident["complete"])
                self.assertEqual([s["path"] for s in ident["shards"]], [str(s) for s in shards])
                self.assertEqual([s["sha256"] for s in ident["shards"]], [sha256(s.read_bytes()) for s in shards])
                self.assertEqual(ident["bytes"], sum(s.stat().st_size for s in shards))
                # No single hash stands for a split artifact.
                self.assertIsNone(ident["sha256"])
            sizes = serving.model_identity(shards[0])
            self.assertNotIn("sha256", keys_of(sizes["shards"]))
            self.assertEqual(sizes["bytes"], sum(s.stat().st_size for s in shards))
            shards[1].unlink()
            partial = serving.model_identity(shards[0], "sha256")
            self.assertFalse(partial["complete"])
            self.assertIsNone(partial["bytes"])
            self.assertTrue(partial["shards"][1]["missing"])

    def test_single_file_model_keeps_its_sha256(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp, "small-Q4_0.gguf")
            model.write_bytes(b"weights")
            ident = serving.model_identity(model, "sha256")
            self.assertEqual(ident["sha256"], sha256(b"weights"))
            self.assertEqual(len(ident["shards"]), 1)

    def test_fingerprint_is_labelled_and_never_a_sha256(self):
        mib = 1 << 20
        with tempfile.TemporaryDirectory() as temp:
            data = bytes(range(256)) * (3 * mib // 256) + b"tail-end"
            for name in ("one-Q2_K.gguf", "two-Q2_K-00001-of-00001.gguf"):
                model = Path(temp, name)
                model.write_bytes(data)
                ident = serving.model_identity(model, "fingerprint", window_mib=1)
                self.assertEqual(ident["hash"], "fingerprint")
                self.assertIsNone(ident["sha256"])
                self.assertNotIn("sha256", keys_of(ident["shards"]))
                shard = ident["shards"][0]
                self.assertEqual(shard["bytes"], len(data))
                self.assertEqual(shard["fingerprint"], {"scheme": "sha256-of-head-and-tail", "window_bytes": mib,
                                                        "head": sha256(data[:mib]), "tail": sha256(data[-mib:])})
                self.assertNotEqual(shard["fingerprint"]["head"], sha256(data))

    def test_hash_and_fingerprint_modes_are_exclusive(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/verify-installed-serving.py"),
                                 "--launcher", "x", "--model", "y", "--output", "z",
                                 "--hash-model", "--fingerprint-model", "64"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("not allowed with", result.stderr)

    def test_backend_launch_line_on_posix_and_windows(self):
        log = ("[launch] /old/llama-server -m /m.gguf\n"
               "[launch] /app/.bin/ik_llama-server-cuda -m /m.gguf --port 8081\n")
        self.assertEqual(serving.launched_backend(log), ("/app/.bin/ik_llama-server-cuda", "-m /m.gguf --port 8081"))
        windows = "[launch] & C:\\Users\\u\\ggrun\\.bin\\llama-server.exe -m C:\\m.gguf --port 8081\n"
        self.assertEqual(serving.launched_backend(windows),
                         ("C:\\Users\\u\\ggrun\\.bin\\llama-server.exe", "-m C:\\m.gguf --port 8081"))
        spaced = "[launch] & 'C:\\Program Files\\ggrun\\llama-server.exe' -m C:\\m.gguf\n"
        self.assertEqual(serving.launched_backend(spaced)[0], "C:\\Program Files\\ggrun\\llama-server.exe")
        self.assertIsNone(serving.launched_backend("[launch] staged expert prefill: raising ubatch\n"))

    def test_companions_in_the_launched_argv_are_identified(self):
        with tempfile.TemporaryDirectory() as temp:
            projector, draft = Path(temp, "mmproj-f16.gguf"), Path(temp, "draft-Q8_0.gguf")
            projector.write_bytes(b"projector")
            draft.write_bytes(b"draft")
            argv = f"-m /m.gguf --mmproj {projector} --model-draft '{draft}' --port 8081"
            found = serving.companion_identity(argv, "sha256")
            self.assertEqual([(c["flag"], c["sha256"]) for c in found],
                             [("--mmproj", sha256(b"projector")), ("--model-draft", sha256(b"draft"))])
            self.assertEqual(serving.companion_identity("-m /m.gguf --port 8081"), [])

    def test_comparison_needs_every_shard_to_match(self):
        old = {"path": "/m.gguf", "bytes": 7, "sha256": "aa"}
        new = {"path": "/m.gguf", "bytes": 7, "sha256": "aa", "shards": [{"path": "/m.gguf", "bytes": 7, "sha256": "aa"}]}
        self.assertEqual(compare.model_key(old), compare.model_key(new))
        split = {"sha256": None, "shards": [{"sha256": "h1"}, {"sha256": "h2"}]}
        changed = {"sha256": None, "shards": [{"sha256": "h1"}, {"sha256": "h3"}]}
        self.assertNotEqual(compare.model_key(split), compare.model_key(changed))
        printed = {"sha256": None, "shards": [{"bytes": 7, "fingerprint": {"head": "x", "tail": "y"}}]}
        self.assertNotEqual(compare.model_key(printed), compare.model_key(new))

if __name__ == "__main__":
    unittest.main()
