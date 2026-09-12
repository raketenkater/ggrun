"""Offline guards for evidence integrity in the large-model GPU test."""
import importlib.util
from pathlib import Path
import tempfile
import socket
import os
import unittest

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

serving = load("verify-installed-serving")
gpu = load("verify-gpu-install")

class GPUCheckTests(unittest.TestCase):
    def test_live_listener_is_rejected(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            with self.assertRaises(OSError):
                serving.check_port_available(server.getsockname()[1])

    @unittest.skipIf(os.name == "nt", "POSIX TIME_WAIT rebinding")
    def test_clean_stop_allows_immediate_relaunch(self):
        with socket.socket() as server, socket.socket() as client:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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

if __name__ == "__main__":
    unittest.main()
