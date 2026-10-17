"""Downloads must be verified, pinned, resumable and interruptible."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def downloader():
    path = Path(__file__).resolve().parents[1] / "tools/download/download_any_gguf.py"
    spec = importlib.util.spec_from_file_location("integrity_downloader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_size_and_digest_both_required(tmp_path):
    module = downloader()
    path = tmp_path / "model.gguf"
    path.write_bytes(b"good")
    identity = (4, "sha256", hashlib.sha256(b"good").hexdigest())
    module.verify_file(path, identity)
    path.write_bytes(b"evil")  # Same size: a size check alone cannot detect it.
    with pytest.raises(ValueError, match="sha256 mismatch"):
        module.verify_file(path, identity)
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="size mismatch"):
        module.verify_file(path, identity)


def test_git_blob_hash_is_not_raw_file_sha1(tmp_path):
    module = downloader()
    path = tmp_path / "tiny.gguf"
    path.write_bytes(b"GGUF")
    module.verify_file(path, (4, "git-sha1", hashlib.sha1(b"blob 4\0GGUF").hexdigest()))


def test_cached_corruption_is_replaced_at_pinned_revision(monkeypatch, tmp_path):
    module = downloader()
    path = tmp_path / "model.gguf"
    path.write_bytes(b"evil")
    sibling = SimpleNamespace(rfilename=path.name, size=4,
                              lfs={"sha256": hashlib.sha256(b"good").hexdigest()})

    class Api:
        def model_info(self, repo, **kwargs):
            return SimpleNamespace(sha="pinned", siblings=[sibling])

    calls = []

    def transfer(repo, filename, output, revision, force=False):
        calls.append((revision, force))
        assert not path.exists()  # HF cannot trust the old final-file metadata.
        path.write_bytes(b"good")
        return path

    monkeypatch.setattr(module, "HfApi", Api)
    monkeypatch.setattr(module, "transfer_file", transfer)
    downloaded, failed = module.download_files("owner/repo", [path.name], tmp_path)
    assert failed == []
    assert downloaded == [(path.name, path)]
    assert calls == [("pinned", False)]


def test_missing_manifest_fails_before_any_transfer(monkeypatch, tmp_path):
    module = downloader()

    class Api:
        def model_info(self, repo, **kwargs):
            return SimpleNamespace(sha="pinned", siblings=[])

    monkeypatch.setattr(module, "HfApi", Api)
    monkeypatch.setattr(module, "transfer_file", lambda *a, **k: pytest.fail("no unverified transfer"))
    assert module.download_files("owner/repo", ["missing.gguf"], tmp_path) == ([], ["missing.gguf"])


@pytest.mark.parametrize("files, expected", [([], 1), (["model.gguf"], 130)])
def test_empty_selection_and_cancel_are_not_success(monkeypatch, tmp_path, files, expected):
    module = downloader()
    monkeypatch.setattr(module, "get_args", lambda: SimpleNamespace(
        repo="owner/repo", dir=str(tmp_path), vram=0, ram=0, quant="Q4_K_M",
        no_repo_search=True, yes=False, cache_dir=None))
    monkeypatch.setattr(module, "clear_screen", lambda: None)
    monkeypatch.setattr(module, "print_header", lambda: None)
    monkeypatch.setattr(module, "resolve_best_gguf_repo", lambda repo, *args: repo)
    monkeypatch.setattr(module, "select_quantization", lambda *args: "Q4_K_M")
    monkeypatch.setattr(module, "get_model_files", lambda *args: files)
    monkeypatch.setattr(module, "get_download_directory", lambda *args: tmp_path)
    monkeypatch.setattr("builtins.input", lambda *args: "n")
    monkeypatch.setattr(module, "download_files", lambda *args: pytest.fail("no transfer"))
    monkeypatch.setattr(module, "update_model_index", lambda *args: pytest.fail("no index update"))
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == expected


def test_wedged_worker_is_killed_and_partial_data_preserved(monkeypatch, tmp_path):
    module = downloader()
    partial = tmp_path / "model.incomplete"
    partial.write_bytes(b"keep me")
    processes = []
    popen = module.subprocess.Popen

    def capture(*args, **kwargs):
        proc = popen(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(module.subprocess, "Popen", capture)
    monkeypatch.setattr(module, "DOWNLOAD_WORKER", "import time; time.sleep(30)")
    with pytest.raises(TimeoutError, match="no download progress"):
        module.transfer_file("owner/repo", "model.gguf", tmp_path, "pinned", stall_seconds=0.3)
    assert processes[0].poll() is not None
    assert partial.read_bytes() == b"keep me"


def test_slow_worker_with_real_progress_is_allowed(monkeypatch, tmp_path):
    module = downloader()
    monkeypatch.setattr(module, "DOWNLOAD_WORKER", '''
import json, time
for i in range(1, 9):
    print(json.dumps({"bytes": i}), flush=True)
    time.sleep(0.3)
print(json.dumps({"path": "model.gguf"}), flush=True)
''')
    assert module.transfer_file("owner/repo", "model.gguf", tmp_path, "pinned", stall_seconds=2) == Path("model.gguf")


def test_worker_tracks_absolute_xet_progress_even_when_hidden(monkeypatch, tmp_path):
    module = downloader()
    # Xet computes each update from tqdm.n; disabled tqdm normally leaves n
    # unchanged. Exercise the real child protocol with that SDK interaction.
    (tmp_path / "huggingface_hub.py").write_text('''
from pathlib import Path
def hf_hub_download(tqdm_class, **kwargs):
    assert kwargs["revision"] == "pinned"
    bar = tqdm_class(total=4, disable=True, name="huggingface_hub.xet_get")
    for position in (1, 2, 3, 4):
        bar.update(position - bar.n)
        assert bar.n == position
    bar.close()
    path = Path(kwargs["local_dir"]) / kwargs["filename"]
    path.write_bytes(b"good")
    return str(path)
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = module.transfer_file("owner/repo", "model.gguf", tmp_path, "pinned", stall_seconds=5)
    assert result.read_bytes() == b"good"
