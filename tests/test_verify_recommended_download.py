"""verify-recommended-download.py reads backend status from current and older launchers."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("verify_rec", ROOT / "scripts/verify-recommended-download.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_current_and_older_status_keys():
    v = load()
    assert v.backend_status({"BackendStatus": "ready"}) == "ready"
    assert v.backend_status({"NeedsBackendBuild": True}) == "needs_setup"
    assert v.backend_status({"NeedsBackendBuild": False}) == "unknown"


def test_published_release_without_status_keys_is_legacy():
    v = load()
    # The published-mode Windows E2E runs the latest release (v3.2.10), whose rows
    # carry neither key.
    assert v.launcher_release("ggrun v3.2.10-go") == (3, 2, 10)
    assert v.backend_status({"repo": "a/b"}, legacy=True) == "legacy"


def test_unknown_shape_still_fails_closed():
    v = load()
    # A build past a tag is not that release, and a newer launcher must carry a key.
    assert v.launcher_release("v3.2.10-174-ge40c127") is None
    assert v.launcher_release("ggrun v3.2.11") == (3, 2, 11)
    with pytest.raises(RuntimeError):
        v.backend_status({"repo": "a/b"})
