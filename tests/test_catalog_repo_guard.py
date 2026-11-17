import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("catalog_repo_guard", ROOT / "tools/models/catalog_repo_guard.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def catalog(*rows):
    return {"candidates": [{"name": n, "repo": r} for n, r in rows]}


def test_repo_change_for_an_existing_row_fails(tmp_path):
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    old.write_text(json.dumps(catalog(("Model A", "org/a-GGUF"), ("Model B", "org/b-GGUF"))))
    new.write_text(json.dumps(catalog(("Model A", "someone/a-GGUF"), ("Model B", "org/b-GGUF"))))
    assert guard.changed_repos(json.loads(old.read_text()), json.loads(new.read_text())) == [
        ("Model A", "org/a-GGUF", "someone/a-GGUF")
    ]
    assert guard.main(["guard", str(old), str(new)]) == 1


def test_new_dropped_and_unchanged_rows_pass(tmp_path):
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    old.write_text(json.dumps(catalog(("Model A", "org/a-GGUF"), ("Gone", "org/gone"), ("Unresolved", ""))))
    new.write_text(json.dumps(catalog(("Model A", "org/a-GGUF"), ("New", "org/new"), ("Unresolved", "org/now-found"))))
    assert guard.main(["guard", str(old), str(new)]) == 0
