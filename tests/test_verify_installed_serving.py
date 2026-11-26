"""revision_matches: exact-main builds and release bundles without a revision."""

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_installed_serving", ROOT / "scripts" / "verify-installed-serving.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

SHA = "05b64fd6140a881d42dce9c24f23076d7c7114d4"
TAG = "v3.2.13"


def ident(revision="", version=""):
    return {"build": {"revision": revision, "version": version}}


def test_exact_main_build_unchanged():
    assert mod.revision_matches(ident(SHA, "05b64fd"), SHA)
    assert mod.revision_matches(ident(SHA[:7]), SHA)
    assert mod.revision_matches(ident("", "v3.2.13-5-g05b64fd"), SHA)
    assert not mod.revision_matches(ident("1" * 40, "v3.2.13-5-g1111111"), SHA)
    assert not mod.revision_matches(ident(), SHA)


def test_release_without_revision_matches_its_tag():
    assert mod.revision_matches(ident("", TAG), SHA, TAG)
    assert mod.revision_matches({"version_text": TAG}, SHA, TAG)
    assert mod.revision_matches(ident("", TAG), "", TAG)


def test_release_check_still_rejects():
    assert not mod.revision_matches(ident("", "v3.2.12"), SHA, TAG)
    assert not mod.revision_matches(ident("", ""), SHA, TAG)
    assert not mod.revision_matches(ident("", "v3.2.13-2-g1111111"), SHA, TAG)
    assert not mod.revision_matches(ident("", TAG), SHA)
    assert not mod.revision_matches(ident("1" * 40, TAG), SHA, TAG)
