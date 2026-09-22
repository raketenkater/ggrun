"""Offline regression tests for installer boundary contracts."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(path, name):
    found = re.search(r"^" + re.escape(name) + r"\(\) \{.*?^\}",
                      (ROOT / path).read_text(), re.M | re.S)
    if not found:
        raise AssertionError(f"missing function {name}")
    return found.group()


class InstallContracts(unittest.TestCase):
    def test_source_checkout_accepts_a_pinned_commit(self):
        # `git clone --branch` rejects a commit id, so LLM_SETUP_REF=<sha> (CI and
        # exact-candidate installs) kept no source checkout while the summary
        # still printed one.
        git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
        with tempfile.TemporaryDirectory() as tmp:
            work, bare = Path(tmp) / "work", Path(tmp) / "origin.git"
            subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
            subprocess.run(git + ["-C", str(work), "commit", "-q", "--allow-empty", "-m", "one"], check=True)
            pinned = subprocess.run(["git", "-C", str(work), "rev-parse", "HEAD"], check=True,
                                    capture_output=True, text=True).stdout.strip()
            subprocess.run(git + ["-C", str(work), "commit", "-q", "--allow-empty", "-m", "two"], check=True)
            subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
            script = function("install.sh", "clone_source_ref")
            for ref, want in ((pinned, pinned), ("main", None)):
                with self.subTest(ref=ref):
                    dest = Path(tmp) / ("co-" + ref[:7])
                    subprocess.run(["bash", "-eu", "-c", script + f'\nclone_source_ref "file://{bare}" "{ref}" "{dest}"\n'],
                                   check=True)
                    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True,
                                          capture_output=True, text=True).stdout.strip()
                    if want:
                        self.assertEqual(head, want)
            missing = Path(tmp) / "co-missing"
            result = subprocess.run(["bash", "-eu", "-c", script +
                                     f'\nclone_source_ref "file://{bare}" "{"0" * 40}" "{missing}"\n'])
            self.assertNotEqual(result.returncode, 0)

    def test_release_does_not_rebuild_checked_launcher(self):
        script = function("install.sh", "install_ggrun_from_source")
        script += "\nensure_source_repo() { echo unexpected-source-build; exit 99; }\n"
        script += "MAIN_IMPL=go INSTALL_MODE=release RELEASE_INSTALLED=1 install_ggrun_from_source\n"
        result = subprocess.run(["bash", "-eu", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertNotIn("unexpected-source-build", result.stdout)

    def test_python_venv_does_not_use_user_install(self):
        for venv in (True, False):
            with self.subTest(venv=venv), tempfile.TemporaryDirectory() as tmp:
                calls = Path(tmp) / "calls"
                script = function("install.sh", "install_python_download_deps")
                script += "\npython_in_venv() { return " + ("0" if venv else "1") + "; }\n"
                script += 'python3() { printf "%s\\n" "$*" >> "$CALLS"; }\n'
                script += "install_python_download_deps\n"
                env = dict(os.environ, CALLS=str(calls))
                subprocess.run(["bash", "-eu", "-c", script], check=True, env=env)
                recorded = calls.read_text()
                self.assertEqual("--user" in recorded, not venv)
                self.assertIn("-m pip install", recorded)

    def test_setup_recognizes_native_macho_and_rejects_scripts(self):
        script = function("scripts/setup-home.sh", "usable_llama_server")
        for magic, expected in [(b"\xcf\xfa\xed\xfe", 0), (b"\xca\xfe\xba\xbe", 0),
                                (b"\x7fELF", 0), (b"#!/b", 1)]:
            with self.subTest(magic=magic), tempfile.TemporaryDirectory() as tmp:
                binary = Path(tmp) / "llama-server"
                binary.write_bytes(magic + b"fixture")
                binary.chmod(0o755)
                result = subprocess.run(["bash", "-c", script + '\nusable_llama_server "$1"', "test", str(binary)])
                self.assertEqual(result.returncode, expected)


if __name__ == "__main__":
    unittest.main()
