"""Exercise release relocation using a real ELF and a private shared library."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cc") and shutil.which("patchelf"), "requires cc and patchelf")
class ReleaseRelocation(unittest.TestCase):
    def test_bundled_dependencies_with_missing_or_wrong_runpath(self):
        for runpath in (None, "$ORIGIN/../missing"):
            with self.subTest(runpath=runpath), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "scripts").mkdir()
                (root / "go").mkdir()
                shutil.copy2(ROOT / "scripts/package-release.sh", root / "scripts/package-release.sh")
                # package-release.sh fails closed on these two, deliberately: an
                # archive shipped without them has a documented command that dies
                # with "scripts/setup-home.sh: No such file or directory". The
                # fixture predates that guard, so it packaged nothing and this
                # test failed for a missing input rather than a runpath fault.
                shutil.copy2(ROOT / "scripts/setup-home.sh", root / "scripts/setup-home.sh")
                shutil.copy2(ROOT / "install.sh", root / "install.sh")
                launcher = root / "go/ggrun"
                launcher.write_text("#!/bin/sh\necho ggrun-fixture\n")
                launcher.chmod(0o755)
                build = root / "build"
                build.mkdir()
                (build / "lib.c").write_text("int fixture(void) { return 0; }\n")
                (build / "main.c").write_text("extern int fixture(void); int main(void) { return fixture(); }\n")
                subprocess.run(["cc", "-shared", "-fPIC", "-Wl,-soname,libllama.so", "-o", str(build / "libllama.so"), str(build / "lib.c")], check=True)
                cmd = ["cc", str(build / "main.c"), "-L" + str(build), "-lllama", "-o", str(build / "llama-server")]
                if runpath:
                    cmd.append("-Wl,-rpath," + runpath)
                subprocess.run(cmd, check=True)
                env = dict(os.environ)
                env.pop("LD_LIBRARY_PATH", None)
                env.pop("LD_PRELOAD", None)
                packaged = subprocess.run(["bash", str(root / "scripts/package-release.sh"), "ggrun-linux-x86_64-cpu.tar.gz", str(build / "llama-server"), str(root / "dist")], env=env, capture_output=True, text=True)
                self.assertEqual(packaged.returncode, 0, packaged.stdout + packaged.stderr)
                unpacked = root / "unpacked"
                unpacked.mkdir()
                subprocess.run(["tar", "-xzf", str(root / "dist/ggrun-linux-x86_64-cpu.tar.gz"), "-C", str(unpacked)], check=True)
                build.rename(root / "hidden-build")
                binary = unpacked / "ggrun-linux-x86_64-cpu/bin/llama-server"
                result = subprocess.run([str(binary), "--version"], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
