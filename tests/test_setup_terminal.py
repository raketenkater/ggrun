"""Run the setup entrypoint with real console attachment and fake install payloads."""
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "linux", "Linux setup and PTY regression")
class SetupTerminalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ggrun-setup-terminal-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "scripts").mkdir()
        (self.root / ".git").mkdir()
        for name in ["setup.sh", "scripts/setup-home.sh"]:
            shutil.copy2(ROOT / name, self.root / name)
        install = self.root / "install.sh"
        install.write_text("""#!/bin/sh
[ "${1:-}" = --discover ] && exit 0
printf '%s\\n' "$LLM_INSTALL_REPO_DIR" > "$GGRUN_TEST_SOURCE_REPO"
mkdir -p "$LLM_INSTALL_PREFIX"
cp "$GGRUN_TEST_INSTALL_PAYLOAD" "$LLM_INSTALL_PREFIX/ggrun"
chmod +x "$LLM_INSTALL_PREFIX/ggrun"
""")
        install.chmod(0o755)
        payload = self.root / "payload"
        payload.write_text("""#!/bin/sh
printf '%s\n' "$*" >> "$GGRUN_TEST_INSTALL_CALLS"
case "$1" in
  version) echo 'ggrun fixture' ;;
  detect) echo '{}' ;;
  recommend) echo 'fixture/repo' ;;
  download)
    if [ ! -t 0 ]; then echo DOWNLOAD_STDIN_NOT_TTY; exit 2; fi
    echo DOWNLOAD_WITH_CONSOLE
    exit 1 ;;
esac
""")
        self.env = dict(os.environ)
        for key in ["LLM_SETUP_NONINTERACTIVE", "LLM_INSTALL_NONINTERACTIVE"]:
            self.env.pop(key, None)
        self.env.update(LLM_APP_HOME=str(self.root / "app"), LLM_SETUP_BACKEND="skip",
                        LLM_SETUP_PY_DEPS="skip", GGRUN_TEST_INSTALL_PAYLOAD=str(payload),
                        GGRUN_TEST_INSTALL_CALLS=str(self.root / "calls"),
                        GGRUN_TEST_SOURCE_REPO=str(self.root / "source-repo"))

    def test_detached_setup_skips_guided_download(self):
        result = subprocess.run([str(self.root / "setup.sh")], env=self.env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                start_new_session=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = (self.root / "calls").read_text()
        self.assertNotIn("recommend", calls)
        self.assertNotIn("download", calls)
        self.assertTrue((self.root / "app/ggrun").is_file())

    def test_linked_worktree_keeps_the_local_source(self):
        (self.root / ".git").rmdir()
        (self.root / ".git").write_text("gitdir: /fixture/worktrees/candidate\n")
        self.test_detached_setup_skips_guided_download()
        self.assertEqual((self.root / "source-repo").read_text().strip(), "")

    def test_source_archive_requests_a_managed_checkout(self):
        (self.root / ".git").rmdir()
        self.test_detached_setup_skips_guided_download()
        self.assertEqual((self.root / "source-repo").read_text().strip(),
                         str(self.root / "app/.src/ggrun"))

    def test_piped_setup_uses_console_for_download_and_handles_failure(self):
        import pty
        pid, fd = pty.fork()
        if pid == 0:
            # curl | bash consumes a pipe; guided prompts use the controlling tty.
            null = os.open(os.devnull, os.O_RDONLY)
            os.dup2(null, 0)
            os.close(null)
            os.execve(str(self.root / "setup.sh"), [str(self.root / "setup.sh")], self.env)
        output = bytearray()
        status = None
        try:
            # Keep install/model directories, skip backend install, download a
            # model, decline shell-profile changes, keep the recommended repo.
            os.write(fd, b"\n\nskip\ny\nn\n\n")
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if select.select([fd], [], [], 0.1)[0]:
                    try:
                        block = os.read(fd, 65536)
                    except OSError:
                        break
                    if not block:
                        break
                    output.extend(block)
                child, status = os.waitpid(pid, os.WNOHANG)
                if child:
                    break
                status = None
            if status is None:
                child, status = os.waitpid(pid, os.WNOHANG)
                if not child:
                    os.killpg(pid, signal.SIGKILL)
                    _, status = os.waitpid(pid, 0)
                    self.fail("setup did not finish: " + output.decode(errors="replace"))
            text = output.decode(errors="replace")
            self.assertEqual(os.waitstatus_to_exitcode(status), 0, text)
            self.assertIn("DOWNLOAD_WITH_CONSOLE", text)
            self.assertIn("Warning: Download failed", text)
            self.assertNotIn("DOWNLOAD_STDIN_NOT_TTY", text)
            self.assertNotIn("command not found", text)
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
