import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.release = self.root / "release"
        self.release.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.dest = self.root / "installed"
        artifact = b"#!/bin/sh\nprintf 'fixture-version\\n'\n"
        (self.release / "modelship_linux_amd64").write_bytes(artifact)
        digest = hashlib.sha256(artifact).hexdigest()
        (self.release / "SHA256SUMS").write_text(f"{digest}  modelship_linux_amd64\n")
        self.executable("uname", "#!/bin/sh\ncase \"$1\" in -s) echo Linux;; -m) echo x86_64;; esac\n")
        self.executable("curl", """#!/bin/sh
while [ "$#" -gt 0 ]; do
  case "$1" in
    https://*) url=$1 ;;
    -o) shift; out=$1 ;;
  esac
  shift
done
cp "$FIXTURE_RELEASE/$(basename "$url")" "$out"
""")
        self.env = {
            **os.environ,
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "MODELSHIP_RELEASE_BASE": "https://release.example/v1",
            "FIXTURE_RELEASE": str(self.release),
            "MODELSHIP_INSTALL_DIR": str(self.dest),
        }

    def executable(self, name, text):
        p = self.bin / name
        p.write_text(text)
        p.chmod(0o755)

    def run_installer(self, *args):
        return subprocess.run(
            ["sh", str(ROOT / "install.sh"), *args],
            env=self.env, capture_output=True, text=True,
        )

    def test_verified_install_and_execution(self):
        result = self.run_installer("version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "fixture-version")
        self.assertTrue(os.access(self.dest / "modelship", os.X_OK))

    def test_corruption_does_not_replace_existing_binary(self):
        self.dest.mkdir()
        existing = self.dest / "modelship"
        existing.write_text("old binary")
        (self.release / "modelship_linux_amd64").write_text("corrupted")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Checksum mismatch", result.stderr)
        self.assertEqual(existing.read_text(), "old binary")

    def test_missing_checksum(self):
        (self.release / "SHA256SUMS").write_text("f" * 64 + "  other-file\n")
        self.assertNotEqual(self.run_installer().returncode, 0)
        self.assertFalse((self.dest / "modelship").exists())

    def test_requires_https(self):
        self.env["MODELSHIP_RELEASE_BASE"] = "http://release.example"
        self.assertNotEqual(self.run_installer().returncode, 0)

    def test_rejects_unsupported_platform(self):
        self.executable("uname", "#!/bin/sh\necho Darwin\n")
        self.assertNotEqual(self.run_installer().returncode, 0)


if __name__ == "__main__":
    unittest.main()
