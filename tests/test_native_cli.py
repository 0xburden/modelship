"""Real Go CLI lifecycle integration with a tiny CPU-only fake native worker.

Opt in with MODELSHIP_TEST_BINARY=/absolute/path/to/modelship on Linux.
Does not run pip, access Hugging Face, load models, or require a GPU.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

BINARY = os.environ.get("MODELSHIP_TEST_BINARY", "")
FAKE_WORKER = '''import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
p = argparse.ArgumentParser()
p.add_argument("--run-id", required=True)
args = p.parse_args()
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = {"ready": True, "model": os.environ["MODEL_ID"], "revision": os.environ["MODEL_REVISION"], "run_id": args.run_id}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"choices":[{"text":"Paris"}]}')
    def log_message(self, *args):
        pass
HTTPServer(("127.0.0.1", int(os.environ["MODEL_PORT"])), Handler).serve_forever()
'''


@unittest.skipUnless(BINARY and sys.platform.startswith("linux"), "set MODELSHIP_TEST_BINARY on Linux for native lifecycle integration")
class NativeCLIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = {**os.environ, "XDG_STATE_HOME": str(self.root)}
        self.dir = self.root / "modelship/native/integration"
        self.dir.mkdir(parents=True, mode=0o700)
        self.dir.chmod(0o700)
        python = self.dir / "venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        (self.dir / "native_entry.py").write_text(FAKE_WORKER)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.cfg = {
            "schema": 1,
            "plan": {
                "backend": "native", "name": "integration", "model": "org/model",
                "revision": "a" * 40, "port": self.port, "task": "text-generation",
                "runtime": "transformers", "context": 4096,
                "gpu": {"uuid": "GPU-fixture"},
            },
            "torch": {},
        }
        (self.dir / "config.json").write_text(json.dumps(self.cfg))
        self.addCleanup(self.cleanup_worker)

    def cli(self, command, *args):
        return subprocess.run(
            [BINARY, command, "--backend", "native", "--name", "integration", *args],
            env=self.env, text=True, capture_output=True, timeout=20,
        )

    def cleanup_worker(self):
        if self.dir.exists():
            self.cli("stop")

    def test_start_smoke_stop_restart_uninstall(self):
        result = self.cli("start")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        status = self.cli("status")
        self.assertEqual(json.loads(status.stdout)["state"], "running")
        smoke = self.cli("smoke")
        self.assertEqual(smoke.returncode, 0, smoke.stderr)
        self.assertEqual(json.loads(smoke.stdout)["choices"][0]["text"], "Paris")
        duplicate = self.cli("start")
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertEqual(self.cli("stop").returncode, 0)
        self.assertEqual(json.loads(self.cli("status").stdout)["state"], "stopped")
        self.assertEqual(self.cli("start").returncode, 0)
        refused = self.cli("uninstall")
        self.assertNotEqual(refused.returncode, 0)
        removed = self.cli("uninstall", "--yes")
        self.assertEqual(removed.returncode, 0, removed.stderr)
        self.assertFalse(self.dir.exists())

    def test_failed_worker_is_retained_for_logs(self):
        (self.dir / "native_entry.py").write_text("raise RuntimeError('fixture startup failure')\n")
        result = self.cli("start")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.dir / "config.json").exists())
        self.assertIn("fixture startup failure", (self.dir / "service.log").read_text())

    def test_occupied_port_fails_without_launching_worker(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", self.port))
            sock.listen()
            result = self.cli("start")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("port unavailable", result.stderr)
            self.assertFalse((self.dir / "run.json").exists())
