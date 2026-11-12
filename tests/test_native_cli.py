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
import urllib.request
import urllib.error

BINARY = os.environ.get("MODELSHIP_TEST_BINARY", "")
FAKE_WORKER = '''import argparse
import json
import os
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
p = argparse.ArgumentParser()
p.add_argument("--run-id", required=True)
args = p.parse_args()
key_path = os.environ.get("MODEL_API_KEY_FILE", "")
key = open(key_path).read().strip() if key_path else ""
class Handler(BaseHTTPRequestHandler):
    def authorized(self):
        if key and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + key):
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'authentication required')
            return False
        return True
    def do_GET(self):
        if not self.authorized(): return
        data = {"ready": True, "model": os.environ["MODEL_ID"], "revision": os.environ["MODEL_REVISION"], "run_id": args.run_id}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    def do_POST(self):
        if not self.authorized(): return
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"choices":[{"text":"Paris"}]}')
    def log_message(self, *args):
        pass
HTTPServer((os.environ.get("MODEL_LISTEN", "127.0.0.1"), int(os.environ["MODEL_PORT"])), Handler).serve_forever()
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

    def test_authenticated_listener_readiness_smoke_and_missing_key_stop(self):
        key_path = self.root / 'private' / 'api.key'
        generated = subprocess.run([BINARY, 'auth-keygen', '--file', str(key_path)], env=self.env, text=True, capture_output=True, timeout=10)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        key = key_path.read_text().strip()
        self.assertNotIn(key, generated.stdout + generated.stderr)
        self.cfg['schema'] = 4
        self.cfg['plan'].update(listen='0.0.0.0', api_key_file=str(key_path))
        (self.dir/'config.json').write_text(json.dumps(self.cfg))
        started = self.cli('start')
        self.assertEqual(started.returncode, 0, started.stderr)
        self.assertNotIn(key, started.stdout+started.stderr)
        url = 'http://127.0.0.1:'+str(self.port)+'/health'
        with self.assertRaises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(url, timeout=5)
        self.assertEqual(denied.exception.code, 401)
        request = urllib.request.Request(url, headers={'Authorization':'Bearer '+key})
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertTrue(json.load(response)['ready'])
        status = self.cli('status')
        self.assertTrue(json.loads(status.stdout)['ready'])
        self.assertTrue(json.loads(status.stdout)['auth_enabled'])
        self.assertNotIn(key,status.stdout)
        smoke = self.cli('smoke')
        self.assertEqual(smoke.returncode,0,smoke.stderr)
        self.assertNotIn(key,smoke.stdout+smoke.stderr)
        key_path.unlink()
        status = json.loads(self.cli('status').stdout)
        self.assertEqual(status['state'],'running')
        self.assertFalse(status['ready'])
        self.assertEqual(self.cli('stop').returncode,0)
        self.assertNotEqual(self.cli('start').returncode,0)
        self.assertNotIn(key,(self.dir/'service.log').read_text())

    def test_configure_refuses_running_then_migrates_stopped_without_inference_changes(self):
        self.assertEqual(self.cli('start').returncode,0)
        key_path = self.root/'private'/'api.key'
        generated = subprocess.run([BINARY,'auth-keygen','--file',str(key_path)],env=self.env,text=True,capture_output=True,timeout=10)
        self.assertEqual(generated.returncode,0,generated.stderr)
        (self.dir/'server.py').write_text('preserved original inference')
        # A tiny original inference module; real entrypoint/auth/uvicorn after
        # migration, but no CUDA, Transformers or model downloads.
        (self.dir/'text_server.py').write_text("import os\nfrom fastapi import FastAPI\napp=FastAPI()\n@app.get('/health')\ndef health():\n return {'ready':True,'model':os.environ['MODEL_ID'],'revision':os.environ['MODEL_REVISION'],'run_id':os.environ['MODEL_RUN_ID']}\n@app.post('/v1/completions')\ndef completions():\n return {'choices':[{'text':'Paris'}]}\n")
        original_text_module = (self.dir/'text_server.py').read_bytes()
        denied = self.cli('configure','--listen','0.0.0.0','--api-key-file',str(key_path))
        self.assertNotEqual(denied.returncode,0)
        self.assertEqual(json.loads(self.cli('status').stdout)['state'],'running')
        self.assertEqual(self.cli('stop').returncode,0)
        configured = self.cli('configure','--listen','0.0.0.0','--api-key-file',str(key_path))
        self.assertEqual(configured.returncode,0,configured.stderr)
        cfg=json.loads((self.dir/'config.json').read_text())
        self.assertEqual(cfg['schema'],4)
        self.assertEqual(cfg['plan']['listen'],'0.0.0.0')
        self.assertEqual(cfg['plan']['api_key_file'],str(key_path))
        self.assertEqual((self.dir/'server.py').read_text(),'preserved original inference')
        self.assertTrue((self.dir/'api_auth.py').is_file())
        self.assertEqual((self.dir/'text_server.py').read_bytes(),original_text_module)
        restarted = self.cli('start')
        self.assertEqual(restarted.returncode,0,restarted.stderr)
        self.assertTrue(json.loads(self.cli('status').stdout)['ready'])
        for path in ['/health','/docs','/openapi.json','/v1/videos/example/content']:
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen('http://127.0.0.1:'+str(self.port)+path,timeout=5)
            self.assertEqual(error.exception.code,401)
        self.assertEqual(self.cli('smoke').returncode,0)
        self.assertEqual(self.cli('stop').returncode,0)
        self.assertNotIn(key_path.read_text().strip(),(self.dir/'service.log').read_text())
