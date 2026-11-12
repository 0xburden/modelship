"""Native ASGI auth and listener guards; no CUDA, weights or network services."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

ASSETS = Path(__file__).resolve().parents[1] / 'internal/modelship/assets'
spec = importlib.util.spec_from_file_location('modelship_api_auth_tests', ASSETS / 'api_auth.py')
auth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auth)
FIXTURE = 'a' * 43


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = self.root / 'api.key'
        self.key.write_text(FIXTURE + '\n')
        self.key.chmod(0o600)
        self.calls = []
        self.app = FastAPI()
        @self.app.api_route('/{path:path}', methods=['GET', 'POST', 'OPTIONS', 'HEAD'])
        async def endpoint(path: str, request: Request):
            self.calls.append((path, await request.body()))
            if path.endswith('/content'):
                return StreamingResponse(iter([b'mp4-fixture']), media_type='video/mp4')
            return {'ready': True}
        self.access = auth.load_access({'MODEL_LISTEN': '0.0.0.0', 'MODEL_API_KEY_FILE': str(self.key)})
        self.client = TestClient(auth.BearerAuth(self.app, self.access))
        self.addCleanup(self.client.close)

    def test_every_path_requires_header_auth_before_routing_or_body(self):
        for path in ['/health', '/v1/models', '/docs', '/redoc', '/openapi.json', '/v1/videos/video-fixture', '/v1/videos/video-fixture/content', '/unknown']:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 401, path)
            self.assertEqual(response.headers['www-authenticate'], 'Bearer realm="modelship"')
            self.assertEqual(response.headers['cache-control'], 'no-store')
        for path in ['/v1/chat/completions', '/v1/completions', '/v1/images/generations', '/v1/images/edits', '/v1/videos/generations']:
            response = self.client.post(path, content='private fixture request')
            self.assertEqual(response.status_code, 401)
            self.assertNotIn('private fixture request', response.text)
        self.assertFalse(self.calls)

    def test_malformed_duplicate_query_and_cookie_credentials_denied(self):
        for value in ['', 'Basic '+FIXTURE, 'Bearer wrong', 'Bearer '+('b'*43), 'Bearer  '+FIXTURE, 'Bearer '+FIXTURE+' extra', 'Bearer '+('a'*1000)]:
            response = self.client.get('/health', headers={'Authorization': value})
            self.assertEqual(response.status_code, 401)
            self.assertNotIn(FIXTURE, response.text)
        response = self.client.get('/health', headers=[('Authorization', 'Bearer '+FIXTURE), ('Authorization', 'Bearer '+FIXTURE)])
        self.assertEqual(response.status_code, 401)
        response = self.client.get('/health?api_key='+FIXTURE+'&token='+FIXTURE, headers={'Cookie': 'token='+FIXTURE})
        self.assertEqual(response.status_code, 401)
        self.assertFalse(self.calls)

    def test_valid_header_allows_json_and_streaming_with_no_store(self):
        for prefix in ['Bearer ', 'bearer ', 'BEARER ']:
            response = self.client.get('/health', headers={'Authorization': prefix+FIXTURE})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['cache-control'], 'no-store')
        response = self.client.get('/v1/videos/video-fixture/content', headers={'Authorization': 'Bearer '+FIXTURE})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'mp4-fixture')
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertNotIn(FIXTURE, response.text)

    def test_rejected_requests_never_receive_body(self):
        sent = []
        async def receive():
            raise AssertionError('unauthorized body was read')
        async def send(message):
            sent.append(message)
        asyncio.run(auth.BearerAuth(self.app, self.access)({'type':'http', 'headers': []}, receive, send))
        self.assertEqual(sent[0]['status'], 401)
        self.assertFalse(self.calls)

    def test_websockets_denied(self):
        sent = []
        async def send(message): sent.append(message)
        asyncio.run(auth.BearerAuth(self.app,self.access)({'type':'websocket'},None,send))
        self.assertEqual(sent, [{'type':'websocket.close', 'code':1008}])

    def test_loopback_legacy_remains_unauthenticated(self):
        access = auth.load_access({})
        self.assertEqual(access.listen, '127.0.0.1')
        self.assertIsNone(access.digest)
        with TestClient(auth.BearerAuth(self.app, access)) as client:
            self.assertEqual(client.get('/health').status_code, 200)

    def test_nonloopback_without_key_and_invalid_listener_fail_closed(self):
        for env in [{'MODEL_LISTEN':'0.0.0.0'}, {'MODEL_LISTEN':'::'}, {'MODEL_LISTEN':'localhost'}, {'MODEL_API_KEY_FILE':'relative'}]:
            with self.assertRaises(RuntimeError): auth.load_access(env)

    def test_bad_key_files_fail_without_echoing_content(self):
        for content in ['short', 'a'*129, 'a'*32+'\n\n', 'a'*32+'\x00']:
            self.key.write_text(content)
            with self.assertRaises(RuntimeError) as error: auth.load_access({'MODEL_API_KEY_FILE':str(self.key)})
            self.assertNotIn(content, str(error.exception))
        self.key.write_text(FIXTURE+'\r\n')
        self.assertIsNotNone(auth.load_access({'MODEL_API_KEY_FILE':str(self.key)}).digest)
        with patch.object(auth.os, 'geteuid', return_value=os.geteuid()+1):
            with self.assertRaises(RuntimeError): auth.load_access({'MODEL_API_KEY_FILE':str(self.key)})
        self.key.chmod(0o644)
        with self.assertRaises(RuntimeError): auth.load_access({'MODEL_API_KEY_FILE':str(self.key)})
        self.key.chmod(0o600)
        link = self.root/'link'
        link.symlink_to(self.key)
        with self.assertRaises(RuntimeError): auth.load_access({'MODEL_API_KEY_FILE':str(link)})
        fifo = self.root/'fifo'
        os.mkfifo(fifo,0o600)
        with self.assertRaises(RuntimeError): auth.load_access({'MODEL_API_KEY_FILE':str(fifo)})

    def test_rotation_requires_restart_not_silent_live_reconfiguration(self):
        self.key.write_text('b'*43)
        response = self.client.get('/health', headers={'Authorization':'Bearer '+FIXTURE})
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/health', headers={'Authorization':'Bearer '+('b'*43)})
        self.assertEqual(response.status_code, 401)
        fresh = auth.load_access({'MODEL_API_KEY_FILE':str(self.key)})
        with TestClient(auth.BearerAuth(self.app,fresh)) as client:
            self.assertEqual(client.get('/health',headers={'Authorization':'Bearer '+('b'*43)}).status_code,200)

    def test_entrypoint_validates_before_inference_import(self):
        for name in ['api_auth.py','native_entry.py']:
            shutil.copyfile(ASSETS/name,self.root/name)
        (self.root/'uvicorn.py').write_text('import json\ndef run(app, **kwargs):\n print(json.dumps({"host":kwargs["host"],"wrapper":type(app).__name__}))\n')
        marker = self.root/'inference-imported'
        (self.root/'text_server.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\napp=object()\n')
        env = {**os.environ, 'MODEL_LISTEN':'0.0.0.0', 'MODEL_API_KEY_FILE':'', 'MODEL_TASK':'text-generation','MODEL_PORT':'18032'}
        cmd = [sys.executable, str(self.root/'native_entry.py'), '--run-id', 'b'*32]
        denied = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=10)
        self.assertNotEqual(denied.returncode,0)
        self.assertFalse(marker.exists(), 'inference imported before auth validation')
        env['MODEL_API_KEY_FILE']=str(self.key)
        allowed = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=10)
        self.assertEqual(allowed.returncode,0,allowed.stderr)
        self.assertEqual(json.loads(allowed.stdout),{'host':'0.0.0.0','wrapper':'BearerAuth'})
        self.assertTrue(marker.exists())
        self.assertNotIn(FIXTURE,allowed.stdout+allowed.stderr)


if __name__ == '__main__':
    unittest.main()
