"""Client contract/decoder tests with fake HTTP and codec; not GPU validation."""
import contextlib
from email.message import Message
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import urllib.error
from urllib.parse import urlparse

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('modelship_live_video_client', ROOT/'tests/live_video_check.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
MP4 = b'\0\0\0\x18ftypmp42fixture-local-mp4'


class Response:
    def __init__(self, value, binary=False):
        self.status = 200
        self.headers = Message()
        self.headers['Content-Type'] = 'video/mp4' if binary else 'application/json'
        self.headers['Content-Length'] = str(len(MP4))
        self.data = value if binary else json.dumps(value).encode()
    def read(self, count=-1):
        return self.data if count < 0 else self.data[:count]
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class FakeService:
    def __init__(self, failed=False, external_content=False):
        self.jobs = {}
        self.failed = failed
        self.external_content = external_content
        self.requests = []
    def error(self, request, code):
        raise urllib.error.HTTPError(request.full_url, code, 'fixture', {}, io.BytesIO(b'{"detail":"fixture"}'))
    def open(self, request, timeout):
        self.requests.append(request.full_url)
        parsed = urlparse(request.full_url)
        assert parsed.netloc == '127.0.0.1:18032', 'foreign request attempted'
        path = parsed.path
        if path == '/health':
            return Response({'ready': True, 'task': 'text-to-video', 'model': 'Wan-AI/Wan2.2-T2V-A14B-Diffusers', 'revision': 'a'*40})
        if request.data is not None:
            if len(request.data) > 32*1024:
                self.error(request, 413)
            body = json.loads(request.data)
            if body.get('prompt') == 'busy probe':
                self.error(request, 503)
            if body.get('seed') not in (40, 42):
                self.error(request, 422)
            job_id = 'video-'+str(len(self.jobs)+1).zfill(32)
            job = {'id': job_id, 'status': 'running', 'model': 'Wan-AI/Wan2.2-T2V-A14B-Diffusers'}
            self.jobs[job_id] = {'public': job, 'body': body, 'polls': 0}
            response = Response(job)
            response.status = 202
            return response
        job_id = path.split('/')[3]
        if job_id not in self.jobs:
            self.error(request, 404)
        item = self.jobs[job_id]
        if path.endswith('/content'):
            if item['public']['status'] == 'running':
                self.error(request, 409)
            return Response(MP4, binary=True)
        item['polls'] += 1
        if item['polls'] > 1:
            item['public']['status'] = 'failed' if self.failed else 'succeeded'
            item['public']['content_url'] = ('http://169.254.169.254/secret' if self.external_content else f'/v1/videos/{job_id}/content')
            if self.failed:
                item['public']['error'] = {'code': 'generation_failed', 'message': 'video generation failed'}
        return Response(item['public'])
    def decode(self, path, pix_fmt):
        assert pix_fmt == 'rgb24'
        item = list(self.jobs.values())[0 if Path(path).stem == 'smoke' else 1]['body']
        size = item['width'], item['height']
        yield {'size': size, 'fps': item['fps'], 'duration': item['num_frames']/item['fps'], 'codec': 'h264'}
        for index in range(item['num_frames']):
            image = Image.new('RGB', size, (index*10, 0, 0))
            image.paste((255, 255, 255), (0, 0, size[0]//2, size[1]//2))
            yield image.tobytes()


class LiveVideoClientTests(unittest.TestCase):
    def run_client(self, root, service):
        with patch.object(sys, 'argv', ['live_video_check.py', '--port', '18032', '--output', str(root)]), \
             patch.object(client.urllib.request, 'urlopen', service.open), \
             patch.object(client.time, 'sleep', lambda _: None), \
             patch.dict(sys.modules, {'imageio_ffmpeg': types.SimpleNamespace(read_frames=service.decode)}), \
             contextlib.redirect_stdout(io.StringIO()):
            client.main()
    def test_complete_bounded_mp4_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service = FakeService()
            self.run_client(root, service)
            report = json.loads((root/'video-report.json').read_text())
            self.assertEqual([clip['decoded']['frames'] for clip in report['clips']], [5, 17])
            self.assertEqual([clip['decoded']['width'] for clip in report['clips']], [256, 512])
            self.assertTrue(all(clip['decoded']['max_adjacent_frame_difference'] > 0 for clip in report['clips']))
            self.assertEqual((root/'quality.mp4').read_bytes(), MP4)
            self.assertTrue((root/'quality-frame-016.png').is_file())
            self.assertTrue(all(item['status'] == item['expected'] for item in report['checks']))
    def test_failed_job_not_treated_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(AssertionError):
                self.run_client(root, FakeService(failed=True))
            self.assertFalse((root/'smoke.mp4').exists())
    def test_foreign_content_url_rejected_without_fetching(self):
        with tempfile.TemporaryDirectory() as temp:
            service = FakeService(external_content=True)
            with self.assertRaises(AssertionError):
                self.run_client(Path(temp), service)
            self.assertTrue(all(urlparse(url).netloc == '127.0.0.1:18032' for url in service.requests))


if __name__ == '__main__':
    unittest.main()
