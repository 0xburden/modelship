"""Optional real-nginx release contract; set MODELSHIP_RELEASE_SITE_URL for local QA."""
import hashlib
import os
from pathlib import Path
import re
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get('MODELSHIP_RELEASE_SITE_URL', '').rstrip('/')


@unittest.skipUnless(BASE, 'set MODELSHIP_RELEASE_SITE_URL to a built release-site container')
class ReleaseHTTPTests(unittest.TestCase):
    def get(self, path, limit=64*1024, method='GET'):
        request = urllib.request.Request(BASE + path, method=method)
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read(limit + 1)
            self.assertLessEqual(len(body), limit)
            self.assertEqual(response.geturl(), BASE + path, 'unexpected redirect')
            return response.headers, body

    def test_latest_is_noncacheable_version_pinned_installer(self):
        version = (ROOT / 'releases/current').read_text().strip()
        headers, latest = self.get('/latest/install.sh')
        _, versioned = self.get(f'/{version}/install.sh')
        self.assertEqual(latest, versioned)
        self.assertIn('no-store', headers.get('Cache-Control', ''))
        match = re.search(rb"release_default='(https://[^']+)'", latest)
        self.assertIsNotNone(match)
        self.assertTrue(match.group(1).endswith(b'/' + version.encode()))
        self.assertNotIn(b'/latest/', match.group(1))
        head, body = self.get('/latest/install.sh', method='HEAD')
        self.assertIn('no-store', head.get('Cache-Control', ''))
        self.assertEqual(body, b'')

    def test_mutable_binary_checksum_and_unknown_aliases_are_not_exposed(self):
        for path in ('/latest/modelship_linux_amd64', '/latest/SHA256SUMS', '/latest/missing.sh', '/latest/'):
            with self.subTest(path=path):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    self.get(path)
                self.assertEqual(raised.exception.code, 404)
                self.assertIn('no-store', raised.exception.headers.get('Cache-Control', ''))
                raised.exception.close()

    def test_all_retained_assets_and_current_checksums_remain_valid(self):
        for line in (ROOT / 'releases/retained.sha256').read_text().splitlines():
            digest, path = line.split()
            _, body = self.get('/' + path, limit=32*1024*1024)
            self.assertEqual(hashlib.sha256(body).hexdigest(), digest, path)
        version = (ROOT / 'releases/current').read_text().strip()
        _, manifest = self.get(f'/{version}/SHA256SUMS')
        for line in manifest.decode().splitlines():
            digest, name = line.split()
            self.assertIn(name, ('modelship_linux_amd64', 'install.sh'))
            _, body = self.get(f'/{version}/{name}', limit=32*1024*1024)
            self.assertEqual(hashlib.sha256(body).hexdigest(), digest, name)


if __name__ == '__main__':
    unittest.main()
