"""Offline release-site retention and immutable-version regression checks."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseSiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('scripts', 'releases', 'archive', 'tools'):
            (self.root / name).mkdir()
        shutil.copyfile(ROOT / 'scripts/build-release-site.sh', self.root / 'scripts/build-release-site.sh')
        (self.root / 'releases/current').write_text('v0.3.0-alpha.1\n')
        (self.root / 'scripts/release.sh').write_text('''#!/bin/sh
set -eu
mkdir -p dist
printf '%s' "$VERSION" > dist/modelship_linux_amd64
printf '%s' "installer:$VERSION" > dist/install.sh
(cd dist; if command -v sha256sum >/dev/null 2>&1; then sha256sum modelship_linux_amd64 install.sh; else shasum -a 256 modelship_linux_amd64 install.sh; fi) > dist/SHA256SUMS
''')
        old = self.root / 'archive/v0.2.0-alpha.1'
        old.mkdir()
        manifest = []
        for name, content in [('modelship_linux_amd64', b'original binary'), ('install.sh', b'original installer'), ('SHA256SUMS', b'original checksums')]:
            (old / name).write_bytes(content)
            manifest.append(f'{hashlib.sha256(content).hexdigest()}  v0.2.0-alpha.1/{name}\n')
        (self.root / 'releases/retained.sha256').write_text(''.join(manifest))
        curl = self.root / 'tools/curl'
        curl.write_text('''#!/usr/bin/env python3
import os, pathlib, sys, urllib.parse
args = sys.argv[1:]
path = pathlib.Path(os.environ['FAKE_ARCHIVE']) / urllib.parse.urlparse(args[-1]).path.lstrip('/')
output = pathlib.Path(args[args.index('--output')+1])
output.write_bytes(path.read_bytes() if path.is_file() else b'not found')
if '--write-out' in args: print('200' if path.is_file() else '404', end='')
if '--fail' in args and not path.is_file(): sys.exit(22)
''')
        curl.chmod(0o755)
        self.env = {**os.environ, 'PATH': str(self.root / 'tools') + os.pathsep + os.environ['PATH'], 'FAKE_ARCHIVE': str(self.root / 'archive'), 'RELEASE_ARCHIVE_BASE': 'https://fixture.invalid', 'RELEASE_VERSION': 'v0.2.0-alpha.1'}

    def run_build(self):
        return subprocess.run(['sh', 'scripts/build-release-site.sh'], cwd=self.root, env=self.env, capture_output=True, text=True)

    def test_retains_original_bytes_and_ignores_stale_build_argument(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('deprecated and ignored', result.stderr)
        for old in (self.root / 'archive/v0.2.0-alpha.1').iterdir():
            self.assertEqual(old.read_bytes(), (self.root / 'dist/site/v0.2.0-alpha.1' / old.name).read_bytes())
        self.assertEqual((self.root / 'dist/site/v0.3.0-alpha.1/modelship_linux_amd64').read_text(), 'v0.3.0-alpha.1')
        self.assertEqual((self.root / 'dist/site/latest/install.sh').read_bytes(), (self.root / 'dist/site/v0.3.0-alpha.1/install.sh').read_bytes())
        self.assertIn('v0.3.0-alpha.1', (self.root / 'dist/site/latest/install.sh').read_text())
        self.assertFalse((self.root / 'dist/site/latest/modelship_linux_amd64').exists())
        self.assertFalse((self.root / 'dist/site/latest/SHA256SUMS').exists())

    def test_latest_advances_without_changing_historical_versions(self):
        # An archive's mutable alias is never fetched/trusted as the selection.
        (self.root / 'archive/latest').mkdir()
        (self.root / 'archive/latest/install.sh').write_text('untrusted/stale alias')
        first = self.run_build()
        self.assertEqual(first.returncode, 0, first.stderr)
        previous = self.root / 'dist/site/v0.3.0-alpha.1'
        original = {p.name: p.read_bytes() for p in previous.iterdir()}
        shutil.copytree(previous, self.root / 'archive/v0.3.0-alpha.1')
        with (self.root / 'releases/retained.sha256').open('a') as pins:
            for name, content in original.items():
                pins.write(f'{hashlib.sha256(content).hexdigest()}  v0.3.0-alpha.1/{name}\n')
        shutil.rmtree(self.root / 'dist')
        (self.root / 'releases/current').write_text('v0.3.0-alpha.2\n')
        second = self.run_build()
        self.assertEqual(second.returncode, 0, second.stderr)
        for name, content in original.items():
            self.assertEqual((self.root / 'dist/site/v0.3.0-alpha.1' / name).read_bytes(), content)
        latest = (self.root / 'dist/site/latest/install.sh').read_bytes()
        self.assertEqual(latest, (self.root / 'dist/site/v0.3.0-alpha.2/install.sh').read_bytes())
        self.assertNotEqual(latest, original['install.sh'])
        self.assertNotIn(b'untrusted/stale', latest)

    def test_corrupted_archive_fails_before_site_publication(self):
        (self.root / 'archive/v0.2.0-alpha.1/modelship_linux_amd64').write_bytes(b'changed')
        self.assertNotEqual(self.run_build().returncode, 0)
        self.assertFalse((self.root / 'dist/site').exists())

    def test_existing_version_cannot_be_overwritten(self):
        published = self.root / 'archive/v0.3.0-alpha.1'
        published.mkdir()
        (published / 'SHA256SUMS').write_text('different published binary\n')
        result = self.run_build()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Refusing to overwrite published', result.stderr)
        self.assertFalse((self.root / 'dist/site').exists())

    def test_identical_version_rebuild_is_allowed(self):
        subprocess.run(['sh', 'scripts/release.sh'], cwd=self.root, env={**self.env, 'VERSION': 'v0.3.0-alpha.1'}, check=True)
        published = self.root / 'archive/v0.3.0-alpha.1'
        published.mkdir()
        shutil.copyfile(self.root / 'dist/SHA256SUMS', published / 'SHA256SUMS')
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
