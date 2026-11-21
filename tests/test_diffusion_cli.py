"""Actual CLI previews and cancellation are download-free with a synthetic GPU."""
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import unittest

BINARY = os.environ.get('MODELSHIP_TEST_BINARY', '')


@unittest.skipUnless(BINARY and sys.platform.startswith('linux'), 'set MODELSHIP_TEST_BINARY on Linux')
class DiffusionCLITests(unittest.TestCase):
    def test_preview_and_decline_do_not_need_runtime_or_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            smi = root / 'nvidia-smi'
            smi.write_text('#!/bin/sh\nprintf "%s\\n" "0, GPU-00000000-0000-0000-0000-000000000001, RTX A6000, 49140, 49140, 610.43.02, 8.6"\n')
            smi.chmod(0o700)
            key = root / 'fixture.key'
            key.touch(mode=0o600)
            key.write_text(secrets.token_urlsafe(32) + '\n')
            env = dict(PATH=f'{temp}:/usr/bin:/bin', HOME=temp, XDG_STATE_HOME=str(root/'state'),
                       HF_HUB_CACHE=str(root/'cache'), HTTPS_PROXY='http://127.0.0.1:1')
            args = ['leejet/Qwen-Image-2.1-GGUF', '--diffusion', '--backend', 'native',
                    '--revision', '0bca76da6edd8a18211e9717454615c1bbaaf6ed', '--experimental', '--research',
                    '--api-key-file', str(key)]
            def run(action, *extra, input=''):
                return subprocess.run([BINARY, action, *args, *extra], input=input, env=env,
                                      capture_output=True, text=True, timeout=10)
            for task, archive in [('text-to-image', 'generation'), ('image-to-image', 'edit')]:
                preview = run('plan', '--task', task, '--json')
                self.assertEqual(preview.returncode, 0, preview.stderr)
                p = json.loads(preview.stdout)
                self.assertEqual(p['task'], task)
                self.assertIn(f'sd-532d2c1-u2204-cuda12.8-{archive}-r1.tar.gz', str(p['warnings']))
                local = run('plan', '--task', task, '--json', '--diffusion-runtime-dir', str(root/'not-present'))
                self.assertEqual(local.returncode, 0, local.stderr)
                self.assertIn('exact verified local executable pair', str(json.loads(local.stdout)['warnings']))
                cancelled = run('deploy', '--task', task, input='n\n')
                self.assertIn('deployment cancelled', cancelled.stderr)
                self.assertNotIn('Downloading verified CUDA runtime', cancelled.stdout)
            bad = run('plan', '--diffusion-runtime-dir=')
            self.assertIn('must be an absolute canonical path', bad.stderr)
            # Emulated Linux AMD64 runners may create ~/.cache for the translator.
            self.assertEqual(set(root.iterdir()) - {root/'.cache'}, {smi, key})
            self.assertFalse((root/'.cache'/'huggingface').exists())
