import hashlib
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import candidate_smoke as smoke


class SmokeTests(unittest.TestCase):
    def session(self, action):
        # Real pipes, synthetic pixels and statuses, no model or CUDA use.
        script = r'''
import struct, sys, time
stdin, out = sys.stdin.buffer, sys.stdout.buffer
def send(payload):
    out.write(struct.pack('!I', len(payload)) + payload)
    out.flush()
action = sys.argv[1]
send(b'L\x01')
length = struct.unpack('!I', stdin.read(4))[0]
request = stdin.read(length)
assert request.startswith(b'\x00\x00\x00\x2bA watercolor') and stdin.read() == b''
if action == 'valid':
    send(b'I\x03' + b'\x01\x02\x03' * (512 * 512))
elif action == 'too-large':
    out.write(b'\xff\xff\xff\xff'); out.flush()
elif action == 'failed':
    send(b'F')
elif action == 'bad-channels':
    send(b'I\x01' + b'\xff' * (512 * 512))
elif action == 'timeout':
    time.sleep(20)
'''
        return subprocess.Popen([sys.executable, "-c", script, action], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def test_clean_single_generation_and_decodable_png(self):
        with self.session("valid") as child:
            channels, pixels, elapsed = smoke._generate_once(child, 5, 5)
            self.assertEqual((channels, len(pixels), pixels[:3]), (3, 512 * 512 * 3, b"\x01\x02\x03"))
            self.assertEqual(child.stdout.read(1), b"")
            self.assertEqual(child.wait(2), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.png"
            smoke._write_png(path, channels, pixels)
            raw = path.read_bytes()
            self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack("!2I", raw[16:24]), (512, 512))
            length = struct.unpack("!I", raw[33:37])[0]
            self.assertEqual(zlib.decompress(raw[41:41 + length]), (b"\0" + b"\1\2\3" * 512) * 512)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                smoke._write_png(path, channels, pixels)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_bounded_response_failure_and_timeout(self):
        for action, error in (("too-large", "frame length"), ("failed", "drained"),
                              ("bad-channels", "drained"), ("timeout", "timed out")):
            with self.subTest(action=action), self.session(action) as child:
                try:
                    with self.assertRaisesRegex(smoke.PreflightError, error):
                        smoke._generate_once(child, 5, 0.2 if action == "timeout" else 5)
                finally:
                    smoke._stop_child(child)
                self.assertIsNotNone(child.returncode)

    @unittest.skipUnless(sys.platform == "linux" and platform.machine() == "x86_64" and shutil.which("cc"),
                         "requires a local Linux AMD64 C compiler")
    def test_launches_verified_elf_and_writes_only_one_image(self):
        source = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    if (argc != 3 || strcmp(argv[1], "--private-stdio") || !getenv("CUDA_VISIBLE_DEVICES") ||
        getenv("LD_LIBRARY_PATH")) return 1;
    const unsigned char loaded[] = {0,0,0,2,'L',1};
    if (fwrite(loaded, 1, sizeof(loaded), stdout) != sizeof(loaded) || fflush(stdout)) return 2;
    unsigned char len[4], payload[1028];
    if (fread(len, 1, 4, stdin) != 4) return 3;
    const unsigned n = (unsigned)len[0]<<24 | (unsigned)len[1]<<16 | (unsigned)len[2]<<8 | len[3];
    if (n < 5 || n > sizeof(payload) || fread(payload, 1, n, stdin) != n || payload[3] != 43 ||
        fgetc(stdin) != EOF) return 4;
    const unsigned char header[] = {0,12,0,2,'I',3};
    static const unsigned char pixels[3 * 512 * 512] = {1,2,3};
    return fwrite(header, 1, sizeof(header), stdout) != sizeof(header) ||
           fwrite(pixels, 1, sizeof(pixels), stdout) != sizeof(pixels) || fflush(stdout);
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            worker = root / "worker"
            subprocess.run(["cc", "-x", "c", "-o", worker, "-"], input=source, text=True, check=True)
            worker.chmod(0o500)
            out = root / "image.png"
            args = SimpleNamespace(worker=worker, worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
                                   bundle=root, gpu="GPU-01234567-1234-abcd-1234-0123456789ab", output=out)
            if os.geteuid() == 0:
                self.skipTest("use an unprivileged account for the launch integration")
            with patch.object(Path, "home", return_value=root), \
                    patch.object(smoke, "verify_generation_bundle") as verify, \
                    patch.object(smoke, "admit_smoke_gpu", return_value=42000) as admit:
                smoke.smoke(args)
                verify.assert_called_once_with(root)
                admit.assert_called_once_with(args.gpu)
            self.assertEqual(out.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_executable_hash_layout_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp).resolve()
            path = directory / "worker"
            raw = b"\x7fELF\x02\x01\x01" + b"\x00" * 9 + b"\x03\x00\x3e\x00" + b"test"
            path.write_bytes(raw)
            path.chmod(0o500)
            digest = hashlib.sha256(raw).hexdigest()
            with smoke._verified_runtime(path, digest) as fd:
                self.assertEqual(os.pread(fd, len(raw), 0), raw)
            with self.assertRaisesRegex(smoke.PreflightError, "SHA-256"):
                with smoke._verified_runtime(path, "0" * 64):
                    self.fail("accepted the wrong digest")
            path.chmod(0o700)
            with self.assertRaisesRegex(smoke.PreflightError, "read-only"):
                with smoke._verified_runtime(path, digest):
                    self.fail("accepted a writable executable")
            gpu = "GPU-01234567-1234-abcd-1234-0123456789ab"
            with smoke._gpu_lock(directory, gpu):
                with self.assertRaisesRegex(smoke.PreflightError, "already owns"):
                    with smoke._gpu_lock(directory, gpu):
                        self.fail("double-acquired a worker lock")
            with smoke._gpu_lock(directory, gpu):
                pass


if __name__ == "__main__":
    unittest.main()
