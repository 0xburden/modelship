import hashlib
import subprocess
import sys
import unittest
from unittest.mock import patch

import candidate_edit_smoke as smoke
import candidate_preflight as preflight


class EditSmokeTests(unittest.TestCase):
    def session(self, mode):
        script = r'''
import hashlib, struct, sys, time
stdin, out = sys.stdin.buffer, sys.stdout.buffer
mode = sys.argv[1]
def send(payload):
    out.write(struct.pack('!I', len(payload)) + payload)
    out.flush()
send(b'L\x01' if mode == 'generation' else b'L\x02')
if mode == 'stalled-input':
    time.sleep(10)
    sys.exit(1)
if mode == 'generation':
    assert stdin.read() == b''
    sys.exit(0)
length = struct.unpack('!I', stdin.read(4))[0]
payload = stdin.read(length)
seed, size = struct.unpack('!IH', payload[:6])
assert seed == 43 and payload[6:6+size].startswith(b'Change the red mug')
assert hashlib.sha256(payload[6+size:]).hexdigest() == sys.argv[2] and stdin.read() == b''
if mode == 'valid':
    send(b'I\x04' + b'\x01\x02\x03\xff' * (512 * 512))
else:
    send(b'F')
'''
        return subprocess.Popen([sys.executable, "-c", script, mode, self.digest], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    @classmethod
    def setUpClass(cls):
        cls.rgb = smoke.reference_pixels()
        cls.digest = hashlib.sha256(cls.rgb).hexdigest()

    def test_fixed_image_and_exact_edit_projector(self):
        self.assertEqual(len(self.rgb), 3 * 512 * 512)
        self.assertEqual(self.digest, "49812638e1e498eaff84ba18736947cea6428137a7664149137119ad290949c2")
        self.assertEqual(sum(item.bytes for item in preflight.EDIT_BUNDLE), 11059819128)
        with patch.object(preflight, "_verify_components") as verify:
            preflight.verify_edit_bundle("bundle")
            verify.assert_called_once_with("bundle", preflight.EDIT_BUNDLE)
        self.assertEqual(preflight.EDIT_BUNDLE[-1].name, "mmproj-Qwen3VL-8B-Instruct-F16.gguf")

    def test_edit_inventory_fails_closed(self):
        uuid = "GPU-01234567-1234-abcd-1234-0123456789ab"
        for value in (b"<processes/>", b"", b"<processes>N/A</processes>",
                      b"<processes><process_info><type>G</type></process_info></processes>"):
            xml = b"<nvidia_smi_log><gpu><uuid>" + uuid.encode() + b"</uuid>" + value + b"</gpu></nvidia_smi_log>"
            with self.subTest(value=value), patch.object(preflight, "admit_smoke_gpu", return_value=42000), \
                    patch.object(preflight.subprocess, "check_output", return_value=xml):
                if value == b"<processes/>":
                    self.assertEqual(preflight.admit_edit_smoke_gpu(uuid), 42000)
                else:
                    with self.assertRaisesRegex(smoke.PreflightError, "empty graphics and compute"):
                        preflight.admit_edit_smoke_gpu(uuid)

    def test_real_pipes_single_request_and_rgba_result(self):
        with self.session("valid") as child:
            channels, pixels, _ = smoke._edit_once(child, self.rgb, 5, 5)
            self.assertEqual((channels, pixels[:4], len(pixels)), (4, b"\1\2\3\xff", 4 * 512 * 512))
            self.assertEqual(child.wait(timeout=5), 0)
            self.assertEqual(child.stdout.read(1), b"")

    def test_generation_handshake_failure_and_large_write_deadline(self):
        for mode, message in (("generation", "did not load"), ("failed", "did not return"),
                              ("stalled-input", "request timed out")):
            with self.subTest(mode=mode), self.session(mode) as child:
                try:
                    with self.assertRaisesRegex(smoke.PreflightError, message):
                        smoke._edit_once(child, self.rgb, 5, 0.2 if mode == "stalled-input" else 5)
                finally:
                    smoke._stop_child(child)


if __name__ == "__main__":
    unittest.main()
