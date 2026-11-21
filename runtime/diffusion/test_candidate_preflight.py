import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import candidate_preflight as preflight


class PreflightTests(unittest.TestCase):
    def test_bundle_verifies_content_not_just_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            file = root / "fixture.gguf"
            file.write_bytes(b"test")
            file.chmod(0o600)
            item = preflight.Component(file.name, 4, hashlib.sha256(b"test").hexdigest())
            with patch.object(preflight, "GENERATION_BUNDLE", (item,)):
                preflight.verify_generation_bundle(root)
                file.write_bytes(b"fail")
                with self.assertRaisesRegex(preflight.PreflightError, "SHA-256"):
                    preflight.verify_generation_bundle(root)
                file.unlink()
                original = root / "other.gguf"
                original.write_bytes(b"test")
                file.symlink_to(original)
                with self.assertRaises(preflight.PreflightError):
                    preflight.verify_generation_bundle(root)
                file.unlink()
                file.hardlink_to(original)
                with self.assertRaises(preflight.PreflightError):
                    preflight.verify_generation_bundle(root)

    def test_gpu_memory_occupancy_and_identity_are_hard_gates(self):
        gpu = "GPU-01234567-1234-abcd-1234-0123456789ab"
        line = f"{gpu}, 49140, 42000, 8.6, 570.26.03\n"
        self.assertEqual(preflight._admit_gpu(line, "", gpu), 42000)
        for rows, procs, uuid in ((line, "932", gpu), (line * 2, "", gpu),
                                  (line.replace("42000", "40959"), "", gpu),
                                  (line.replace("42000", "490000"), "", gpu),
                                  (line.replace("570.26", "570.25"), "", gpu),
                                  (line.replace("8.6", "7.5"), "", gpu),
                                  (line, "", gpu.replace("ab", "cd")),
                                  (line, "", "0")):
            with self.subTest(rows=rows, procs=procs, uuid=uuid):
                with self.assertRaises(preflight.PreflightError):
                    preflight._admit_gpu(rows, procs, uuid)

    def test_pinned_sizes_match_the_worker(self):
        source = Path(__file__).with_name("worker.cpp").read_text()
        for item in preflight.GENERATION_BUNDLE:
            self.assertIn(item.name, source)
            self.assertIn(str(item.bytes), source)
        self.assertEqual(sum(item.bytes for item in preflight.GENERATION_BUNDLE), 9900789304)


if __name__ == "__main__":
    unittest.main()
