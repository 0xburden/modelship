"""MOSS policy tests. No upstream Python is executed and no weights are fetched."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("modelship_code_policy", ROOT / "internal/modelship/assets/model_code.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class ModelCodeTests(unittest.TestCase):
    def test_no_implicit_research_to_code_consent(self):
        with patch.dict(os.environ, {"MODEL_TRUST_CODE": "false", "MODEL_RESEARCH": "true"}):
            with self.assertRaisesRegex(RuntimeError, "--trust-model-code"):
                policy.load_moss_pipeline(policy.EXPECTED_REPO, policy.EXPECTED_REVISION, {})

    def test_wrong_repo_revision_and_manifest_rejected_before_download(self):
        with patch.dict(os.environ, {"MODEL_TRUST_CODE": "true", "MODEL_CODE_HASHES": "{}"}):
            for model, revision in (("other/model", policy.EXPECTED_REVISION),
                                    (policy.EXPECTED_REPO, "main")):
                with self.assertRaisesRegex(RuntimeError, "reviewed immutable"):
                    policy.load_moss_pipeline(model, revision, {})
            with self.assertRaisesRegex(RuntimeError, "manifest"):
                policy.load_moss_pipeline(policy.EXPECTED_REPO, policy.EXPECTED_REVISION, {})

    def test_hash_and_size_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "code.py"
            path.write_bytes(b"# never executed")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                policy.verify_code(path, "0" * 64)
            path.write_bytes(b"x" * (1024 * 1024 + 1))
            with self.assertRaisesRegex(RuntimeError, "size limit"):
                policy.verify_code(path, "0" * 64)

    def test_verified_local_safetensors_only_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = b"# reviewed fixture, never imported\n"
            digest = hashlib.sha256(code).hexdigest()
            for name in policy.CODE_PATHS:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(code)
            index = {"_class_name": "MOSSImagePipeline", **policy.COMPONENTS}
            (root / "model_index.json").write_text(json.dumps(index))
            calls = []
            def download(model, name, revision):
                calls.append(("code", name))
                return str(root / name)
            def snapshot(**kwargs):
                calls.append(("snapshot", kwargs))
                return str(root)
            class Scheduler:
                config = {}
                @classmethod
                def from_config(cls, config):
                    return cls()
            def load(path, **kwargs):
                calls.append(("load", kwargs))
                return types.SimpleNamespace(scheduler=Scheduler())
            env = {
                "MODEL_TRUST_CODE": "true", "MODEL_CODE_REVISION": policy.EXPECTED_REVISION,
                "MODEL_CODE_HASHES": json.dumps({n: digest for n in policy.CODE_PATHS}),
                "MODEL_WEIGHT_FILES": json.dumps(["transformer/diffusion_pytorch_model.safetensors"]),
            }
            modules = {
                "huggingface_hub": types.SimpleNamespace(hf_hub_download=download, snapshot_download=snapshot),
                "diffusers": types.SimpleNamespace(DiffusionPipeline=types.SimpleNamespace(from_pretrained=load)),
            }
            with patch.object(policy, "EXPECTED_HASH", digest), patch.dict(os.environ, env), patch.dict(sys.modules, modules):
                policy.load_moss_pipeline(policy.EXPECTED_REPO, policy.EXPECTED_REVISION, {"default": "bf16", "vae": "fp32"})
                self.assertEqual([c[0] for c in calls], ["code", "code", "code", "snapshot", "load"])
                kwargs = calls[-1][1]
                self.assertTrue(kwargs["local_files_only"])
                self.assertTrue(kwargs["use_safetensors"])
                self.assertTrue(kwargs["trust_remote_code"])
                self.assertTrue(kwargs["custom_pipeline"].endswith("/pipeline_moss_image.py"))
                calls.clear()
                index["scheduler"] = ["unreviewed_code", "Unknown"]
                (root / "model_index.json").write_text(json.dumps(index))
                with self.assertRaisesRegex(RuntimeError, "component manifest"):
                    policy.load_moss_pipeline(policy.EXPECTED_REPO, policy.EXPECTED_REVISION, {})
                self.assertNotIn("load", [c[0] for c in calls])
                env["MODEL_WEIGHT_FILES"] = '["../bad.safetensors"]'
                with patch.dict(os.environ, env):
                    with self.assertRaisesRegex(RuntimeError, "unsafe selected"):
                        policy.load_moss_pipeline(policy.EXPECTED_REPO, policy.EXPECTED_REVISION, {})


if __name__ == "__main__":
    unittest.main()
