"""HTTP contract tests with a fake GPU pipeline; no model download or CUDA needed."""
from contextlib import nullcontext
import base64
import importlib.util
import io
import os
from unittest.mock import patch
from pathlib import Path
import sys
import types
import unittest

from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CALLS = []


class FakePipeline:
    def __init__(self):
        self.vae = types.SimpleNamespace(enable_slicing=lambda: None, enable_tiling=lambda: None)

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        CALLS.append(("load", model, kwargs))
        return cls()

    def to(self, device):
        return self

    def set_progress_bar_config(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        CALLS.append(("generate", kwargs))
        return types.SimpleNamespace(images=[Image.new("RGB", (32, 32))])


class Generator:
    def __init__(self, **kwargs):
        CALLS.append(("generator", kwargs))

    def manual_seed(self, seed):
        return self


class OOM(Exception):
    pass


sys.modules["torch"] = types.SimpleNamespace(
    float16="fp16", bfloat16="bf16", float32="fp32", Generator=Generator,
    inference_mode=nullcontext, OutOfMemoryError=OOM,
    cuda=types.SimpleNamespace(empty_cache=lambda: None),
)
sys.modules["diffusers"] = types.SimpleNamespace(
    AutoPipelineForImage2Image=FakePipeline, AutoPipelineForText2Image=FakePipeline,
)
spec = importlib.util.spec_from_file_location("modelship_image_server", ROOT / "internal/modelship/assets/server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class ImageAPITests(unittest.TestCase):
    def setUp(self):
        CALLS.clear()
        server.MODEL = "org/model"
        server.REVISION = "a" * 40
        server.TASK = "text-to-image"
        server.PIPELINE = "StableDiffusionXLPipeline"
        self.env = patch.dict(os.environ, {"MODEL_COMPONENT_DTYPES": "{}"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_generation_and_loading_contract(self):
        with TestClient(server.app) as client:
            self.assertTrue(client.get("/health").json()["ready"])
            response = client.post("/v1/images/generations", json={"prompt": "a mountain", "seed": 42})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["seed"], 42)
            image = base64.b64decode(response.json()["data"][0]["b64_json"])
            self.assertTrue(image.startswith(b"\x89PNG"))
        load = CALLS[0][2]
        self.assertEqual(load["revision"], "a" * 40)
        self.assertTrue(load["use_safetensors"])
        self.assertFalse(load["trust_remote_code"])

    def test_rejects_bad_fields_and_task(self):
        with TestClient(server.app) as client:
            for body in (
                {"prompt": ""},
                {"prompt": "x", "width": 513},
                {"prompt": "x", "steps": 101},
                {"prompt": "x", "image_url": "http://169.254.169.254"},
                {"prompt": "x", "image_b64": "anything"},
            ):
                response = client.post("/v1/images/generations", json=body)
                self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(client.post("/v1/images/edits", json={"prompt": "x"}).status_code, 409)

    def test_image_to_image_resizes_and_requires_base64(self):
        server.TASK = "image-to-image"
        image = io.BytesIO()
        Image.new("RGB", (64, 64)).save(image, "PNG")
        encoded = base64.b64encode(image.getvalue()).decode()
        with TestClient(server.app) as client:
            for value in (None, "https://example.com/image.png", "invalid base64"):
                body = {"prompt": "paint it", "image_b64": value}
                self.assertEqual(client.post("/v1/images/edits", json=body).status_code, 422)
            response = client.post("/v1/images/edits", json={"prompt": "paint it", "image_b64": encoded})
            self.assertEqual(response.status_code, 200, response.text)
        args = next(call[1] for call in CALLS if call[0] == "generate")
        self.assertEqual(args["image"].size, (512, 512))
        self.assertNotIn("width", args)

    def test_gpu_busy(self):
        with TestClient(server.app) as client:
            server.BUSY.acquire()
            try:
                response = client.post("/v1/images/generations", json={"prompt": "x"})
                self.assertEqual(response.status_code, 503)
            finally:
                server.BUSY.release()

    def test_flux_negative_prompt_rejected(self):
        server.PIPELINE = "FluxPipeline"
        with TestClient(server.app) as client:
            response = client.post("/v1/images/generations", json={"prompt": "x", "negative_prompt": "y"})
            self.assertEqual(response.status_code, 422)
        self.assertEqual(CALLS[0][2]["torch_dtype"], "bf16")

    def test_qwen_image_license_loader_and_guidance(self):
        server.PIPELINE = "QwenImage21Pipeline"
        fake = types.SimpleNamespace(QwenImage21Pipeline=FakePipeline)
        with patch.dict(sys.modules, {"diffusers": fake}):
            with self.assertRaisesRegex(RuntimeError, "--research"):
                server.load_pipeline()
            with patch.dict(os.environ, {"MODEL_RESEARCH": "true"}):
                with TestClient(server.app) as client:
                    result = client.post("/v1/images/generations", json={"prompt": "a mountain"})
                    self.assertEqual(result.status_code, 200, result.text)
                    for body in ({"prompt": "x", "width": 528},
                                 {"prompt": "x", "strength": 0.7},
                                 {"prompt": "x", "negative_prompt": "clouds"}):
                        self.assertEqual(client.post("/v1/images/generations", json=body).status_code, 422)
                    self.assertEqual(client.post("/v1/images/generations", json={
                        "prompt": "x", "guidance_scale": 2, "negative_prompt": "clouds"
                    }).status_code, 200)
        load = CALLS[0][2]
        self.assertEqual(load["torch_dtype"], {"default": "bf16", "vae": "fp32"})
        self.assertFalse(load["trust_remote_code"])
        args = next(c[1] for c in CALLS if c[0] == "generate")
        self.assertEqual(args["true_cfg_scale"], 1)
        self.assertNotIn("guidance_scale", args)
        self.assertNotIn("negative_prompt", args)

    def test_qwen_image_editing_uses_conditioning_without_strength(self):
        server.PIPELINE = "QwenImage21Pipeline"
        server.TASK = "image-to-image"
        image = io.BytesIO()
        Image.new("RGB", (64, 64)).save(image, "PNG")
        encoded = base64.b64encode(image.getvalue()).decode()
        with patch.dict(sys.modules, {"diffusers": types.SimpleNamespace(QwenImage21Pipeline=FakePipeline)}), patch.dict(os.environ, {"MODEL_RESEARCH": "true"}):
            with TestClient(server.app) as client:
                self.assertEqual(client.post("/v1/images/edits", json={"prompt": "x"}).status_code, 422)
                result = client.post("/v1/images/edits", json={"prompt": "paint it", "image_b64": encoded})
                self.assertEqual(result.status_code, 200, result.text)
        args = next(c[1] for c in CALLS if c[0] == "generate")
        self.assertEqual(args["image"].size, (512, 512))
        self.assertEqual(args["width"], 512)
        self.assertNotIn("strength", args)

    def test_moss_loader_dtypes_cpu_noise_and_prompt_bound(self):
        server.PIPELINE = "MOSSImagePipeline"
        def load_moss(model, revision, dtypes):
            CALLS.append(("moss-load", model, revision, dtypes))
            return FakePipeline()
        with patch.dict(sys.modules, {"model_code": types.SimpleNamespace(load_moss_pipeline=load_moss)}):
            with TestClient(server.app) as client:
                self.assertEqual(client.post("/v1/images/generations", json={"prompt": "x", "width": 520}).status_code, 422)
                result = client.post("/v1/images/generations", json={"prompt": "a mountain"})
                self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(CALLS[0][3], {"default": "bf16", "vae": "fp32"})
        self.assertIn(("generator", {"device": "cpu"}), CALLS)
        args = next(c[1] for c in CALLS if c[0] == "generate")
        self.assertEqual(args["guidance_scale"], 3)
        self.assertEqual(args["max_sequence_length"], 512)

    def test_body_limit(self):
        with TestClient(server.app) as client:
            response = client.post("/v1/images/generations", content=b"x" * (server.MAX_BODY + 1))
            self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
