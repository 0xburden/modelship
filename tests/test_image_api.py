"""HTTP contract tests with a fake GPU pipeline; no model download or CUDA needed."""
from contextlib import nullcontext
import base64
import importlib.util
import io
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
        pass

    def manual_seed(self, seed):
        return self


class OOM(Exception):
    pass


sys.modules["torch"] = types.SimpleNamespace(
    float16="fp16", bfloat16="bf16", Generator=Generator,
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

    def test_body_limit(self):
        with TestClient(server.app) as client:
            response = client.post("/v1/images/generations", content=b"x" * (server.MAX_BODY + 1))
            self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
