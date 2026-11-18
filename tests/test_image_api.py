"""HTTP contract tests with a fake GPU pipeline; no model download or CUDA needed."""
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import hashlib
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
        self.vae = types.SimpleNamespace(dtype="fp32", enable_slicing=lambda: None, enable_tiling=lambda: None)

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        CALLS.append(("load", model, kwargs))
        return cls()

    def to(self, device):
        return self

    def _encode_vae_image(self, image, generator):
        assert image.dtype == self.vae.dtype, "conditioning image dtype must match VAE"
        CALLS.append(("vae-encode", image.dtype, generator))
        return image

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


def encoded_input():
    image = io.BytesIO()
    Image.new("RGB", (64, 64)).save(image, "PNG")
    return base64.b64encode(image.getvalue()).decode()


class DualImageAPITests(unittest.TestCase):
    GEN = "/v1/images/generations"
    EDIT = "/v1/images/edits"

    def configure(self, *, qwen):
        model, revision, pipeline, recipe, dtype, version, loader, component_dtypes, research = (
            ("Qwen/Qwen-Image-2.1", "d26bb61231c349cf6b7896fa83353113880e1ba3", "QwenImage21Pipeline",
             "qwen-image21", "bfloat16", "0.42.1", "qwen-image21", '{"vae":"float32"}', "true") if qwen else
            ("stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b", "StableDiffusionXLPipeline",
             "sdxl", "float16", "0.35.1", "auto-pipeline", "{}", "false")
        )
        CALLS.clear()
        self.enterContext(patch.multiple(server, MODEL=model, REVISION=revision, PIPELINE=pipeline, TASK=server.DUAL))
        self.enterContext(patch.dict(os.environ, {
            "MODEL_RECIPE": recipe, "MODEL_DTYPE": dtype, "MODEL_COMPONENT_DTYPES": component_dtypes,
            "MODEL_NATIVE_LOADER": loader, "MODEL_RESEARCH": research, "MODEL_RUN_ID": "a" * 32,
        }))

        class Scheduler:
            config = {"fixture": True}

            @classmethod
            def from_config(cls, config):
                assert config == cls.config
                return cls()

        class XL(FakePipeline):
            def __init__(self):
                super().__init__()
                self.unet, self.text_encoder, self.text_encoder_2 = object(), object(), object()
                self.tokenizer, self.tokenizer_2 = object(), object()
                self.scheduler = Scheduler()

            def to(self, device):
                CALLS.append(("move", device))
                return self

        class XLImg2Img(FakePipeline):
            pass

        class AutoEdit:
            @classmethod
            def from_pipe(cls, pipe, *, scheduler):
                edit = XLImg2Img()
                edit.__dict__.update(pipe.__dict__, scheduler=scheduler)
                CALLS.append(("wrap", pipe, edit))
                return edit

        self.enterContext(patch.object(server, "AutoPipelineForText2Image", XL))
        self.enterContext(patch.object(server, "AutoPipelineForImage2Image", AutoEdit))
        self.enterContext(patch.dict(sys.modules, {"diffusers": types.SimpleNamespace(
            __version__=version, QwenImage21Pipeline=FakePipeline,
            EulerDiscreteScheduler=Scheduler, StableDiffusionXLPipeline=XL, StableDiffusionXLImg2ImgPipeline=XLImg2Img,
        )}))
        self.enterContext(patch.dict(sys.modules, {"qwen_image_decode": types.SimpleNamespace(
            install_tiled_decode=lambda vae: CALLS.append(("install-tiled-decode", vae)),
        )}))
        self.enterContext(patch.object(server.torch.cuda, "synchronize", lambda: CALLS.append(("drain",)), create=True))

    def test_shared_routes_and_distinct_request_arguments(self):
        for qwen in (False, True):
            with self.subTest(qwen=qwen):
                self.configure(qwen=qwen)
                with TestClient(server.app) as client:
                    health = client.get("/health").json()
                    self.assertTrue(health["ready"])
                    self.assertEqual(health["task"], server.GENERATE)
                    self.assertEqual(health["image_tasks"], [server.GENERATE, server.EDIT])
                    self.assertEqual(server.PIPE is server.EDIT_PIPE, qwen)
                    self.assertEqual([c[1] for c in CALLS if c[0] == "install-tiled-decode"],
                                     [server.PIPE.vae] if qwen else [])
                    before = {"prompt": "fixture", "seed": 42, "width": 512, "height": 768}
                    for route, body in [(self.GEN, before), (self.EDIT, {**before, "image_b64": encoded_input()}), (self.GEN, before)]:
                        response = client.post(route, json=body)
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual(response.json()["seed"], 42)
                    self.assertEqual(client.post(self.GEN, json={**before, "image_b64": encoded_input()}).status_code, 422)
                    self.assertEqual(client.post(self.EDIT, json=before).status_code, 422)
                loads = [c for c in CALLS if c[0] == "load"]
                self.assertEqual(len(loads), 1)
                self.assertEqual(loads[0][1], server.MODEL)
                self.assertEqual(loads[0][2]["revision"], server.REVISION)
                self.assertTrue(loads[0][2]["use_safetensors"])
                self.assertFalse(loads[0][2]["trust_remote_code"])
                self.assertEqual([c for c in CALLS if c[0] == "move"], [] if qwen else [("move", "cuda")])
                gen, edit, again = [c[1] for c in CALLS if c[0] == "generate"]
                self.assertNotIn("image", gen)
                self.assertNotIn("image", again)
                self.assertEqual(edit["image"].size, (512, 768))
                self.assertEqual(edit["image"].mode, "RGBA" if qwen else "RGB")
                self.assertEqual(edit.get("strength"), None if qwen else 0.7)
                if qwen:
                    self.assertEqual(gen["output_resolution"], 608)
                    self.assertEqual(edit["output_resolution"], 512)
                    self.assertEqual((edit["width"], edit["height"]), (512, 768))
                    self.assertEqual(again["output_resolution"], gen["output_resolution"])
                else:
                    self.assertNotIn("width", edit)
                    _, pipe, wrapper = next(c for c in CALLS if c[0] == "wrap")
                    self.assertIs(pipe.vae, wrapper.vae)
                    self.assertIs(pipe.unet, wrapper.unet)
                    self.assertIsNot(pipe.scheduler, wrapper.scheduler)
                self.assertEqual(sum(c == ("drain",) for c in CALLS), 3)

    def test_qwen_edit_reference_budget_is_independent_of_output_size(self):
        self.configure(qwen=True)
        body = {"prompt": "fixture", "image_b64": encoded_input(), "seed": 8,
                "steps": 40, "guidance_scale": 1.0}
        with TestClient(server.app) as client:
            for width, height, reference in [(1024, 1024, 512), (256, 1024, 512), (512, 512, 512), (256, 512, 352)]:
                response = client.post(self.EDIT, json={**body, "width": width, "height": height})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["seed"], 8)
                args = [c[1] for c in CALLS if c[0] == "generate"][-1]
                self.assertEqual((args["width"], args["height"]), (width, height))
                self.assertEqual(args["output_resolution"], reference)
                self.assertEqual(args["image"].size, (width, height))
                self.assertEqual(args["true_cfg_scale"], 1.0)
                self.assertEqual(args["num_inference_steps"], 40)
        # Boot actual edit-only lifespans, not just a dual worker with a
        # changed task label. An unrelated revision keeps its prior behavior.
        for revision, reference in [(server.REVISION, 512), ("f" * 40, 1024)]:
            with patch.multiple(server, TASK=server.EDIT, REVISION=revision), TestClient(server.app) as client:
                self.assertTrue(client.get("/health").json()["ready"])
                self.assertIsNone(server.EDIT_PIPE)
                response = client.post(self.EDIT, json={**body, "width": 1024, "height": 1024})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual([c[1] for c in CALLS if c[0] == "generate"][-1]["output_resolution"], reference)

    def test_qwen_decode_profile_only_on_the_pinned_native_worker(self):
        self.configure(qwen=True)
        for task in (server.GENERATE, server.EDIT, server.DUAL):
            with patch.object(server, "TASK", task), TestClient(server.app):
                self.assertEqual([c[1] for c in CALLS if c[0] == "install-tiled-decode"], [server.PIPE.vae])
            CALLS.clear()
        with patch.object(server, "TASK", server.GENERATE):
            with patch.object(server, "REVISION", "f" * 40), TestClient(server.app):
                self.assertFalse(any(c[0] == "install-tiled-decode" for c in CALLS))
            with patch.dict(os.environ, {"MODEL_NATIVE_LOADER": "auto-pipeline"}), TestClient(server.app):
                self.assertFalse(any(c[0] == "install-tiled-decode" for c in CALLS))

    def test_reviewed_lora_contract_and_scope(self):
        self.configure(qwen=True)
        self.enterContext(patch.dict(os.environ, {"MODEL_LORA_ENABLED": "1"}))
        lora_spec = importlib.util.spec_from_file_location("qwen_lora", ROOT / "internal/modelship/assets/qwen_lora.py")
        lora = importlib.util.module_from_spec(lora_spec)
        lora_spec.loader.exec_module(lora)
        profile = lora.adapter_profile()
        adapter = lora.ImageAdapter(profile)
        lora.install_adapter = lambda transformer, spec: adapter
        self.enterContext(patch.dict(sys.modules, {"qwen_lora": lora}))
        self.enterContext(patch.object(FakePipeline, "transformer", object(), create=True))
        original = FakePipeline.__call__
        seen = []

        def invoke(pipe, **kwargs):
            seen.append(adapter.strength)
            self.assertNotIn("lora_strength", kwargs)
            self.assertNotIn("expected_run_id", kwargs)
            if kwargs["prompt"] == "fail":
                raise RuntimeError("fixture failure")
            return original(pipe, **kwargs)

        self.enterContext(patch.object(FakePipeline, "__call__", invoke))
        body = {"prompt": "fixture", "expected_run_id": "a" * 32}
        with TestClient(server.app) as client:
            self.assertEqual(client.get("/health").json()["lora_revision"], profile["revision"])
            caps = client.get("/v1/capabilities").json()
            self.assertEqual(caps["image_lora"]["tasks"], [server.GENERATE, server.EDIT])
            for bad in (True, "0.8", None, -0.1, 1.1):
                self.assertEqual(client.post(self.GEN, json={**body, "lora_strength": bad}).status_code, 422)
            self.assertEqual(client.post(self.GEN, json=body).status_code, 422)
            self.assertEqual(client.post(self.GEN, json={"prompt": "fixture", "lora_strength": 0}).status_code, 422)
            self.assertEqual(client.post(self.GEN, json={**body, "lora_strength": 0, "expected_run_id": "b" * 32}).status_code, 409)
            self.assertEqual(seen, [])
            self.assertEqual(client.post(self.GEN, json={**body, "lora_strength": 0.75}).status_code, 200)
            self.assertEqual(client.post(self.EDIT, json={**body, "lora_strength": 1, "image_b64": encoded_input()}).status_code, 200)
            self.assertEqual(client.post(self.GEN, json={**body, "lora_strength": 0}).status_code, 200)
            self.assertEqual(client.post(self.GEN, json={"prompt": "fixture"}).status_code, 200)
            self.assertEqual(seen, [0.75, 1, 0, 0])
            self.assertEqual(client.post(self.GEN, json={**body, "lora_strength": .5, "prompt": "fail"}).status_code, 500)
            self.assertEqual(adapter.strength, 0)
            self.assertFalse(client.get("/health").json()["ready"])

    def test_legacy_worker_refuses_explicit_lora_even_zero(self):
        self.configure(qwen=True)
        with TestClient(server.app) as client:
            self.assertEqual(client.get("/v1/capabilities").status_code, 404)
            self.assertEqual(client.post(self.GEN, json={"prompt": "fixture", "lora_strength": 0,
                                                        "expected_run_id": "a" * 32}).status_code, 409)
            self.assertFalse(any(c[0] == "generate" for c in CALLS))

    def test_cross_route_busy_through_drain_in_both_directions(self):
        self.configure(qwen=True)
        for first, second in ((self.GEN, self.EDIT), (self.EDIT, self.GEN)):
            draining, release = Event(), Event()

            def synchronize():
                draining.set()
                if not release.wait(3):
                    raise AssertionError("fixture failed to release drain")

            def request(client, route):
                return client.post(route, json={"prompt": "fixture", **({"image_b64": encoded_input()} if route == self.EDIT else {})})

            with patch.object(server.torch.cuda, "synchronize", synchronize), TestClient(server.app) as client, ThreadPoolExecutor(1) as pool:
                pending = pool.submit(request, client, first)
                try:
                    self.assertTrue(draining.wait(2))
                    self.assertEqual(request(client, second).status_code, 503)
                finally:
                    release.set()
                self.assertEqual(pending.result(timeout=2).status_code, 200)

    def test_inference_or_drain_failure_disables_both_routes(self):
        self.configure(qwen=False)
        for error_at in ("pipeline", "drain"):
            with self.subTest(error_at=error_at):
                def fail(*args, **kwargs):
                    raise OOM("private fixture details")

                with TestClient(server.app) as client, (
                    patch.object(FakePipeline, "__call__", fail) if error_at == "pipeline" else
                    patch.object(server.torch.cuda, "synchronize", fail)
                ):
                    response = client.post(self.GEN, json={"prompt": "fixture"})
                    self.assertEqual(response.status_code, 503, response.text)
                    self.assertNotIn("private fixture details", response.text)
                    health = client.get("/health").json()
                    self.assertFalse(health["ready"])
                    self.assertNotIn("image_tasks", health)
                    self.assertEqual(client.post(self.EDIT, json={"prompt": "fixture", "image_b64": encoded_input()}).status_code, 503)

    def test_reserved_qwen_markers_do_not_poison_worker(self):
        self.configure(qwen=True)
        edit = {"prompt": "fixture", "image_b64": encoded_input()}
        with TestClient(server.app) as client:
            for field in ("prompt", "negative_prompt"):
                response = client.post(self.EDIT, json={**edit, "guidance_scale": 2, field: "private <|image_pad|>"})
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNotIn("private", response.text)
            self.assertFalse(any(call[0] in ("generate", "drain") for call in CALLS))
            self.assertEqual(client.get("/health").json()["image_tasks"], [server.GENERATE, server.EDIT])
            self.assertEqual(client.post(self.GEN, json={"prompt": "valid"}).status_code, 200)
            self.assertEqual(client.post(self.EDIT, json=edit).status_code, 200)

    def test_bad_runtime_or_missing_consent_fails_before_loading(self):
        self.configure(qwen=True)
        for key, value in [("MODEL_RESEARCH", "false"), ("MODEL_DTYPE", "float16"), ("MODEL_RECIPE", "wrong")]:
            with patch.dict(os.environ, {key: value}), self.assertRaisesRegex(RuntimeError, "dual-image"):
                server.load_pipeline()
        self.assertEqual(CALLS, [])

    def test_both_routes_and_health_remain_behind_native_auth(self):
        self.configure(qwen=True)
        spec = importlib.util.spec_from_file_location("image_test_auth", ROOT / "internal/modelship/assets/api_auth.py")
        auth = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auth)
        key = "a" * 43
        protected = auth.BearerAuth(server.app, types.SimpleNamespace(digest=hashlib.sha256(key.encode()).digest()))
        with TestClient(protected) as client:
            self.assertEqual(client.get("/health").status_code, 401)
            for route in (self.GEN, self.EDIT):
                self.assertEqual(client.post(route, content=b"not a JSON body").status_code, 401)
            self.assertEqual(client.get("/health", headers={"Authorization": f"Bearer {key}"}).json()["image_tasks"], [server.GENERATE, server.EDIT])
            self.assertFalse(any(c[0] == "generate" for c in CALLS))


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

    def test_qwen_edit_encoder_bridges_fp32_vae_and_bf16_transformer(self):
        server.PIPELINE = "QwenImage21Pipeline"
        class Tensor:
            def __init__(self, dtype):
                self.dtype = dtype
            def to(self, *, dtype):
                return Tensor(dtype)
        with patch.dict(sys.modules, {"diffusers": types.SimpleNamespace(QwenImage21Pipeline=FakePipeline)}), patch.dict(os.environ, {"MODEL_RESEARCH": "true"}):
            pipe = server.load_pipeline()
            for dtype in ("bf16", "fp16"):
                encoded = pipe._encode_vae_image(Tensor(dtype), "generator")
                self.assertEqual(encoded.dtype, dtype)
        self.assertEqual([call for call in CALLS if call[0] == "vae-encode"],
                         [("vae-encode", "fp32", "generator")] * 2)

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
