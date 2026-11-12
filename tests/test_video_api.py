"""Wan HTTP/lifecycle contract tests. All Torch, Diffusers and MP4 work is fake.

These tests deliberately do NOT claim real model, codec, or GPU validation.
"""
from contextlib import nullcontext
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
CALLS = []
MP4 = b"\x00\x00\x00\x18ftypmp42fake-local-video"
SECRET = "private prompt hf_secret_token /private/path"


class OOM(Exception):
    pass


class FakePipeline:
    failure = None
    gate = None
    entered = threading.Event()

    def __init__(self):
        self.vae = types.SimpleNamespace(
            enable_slicing=lambda: CALLS.append(("slicing",)),
            enable_tiling=lambda: CALLS.append(("tiling",)),
        )

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        CALLS.append(("load", model, kwargs))
        return cls()

    def to(self, device):
        CALLS.append(("device", device))
        return self

    def set_progress_bar_config(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        CALLS.append(("generate", kwargs))
        self.entered.set()
        if self.gate is not None and not self.gate.wait(5):
            raise RuntimeError("test gate timed out")
        if self.failure is not None:
            raise self.failure
        return types.SimpleNamespace(frames=[["fake-frame"]])


class Generator:
    def __init__(self, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


EXPORT_FAILURE = None


def export(frames, destination, fps):
    CALLS.append(("export", frames, destination, fps))
    Path(destination).write_bytes(MP4)
    if EXPORT_FAILURE is not None:
        raise EXPORT_FAILURE


fake_torch = types.SimpleNamespace(
    bfloat16="bf16", float32="fp32", Generator=Generator,
    inference_mode=nullcontext, OutOfMemoryError=OOM,
    cuda=types.SimpleNamespace(empty_cache=lambda: CALLS.append(("empty_cache",))),
)
spec = importlib.util.spec_from_file_location(
    "modelship_video_server", ROOT / "internal/modelship/assets/video_server.py",
)
server = importlib.util.module_from_spec(spec)
# Do not poison other test modules' Torch/Diffusers stubs during discovery.
with patch.dict(sys.modules, {
    "torch": fake_torch,
    "diffusers": types.SimpleNamespace(WanPipeline=FakePipeline),
    "diffusers.utils": types.SimpleNamespace(export_to_video=export),
}):
    spec.loader.exec_module(server)


class VideoAPITests(unittest.TestCase):
    def setUp(self):
        global EXPORT_FAILURE
        CALLS.clear()
        FakePipeline.failure = None
        FakePipeline.gate = None
        FakePipeline.entered.clear()
        EXPORT_FAILURE = None
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {"MODEL_PIPELINE": "WanPipeline", "MODEL_DTYPE": "bfloat16"})
        self.env.start()
        self.addCleanup(self.env.stop)
        server.MODEL = server.OFFICIAL_MODEL
        server.REVISION = "a" * 40
        server.TASK = "text-to-video"
        server.RUN_ID = "fixture-run"
        server.ARTIFACT_DIR = self.tmp.name
        server.MAX_JOBS = 8
        server.JOB_TTL = 3600
        server.MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
        server.MAX_TOTAL_BYTES = 256 * 1024 * 1024

    def submit(self, client, **fields):
        response = client.post("/v1/videos/generations", json={"prompt": "a mountain", **fields})
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        self.assertEqual(job["status"], "running")
        self.assertRegex(job["id"], r"^video-[0-9a-f]{32}$")
        return job

    def finished(self, client, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            response = client.get(f"/v1/videos/{job_id}")
            self.assertEqual(response.status_code, 200, response.text)
            if response.json()["status"] != "running":
                return response.json()
            time.sleep(0.005)
        self.fail("job did not complete")

    def test_loading_generation_content_and_health(self):
        with TestClient(server.app) as client:
            self.assertEqual(client.get("/health").json(), {
                "ready": True, "model": server.OFFICIAL_MODEL, "revision": "a" * 40,
                "task": "text-to-video", "run_id": "fixture-run",
            })
            accepted = self.submit(client, seed=42, prompt=SECRET, negative_prompt="flicker")
            job = self.finished(client, accepted["id"])
            self.assertEqual(job["status"], "succeeded")
            self.assertEqual(job["seed"], 42)
            self.assertIsNone(job["error"])
            self.assertNotIn(SECRET, json.dumps(job))
            self.assertNotIn("_size", job)
            response = client.get(job["content_url"])
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, MP4)
            self.assertEqual(response.headers["content-type"], "video/mp4")
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertIn("attachment", response.headers["content-disposition"])
            output = server.STORE._path(job["id"])
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(output.parent.stat().st_mode), 0o700)
        self.assertIsNone(server.PIPE)
        self.assertIsNone(server.STORE)
        self.assertEqual(list(Path(self.tmp.name).rglob("*.mp4")), [])
        load = next(call for call in CALLS if call[0] == "load")
        self.assertEqual(load[1], server.OFFICIAL_MODEL)
        self.assertEqual(load[2]["revision"], "a" * 40)
        self.assertEqual(load[2]["torch_dtype"], {"default": "bf16", "vae": "fp32"})
        self.assertTrue(load[2]["use_safetensors"])
        self.assertFalse(load[2]["trust_remote_code"])
        self.assertEqual(load[2]["device_map"], "cuda")
        generated = next(call[1] for call in CALLS if call[0] == "generate")
        self.assertEqual(generated["num_frames"], 17)
        self.assertEqual(generated["generator"].device, "cuda")
        self.assertEqual(generated["generator"].seed, 42)
        self.assertEqual(generated["negative_prompt"], "flicker")
        exported = next(call for call in CALLS if call[0] == "export")
        self.assertTrue(exported[2].endswith(".partial.mp4"))
        self.assertEqual(exported[3], 16)

    def test_bounded_parameters_unknown_fields_and_sanitized_validation(self):
        invalid = [
            {"prompt": ""}, {"prompt": "  "}, {"prompt": "x" * 8193},
            {"negative_prompt": "x" * 8193}, {"width": 255}, {"width": 784},
            {"width": 257}, {"height": 767}, {"height": 1024},
            {"num_frames": 0}, {"num_frames": 16}, {"num_frames": 53},
            {"num_frames": 49, "width": 768, "height": 768},
            {"steps": 0}, {"steps": 51}, {"fps": 0}, {"fps": 31},
            {"guidance_scale": 11}, {"seed": -1}, {"seed": 2**63},
            {"width": "512"}, {"steps": True}, {"num_frames": 17.0},
            {"image_url": "http://169.254.169.254/latest/meta-data"},
            {"video_url": "https://example.com/a.mp4"},
            {"image_b64": "abc"}, {"model": "other"}, {"callback_url": "https://example.com"},
        ]
        with TestClient(server.app) as client:
            for fields in invalid:
                with self.subTest(fields=fields):
                    body = {"prompt": SECRET, **fields}
                    response = client.post("/v1/videos/generations", json=body)
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertNotIn(SECRET, response.text)
                    self.assertNotIn("169.254", response.text)
            self.assertFalse(any(call[0] == "generate" for call in CALLS))
            self.assertEqual(client.post("/v1/videos/generations", content=b"{").status_code, 422)
            self.assertEqual(client.post("/v1/videos/generations", content=b"null").status_code, 422)

    def test_valid_minimum_and_maximum_frames(self):
        with TestClient(server.app) as client:
            for frames in (1, 5, 49):
                job = self.submit(client, width=256, height=256, num_frames=frames, steps=1, fps=30)
                self.assertEqual(self.finished(client, job["id"])["status"], "succeeded")

    def test_body_cap_and_streamed_body_cap(self):
        with TestClient(server.app) as client:
            response = client.post("/v1/videos/generations", content=b"x" * (server.MAX_BODY + 1))
            self.assertEqual(response.status_code, 413)
            def chunks():
                for _ in range(5):
                    yield b"x" * 8192
            self.assertEqual(client.post("/v1/videos/generations", content=chunks()).status_code, 413)
            self.assertFalse(any(call[0] == "generate" for call in CALLS))

    def test_one_active_job_no_backlog_and_pending_content(self):
        gate = threading.Event()
        FakePipeline.gate = gate
        try:
            with TestClient(server.app) as client:
                first = self.submit(client)
                self.assertTrue(FakePipeline.entered.wait(2))
                self.assertEqual(client.get(f"/v1/videos/{first['id']}/content").status_code, 409)
                for _ in range(3):
                    response = client.post("/v1/videos/generations", json={"prompt": SECRET})
                    self.assertEqual(response.status_code, 503)
                    self.assertIn("Retry-After", response.headers)
                    self.assertNotIn(SECRET, response.text)
                self.assertEqual(len(server.STORE.jobs), 1)
                gate.set()
                self.finished(client, first["id"])
                second = self.submit(client)
                self.finished(client, second["id"])
        finally:
            gate.set()

    def test_concurrent_submissions_only_accept_one(self):
        gate = threading.Event()
        FakePipeline.gate = gate
        try:
            with TestClient(server.app) as client:
                barrier = threading.Barrier(5)
                responses = []
                def post():
                    barrier.wait()
                    responses.append(client.post("/v1/videos/generations", json={"prompt": "x"}))
                threads = [threading.Thread(target=post) for _ in range(5)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(3)
                    self.assertFalse(thread.is_alive())
                self.assertEqual(sorted(r.status_code for r in responses), [202, 503, 503, 503, 503])
                gate.set()
        finally:
            gate.set()

    def test_unknown_jobs_and_path_traversal_cannot_read_files(self):
        with TestClient(server.app) as client:
            for url in ("/v1/videos/missing", "/v1/videos/missing/content",
                        "/v1/videos/%2e%2e%2f.lock/content"):
                self.assertEqual(client.get(url).status_code, 404)

    def assert_failure(self, failure=None, export_failure=None, code="generation_failed"):
        global EXPORT_FAILURE
        FakePipeline.failure = failure
        EXPORT_FAILURE = export_failure
        with TestClient(server.app) as client:
            job = self.submit(client, prompt=SECRET)
            failed = self.finished(client, job["id"])
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["error"]["code"], code)
            self.assertNotIn(SECRET, json.dumps(failed))
            self.assertEqual(client.get(f"/v1/videos/{job['id']}/content").status_code, 409)
            self.assertEqual(list(server.STORE.root.glob("*.mp4")), [])
            self.assertIsNone(server.STORE.active)
            FakePipeline.failure = None
            EXPORT_FAILURE = None
            next_job = self.submit(client)
            self.assertEqual(self.finished(client, next_job["id"])["status"], "succeeded")
        return failed

    def test_inference_error_sanitized_and_capacity_released(self):
        self.assert_failure(failure=RuntimeError(SECRET))

    def test_oom_is_failed_job_with_503_and_no_secret(self):
        job = self.assert_failure(failure=OOM(SECRET), code="gpu_out_of_memory")
        self.assertEqual(job["error"]["status_code"], 503)
        self.assertIn(("empty_cache",), CALLS)

    def test_partial_export_is_removed_on_failure(self):
        self.assert_failure(export_failure=RuntimeError(SECRET))

    def test_artifact_size_cap(self):
        server.MAX_ARTIFACT_BYTES = 1
        with TestClient(server.app) as client:
            job = self.submit(client)
            self.assertEqual(self.finished(client, job["id"])["status"], "failed")
            self.assertEqual(list(server.STORE.root.glob("*.mp4")), [])

    def test_job_count_bounded_and_old_content_removed(self):
        server.MAX_JOBS = 2
        with TestClient(server.app) as client:
            jobs = []
            for _ in range(4):
                job = self.submit(client)
                self.finished(client, job["id"])
                jobs.append(job["id"])
            self.assertEqual(len(server.STORE.jobs), 2)
            self.assertEqual(len(list(server.STORE.root.glob("*.mp4"))), 2)
            for job_id in jobs[:2]:
                self.assertEqual(client.get(f"/v1/videos/{job_id}").status_code, 404)
                self.assertEqual(client.get(f"/v1/videos/{job_id}/content").status_code, 404)

    def test_total_artifact_bytes_bounded(self):
        server.MAX_TOTAL_BYTES = 2 * len(MP4)
        with TestClient(server.app) as client:
            jobs = []
            for _ in range(3):
                job = self.submit(client)
                self.finished(client, job["id"])
                jobs.append(job["id"])
            self.assertEqual(client.get(f"/v1/videos/{jobs[0]}").status_code, 404)
            self.assertLessEqual(sum(p.stat().st_size for p in server.STORE.root.glob("*.mp4")),
                                 server.MAX_TOTAL_BYTES)

    def test_expiry_removes_job_and_artifact(self):
        with TestClient(server.app) as client:
            job = self.submit(client)
            self.finished(client, job["id"])
            with server.STORE.lock:
                server.STORE.jobs[job["id"]]["_expires"] = time.monotonic() - 1
            self.assertEqual(client.get(f"/v1/videos/{job['id']}").status_code, 404)
            self.assertEqual(list(server.STORE.root.glob("*.mp4")), [])

    def test_failed_job_records_also_bounded(self):
        server.MAX_JOBS = 2
        FakePipeline.failure = RuntimeError(SECRET)
        with TestClient(server.app) as client:
            for _ in range(5):
                job = self.submit(client)
                self.assertEqual(self.finished(client, job["id"])["status"], "failed")
            self.assertEqual(len(server.STORE.jobs), 2)

    def test_restart_discards_old_jobs_and_cleans_crash_exports(self):
        root = Path(self.tmp.name) / "video-jobs"
        root.mkdir()
        stale = root / ("video-" + "b" * 32 + ".partial.mp4")
        complete = root / ("video-" + "c" * 32 + ".mp4")
        stale.write_bytes(b"partial")
        complete.write_bytes(MP4)
        with TestClient(server.app) as client:
            self.assertFalse(stale.exists())
            self.assertFalse(complete.exists())
            job = self.submit(client)
            self.finished(client, job["id"])
        with TestClient(server.app) as client:
            self.assertEqual(client.get(f"/v1/videos/{job['id']}").status_code, 404)
            self.assertEqual(list(root.glob("*.mp4")), [])

    def test_exclusive_artifact_directory(self):
        with TestClient(server.app):
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                server.JobStore(self.tmp.name)

    def test_symlink_artifact_directory_is_rejected(self):
        external = Path(self.tmp.name) / "external"
        external.mkdir()
        (Path(self.tmp.name) / "video-jobs").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            with TestClient(server.app):
                pass
        self.assertFalse(any(call[0] == "load" for call in CALLS))

    def test_shutdown_drains_worker_before_releasing_pipeline_and_files(self):
        gate = threading.Event()
        FakePipeline.gate = gate
        client = TestClient(server.app)
        client.__enter__()
        job = self.submit(client)
        self.assertTrue(FakePipeline.entered.wait(2))
        store = server.STORE
        errors = []
        def close():
            try:
                client.__exit__(None, None, None)
            except BaseException as exc:
                errors.append(exc)
        closer = threading.Thread(target=close)
        closer.start()
        try:
            deadline = time.monotonic() + 2
            while not store.closing and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(store.closing)
            self.assertTrue(closer.is_alive())
            self.assertIsNotNone(server.PIPE)
            self.assertIsNotNone(store.file_lock)
            self.assertEqual(store.active, job["id"])
        finally:
            gate.set()
            closer.join(5)
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])
        self.assertIsNone(server.PIPE)
        self.assertIsNone(store.file_lock)
        self.assertEqual(list(store.root.glob("*.mp4")), [])

    def test_repeated_shutdown_cancellation_still_drains_worker(self):
        gate = threading.Event()
        FakePipeline.gate = gate

        async def scenario():
            lifetime = server.lifespan(server.app)
            await lifetime.__aenter__()
            store = server.STORE
            store.submit(server.Generation(prompt="x"))
            self.assertTrue(FakePipeline.entered.wait(2))
            closing = asyncio.create_task(lifetime.__aexit__(None, None, None))
            try:
                deadline = time.monotonic() + 2
                while not store.closing and time.monotonic() < deadline:
                    await asyncio.sleep(0.001)
                self.assertTrue(store.closing)
                for _ in range(3):
                    closing.cancel()
                    await asyncio.sleep(0.005)
                    self.assertFalse(closing.done())
                    self.assertIsNotNone(server.PIPE)
                    self.assertIs(server.STORE, store)
                    self.assertIsNotNone(store.file_lock)
                    self.assertIsNotNone(store.active)
                gate.set()
                with self.assertRaises(asyncio.CancelledError):
                    await closing
                self.assertIsNone(server.PIPE)
                self.assertIsNone(server.STORE)
                self.assertIsNone(store.file_lock)
                self.assertEqual(list(store.root.glob("*.mp4")), [])
            finally:
                gate.set()
                try:
                    await closing
                except asyncio.CancelledError:
                    pass
        asyncio.run(scenario())

    def test_failed_or_cancelled_sweeper_does_not_bypass_shutdown_cleanup(self):
        for failure in (RuntimeError(SECRET), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                with patch.object(server, "sweep", new=AsyncMock(side_effect=failure)):
                    with TestClient(server.app) as client:
                        job = self.submit(client)
                        self.finished(client, job["id"])
                        store = server.STORE
                self.assertIsNone(server.PIPE)
                self.assertIsNone(server.STORE)
                self.assertIsNone(store.file_lock)
                self.assertEqual(list(store.root.glob("*.mp4")), [])

    def test_download_cap_and_disconnect_release(self):
        with TestClient(server.app) as client:
            job = self.submit(client)
            self.finished(client, job["id"])
            first = server.STORE.content(job["id"])
            second = server.STORE.content(job["id"])
            try:
                self.assertEqual(client.get(f"/v1/videos/{job['id']}/content").status_code, 503)
                with patch.object(server.StreamingResponse, "__call__", new=AsyncMock(side_effect=OSError("disconnected"))):
                    with self.assertRaises(OSError):
                        asyncio.run(first({}, None, None))
                self.assertEqual(server.STORE.downloads, 1)
                self.assertEqual(client.get(f"/v1/videos/{job['id']}/content").content, MP4)
            finally:
                first.release()
                second.release()
            self.assertEqual(server.STORE.downloads, 0)

    def test_cleanup_storage_fault_fails_closed_without_exception_leak(self):
        original_unlink = Path.unlink
        failed = threading.Event()
        def unlink(path, *args, **kwargs):
            if path.name.endswith(".partial.mp4") and not failed.is_set():
                failed.set()
                raise PermissionError(SECRET)
            return original_unlink(path, *args, **kwargs)
        with TestClient(server.app) as client:
            with patch.object(Path, "unlink", unlink):
                job = self.submit(client)
                finished = self.finished(client, job["id"])
                self.assertEqual(finished["status"], "failed")
                self.assertNotIn(SECRET, json.dumps(finished))
                self.assertTrue(server.STORE.closing)
                self.assertFalse(client.get("/health").json()["ready"])
                response = client.post("/v1/videos/generations", json={"prompt": "x"})
                self.assertEqual(response.status_code, 503)
                self.assertIsNone(server.STORE.active)
        self.assertEqual(list(Path(self.tmp.name).rglob("*.mp4")), [])

    def test_model_revision_task_and_dtype_fail_closed(self):
        for name, value in (
            ("MODEL", "Wan-AI/Wan2.2-T2V-A14B"), ("MODEL", "org/other-Diffusers"),
            ("REVISION", "main"), ("REVISION", ""), ("REVISION", "a" * 39),
            ("TASK", "image-to-video"), ("ARTIFACT_DIR", "relative/path"),
        ):
            with self.subTest(name=name, value=value), patch.object(server, name, value):
                with self.assertRaises(RuntimeError):
                    with TestClient(server.app):
                        pass
        for env in ({"MODEL_DTYPE": "float16"}, {"MODEL_PIPELINE": "AutoPipeline"}):
            with patch.dict(os.environ, env):
                with self.assertRaises(RuntimeError):
                    with TestClient(server.app):
                        pass
        self.assertFalse(any(call[0] == "load" for call in CALLS))

    def test_loader_failure_releases_artifact_lock(self):
        with patch.object(FakePipeline, "from_pretrained", side_effect=RuntimeError(SECRET)):
            with self.assertRaisesRegex(RuntimeError, "^Wan pipeline initialization failed$"):
                with TestClient(server.app):
                    pass
        self.assertIsNone(server.PIPE)
        self.assertIsNone(server.STORE)
        with TestClient(server.app) as client:
            self.assertTrue(client.get("/health").json()["ready"])


if __name__ == "__main__":
    unittest.main()
