"""Bounded native Wan text-to-video jobs; no user URLs or original PTH loaders.

Jobs are intentionally ephemeral: restart invalidates IDs and removes old exports.
Shutdown drains the sole worker before releasing the pipeline/artifact directory.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import fcntl
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import time
import uuid

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.responses import StreamingResponse

OFFICIAL_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODEL = os.environ.get("MODEL_ID", "")
REVISION = os.environ.get("MODEL_REVISION", "")
TASK = os.environ.get("MODEL_TASK", "text-to-video")
RUN_ID = os.environ.get("MODEL_RUN_ID", "")
ARTIFACT_DIR = os.environ.get("MODEL_ARTIFACT_DIR", "")
MAX_BODY = 32 * 1024
MAX_PIXELS_FRAMES = 512 * 512 * 49
MAX_JOBS = 8
JOB_TTL = 3600
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_DOWNLOADS = 2
ARTIFACT_NAME = re.compile(r"video-[0-9a-f]{32}(?:\.partial)?\.mp4\Z")
PIPE = None
STORE = None


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(min_length=1, max_length=8192)
    negative_prompt: str | None = Field(default=None, max_length=8192)
    width: int = Field(default=512, ge=256, le=768)
    height: int = Field(default=512, ge=256, le=768)
    num_frames: int = Field(default=17, ge=1, le=49)
    steps: int = Field(default=30, ge=1, le=50)
    fps: int = Field(default=16, ge=1, le=30)
    guidance_scale: float = Field(default=4.0, ge=0, le=10, allow_inf_nan=False)
    seed: int | None = Field(default=None, ge=0, lt=2**63)

    @model_validator(mode="after")
    def bounded_work(self):
        if not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        if self.width % 16 or self.height % 16:
            raise ValueError("dimensions must be divisible by 16")
        if (self.num_frames - 1) % 4:
            raise ValueError("num_frames must be 4k+1")
        if self.width * self.height * self.num_frames > MAX_PIXELS_FRAMES:
            raise ValueError("pixel/frame budget exceeded")
        return self


def render(req, seed, destination):
    """The only inference path: pinned native Diffusers WanPipeline on CUDA."""
    args = dict(prompt=req.prompt, width=req.width, height=req.height,
                num_frames=req.num_frames, num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(seed))
    if req.negative_prompt is not None:
        args["negative_prompt"] = req.negative_prompt
    with torch.inference_mode():
        frames = PIPE(**args).frames[0]
    # A local .mp4 suffix is required by imageio's ffmpeg writer.
    export_to_video(frames, str(destination), fps=req.fps)


class VideoResponse(StreamingResponse):
    """Release the open file/download slot even if disconnected before iteration."""
    def __init__(self, chunks, release, headers):
        super().__init__(chunks, media_type="video/mp4", headers=headers)
        self.release = release

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.release()


class JobStore:
    def __init__(self, root):
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None
        self.closing = False
        self.downloads = 0
        self.executor = None
        self.file_lock = None
        base = Path(root)
        if not root or not base.is_absolute():
            raise RuntimeError("absolute MODEL_ARTIFACT_DIR is required")
        # The harness owns base. Never follow a preexisting video-jobs symlink.
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root = base / "video-jobs"
        self.root.mkdir(mode=0o700, exist_ok=True)
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError("unsafe video artifact directory")
        self.root.chmod(0o700)
        fd = os.open(self.root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        self.file_lock = os.fdopen(fd, "a+b")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file_lock.close()
            self.file_lock = None
            raise RuntimeError("video artifact directory is already in use") from None
        try:
            self._clean_exports()
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wan-video")
        except BaseException:
            self.file_lock.close()
            raise

    def _clean_exports(self):
        for path in self.root.iterdir():
            if ARTIFACT_NAME.fullmatch(path.name):
                path.unlink(missing_ok=True)

    def _path(self, job_id, partial=False):
        return self.root / (job_id + (".partial" if partial else "") + ".mp4")

    def _remove(self, job_id):
        try:
            self._path(job_id).unlink(missing_ok=True)
            self._path(job_id, True).unlink(missing_ok=True)
        except OSError:
            self.closing = True
            raise HTTPException(503, "video artifact storage unavailable") from None
        self.jobs.pop(job_id)

    def _prune(self, reserve=0):
        now = time.monotonic()
        for job_id, job in list(self.jobs.items()):
            if job_id != self.active and job["_expires"] <= now:
                self._remove(job_id)
        # Insertion order provides oldest-first retention. Never evict active work.
        for job_id in list(self.jobs):
            size = sum(job["_size"] for job in self.jobs.values())
            if len(self.jobs) <= MAX_JOBS - reserve and size <= MAX_TOTAL_BYTES:
                break
            if job_id != self.active:
                self._remove(job_id)

    def prune(self):
        with self.lock:
            self._prune()

    @staticmethod
    def public(job):
        return {key: value for key, value in job.items() if not key.startswith("_")}

    def submit(self, req):
        with self.lock:
            if self.closing or PIPE is None:
                raise HTTPException(503, "video service unavailable")
            if self.active is not None:
                raise HTTPException(503, "GPU busy; retry after the active video job finishes",
                                    headers={"Retry-After": "5"})
            self._prune(reserve=1)
            job_id = "video-" + uuid.uuid4().hex
            seed = req.seed if req.seed is not None else secrets.randbelow(2**63)
            job = dict(id=job_id, object="video", status="running", model=MODEL,
                       created=int(time.time()), completed_at=None, expires_at=None,
                       seed=seed, width=req.width, height=req.height,
                       num_frames=req.num_frames, fps=req.fps, error=None,
                       content_url=None, _expires=float("inf"), _size=0)
            self.jobs[job_id] = job
            self.active = job_id
            # Snapshot before starting: even a fast worker cannot change the 202 object.
            response = self.public(job)
            try:
                self.executor.submit(self._run, job_id, req, seed)
            except Exception:
                self.active = None
                self.jobs.pop(job_id)
                raise HTTPException(503, "video service unavailable") from None
            return response

    def _run(self, job_id, req, seed):
        partial, final = self._path(job_id, True), self._path(job_id)
        error = None
        size = 0
        try:
            render(req, seed, partial)
            info = partial.lstat()
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_ARTIFACT_BYTES:
                raise RuntimeError("invalid export")
            partial.chmod(0o600)
            size = info.st_size
            os.replace(partial, final)
        except torch.OutOfMemoryError:
            error = {"code": "gpu_out_of_memory", "message": "GPU out of memory; reduce video dimensions or frames",
                     "status_code": 503}
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            # Never expose exception strings: model errors can contain prompts/tokens/paths.
            error = {"code": "generation_failed", "message": "video generation failed", "status_code": 500}
        finally:
            cleanup_failed = False
            try:
                partial.unlink(missing_ok=True)
                if error is not None:
                    final.unlink(missing_ok=True)
            except OSError:
                # Do not admit more work if a storage fault prevents bounded cleanup.
                cleanup_failed = True
                error = {"code": "generation_failed", "message": "video generation failed", "status_code": 500}
            with self.lock:
                if cleanup_failed:
                    self.closing = True
                job = self.jobs[job_id]
                job.update(status="failed" if error else "succeeded", error=error,
                           completed_at=int(time.time()), expires_at=int(time.time()) + JOB_TTL,
                           _expires=time.monotonic() + JOB_TTL, _size=0 if error else size,
                           content_url=None if error else f"/v1/videos/{job_id}/content")
                self.active = None
                self._prune()

    def get(self, job_id):
        with self.lock:
            self._prune()
            job = self.jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "video job not found or expired")
            return self.public(job)

    def content(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job["status"] != "succeeded":
                raise HTTPException(409, "video content is not available")
            if self.downloads >= MAX_DOWNLOADS:
                raise HTTPException(503, "video downloads busy")
            try:
                fd = os.open(self._path(job_id), os.O_RDONLY | os.O_NOFOLLOW)
                stream = os.fdopen(fd, "rb")
            except OSError:
                raise HTTPException(404, "video content is unavailable") from None
            self.downloads += 1
        closed = False

        def release():
            nonlocal closed
            with self.lock:
                if not closed:
                    closed = True
                    stream.close()
                    self.downloads -= 1

        def chunks():
            try:
                while data := stream.read(64 * 1024):
                    yield data
            finally:
                release()

        return VideoResponse(chunks(), release, headers={
            "Content-Disposition": f'attachment; filename="{job_id}.mp4"',
            "Content-Length": str(os.fstat(fd).st_size), "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        })

    def close(self):
        with self.lock:
            self.closing = True
        # Do not cancel an asyncio wrapper while its CUDA/export thread is still alive.
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=False)
        try:
            with self.lock:
                self._clean_exports()
                self.jobs.clear()
        finally:
            if self.file_lock is not None:
                self.file_lock.close()
                self.file_lock = None


async def sweep(store, stop):
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=30)
        except asyncio.TimeoutError:
            try:
                store.prune()
            except HTTPException:
                # A cleanup fault already marks the store unavailable. Still drain
                # the inference worker during shutdown; do not leave a failed task.
                return


async def drain(store, sweeper):
    try:
        if sweeper is not None:
            try:
                await sweeper
            except (Exception, asyncio.CancelledError):
                # Even a failed/cancelled sweeper must not bypass worker cleanup.
                pass
    finally:
        await asyncio.to_thread(store.close)


@asynccontextmanager
async def lifespan(app):
    global PIPE, STORE
    if MODEL != OFFICIAL_MODEL:
        raise RuntimeError("only the official Wan2.2 Diffusers repository is supported")
    if not re.fullmatch(r"[0-9a-f]{40}", REVISION):
        raise RuntimeError("immutable 40-hex MODEL_REVISION is required")
    if TASK != "text-to-video" or os.environ.get("MODEL_PIPELINE", "WanPipeline") != "WanPipeline":
        raise RuntimeError("only WanPipeline text-to-video is supported")
    if os.environ.get("MODEL_DTYPE", "bfloat16") != "bfloat16":
        raise RuntimeError("Wan requires the bfloat16 recipe")
    store = JobStore(ARTIFACT_DIR)
    stop = asyncio.Event()
    sweeper = None
    try:
        try:
            PIPE = WanPipeline.from_pretrained(
                MODEL, revision=REVISION,
                torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
                use_safetensors=True, trust_remote_code=False, low_cpu_mem_usage=True, device_map="cuda",
            )
            for method_name in ("enable_slicing", "enable_tiling"):
                method = getattr(PIPE.vae, method_name, None)
                if callable(method):
                    method()
            PIPE.set_progress_bar_config(disable=True)
        except Exception:
            raise RuntimeError("Wan pipeline initialization failed") from None
        STORE = store
        sweeper = asyncio.create_task(sweep(store, stop))
        yield
    finally:
        with store.lock:
            store.closing = True
        stop.set()
        # Cancellation of to_thread only cancels its asyncio wrapper, NOT the CUDA
        # thread. Shield a separate drain task and defer even repeated cancellation
        # until it has actually stopped before releasing PIPE or the artifact lock.
        cleanup = asyncio.create_task(drain(store, sweeper))
        cancelled = False
        try:
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
        finally:
            STORE = None
            PIPE = None
            torch.cuda.empty_cache()
        if cancelled:
            raise asyncio.CancelledError


app = FastAPI(title="modelship native video", lifespan=lifespan)


@app.get("/health")
def health():
    store = STORE
    return {"ready": PIPE is not None and store is not None and not store.closing,
            "model": MODEL, "revision": REVISION, "task": TASK, "run_id": RUN_ID}


async def parse_request(request):
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BODY:
            raise HTTPException(413, "request exceeds 32 KiB")
        data.extend(chunk)
    try:
        return Generation.model_validate_json(data)
    except ValidationError:
        raise HTTPException(422, "invalid request fields; see /docs") from None


REQUEST_SCHEMA = {"requestBody": {"required": True, "content": {
    "application/json": {"schema": Generation.model_json_schema()}
}}}


@app.post("/v1/videos/generations", status_code=202, openapi_extra=REQUEST_SCHEMA)
async def generations(request: Request):
    req = await parse_request(request)
    if STORE is None:
        raise HTTPException(503, "video service unavailable")
    return STORE.submit(req)


@app.get("/v1/videos/{job_id}")
def video(job_id: str):
    store = STORE
    if store is None:
        raise HTTPException(503, "video service unavailable")
    return store.get(job_id)


@app.get("/v1/videos/{job_id}/content")
def content(job_id: str):
    store = STORE
    if store is None:
        raise HTTPException(503, "video service unavailable")
    return store.content(job_id)
