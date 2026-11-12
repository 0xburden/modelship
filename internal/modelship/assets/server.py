"""Single-model Diffusers service. JSON/base64 only; never fetches user URLs."""
import base64
import binascii
import io
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager

import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool

MAX_BODY = 24 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 16_777_216
MODEL = os.environ.get("MODEL_ID", "")
REVISION = os.environ.get("MODEL_REVISION", "")
TASK = os.environ.get("MODEL_TASK", "text-to-image")
PIPELINE = os.environ.get("MODEL_PIPELINE", "")
PIPE = None
BUSY = threading.Lock()


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=8192)
    negative_prompt: str | None = Field(default=None, max_length=8192)
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    steps: int = Field(default=28, ge=1, le=100)
    guidance_scale: float = Field(default=7.0, ge=0, le=20)
    seed: int | None = Field(default=None, ge=0, lt=2**63)
    image_b64: str | None = Field(default=None, max_length=22_369_624)
    strength: float = Field(default=0.7, gt=0, le=1)

    @model_validator(mode="after")
    def dimensions(self):
        if self.width % 8 or self.height % 8:
            raise ValueError("width and height must be multiples of 8")
        if int(self.steps * self.strength) < 1 and self.image_b64:
            raise ValueError("steps * strength must be >= 1")
        return self


def decode_image(encoded, width, height):
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("image exceeds 16 MiB")
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError("input image exceeds pixel limit")
            image.load()
            return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError,
            Image.DecompressionBombError) as exc:
        raise HTTPException(422, "invalid or oversized base64 image") from exc


@asynccontextmanager
async def lifespan(app):
    global PIPE
    if not MODEL or not REVISION:
        raise RuntimeError("MODEL_ID and immutable MODEL_REVISION are required")
    loader = AutoPipelineForImage2Image if TASK == "image-to-image" else AutoPipelineForText2Image
    dtype = torch.bfloat16 if PIPELINE == "FluxPipeline" else torch.float16
    PIPE = loader.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=dtype,
        use_safetensors=True, trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to("cuda")
    PIPE.vae.enable_slicing()
    PIPE.vae.enable_tiling()
    PIPE.set_progress_bar_config(disable=True)
    yield
    PIPE = None
    torch.cuda.empty_cache()


app = FastAPI(title="modelship images", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ready": PIPE is not None, "model": MODEL, "revision": REVISION,
            "task": TASK, "run_id": os.environ.get("MODEL_RUN_ID", "")}


def generate(req):
    if not BUSY.acquire(blocking=False):
        raise HTTPException(503, "GPU busy; retry after the current request finishes")
    try:
        seed = req.seed if req.seed is not None else secrets.randbelow(2**63)
        args = dict(
            prompt=req.prompt, num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            width=req.width, height=req.height,
        )
        if PIPELINE == "FluxPipeline":
            # FLUX defaults may otherwise consume far more memory than this service budgets.
            args["max_sequence_length"] = 256
            if req.negative_prompt is not None:
                raise HTTPException(422, "negative_prompt is not supported for FLUX in this runtime")
        elif req.negative_prompt is not None:
            args["negative_prompt"] = req.negative_prompt
        if TASK == "image-to-image":
            if not req.image_b64:
                raise HTTPException(422, "image_b64 is required for image-to-image")
            args["image"] = decode_image(req.image_b64, req.width, req.height)
            args["strength"] = req.strength
            # SD/SDXL img2img gets dimensions from the resized input image.
            if PIPELINE != "FluxPipeline":
                args.pop("width")
                args.pop("height")
        elif req.image_b64 is not None:
            raise HTTPException(422, "this deployment serves text-to-image, not image-to-image")
        with torch.inference_mode():
            image = PIPE(**args).images[0]
        output = io.BytesIO()
        image.save(output, format="PNG")
        return {
            "created": int(time.time()), "model": MODEL, "seed": seed,
            "data": [{"b64_json": base64.b64encode(output.getvalue()).decode("ascii")}],
        }
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise HTTPException(503, "GPU out of memory; reduce dimensions or use a larger GPU") from exc
    finally:
        BUSY.release()


async def parse_request(request):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BODY:
            raise HTTPException(413, "request exceeds 24 MiB")
    try:
        return Generation.model_validate_json(data)
    except ValidationError as exc:
        # Do not echo base64 image data or prompts in validation errors/logs.
        raise HTTPException(422, "invalid request fields; see /docs") from exc


REQUEST_SCHEMA = {"requestBody": {"required": True, "content": {
    "application/json": {"schema": Generation.model_json_schema()}
}}}


@app.post("/v1/images/generations", openapi_extra=REQUEST_SCHEMA)
async def generations(request: Request):
    if TASK != "text-to-image":
        raise HTTPException(409, "this deployment serves /v1/images/edits")
    return await run_in_threadpool(generate, await parse_request(request))


@app.post("/v1/images/edits", openapi_extra=REQUEST_SCHEMA)
async def edits(request: Request):
    if TASK != "image-to-image":
        raise HTTPException(409, "this deployment serves /v1/images/generations")
    return await run_in_threadpool(generate, await parse_request(request))
