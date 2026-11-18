"""Single-model Diffusers service. JSON/base64 only; never fetches user URLs."""
import base64
import binascii
import io
import json
import math
import re
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

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
GENERATE = "text-to-image"
EDIT = "image-to-image"
DUAL = "image-generate-and-edit"
PIPE = None
EDIT_PIPE = None
UNHEALTHY = False
BUSY = threading.Lock()


def check_dual_profile():
    import diffusers

    profiles = json.loads(Path(__file__).with_name("dual_image_profiles.json").read_text(encoding="utf-8"))
    for profile in profiles:
        if (MODEL, REVISION, PIPELINE, os.environ.get("MODEL_RECIPE")) != (
            profile["model"], profile["revision"], profile["pipeline"], profile["recipe"]
        ):
            continue
        if (diffusers.__version__ != profile["diffusers"]
                or os.environ.get("MODEL_NATIVE_LOADER") != profile["loader"]
                or os.environ.get("MODEL_DTYPE") != profile["dtype"]
                or json.loads(os.environ.get("MODEL_COMPONENT_DTYPES", "{}")) != profile["component_dtypes"]):
            raise RuntimeError("dual-image runtime does not match the reviewed profile")
        if profile["research"] and os.environ.get("MODEL_RESEARCH") != "true":
            raise RuntimeError("dual-image Qwen requires --research acknowledgment of non-commercial use")
        return
    raise RuntimeError("dual-image model does not match a reviewed profile")


def make_edit_pipeline(pipe):
    if PIPELINE == "QwenImage21Pipeline":
        return pipe
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

    if type(pipe) is not StableDiffusionXLPipeline or type(pipe.scheduler) is not EulerDiscreteScheduler:
        raise RuntimeError("unexpected SDXL generation pipeline or scheduler")
    scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    edit = AutoPipelineForImage2Image.from_pipe(pipe, scheduler=scheduler)
    if (type(edit) is not StableDiffusionXLImg2ImgPipeline
            or edit.scheduler is not scheduler or scheduler is pipe.scheduler
            or any(getattr(edit, name, None) is not getattr(pipe, name)
                   for name in ("vae", "unet", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"))):
        raise RuntimeError("unexpected SDXL edit pipeline or component placement")
    return edit


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=8192)
    negative_prompt: str | None = Field(default=None, max_length=8192)
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    steps: int = Field(default=28, ge=1, le=100)
    guidance_scale: float | None = Field(default=None, ge=0, le=20)
    seed: int | None = Field(default=None, ge=0, lt=2**63)
    image_b64: str | None = Field(default=None, max_length=22_369_624)
    strength: float | None = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def dimensions(self):
        if self.width % 8 or self.height % 8:
            raise ValueError("width and height must be multiples of 8")
        if PIPELINE == "QwenImage21Pipeline":
            if self.width % 32 or self.height % 32:
                raise ValueError("Qwen Image dimensions must be multiples of 32")
            if self.strength is not None:
                raise ValueError("Qwen Image uses image conditioning, not strength-based img2img")
        elif PIPELINE == "MOSSImagePipeline" and (self.width % 16 or self.height % 16):
            raise ValueError("MOSS dimensions must be multiples of 16")
        if PIPELINE in ("QwenImage21Pipeline", "MOSSImagePipeline") and len(self.prompt) > 4096:
            raise ValueError("this pipeline accepts at most 4096 prompt characters")
        if PIPELINE != "QwenImage21Pipeline" and int(self.steps * (self.strength or 0.7)) < 1 and self.image_b64:
            raise ValueError("steps * strength must be >= 1")
        return self


def decode_image(encoded, width, height, mode="RGB"):
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("image exceeds 16 MiB")
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError("input image exceeds pixel limit")
            image.load()
            return image.convert(mode).resize((width, height), Image.Resampling.LANCZOS)
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError,
            Image.DecompressionBombError) as exc:
        raise HTTPException(422, "invalid or oversized base64 image") from exc


def load_pipeline():
    if not MODEL or not re.fullmatch(r"[a-f0-9]{40}", REVISION):
        raise RuntimeError("MODEL_ID and immutable MODEL_REVISION are required")
    if TASK == DUAL:
        check_dual_profile()
    modern = PIPELINE in ("QwenImage21Pipeline", "MOSSImagePipeline")
    dtype_name = os.environ.get("MODEL_DTYPE", "bfloat16" if modern or PIPELINE == "FluxPipeline" else "float16")
    if dtype_name not in ("float16", "bfloat16"):
        raise RuntimeError("unsupported image recipe dtype")
    dtype = getattr(torch, dtype_name)
    components = json.loads(os.environ.get("MODEL_COMPONENT_DTYPES", "{}"))
    if modern:
        components["vae"] = "float32"
    if any(value not in ("float16", "bfloat16", "float32") for value in components.values()):
        raise RuntimeError("unsupported component dtype")
    dtypes = {"default": dtype, **{name: getattr(torch, value) for name, value in components.items()}} if components else dtype
    if PIPELINE == "QwenImage21Pipeline":
        if os.environ.get("MODEL_RESEARCH") != "true":
            raise RuntimeError("Qwen Image requires --research acknowledgment of non-commercial use")
        from diffusers import QwenImage21Pipeline

        class ModelshipQwenImage21Pipeline(QwenImage21Pipeline):
            def _encode_vae_image(self, image, generator):
                # Diffusers 0.42.1 casts conditioning pixels to the text/DiT
                # dtype, even when the VAE is explicitly FP32. Encode in VAE
                # precision, normalize there, then return latents in the input
                # precision so concatenation cannot upcast the BF16 DiT input.
                input_dtype = image.dtype
                encoded = super()._encode_vae_image(image.to(dtype=self.vae.dtype), generator)
                return encoded.to(dtype=input_dtype)

        loader = ModelshipQwenImage21Pipeline
    elif PIPELINE == "MOSSImagePipeline":
        if TASK != "text-to-image":
            raise RuntimeError("MOSS only supports text-to-image")
        from model_code import load_moss_pipeline
        return load_moss_pipeline(MODEL, REVISION, dtypes)
    else:
        loader = AutoPipelineForImage2Image if TASK == "image-to-image" else AutoPipelineForText2Image
    return loader.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=dtypes,
        use_safetensors=True, trust_remote_code=False, low_cpu_mem_usage=True,
        **({"device_map": "cuda"} if modern else {}),
    )


@asynccontextmanager
async def lifespan(app):
    global PIPE, EDIT_PIPE, UNHEALTHY
    UNHEALTHY = False
    pipe = load_pipeline()
    if PIPELINE not in ("QwenImage21Pipeline", "MOSSImagePipeline"):
        pipe = pipe.to("cuda")
    for name in ("enable_slicing", "enable_tiling"):
        method = getattr(pipe.vae, name, None)
        if callable(method):
            method()
    edit = make_edit_pipeline(pipe) if TASK == DUAL else None
    pipe.set_progress_bar_config(disable=True)
    if edit is not None and edit is not pipe:
        edit.set_progress_bar_config(disable=True)
    PIPE, EDIT_PIPE = pipe, edit
    try:
        yield
    finally:
        if not UNHEALTHY:
            PIPE = EDIT_PIPE = pipe = edit = None
            torch.cuda.empty_cache()


app = FastAPI(title="modelship images", lifespan=lifespan)


@app.get("/health")
def health():
    ready = PIPE is not None and not UNHEALTHY and (TASK != DUAL or EDIT_PIPE is not None)
    return {"ready": ready, "model": MODEL, "revision": REVISION,
            "task": GENERATE if TASK == DUAL else TASK, "pipeline": PIPELINE,
            "recipe": os.environ.get("MODEL_RECIPE", ""), "run_id": os.environ.get("MODEL_RUN_ID", ""),
            **({"image_tasks": [GENERATE, EDIT]} if TASK == DUAL and ready else {})}


def generate(req, task):
    global UNHEALTHY
    if not BUSY.acquire(blocking=False):
        raise HTTPException(503, "GPU busy; retry after the current request finishes")
    try:
        if UNHEALTHY or PIPE is None or (TASK == DUAL and EDIT_PIPE is None):
            raise HTTPException(503, "image worker unavailable; check readiness and service logs")
        seed = req.seed if req.seed is not None else secrets.randbelow(2**63)
        args = dict(
            prompt=req.prompt, num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale if req.guidance_scale is not None else (
                1.0 if PIPELINE == "QwenImage21Pipeline" else 3.0 if PIPELINE == "MOSSImagePipeline" else 7.0),
            width=req.width, height=req.height,
        )
        if PIPELINE == "QwenImage21Pipeline":
            # The pinned edit processor expands these strings into visual
            # features. User-supplied markers can invalidate that alignment
            # before denoising; reject them before entering the pipeline.
            if TASK == DUAL and any(marker in text
                    for text in (req.prompt, req.negative_prompt or "")
                    for marker in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>",
                                   "<|vision_end|>", "<|placeholder|>")):
                raise HTTPException(422, "Qwen Image prompts cannot include reserved vision markers")
            args["true_cfg_scale"] = args.pop("guidance_scale")
            # Condition images otherwise default to 1024-pixel preprocessing even
            # for a 256/512 request, escaping this service's image memory budget.
            args["output_resolution"] = math.isqrt(req.width * req.height) // 32 * 32
            if req.negative_prompt is not None and args["true_cfg_scale"] <= 1:
                raise HTTPException(422, "negative_prompt requires guidance_scale > 1 for Qwen Image")
            if args["true_cfg_scale"] > 1:
                args["negative_prompt"] = req.negative_prompt or ""
        elif PIPELINE == "MOSSImagePipeline":
            args["max_sequence_length"] = 512
            if req.negative_prompt is not None:
                args["negative_prompt"] = req.negative_prompt
        elif PIPELINE == "FluxPipeline":
            # FLUX defaults may otherwise consume far more memory than this service budgets.
            args["max_sequence_length"] = 256
            if req.negative_prompt is not None:
                raise HTTPException(422, "negative_prompt is not supported for FLUX in this runtime")
        elif req.negative_prompt is not None:
            args["negative_prompt"] = req.negative_prompt
        if task == EDIT:
            if not req.image_b64:
                raise HTTPException(422, "image_b64 is required for image-to-image")
            args["image"] = decode_image(req.image_b64, req.width, req.height, "RGBA" if PIPELINE == "QwenImage21Pipeline" else "RGB")
            if PIPELINE != "QwenImage21Pipeline":
                args["strength"] = req.strength if req.strength is not None else 0.7
            # SD/SDXL img2img gets dimensions from the resized input image.
            if PIPELINE not in ("FluxPipeline", "QwenImage21Pipeline"):
                args.pop("width")
                args.pop("height")
        elif req.image_b64 is not None:
            raise HTTPException(422, "this deployment serves text-to-image, not image-to-image")
        pipe = EDIT_PIPE if TASK == DUAL and task == EDIT else PIPE
        try:
            args["generator"] = torch.Generator(device="cpu" if PIPELINE == "MOSSImagePipeline" else "cuda").manual_seed(seed)
            with torch.inference_mode():
                image = pipe(**args).images[0]
        except Exception as exc:
            if TASK != DUAL:
                raise
            # SDXL can upcast its shared VAE in place before a failure. A CUDA
            # drain alone cannot certify that the pipeline state was restored.
            UNHEALTHY = True
            if isinstance(exc, torch.OutOfMemoryError):
                raise HTTPException(503, "GPU out of memory; restart worker before reducing the workload") from None
            raise HTTPException(500, "image pipeline failed; restart worker before submitting again") from None
        finally:
            if TASK == DUAL:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    # A failed drain cannot establish that another request is safe.
                    # Keep resident pipelines and fail readiness until process restart.
                    UNHEALTHY = True
                    raise HTTPException(503, "image worker CUDA drain failed; restart required") from None
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
    if TASK not in (GENERATE, DUAL):
        raise HTTPException(409, "this deployment serves /v1/images/edits")
    return await run_in_threadpool(generate, await parse_request(request), GENERATE)


@app.post("/v1/images/edits", openapi_extra=REQUEST_SCHEMA)
async def edits(request: Request):
    if TASK not in (EDIT, DUAL):
        raise HTTPException(409, "this deployment serves /v1/images/generations")
    return await run_in_threadpool(generate, await parse_request(request), EDIT)
