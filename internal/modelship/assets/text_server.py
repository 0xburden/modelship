"""Single-request native Transformers API; reviewed Gemma chat controls are opt-in."""
import os
import asyncio
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForCausalLM, AutoTokenizer
from vision_input import VisionMessage, chat_capabilities, image_count, normalize_messages

MODEL = os.environ.get("MODEL_ID", "")
REVISION = os.environ.get("MODEL_REVISION", "")
RUN_ID = os.environ.get("MODEL_RUN_ID", "")
RECIPE = os.environ.get("MODEL_RECIPE", "")
TEMPLATE_COMPLETIONS = os.environ.get("MODEL_TEMPLATE_COMPLETIONS", "false") == "true"
DISABLE_THINKING = os.environ.get("MODEL_DISABLE_THINKING", "false") == "true"
CONTEXT = int(os.environ.get("MODEL_CONTEXT", "4096"))
TEXT_PROFILE_ID = os.environ.get("MODEL_TEXT_PROFILE", "")
TEXT_FEATURES = None
ACTIVE_JOB = None
POISONED = threading.Event()
DRAIN_REFS = []  # Retain GPU references after an uncertain drain until process exit.
VISION_PROFILE = os.environ.get("MODEL_VISION_PROFILE", "")
TOKENIZER = None
IMAGE_PROCESSOR = None
NETWORK = None
BUSY = threading.Lock()
MAX_BODY = (8 if VISION_PROFILE else 1) * 1024 * 1024


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    max_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    stream: StrictBool = False
    seed: int | None = Field(default=None, ge=0, lt=2**63)


class Completion(Parameters):
    prompt: str = Field(min_length=1, max_length=65536)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32768)


class ThinkingKwargs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable_thinking: StrictBool = False


class ChatParameters(Parameters):
    chat_template_kwargs: ThinkingKwargs | None = None
    expected_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class Chat(ChatParameters):
    messages: list[Message] = Field(min_length=1, max_length=128)


class VisionChat(ChatParameters):
    messages: list[VisionMessage] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def one_image(self):
        if image_count(self.messages) > 1:
            raise ValueError("one image is supported across a conversation request")
        return self


@asynccontextmanager
async def lifespan(app):
    global NETWORK, TOKENIZER, IMAGE_PROCESSOR, TEXT_FEATURES
    if not MODEL or not REVISION:
        raise RuntimeError("MODEL_ID and immutable MODEL_REVISION are required")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    loader_name = os.environ.get("MODEL_NATIVE_LOADER", "causal-lm")
    dtype_name = os.environ.get("MODEL_DTYPE", "float16")
    if dtype_name not in ("float16", "bfloat16"):
        raise RuntimeError("unsupported recipe dtype")
    dtype = getattr(torch, dtype_name)
    from native_text import load_text_profile
    TEXT_FEATURES = load_text_profile(TEXT_PROFILE_ID, MODEL, REVISION, RECIPE, loader_name, dtype_name, TOKENIZER)
    if VISION_PROFILE:
        from gemma_vision import load_image_processor
        IMAGE_PROCESSOR = load_image_processor(VISION_PROFILE, MODEL, REVISION, RECIPE, loader_name, dtype_name, TOKENIZER)
    if loader_name == "multimodal-lm":
        from transformers import AutoModelForMultimodalLM
        loader = AutoModelForMultimodalLM
        dtype_options = {"dtype": dtype, "device_map": "cuda", "low_cpu_mem_usage": True}
    elif loader_name == "causal-lm":
        loader = AutoModelForCausalLM
        dtype_options = {"torch_dtype": dtype}
    else:
        raise RuntimeError("unsupported native text loader")
    NETWORK = loader.from_pretrained(
        MODEL, revision=REVISION, **dtype_options,
        use_safetensors=True, trust_remote_code=False,
        attn_implementation="eager",
    )
    if loader_name == "causal-lm":
        NETWORK = NETWORK.to("cuda")
    NETWORK = NETWORK.eval()
    try:
        yield
    finally:
        if ACTIVE_JOB is not None and not ACTIVE_JOB.done.is_set():
            ACTIVE_JOB.cancel()
            task = asyncio.create_task(run_in_threadpool(ACTIVE_JOB.done.wait))
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
    if not POISONED.is_set():
        NETWORK = None
        TOKENIZER = None
        IMAGE_PROCESSOR = None
        torch.cuda.empty_cache()


app = FastAPI(title="modelship native text", lifespan=lifespan)


@app.get("/health")
def health():
    draining = ACTIVE_JOB is not None and ACTIVE_JOB.stop.is_set() and not ACTIVE_JOB.done.is_set()
    return {"ready": NETWORK is not None and not POISONED.is_set() and not draining, "draining": draining, "model": MODEL, "revision": REVISION,
            "task": "text-generation", "run_id": RUN_ID, "recipe": RECIPE,
            "input_modalities": ["text", "image"] if VISION_PROFILE else ["text"]}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "modelship",
                                          "input_modalities": ["text", "image"] if VISION_PROFILE else ["text"]}]}


@app.get("/v1/capabilities")
def capabilities():
    from native_text import chat_controls
    controls = chat_capabilities(bool(VISION_PROFILE), 280)
    controls.update(chat_controls(TEXT_FEATURES))
    return {"schema_version": 1, "model": MODEL, "revision": REVISION, "run_id": RUN_ID,
            "runtime": "transformers", "profile": VISION_PROFILE or None,
            "input_modalities": ["text", "image"] if VISION_PROFILE else ["text"],
            "output_modalities": ["text"], "tasks": ["text-generation"],
            "context_tokens": CONTEXT, "chat": controls, "completion": {"streaming": False}}


def prepare_inputs(req, chat=False, thinking=False, move=True):
    images = []
    try:
        templated = chat or TEMPLATE_COMPLETIONS
        if templated:
            if not TOKENIZER.chat_template:
                raise HTTPException(422, "model has no chat template for this recipe")
            if chat and VISION_PROFILE:
                try:
                    messages, images = normalize_messages(req.messages)
                except (ValueError, OSError):
                    raise HTTPException(422, "invalid bounded image chat input; use one PNG/JPEG data image and accompanying text") from None
            else:
                messages = [message.model_dump() for message in req.messages] if chat else [{"role": "user", "content": req.prompt}]
            prompt = TOKENIZER.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **({"enable_thinking": thinking} if DISABLE_THINKING else {}),
            )
        else:
            prompt = req.prompt
        if VISION_PROFILE:
            try:
                inputs = IMAGE_PROCESSOR.prepare(prompt, images, TOKENIZER)
            except (ValueError, TypeError, KeyError):
                raise HTTPException(422, "image preprocessing or placeholder/token layout is unsupported") from None
        else:
            inputs = TOKENIZER(prompt, return_tensors="pt", add_special_tokens=not templated)
        count = inputs["input_ids"].shape[1]
        if count + req.max_tokens > CONTEXT:
            raise HTTPException(422, "prompt plus max_tokens exceeds the deployment context limit")
        # Preserve token/position/mask integer dtypes. Only image pixels use the
        # reviewed compute dtype; never cast IDs or positions to BF16.
        if move:
            inputs = {key: value.to(device="cuda", dtype=torch.bfloat16) if key == "pixel_values"
                      else value.to("cuda") for key, value in inputs.items()}

        return inputs, count, images
    except BaseException:
        for image in images:
            image.close()
        raise


def generation_kwargs(req):
    kwargs = {"max_new_tokens": req.max_tokens, "do_sample": req.temperature > 0}
    if req.temperature > 0:
        kwargs.update(temperature=req.temperature, top_p=req.top_p)
    if TOKENIZER.eos_token_id is not None:
        kwargs["pad_token_id"] = TOKENIZER.eos_token_id
    if req.seed is not None:
        torch.manual_seed(req.seed)
    return kwargs


def check_model(req):
    if req.model not in (None, MODEL):
        raise HTTPException(404, "requested model is not loaded")
    if POISONED.is_set():
        raise HTTPException(503, "CUDA drain failed; explicitly restart this worker")


def generate(req, chat=False):
    check_model(req)
    if req.stream:
        raise HTTPException(422, "streaming is unavailable for this execution profile/operation")
    if chat and (req.chat_template_kwargs is not None or req.expected_run_id is not None):
        raise HTTPException(422, "thinking overrides are unavailable for this execution profile")
    if not BUSY.acquire(blocking=False):
        raise HTTPException(503, "GPU busy; retry after the current request finishes")
    images = []
    try:
        inputs, count, images = prepare_inputs(req, chat)
        with torch.inference_mode():
            output = NETWORK.generate(**inputs, **generation_kwargs(req))
        tokens = output[0, count:]
        text = TOKENIZER.decode(tokens, skip_special_tokens=True)
        completion_tokens = len(tokens)
        choice = {"index": 0, "finish_reason": "length" if completion_tokens >= req.max_tokens else "stop"}
        choice["message" if chat else "text"] = {"role": "assistant", "content": text} if chat else text
        return {"id": ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex,
                "object": "chat.completion" if chat else "text_completion",
                "created": int(time.time()), "model": MODEL, "choices": [choice],
                "usage": {"prompt_tokens": count, "completion_tokens": completion_tokens,
                          "total_tokens": count + completion_tokens}}
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise HTTPException(503, "GPU out of memory; reduce context or stop competing workloads") from exc
    finally:
        for image in images:
            image.close()
        BUSY.release()


def prepare_feature_job(req):
    from native_text import NativeTextJob
    from transformers import StoppingCriteria, StoppingCriteriaList
    global ACTIVE_JOB
    check_model(req)
    advanced = req.stream or req.chat_template_kwargs is not None
    if (advanced and req.expected_run_id != RUN_ID) or (req.expected_run_id is not None and req.expected_run_id != RUN_ID):
        raise HTTPException(409, "worker identity changed; fetch capabilities and use expected_run_id before submitting")
    if not BUSY.acquire(blocking=False):
        raise HTTPException(503, "GPU busy; retry after the current request finishes")
    images, inputs, cuda_started = [], {}, False
    try:
        thinking = bool(req.chat_template_kwargs and req.chat_template_kwargs.enable_thinking)
        inputs, count, images = prepare_inputs(req, True, thinking, move=False)
        cuda_started = True
        inputs = {key: value.to(device="cuda", dtype=torch.bfloat16) if key == "pixel_values"
                  else value.to("cuda") for key, value in inputs.items()}
        kwargs = generation_kwargs(req)
        def work(streamer, cancelled):
            class Cancel(StoppingCriteria):
                def __call__(self, input_ids, scores, **extra):
                    return cancelled()
            with torch.inference_mode():
                output = NETWORK.generate(**inputs, **kwargs, streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([Cancel()]))
            return len(output[0, count:])
        def cleanup(error):
            drained = True
            try:
                torch.cuda.synchronize()
                if error == 'oom':
                    torch.cuda.empty_cache()
            except Exception:
                POISONED.set()
                drained = False
            finally:
                for image in images:
                    try:
                        image.close()
                    except Exception:
                        POISONED.set()
                        drained = False
                if drained:
                    inputs.clear()
                    BUSY.release()
                else:
                    DRAIN_REFS.append(inputs)
            return drained
        identity = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": MODEL}
        job = NativeTextJob(TOKENIZER, thinking, work, cleanup, count, req.max_tokens, identity, torch.OutOfMemoryError)
        ACTIVE_JOB = job
    except BaseException as exc:
        drained = True
        if cuda_started:
            try:
                torch.cuda.synchronize()
            except Exception:
                POISONED.set()
                drained = False
        for image in images:
            image.close()
        if drained:
            inputs.clear()
            BUSY.release()
        else:
            DRAIN_REFS.append(inputs)
        if isinstance(exc, torch.OutOfMemoryError):
            if drained:
                torch.cuda.empty_cache()
            raise HTTPException(503, "GPU out of memory during input preparation") from None
        raise
    try:
        job.start()
    except Exception:
        raise HTTPException(503, "could not start generation; no retry was made") from None
    return job


async def feature_chat(request, req):
    from native_text import NativeTextResponse, ERRORS
    job = await run_in_threadpool(prepare_feature_job, req)
    if req.stream:
        return NativeTextResponse(job)
    text, reasoning, final = [], [], None
    try:
        async for kind, value in job.events():
            if await request.is_disconnected():
                raise HTTPException(499, "client disconnected; GPU work is draining")
            if kind == 'error':
                raise HTTPException(503 if value in ('oom', 'timeout', 'drain') else 500, ERRORS[value])
            if kind == 'heartbeat':
                continue
            if kind == 'delta':
                text.append(value.get('content', ''))
                reasoning.append(value.get('reasoning_content', ''))
            else:
                final = value
        if final is None:
            raise HTTPException(500, "generation ended without a result")
        message = {"role": "assistant", "content": ''.join(text)}
        if any(reasoning):
            message['reasoning_content'] = ''.join(reasoning)
        return dict(job.identity, object='chat.completion',
                    choices=[{"index": 0, "message": message, "finish_reason": final['finish_reason']}],
                    usage=final['usage'])
    finally:
        job.cancel()


async def parse_request(request, schema):
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BODY:
            raise HTTPException(413, "request exceeds the deployment body limit")
        data.extend(chunk)
    try:
        return schema.model_validate_json(data)
    except ValidationError as exc:
        raise HTTPException(422, "invalid request fields; see /docs") from exc


def request_schema(schema):
    definition = schema.model_json_schema()
    definitions = definition.pop("$defs", {})
    def inline(value):
        if isinstance(value, list):
            return [inline(item) for item in value]
        if isinstance(value, dict):
            if "$ref" in value:
                return inline(definitions[value["$ref"].rsplit("/", 1)[-1]])
            return {key: inline(item) for key, item in value.items() if key != "discriminator"}
        return value
    return {"requestBody": {"required": True, "content": {
        "application/json": {"schema": inline(definition)}
    }}}


@app.post("/v1/completions", openapi_extra=request_schema(Completion))
async def completions(request: Request):
    return await run_in_threadpool(generate, await parse_request(request, Completion))


@app.post("/v1/chat/completions", openapi_extra=request_schema(VisionChat if VISION_PROFILE else Chat))
async def chats(request: Request):
    req = await parse_request(request, VisionChat if VISION_PROFILE else Chat)
    if TEXT_FEATURES:
        return await feature_chat(request, req)
    return await run_in_threadpool(generate, req, True)
