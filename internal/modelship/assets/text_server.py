"""Minimal non-streaming, single-request Transformers API for native CUDA hosts."""
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL_ID", "")
REVISION = os.environ.get("MODEL_REVISION", "")
RUN_ID = os.environ.get("MODEL_RUN_ID", "")
RECIPE = os.environ.get("MODEL_RECIPE", "")
TEMPLATE_COMPLETIONS = os.environ.get("MODEL_TEMPLATE_COMPLETIONS", "false") == "true"
DISABLE_THINKING = os.environ.get("MODEL_DISABLE_THINKING", "false") == "true"
CONTEXT = int(os.environ.get("MODEL_CONTEXT", "4096"))
TOKENIZER = None
NETWORK = None
BUSY = threading.Lock()
MAX_BODY = 1024 * 1024


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    max_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    stream: bool = False
    seed: int | None = Field(default=None, ge=0, lt=2**63)


class Completion(Parameters):
    prompt: str = Field(min_length=1, max_length=65536)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32768)


class Chat(Parameters):
    messages: list[Message] = Field(min_length=1, max_length=128)


@asynccontextmanager
async def lifespan(app):
    global NETWORK, TOKENIZER
    if not MODEL or not REVISION:
        raise RuntimeError("MODEL_ID and immutable MODEL_REVISION are required")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    loader_name = os.environ.get("MODEL_NATIVE_LOADER", "causal-lm")
    dtype_name = os.environ.get("MODEL_DTYPE", "float16")
    if dtype_name not in ("float16", "bfloat16"):
        raise RuntimeError("unsupported recipe dtype")
    dtype = getattr(torch, dtype_name)
    if loader_name == "multimodal-lm":
        from transformers import AutoModelForMultimodalLM
        loader = AutoModelForMultimodalLM
        dtype_options = {"dtype": dtype}
    elif loader_name == "causal-lm":
        loader = AutoModelForCausalLM
        dtype_options = {"torch_dtype": dtype}
    else:
        raise RuntimeError("unsupported native text loader")
    NETWORK = loader.from_pretrained(
        MODEL, revision=REVISION, **dtype_options,
        use_safetensors=True, trust_remote_code=False,
        attn_implementation="eager",
    ).to("cuda").eval()
    yield
    NETWORK = None
    TOKENIZER = None
    torch.cuda.empty_cache()


app = FastAPI(title="modelship native text", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ready": NETWORK is not None, "model": MODEL, "revision": REVISION,
            "task": "text-generation", "run_id": RUN_ID, "recipe": RECIPE}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "modelship"}]}


def generate(req, chat=False):
    if req.model not in (None, MODEL):
        raise HTTPException(404, "requested model is not loaded")
    if req.stream:
        raise HTTPException(422, "streaming is not implemented by the native backend")
    if not BUSY.acquire(blocking=False):
        raise HTTPException(503, "GPU busy; retry after the current request finishes")
    try:
        templated = chat or TEMPLATE_COMPLETIONS
        if templated:
            if not TOKENIZER.chat_template:
                raise HTTPException(422, "model has no chat template for this recipe")
            messages = [message.model_dump() for message in req.messages] if chat else [{"role": "user", "content": req.prompt}]
            prompt = TOKENIZER.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **({"enable_thinking": False} if DISABLE_THINKING else {}),
            )
        else:
            prompt = req.prompt
        inputs = TOKENIZER(prompt, return_tensors="pt", add_special_tokens=not templated)
        count = inputs["input_ids"].shape[1]
        if count + req.max_tokens > CONTEXT:
            raise HTTPException(422, "prompt plus max_tokens exceeds the deployment context limit")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        kwargs = {"max_new_tokens": req.max_tokens, "do_sample": req.temperature > 0}
        if req.temperature > 0:
            kwargs.update(temperature=req.temperature, top_p=req.top_p)
        if TOKENIZER.eos_token_id is not None:
            kwargs["pad_token_id"] = TOKENIZER.eos_token_id
        if req.seed is not None:
            torch.manual_seed(req.seed)
        with torch.inference_mode():
            output = NETWORK.generate(**inputs, **kwargs)
        tokens = output[0, count:]
        text = TOKENIZER.decode(tokens, skip_special_tokens=True)
        completion_tokens = len(tokens)
        choice = {"index": 0, "finish_reason": "length" if completion_tokens >= req.max_tokens else "stop"}
        if chat:
            choice["message"] = {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        return {
            "id": ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex,
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()), "model": MODEL, "choices": [choice],
            "usage": {"prompt_tokens": count, "completion_tokens": completion_tokens,
                      "total_tokens": count + completion_tokens},
        }
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise HTTPException(503, "GPU out of memory; reduce context or stop competing workloads") from exc
    finally:
        BUSY.release()


async def parse_request(request, schema):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BODY:
            raise HTTPException(413, "request exceeds 1 MiB")
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
            return {key: inline(item) for key, item in value.items()}
        return value
    return {"requestBody": {"required": True, "content": {
        "application/json": {"schema": inline(definition)}
    }}}


@app.post("/v1/completions", openapi_extra=request_schema(Completion))
async def completions(request: Request):
    return await run_in_threadpool(generate, await parse_request(request, Completion))


@app.post("/v1/chat/completions", openapi_extra=request_schema(Chat))
async def chats(request: Request):
    return await run_in_threadpool(generate, await parse_request(request, Chat), True)
