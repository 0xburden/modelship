"""Opt-in, hash-verified loading of the one reviewed MOSS repository revision.

This is NOT a sandbox. Reviewed repository Python executes as the service user.
No code is fetched or imported before both the consent and immutable manifest checks.
"""
import hashlib
import json
import os
from pathlib import Path
import re

EXPECTED_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
EXPECTED_REVISION = "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9"
EXPECTED_HASH = "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31"
CODE_PATHS = ("pipeline_moss_image.py", "scheduler/pipeline_moss_image.py",
              "transformer/pipeline_moss_image.py")
COMPONENTS = {
    "scheduler": ["pipeline_moss_image", "MOSSImageScheduler"],
    "transformer": ["pipeline_moss_image", "MOSSImageTransformer"],
    "text_encoder": ["transformers", "Qwen3_5Model"],
    "tokenizer": ["transformers", "Qwen2Tokenizer"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}


def verify_code(path, expected):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise RuntimeError("model code exceeds size limit")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("model code SHA-256 mismatch; refusing execution")


def load_moss_pipeline(model, revision, dtypes):
    if os.environ.get("MODEL_TRUST_CODE") != "true":
        raise RuntimeError("MOSS requires explicit --trust-model-code approval; research is not code consent")
    if model != EXPECTED_REPO or revision != EXPECTED_REVISION:
        raise RuntimeError("MOSS code is only approved at the reviewed immutable revision")
    manifest = json.loads(os.environ.get("MODEL_CODE_HASHES", "{}"))
    if manifest != {name: EXPECTED_HASH for name in CODE_PATHS}:
        raise RuntimeError("missing or unexpected reviewed code manifest")
    if os.environ.get("MODEL_CODE_REVISION") != revision:
        raise RuntimeError("model and code revisions must match")
    weights = json.loads(os.environ.get("MODEL_WEIGHT_FILES", "[]"))
    if not weights or any(not isinstance(name, str) or not re.fullmatch(
            r"(?:transformer|text_encoder|vae)/[A-Za-z0-9_.-]+\.safetensors", name)
            or ".." in name for name in weights):
        raise RuntimeError("missing or unsafe selected safetensors manifest")

    from huggingface_hub import hf_hub_download, snapshot_download
    from diffusers import DiffusionPipeline

    # Verify the small code files before fetching model tensors, and again before import.
    for name, digest in manifest.items():
        verify_code(hf_hub_download(model, name, revision=revision), digest)
    snapshot = Path(snapshot_download(
        repo_id=model, revision=revision,
        allow_patterns=weights + list(CODE_PATHS) + [
            "*.json", "tokenizer/*.txt", "tokenizer/*.model", "tokenizer/*.tiktoken",
            "tokenizer/chat_template.jinja",
        ],
    ))
    for name, digest in manifest.items():
        verify_code(snapshot / name, digest)
    index = json.loads((snapshot / "model_index.json").read_text())
    if index.get("_class_name") != "MOSSImagePipeline" or any(
            index.get(name) != spec for name, spec in COMPONENTS.items()) or any(
            not name.startswith("_") and name not in COMPONENTS for name in index):
        raise RuntimeError("MOSS component manifest differs from the reviewed pipeline")
    pipe = DiffusionPipeline.from_pretrained(
        str(snapshot), custom_pipeline=str(snapshot / "pipeline_moss_image.py"),
        local_files_only=True, trust_remote_code=True, use_safetensors=True,
        torch_dtype=dtypes, low_cpu_mem_usage=True, device_map="cuda",
    )
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    return pipe
