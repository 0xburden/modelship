"""Real pinned PIL preprocessing and upstream expansion checks; no model inference.

Run with tests/processor-requirements.txt in an isolated Torch-free environment:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tests/qwen_processor_check.py
Only synthetic local images and a checked-in 390-byte metadata fixture are used.
The tokenizer fixture tests framing/order, not actual tokenizer/model inference.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if importlib.util.find_spec("torch") or importlib.util.find_spec("torchvision"):
    raise RuntimeError("this check requires a clean, Torch-free CPU environment")

import huggingface_hub
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "internal/modelship/assets"
sys.path.insert(0, str(ASSETS))
from qwen_vision import QwenImageProcessor, PROCESSOR_PIN, REPO, REVISION, load_image_processor
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.processing_utils import ProcessorMixin

fixture = ROOT / "internal/modelship/testdata/qwen38-preprocessor-config.json"
profile = {
    "id": "qwen38-27b-images-v1", "repo": REPO, "revision": REVISION,
    "recipe": "qwen3.8-multimodal", "architecture": "Qwen3_5ForConditionalGeneration",
    "processor": dict(PROCESSOR_PIN), "max_image_tokens": 1024,
    "activation_allowance_mib": 4096,
    "images": {"max_images": 4, "max_total_image_bytes": 8 * 1024 * 1024,
               "max_total_image_pixels": 8 * 1024 * 1024},
}
# Check catalog integration too when the leader has added the new profile.
catalog = json.loads((ASSETS / "pytorch_vision_profiles.json").read_text())
for entry in catalog["profiles"]:
    if entry["id"] == profile["id"]:
        assert entry == profile, (entry, profile)


class TokenizerFixture:
    tokens = {"<|vision_start|>": 248053, "<|vision_end|>": 248054,
              "<|image_pad|>": 248056, "<|video_pad|>": 248057}
    pattern = re.compile("(" + "|".join(re.escape(t) for t in tokens) + ")")

    def convert_tokens_to_ids(self, token):
        return self.tokens[token]

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "np" and add_special_tokens is False
        self.last_prompt = text
        ids = []
        for part in self.pattern.split(text):
            ids.extend([self.tokens[part]] if part in self.tokens else [ord(c) + 100 for c in part])
        return {"input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}


def rejected(fn, label):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("accepted invalid " + label)


def check_images(adapter, tokenizer, images):
    marker = adapter.boi_token + adapter.image_token + adapter.eoi_token
    # Two images in the same user turn, then later turns after an assistant reply.
    # The first images deliberately have different counts; catches replace()-loops
    # that incorrectly consume the first newly expanded placeholder a second time.
    chunks = ["<|im_start|>user\ncompare ", marker, " then "]
    for index in range(1, len(images)):
        if index > 1:
            chunks.append("<|im_end|>\n<|im_start|>assistant\nnoted<|im_end|>\n<|im_start|>user\n")
        chunks.extend([marker, " next "])
    chunks.append("<|im_end|>\n<|im_start|>assistant\n")
    prompt = "".join(chunks)
    inputs = adapter.prepare(prompt, images, tokenizer, return_tensors="np")
    expanded = tokenizer.last_prompt
    grid, values = inputs["image_grid_thw"], inputs["pixel_values"]
    counts = grid.prod(axis=-1) // 4
    assert grid.shape == (len(images), 3) and grid.dtype == np.int64
    assert np.all(grid[:, 0] == 1) and np.all(grid[:, 1:] % 2 == 0)
    assert np.all((counts >= 1) & (counts <= 1024))
    assert values.shape == (int(grid.prod(axis=-1).sum()), 1536)
    assert values.dtype == np.float32 and np.isfinite(values).all()
    assert values.min() >= -1.00001 and values.max() <= 1.00001
    assert "position_ids" not in inputs and "video_grid_thw" not in inputs
    assert "pixel_values_videos" not in inputs

    reference = SimpleNamespace(
        image_processor=adapter.processor, image_token=adapter.image_token,
        all_special_multimodal_tokens=[adapter.image_token],
        image_token_ids=[248056], video_token_ids=[248057], audio_token_ids=[],
    )
    replacements = [Qwen3VLProcessor.replace_image_token(reference, inputs, image_idx=index)
                    for index in range(len(images))]
    upstream_text, offsets = ProcessorMixin.get_text_with_replacements(reference, [prompt], replacements)
    assert expanded == upstream_text[0]
    assert len(offsets[0]) == len(images)
    reference_types = ProcessorMixin.create_mm_token_type_ids(reference, inputs["input_ids"])
    np.testing.assert_array_equal(inputs["mm_token_type_ids"], reference_types)
    assert int(inputs["mm_token_type_ids"].sum()) == int(counts.sum())
    # Verify both concatenated patches and grid rows preserve image/turn order.
    cursor = 0
    for index, image in enumerate(images):
        single = adapter.processor(images=[image], return_tensors="np")
        n = len(single["pixel_values"])
        np.testing.assert_array_equal(values[cursor:cursor + n], single["pixel_values"])
        np.testing.assert_array_equal(grid[index], single["image_grid_thw"][0])
        cursor += n
    print(f"REAL_QWEN_PIL_CPU_PASS images={len(images)} tokens={counts.tolist()} grid={grid.tolist()}", flush=True)
    return inputs


# Fail closed on accidental Hub/network access, even outside from_pretrained().
with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")), \
     patch.object(huggingface_hub, "hf_hub_download", return_value=str(fixture)) as download:
    tokenizer = TokenizerFixture()
    adapter = QwenImageProcessor(profile, tokenizer)
    download.assert_called_once_with(REPO, "preprocessor_config.json", revision=REVISION)
    assert adapter.images_policy == profile["images"] and adapter.max_tokens == 1024
    assert adapter.processor.backend == "pil"
    assert adapter.processor.size == {"shortest_edge": 65536, "longest_edge": 1048576}

    images = [Image.new("RGB", (224, 224), "red"), Image.new("RGB", (480, 640), "green"),
              Image.new("RGB", (640, 480), "blue"), Image.new("RGB", (1024, 1024), "white")]
    try:
        for count in range(1, 5):
            check_images(adapter, tokenizer, images[:count])
        for size in ((1, 1), (4096, 32), (32, 4096), (2048, 2048)):
            with Image.new("RGB", size, "black") as image:
                check_images(adapter, tokenizer, [image])
        for mode in ("L", "RGBA"):
            with Image.new(mode, (256, 256)) as image:
                check_images(adapter, tokenizer, [image])
        with Image.new("RGB", (2048, 2048), "red") as large:
            # At exactly the decoded aggregate pixel cap, 2x4MP is valid.
            check_images(adapter, tokenizer, [large, large])
            rejected(lambda: adapter.prepare((adapter.boi_token + adapter.image_token + adapter.eoi_token) * 3,
                                              [large] * 3, tokenizer, return_tensors="np"), "aggregate pixels")
        marker = adapter.boi_token + adapter.image_token + adapter.eoi_token
        for prompt, supplied in ((marker, []), ("text only", images[:1]), (marker * 2, images[:1]),
                                 (adapter.image_token, images[:1]), (marker, images[:2]),
                                 (marker * 5, images + images[:1]),
                                 (marker + "<|video_pad|>", images[:1]),
                                 (marker + "<|audio_pad|>", images[:1]),
                                 (marker + adapter.boi_token, images[:1])):
            rejected(lambda: adapter.prepare(prompt, supplied, tokenizer, return_tensors="np"), "placeholder/modality/count")
        for size in ((4096, 1), (4097, 32), (2049, 2048)):
            with Image.new("RGB", size) as image:
                rejected(lambda: adapter.prepare(marker, [image], tokenizer, return_tensors="np"), "source dimensions")
        rejected(lambda: adapter.prepare(marker, [np.zeros((32, 32, 3))], tokenizer, return_tensors="np"), "non-PIL input")
        rejected(lambda: adapter.prepare(marker, images[:1], tokenizer, return_tensors="tf"), "tensor backend")
        text = adapter.prepare("plain text", [], tokenizer, return_tensors="np")
        assert set(text) == {"input_ids", "attention_mask"}

        class BrokenTokenizer(TokenizerFixture):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                result["input_ids"][result["input_ids"] == 248056] = 123
                return result

        rejected(lambda: adapter.prepare(marker, images[:1], BrokenTokenizer(), return_tensors="np"), "image token mismatch")
        for key, value in (("id", "unreviewed"), ("revision", "0" * 40), ("max_image_tokens", 1025), ("images", {}),
                           ("activation_allowance_mib", 0), ("architecture", "Qwen3VLForConditionalGeneration")):
            rejected(lambda: QwenImageProcessor(dict(profile, **{key: value}), tokenizer), "profile tampering")
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "preprocessor_config.json"
            original = fixture.read_bytes()
            for data in (original + b" ", original.replace(b"16777216", b"16777217")):
                tampered.write_bytes(data)
                with patch.object(huggingface_hub, "hf_hub_download", return_value=str(tampered)):
                    rejected(lambda: QwenImageProcessor(profile, tokenizer), "processor size/digest mismatch")
        with patch("qwen_vision.package_version", return_value="5.22.3"):
            rejected(lambda: QwenImageProcessor(profile, tokenizer), "Transformers version")
        wrong_tokenizer = TokenizerFixture()
        wrong_tokenizer.tokens = dict(tokenizer.tokens, **{"<|image_pad|>": 1})
        rejected(lambda: QwenImageProcessor(profile, wrong_tokenizer), "special token ID")
        if any(entry["id"] == profile["id"] for entry in catalog["profiles"]):
            args = (profile["id"], REPO, REVISION, "qwen3.8-multimodal", "multimodal-lm", "bfloat16", tokenizer)
            with patch.dict(os.environ, {"MODEL_VISION_PROFILE_SPEC": json.dumps(profile)}):
                loaded = load_image_processor(*args)
                assert loaded.max_tokens == 1024 and loaded.images_policy == profile["images"]
            with patch.dict(os.environ, {"MODEL_VISION_PROFILE_SPEC": "null"}):
                rejected(lambda: load_image_processor(*args), "missing approved profile")
    finally:
        for image in images:
            image.close()

assert importlib.util.find_spec("torch") is None
assert importlib.util.find_spec("torchvision") is None
print("QWEN_PROCESSOR_CPU_VALIDATED: real pinned PIL/expansion, no Hub/Torch/torchvision/weights/GPU/model inference", flush=True)
