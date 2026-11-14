"""Pinned Qwen3.8 image-only composition of Transformers 5.22.2's PIL backend.

No repository Python, AutoProcessor, video processor or torchvision. The caller
supplies decoded PIL images in chat-placeholder order (including earlier turns).
The existing model runtime supplies Torch only for return_tensors="pt"; NumPy
preprocessing is independently testable without Torch, weights, Hub or a GPU.
"""
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path


PROFILE_ID = "qwen38-27b-images-v1"
REPO = "Qwen/Qwen3.8-27B"
REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
PROCESSOR_PIN = {
    "name": "preprocessor_config.json",
    "size": 390,
    "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
}
MAX_IMAGE_TOKENS = 1024
ACTIVATION_ALLOWANCE_MIB = 4096


class QwenImageProcessor:
    image_token = "<|image_pad|>"
    boi_token = "<|vision_start|>"
    eoi_token = "<|vision_end|>"
    image_token_id = 248056

    def __init__(self, profile, tokenizer):
        if package_version("transformers") != "5.22.2":
            raise RuntimeError("Qwen vision requires the reviewed Transformers 5.22.2")
        from huggingface_hub import hf_hub_download
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

        if (profile.get("id") != PROFILE_ID or profile.get("repo") != REPO or profile.get("revision") != REVISION
                or profile.get("recipe") != "qwen3.8-multimodal"
                or profile.get("architecture") != "Qwen3_5ForConditionalGeneration"
                or profile.get("processor") != PROCESSOR_PIN
                or profile.get("max_image_tokens") != MAX_IMAGE_TOKENS
                or profile.get("activation_allowance_mib") != ACTIVATION_ALLOWANCE_MIB
                or profile.get("images") != {"max_images": 4, "max_total_image_bytes": 8 * 1024 * 1024,
                                             "max_total_image_pixels": 8 * 1024 * 1024}):
            raise RuntimeError("Qwen vision requires the exact reviewed image profile")
        for token, expected in ((self.image_token, self.image_token_id), (self.boi_token, 248053),
                                (self.eoi_token, 248054), ("<|video_pad|>", 248057)):
            if tokenizer.convert_tokens_to_ids(token) != expected:
                raise RuntimeError("Qwen image special tokens differ from the reviewed profile")
        path = Path(hf_hub_download(REPO, PROCESSOR_PIN["name"], revision=REVISION))
        with path.open("rb") as stream:
            data = stream.read(1024 * 1024 + 1)
        if len(data) != PROCESSOR_PIN["size"] or hashlib.sha256(data).hexdigest() != PROCESSOR_PIN["sha256"]:
            raise RuntimeError("Qwen processor configuration differs from its reviewed size/digest")
        config = json.loads(data)
        if config != {
            "size": {"longest_edge": 16777216, "shortest_edge": 65536},
            "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "processor_class": "Qwen3VLProcessor", "image_processor_type": "Qwen2VLImageProcessorFast",
        }:
            raise RuntimeError("Qwen processor shape differs from the reviewed image-only adapter")
        # Explicit public PIL class, not the config's Fast/Torchvision backend.
        # Cap resized pixels to 1024 merged tokens; retain upstream minimum.
        self.processor = Qwen2VLImageProcessorPil(
            size={"shortest_edge": 65536, "longest_edge": 1048576},
            patch_size=16, temporal_patch_size=2, merge_size=2,
            do_resize=True, resample=3, do_convert_rgb=True,
            do_rescale=True, rescale_factor=1 / 255, do_normalize=True,
            image_mean=config["image_mean"], image_std=config["image_std"],
        )
        self.max_tokens = MAX_IMAGE_TOKENS
        self.images_policy = dict(profile["images"])
        if self.processor.backend != "pil":
            raise RuntimeError("Qwen image adapter requires the reviewed PIL backend")

    def prepare(self, prompt, images, tokenizer, *, return_tensors="pt"):
        import numpy as np
        from PIL import Image
        from transformers.feature_extraction_utils import BatchFeature

        if return_tensors not in ("pt", "np"):
            raise ValueError("unsupported image tensor backend")
        if any(token in prompt for token in ("<|video_pad|>", "<|vision_pad|>",
                                               "<|audio_pad|>", "<|audio_start|>", "<|audio_end|>")):
            raise ValueError("only uploaded still images are supported")
        marker = self.boi_token + self.image_token + self.eoi_token
        count = len(images)
        if (count > 4 or prompt.count(marker) != count or prompt.count(self.image_token) != count
                or prompt.count(self.boi_token) != count or prompt.count(self.eoi_token) != count):
            raise ValueError("image placeholders do not match uploaded images")
        if not images:
            return tokenizer(prompt, return_tensors=return_tensors, add_special_tokens=False)
        pixels = 0
        for image in images:
            if not isinstance(image, Image.Image):
                raise ValueError("image inputs must be decoded PIL images")
            width, height = image.size
            if (min(width, height) < 1 or max(width, height) > 4096
                    or max(width, height) / min(width, height) > 200
                    or width * height > 4 * 1024 * 1024):
                raise ValueError("image dimensions exceed the reviewed Qwen shape bounds")
            pixels += width * height
        if pixels > 8 * 1024 * 1024:
            raise ValueError("aggregate image pixels exceed the reviewed bounds")

        processed = self.processor(images=images, return_tensors="np")
        grid = processed["image_grid_thw"]
        values = processed["pixel_values"]
        if (grid.shape != (count, 3) or grid.dtype != np.int64
                or not np.all(grid[:, 0] == 1) or np.any(grid[:, 1:] < 2)
                or np.any(grid[:, 1:] > 1024) or np.any(grid[:, 1:] % 2)):
            raise ValueError("processor returned an unsupported image grid")
        patches = grid.prod(axis=-1)
        counts = patches // 4
        if (np.any(counts < 1) or np.any(counts > self.max_tokens)
                or values.shape != (int(patches.sum()), 1536) or values.dtype != np.float32
                or not np.isfinite(values).all()):
            raise ValueError("processor returned an unsupported image patch layout")
        # Split on original markers: expanding by repeated replace() would consume
        # a newly inserted image_pad on the second image and corrupt turn order.
        parts = prompt.split(marker)
        expanded = parts[0]
        for index, tokens in enumerate(counts):
            expanded += self.boi_token + self.image_token * int(tokens) + self.eoi_token + parts[index + 1]
        inputs = dict(tokenizer(expanded, return_tensors="np", add_special_tokens=False))
        ids = np.asarray(inputs["input_ids"])
        modality = (ids == self.image_token_id).astype(np.int64)
        # Check per-image runs, not just totals, so framing/order cannot silently
        # differ from the image_grid_thw consumed sequentially by model M-RoPE.
        starts = np.flatnonzero(ids[0] == 248053) if ids.ndim == 2 and ids.shape[0] == 1 else []
        ends = np.flatnonzero(ids[0] == 248054) if len(starts) else []
        if (len(starts) != count or len(ends) != count or int(modality.sum()) != int(counts.sum())
                or any(end - start - 1 != int(tokens) or not np.all(ids[0, start + 1:end] == self.image_token_id)
                       for start, end, tokens in zip(starts, ends, counts))):
            raise ValueError("image tokenization does not match encoded image features")
        inputs["mm_token_type_ids"] = modality
        inputs["pixel_values"] = values
        inputs["image_grid_thw"] = grid
        # No position_ids: pinned Qwen3_5 computes 3D M-RoPE/rope_deltas using
        # image_grid_thw and mm_token_type_ids. Ordinary 1D positions are wrong.
        return BatchFeature(inputs, tensor_type=return_tensors)


def load_image_processor(profile_id, model, revision, recipe, loader, dtype, tokenizer):
    path = Path(__file__).with_name("pytorch_vision_profiles.json")
    if path.stat().st_size > 65536:
        raise RuntimeError("invalid bundled vision profiles")
    catalog = json.loads(path.read_text())
    if catalog.get("schema") != 1 or not 1 <= len(catalog.get("profiles", [])) <= 8:
        raise RuntimeError("invalid bundled vision profiles")
    profiles = [p for p in catalog["profiles"] if p.get("id") == profile_id]
    try:
        approved = json.loads(os.environ.get("MODEL_VISION_PROFILE_SPEC", "null"))
    except ValueError:
        raise RuntimeError("invalid saved vision profile") from None
    if (len(profiles) != 1 or profiles[0] != approved or model != REPO or revision != REVISION
            or recipe != "qwen3.8-multimodal" or loader != "multimodal-lm" or dtype != "bfloat16"):
        raise RuntimeError("vision requires the exact reviewed Qwen image profile")
    return QwenImageProcessor(profiles[0], tokenizer)
