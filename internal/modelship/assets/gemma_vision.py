"""Reviewed Gemma4 image-only processor composition; no repository Python.

Uses Transformers 5.22.2's public PIL image processor and the reviewed tokenizer
chat template. Placeholder expansion/modality IDs follow that release's
Gemma4Processor/ProcessorMixin, without loading unused audio/video processors or
installing torchvision. Model inference still requires the existing CUDA Torch.
"""
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path


class GemmaImageProcessor:
    image_token = "<|image|>"
    boi_token = "<|image>"
    eoi_token = "<image|>"
    image_token_id = 258880

    def __init__(self, profile, tokenizer):
        if package_version("transformers") != "5.22.2":
            raise RuntimeError("Gemma vision requires the reviewed Transformers 5.22.2")
        from huggingface_hub import hf_hub_download
        from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil

        for token, expected in ((self.image_token, self.image_token_id), (self.boi_token, 255999), (self.eoi_token, 258882)):
            if tokenizer.convert_tokens_to_ids(token) != expected:
                raise RuntimeError("Gemma image special tokens differ from the reviewed profile")
        pin = profile["processor"]
        path = Path(hf_hub_download(profile["repo"], pin["name"], revision=profile["revision"]))
        with path.open("rb") as stream:
            data = stream.read(1024 * 1024 + 1)
        if len(data) != pin["size"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
            raise RuntimeError("Gemma processor configuration differs from its reviewed size/digest")
        config = json.loads(data)
        image = config.get("image_processor", {})
        if config.get("processor_class") != "Gemma4Processor" or config.get("image_seq_length") != 280 or image.get("image_processor_type") != "Gemma4ImageProcessor" or image.get("max_soft_tokens") != 280 or image.get("patch_size") != 16 or image.get("pooling_kernel_size") != 3:
            raise RuntimeError("Gemma processor shape differs from the reviewed image-only adapter")
        # Only reviewed image preprocessing settings, never generic Hub URLs,
        # audio/video processor settings or arbitrary processor kwargs.
        keys = ("do_convert_rgb", "do_normalize", "do_rescale", "do_resize", "image_mean", "image_std",
                "max_soft_tokens", "patch_size", "pooling_kernel_size", "resample", "rescale_factor")
        self.processor = Gemma4ImageProcessorPil(**{key: image[key] for key in keys})
        self.max_tokens = profile["max_image_tokens"]
        if self.processor.backend != "pil":
            raise RuntimeError("Gemma image adapter requires the reviewed PIL backend")

    def prepare(self, prompt, images, tokenizer):
        import torch
        if not images:
            if self.image_token in prompt:
                raise ValueError("image markers require an uploaded image")
            return tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if len(images) != 1 or prompt.count(self.image_token) != 1:
            raise ValueError("image placeholders do not match the uploaded image")
        processed = self.processor(images=images, max_soft_tokens=self.max_tokens, return_tensors="pt")
        count = int(processed["num_soft_tokens_per_image"][0])
        if not 1 <= count <= self.max_tokens or tuple(processed["pixel_values"].shape) != (1, 2520, 768) or tuple(processed["image_position_ids"].shape) != (1, 2520, 2):
            raise ValueError("processor returned an unsupported image patch layout")
        expanded = prompt.replace(self.image_token, self.boi_token + self.image_token * count + self.eoi_token, 1)
        inputs = dict(tokenizer(expanded, return_tensors="pt", add_special_tokens=False))
        modality = inputs["input_ids"].eq(self.image_token_id).to(dtype=torch.long)
        if int(modality.sum().item()) != count:
            raise ValueError("image tokenization does not match encoded image features")
        inputs["mm_token_type_ids"] = modality
        inputs["pixel_values"] = processed["pixel_values"]
        inputs["image_position_ids"] = processed["image_position_ids"]
        return inputs


def load_image_processor(profile_id, model, revision, recipe, loader, dtype, tokenizer):
    path = Path(__file__).with_name("pytorch_vision_profiles.json")
    if path.stat().st_size > 16384:
        raise RuntimeError("invalid bundled vision profiles")
    catalog = json.loads(path.read_text())
    if catalog.get("schema") != 1 or len(catalog.get("profiles", [])) != 1:
        raise RuntimeError("invalid bundled vision profiles")
    profile = next((p for p in catalog["profiles"] if p["id"] == profile_id), None)
    try:
        approved = json.loads(os.environ.get("MODEL_VISION_PROFILE_SPEC", "null"))
    except ValueError:
        raise RuntimeError("invalid saved vision profile") from None
    if profile is None or profile != approved or profile.get("max_image_tokens") != 280 or profile.get("activation_allowance_mib") != 2048 or model != profile["repo"] or revision != profile["revision"] or recipe != profile["recipe"] or loader != "multimodal-lm" or dtype != "bfloat16":
        raise RuntimeError("vision requires the exact reviewed Gemma image profile")
    return GemmaImageProcessor(profile, tokenizer)
