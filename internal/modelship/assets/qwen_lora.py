"""Exact rank-32 Qwen Image 2.1 adapter; no PEFT, pickle, or fusion."""
import json
from contextlib import contextmanager
from pathlib import Path

import torch

TARGETS = {
    "attn.to_q": (4096, 4096), "attn.to_k": (4096, 4096),
    "attn.to_v": (4096, 4096), "attn.to_out.0": (4096, 4096),
    "img_mlp.gate_layer": (4096, 12288), "img_mlp.proj": (4096, 12288),
    "img_mlp.out": (12288, 4096),
}


class ImageAdapter:
    def __init__(self, profile):
        self.profile = profile
        self.strength = 0.0

    @contextmanager
    def request(self, strength):
        # The caller holds the exclusive pipeline lock through CUDA drain.
        self.strength = strength
        try:
            yield
        finally:
            self.strength = 0.0

    def hook(self, a, b):
        def apply(_module, inputs, output):
            if not self.strength:
                return output  # No adapter kernels and no base output arithmetic.
            x, = inputs
            linear = torch.nn.functional.linear
            return output + self.strength * linear(linear(x, a), b)
        return apply


def adapter_profile():
    return json.loads(Path(__file__).with_name("qwen_lora_profile.json").read_text(encoding="utf-8"))


def install_adapter(transformer, profile):
    import diffusers
    from diffusers import QwenImage21Transformer2DModel
    from safetensors import safe_open

    if diffusers.__version__ != "0.42.1" or type(transformer) is not QwenImage21Transformer2DModel:
        raise RuntimeError("unexpected LoRA transformer class or runtime version")
    root = Path(__file__).parent
    adapter = ImageAdapter(profile)
    modules = []
    for block in range(32):
        for target, (inputs, outputs) in TARGETS.items():
            path = f"transformer_blocks.{block}.{target}"
            module = transformer.get_submodule(path)
            if (type(module) is not torch.nn.Linear or module.weight.dtype != torch.bfloat16
                    or str(module.weight.device) != "cuda:0" or tuple(module.weight.shape) != (outputs, inputs)):
                raise RuntimeError("LoRA target layout or GPU placement does not match the reviewed profile")
            modules.append((path, module, (32, inputs), (outputs, 32)))
    # The CLI validates the private local file's exact digest before spawning.
    # Open only that file, enforce every key/shape/dtype, and transfer all
    # adapters to the same device as base weights (never offload or fuse).
    with safe_open(root / profile["file"], framework="pt", device="cpu") as data:
        expected = {f"{path}.lora_{key}.default.weight" for path, *_ in modules for key in ("A", "B")}
        if set(data.keys()) != expected:
            raise RuntimeError("LoRA tensor names do not match the reviewed profile")
        for path, module, a_shape, b_shape in modules:
            pair = []
            for key, shape in (("A", a_shape), ("B", b_shape)):
                weight = data.get_tensor(f"{path}.lora_{key}.default.weight")
                if tuple(weight.shape) != shape or weight.dtype != torch.bfloat16:
                    raise RuntimeError("LoRA rank, shape or dtype does not match the reviewed profile")
                pair.append(weight.to(device=module.weight.device))
            module.register_forward_hook(adapter.hook(*pair))
    torch.cuda.synchronize()
    return adapter
