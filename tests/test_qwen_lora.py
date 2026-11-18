"""Exercise reviewed target and hook behavior without loading tensors or CUDA."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class QwenLoRATests(unittest.TestCase):
    def test_reviewed_linear_layout_and_unfused_scale(self):
        calls = []
        tensors = {}
        modules = {}

        class Weight:
            def __init__(self, shape, dtype="bf16", device="cpu"):
                self.shape, self.dtype, self.device = shape, dtype, device

            def to(self, *, device):
                calls.append("transfer")
                return Weight(self.shape, self.dtype, device)

        class Linear:
            def __init__(self, width, height):
                self.weight = Weight((height, width), device="cuda:0")

            def register_forward_hook(self, hook):
                self.hook = hook

        class Transformer:
            def get_submodule(self, path):
                return modules[path]

        class Safetensors:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def keys(self):
                return tensors.keys()

            def get_tensor(self, key):
                return tensors[key]

        def linear(value, weight):
            self.assertEqual(weight.device, "cuda:0")
            calls.append("linear")
            return value * 2

        def open_adapter(path, **kwargs):
            self.assertEqual(path.name, "p0laroid_qwen_v1.safetensors")
            self.assertEqual(kwargs, {"framework": "pt", "device": "cpu"})
            return Safetensors()

        with patch.dict(sys.modules, {
            "torch": types.SimpleNamespace(bfloat16="bf16", nn=types.SimpleNamespace(
                Linear=Linear, functional=types.SimpleNamespace(linear=linear)),
                cuda=types.SimpleNamespace(synchronize=lambda: calls.append("drain"))),
            "diffusers": types.SimpleNamespace(__version__="0.42.1", QwenImage21Transformer2DModel=Transformer),
            "safetensors": types.SimpleNamespace(safe_open=open_adapter),
        }):
            path = Path(__file__).resolve().parents[1] / "internal/modelship/assets/qwen_lora.py"
            spec = importlib.util.spec_from_file_location("qwen_lora_test", path)
            lora = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lora)
            for index in range(32):
                for target, (width, height) in lora.TARGETS.items():
                    name = f"transformer_blocks.{index}.{target}"
                    modules[name] = Linear(width, height)
                    tensors[f"{name}.lora_A.default.weight"] = Weight((32, width))
                    tensors[f"{name}.lora_B.default.weight"] = Weight((height, 32))
            profile = lora.adapter_profile()
            tensors["unexpected"] = Weight((1,))
            with self.assertRaisesRegex(RuntimeError, "tensor names"):
                lora.install_adapter(Transformer(), profile)
            del tensors["unexpected"]
            self.assertEqual(calls, [])
            target = modules["transformer_blocks.0.attn.to_q"]
            target.weight.device = "cpu"
            with self.assertRaisesRegex(RuntimeError, "GPU placement"):
                lora.install_adapter(Transformer(), profile)
            target.weight.device = "cuda:0"
            adapter = lora.install_adapter(Transformer(), profile)
            self.assertEqual(calls, ["transfer"] * 448 + ["drain"])
            self.assertTrue(all(hasattr(module, "hook") for module in modules.values()))
            calls.clear()
            marker = object()
            self.assertIs(target.hook(target, (3,), marker), marker)
            self.assertEqual(calls, [])
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with adapter.request(.75):
                    self.assertEqual(target.hook(target, (3,), 7), 16)
                    raise RuntimeError("fixture failure")
            self.assertEqual(calls, ["linear", "linear"])
            self.assertIs(target.hook(target, (3,), marker), marker)
