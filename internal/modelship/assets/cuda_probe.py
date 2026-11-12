"""Native preflight: validate actual CUDA computation without installing torch."""
import importlib.metadata
import json
import os

import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("expected exactly one CUDA-visible GPU selected by UUID")
value = (torch.ones(4, device="cuda") ** 2).sum().item()
if value != 4.0:
    raise RuntimeError("CUDA computation smoke test failed")
print(json.dumps({
    "torch_version": torch.__version__,
    "torch_package_version": importlib.metadata.version("torch"),
    "torch_file": os.path.realpath(torch.__file__),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
}))
