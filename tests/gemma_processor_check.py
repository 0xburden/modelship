"""Execute the pinned real PIL processor on synthetic images, without Torch/weights.

Run in an isolated CPU environment with tests/processor-requirements.txt.
This proves preprocessing/placeholder compatibility, NOT model/GPU inference.
"""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import huggingface_hub

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'internal/modelship/assets'
sys.path.insert(0, str(ASSETS))

if importlib.util.find_spec('torch') or importlib.util.find_spec('torchvision'):
    raise RuntimeError('this check requires a clean, Torch-free CPU environment')

from gemma_vision import GemmaImageProcessor
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor

profile = json.loads((ASSETS / 'pytorch_vision_profiles.json').read_text())['profiles'][0]
fixture = ROOT / 'internal/modelship/testdata/gemma4-processor-config.json'


class TokenizerFixture:
    def convert_tokens_to_ids(self, token):
        return {'<|image|>': 258880, '<|image>': 255999, '<image|>': 258882}[token]


# No Hub access: only a checked-in, size/digest-pinned small metadata fixture.
with patch.object(huggingface_hub, 'hf_hub_download', return_value=str(fixture)):
    adapter = GemmaImageProcessor(profile, TokenizerFixture())

for width, height in ((224, 224), (480, 640), (640, 480), (4096, 1)):
    with Image.new('RGB', (width, height), 'red') as image:
        inputs = adapter.processor(images=[image], max_soft_tokens=280, return_tensors='np')
    pixels = inputs['pixel_values']
    positions = inputs['image_position_ids']
    count = int(inputs['num_soft_tokens_per_image'][0])
    assert pixels.shape == (1, 2520, 768), pixels.shape
    assert pixels.dtype == np.float32, pixels.dtype
    assert positions.shape == (1, 2520, 2), positions.shape
    assert positions.dtype == np.int64, positions.dtype
    assert 1 <= count <= 280, count
    assert np.isfinite(pixels).all() and pixels.min() >= 0 and pixels.max() <= 1
    assert int(np.all(positions[0] >= 0, axis=-1).sum()) == count * 9
    reference = Gemma4Processor.replace_image_token(
        SimpleNamespace(boi_token=adapter.boi_token, image_token=adapter.image_token, eoi_token=adapter.eoi_token),
        inputs, image_idx=0,
    )
    assert reference == adapter.boi_token + adapter.image_token * count + adapter.eoi_token
    print(f'REAL_PIL_PROCESSOR_CPU_PASS {width}x{height}: {count} soft tokens, padded 2520 patches', flush=True)
print('GEMMA_PROCESSOR_CPU_VALIDATED: no Torch, torchvision, Hub network, model weights or GPU inference', flush=True)

# Distinct size/color inputs expose flattening, reordering and shape errors.
from contextlib import ExitStack
with ExitStack() as stack:
    images = [stack.enter_context(Image.new('RGB', size, color)) for size, color in
              [((224,224),'red'), ((480,640),'blue'), ((640,480),'green'), ((4096,1),'white')]]
    batch = adapter.processor(images=images, max_soft_tokens=280, return_tensors='np')
    assert batch['pixel_values'].shape == (4,2520,768)
    assert batch['image_position_ids'].shape == (4,2520,2)
    assert list(batch['num_soft_tokens_per_image']) == [256,266,266,280]
    for i, image in enumerate(images):
        single = adapter.processor(images=[image], max_soft_tokens=280, return_tensors='np')
        np.testing.assert_array_equal(batch['pixel_values'][i], single['pixel_values'][0])
        np.testing.assert_array_equal(batch['image_position_ids'][i], single['image_position_ids'][0])
print('GEMMA_MULTI_PROCESSOR_CPU_VALIDATED: 4 ordered images equal independent reference calls; no inference')
