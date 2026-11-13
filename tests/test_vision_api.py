"""Gemma image-understanding HTTP/processor contracts: CPU fixtures, never weights/CUDA."""
import base64
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import struct
import tempfile
import sys
import types
import unittest
from unittest.mock import patch
import zlib
from contextlib import nullcontext

from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'internal/modelship/assets'
if str(ASSETS) not in sys.path:
    sys.path.insert(0, str(ASSETS))
import gemma_vision
import vision_input
from api_auth import Access, BearerAuth

PROFILE = json.loads((ASSETS / 'pytorch_vision_profiles.json').read_text())['profiles'][0]
CALLS = []


class Tensor:
    def __init__(self, shape, dtype='int64', values=None):
        self.shape, self.dtype, self.values = shape, dtype, values

    def to(self, device=None, dtype=None):
        CALLS.append(('move', self.shape, device, dtype))
        return Tensor(self.shape, dtype or self.dtype, self.values)

    def eq(self, value):
        return Tensor(self.shape, 'bool', [int(v == value) for v in self.values])

    def sum(self):
        return types.SimpleNamespace(item=lambda: sum(self.values))


class Tokenizer:
    eos_token_id = 1
    chat_template = 'reviewed-fixture'

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        CALLS.append(('tokenizer-load', model, kwargs))
        return cls()

    def convert_tokens_to_ids(self, token):
        return {'<|image|>': 258880, '<|image>': 255999, '<image|>': 258882}[token]

    def apply_chat_template(self, messages, **kwargs):
        CALLS.append(('template', messages, kwargs))
        # Test the actual template handoff without giving any URL to it.
        image_parts = sum(p.get('type') == 'image' for m in messages
                          if isinstance(m['content'], list) for p in m['content'])
        return 'user ' + '<|image|>' * image_parts + ' describe assistant'

    def __call__(self, prompt, **kwargs):
        count = prompt.count('<|image|>')
        CALLS.append(('tokenize', prompt, kwargs))
        values = [2, 3] + [258880] * count + [4, 5]
        return {'input_ids': Tensor((1, len(values)), values=values),
                'attention_mask': Tensor((1, len(values)), values=[1] * len(values))}

    def decode(self, tokens, **kwargs):
        return 'A red square.'


class Network:
    @classmethod
    def from_pretrained(cls, model, **kwargs):
        CALLS.append(('network-load', model, kwargs))
        return cls()

    def eval(self):
        return self

    def generate(self, **kwargs):
        CALLS.append(('generate', kwargs))
        return Output()


class Output:
    def __getitem__(self, key):
        return [7, 8, 9]


class OOM(Exception):
    pass


def processor_pixels(**kwargs):
    CALLS.append(('pixels', kwargs))
    return {'pixel_values': Tensor((1, 2520, 768), 'fp32'),
            'image_position_ids': Tensor((1, 2520, 2), 'int64'),
            'num_soft_tokens_per_image': [256]}


def data_image(mime='image/png', size=(2, 2)):
    image = Image.new('RGB', size, 'red')
    output = BytesIO()
    image.save(output, format='PNG' if mime == 'image/png' else 'JPEG')
    image.close()
    return 'data:' + mime + ';base64,' + base64.b64encode(output.getvalue()).decode()


def chat(url=None):
    return {'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Describe this.'},
        {'type': 'image_url', 'image_url': {'url': url or data_image()}},
    ]}]}


class VisionAPITests(unittest.TestCase):
    def setUp(self):
        CALLS.clear()
        self.torch = types.SimpleNamespace(float16='fp16', bfloat16='bf16', long='int64',
            inference_mode=nullcontext, OutOfMemoryError=OOM, manual_seed=lambda value: None,
            cuda=types.SimpleNamespace(empty_cache=lambda: None))
        modules = {'torch': self.torch, 'transformers': types.SimpleNamespace(
            AutoModelForCausalLM=Network, AutoTokenizer=Tokenizer, AutoModelForMultimodalLM=Network)}
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start(); self.addCleanup(self.modules.stop)
        self.env = patch.dict(os.environ, {'MODEL_VISION_PROFILE': PROFILE['id'], 'MODEL_ID': PROFILE['repo'],
            'MODEL_REVISION': PROFILE['revision'], 'MODEL_RECIPE': PROFILE['recipe'],
            'MODEL_NATIVE_LOADER': 'multimodal-lm', 'MODEL_DTYPE': 'bfloat16',
            'MODEL_TEMPLATE_COMPLETIONS': 'true', 'MODEL_DISABLE_THINKING': 'true',
            'MODEL_VISION_PROFILE_SPEC': json.dumps(PROFILE)})
        self.env.start(); self.addCleanup(self.env.stop)
        spec = importlib.util.spec_from_file_location('modelship_vision_server_fixture', ASSETS / 'text_server.py')
        self.server = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.server)
        # Keep the real class for prepare(); only replace the loader function.
        self.real_processor_class = gemma_vision.GemmaImageProcessor
        self.loader = patch.object(gemma_vision, 'GemmaImageProcessor', side_effect=self.load)
        self.loader.start(); self.addCleanup(self.loader.stop)

    def load(self, profile, tokenizer):
        result = self.real_processor_class.__new__(self.real_processor_class)
        result.max_tokens, result.processor = 280, processor_pixels
        return result

    def test_png_jpeg_full_message_pixel_dtype_and_capability_contract(self):
        with TestClient(self.server.app) as client:
            caps = client.get('/v1/capabilities').json()
            self.assertEqual(caps['schema_version'], 1)
            self.assertEqual(caps['model'], PROFILE['repo'])
            self.assertEqual(caps['chat']['max_image_tokens'], 280)
            self.assertTrue(caps['chat']['image_input'])
            self.assertNotIn('#/$defs/', json.dumps(client.get('/openapi.json').json()))
            for mime in ('image/png', 'image/jpeg'):
                response = client.post('/v1/chat/completions', json=chat(data_image(mime)))
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['choices'][0]['message']['content'], 'A red square.')
                self.assertEqual(response.json()['usage']['prompt_tokens'], 260)
            text = client.post('/v1/completions', json={'prompt': 'Text-only still works'})
            self.assertEqual(text.status_code, 200, text.text)
        inputs = next(c[1] for c in CALLS if c[0] == 'generate')
        self.assertEqual(inputs['pixel_values'].dtype, 'bf16')
        for key in ('input_ids', 'attention_mask', 'image_position_ids', 'mm_token_type_ids'):
            self.assertEqual(inputs[key].dtype, 'int64', key)
        self.assertNotIn('num_soft_tokens_per_image', inputs)
        messages = next(c[1] for c in CALLS if c[0] == 'template')
        self.assertEqual(messages[0]['content'][1], {'type': 'image'})
        self.assertNotIn('data:image', repr(messages))
        self.assertFalse(next(c[2] for c in CALLS if c[0] == 'template')['enable_thinking'])
        self.assertTrue(next(c[2] for c in CALLS if c[0] == 'network-load')['use_safetensors'])
        self.assertFalse(next(c[2] for c in CALLS if c[0] == 'network-load')['trust_remote_code'])

    def test_rejects_remote_file_multiple_unknown_or_wrong_role_images_before_generation(self):
        with TestClient(self.server.app) as client:
            for url in ('https://example.com/secret.png', 'http://169.254.169.254/metadata', 'file:///etc/passwd',
                        'data:image/webp;base64,abcd', 'data:image/png;base64,invalid!', data_image().replace('image/png', 'image/jpeg')):
                response = client.post('/v1/chat/completions', json=chat(url))
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNotIn(url, response.text)
            for change in ('multiple', 'assistant', 'detail', 'no-text'):
                body = chat()
                if change == 'multiple': body['messages'].append(body['messages'][0])
                elif change == 'assistant': body['messages'][0]['role'] = 'assistant'
                elif change == 'detail': body['messages'][0]['content'][1]['image_url']['detail'] = 'high'
                else: body['messages'][0]['content'].pop(0)
                self.assertEqual(client.post('/v1/chat/completions', json=body).status_code, 422, change)
        self.assertFalse(any(c[0] in ('generate', 'pixels') for c in CALLS))

    def test_image_tokens_count_toward_context_before_cuda_transfer(self):
        self.server.CONTEXT = 128
        with TestClient(self.server.app) as client:
            response = client.post('/v1/chat/completions', json=chat())
            self.assertEqual(response.status_code, 422, response.text)
        self.assertFalse(any(c[0] == 'generate' or c[0] == 'move' and c[2] == 'cuda' for c in CALLS))

    def test_busy_oom_and_auth_before_media_parsing(self):
        key = 'a' * 43
        wrapped = BearerAuth(self.server.app, Access('0.0.0.0', hashlib.sha256(key.encode()).digest()))
        with TestClient(wrapped) as client:
            self.assertEqual(client.get('/v1/capabilities').status_code, 401)
            self.assertEqual(client.post('/v1/chat/completions', json=chat('file:///secret')).status_code, 401)
            auth = {'Authorization': 'Bearer ' + key}
            self.assertEqual(client.get('/v1/capabilities', headers=auth).status_code, 200)
            self.server.BUSY.acquire()
            try: self.assertEqual(client.post('/v1/chat/completions', json=chat(), headers=auth).status_code, 503)
            finally: self.server.BUSY.release()
            def oom(**kwargs): raise OOM()
            self.server.NETWORK.generate = oom
            self.assertEqual(client.post('/v1/chat/completions', json=chat(), headers=auth).status_code, 503)
            self.assertFalse(self.server.BUSY.locked())

    def test_invalid_profile_fails_before_model_weight_loading(self):
        self.server.REVISION = 'f' * 40
        with self.assertRaises(RuntimeError):
            with TestClient(self.server.app): pass
        self.assertFalse(any(c[0] == 'network-load' for c in CALLS))

    def test_body_limit_is_explicit_and_does_not_echo_media(self):
        with TestClient(self.server.app) as client:
            response = client.post('/v1/chat/completions', content=b'x' * (self.server.MAX_BODY + 1))
            self.assertEqual(response.status_code, 413)
            self.assertNotIn('xxxxx', response.text)


class VisionProfileBindingTests(unittest.TestCase):
    def test_bundle_and_saved_profile_must_match_before_processor_construction(self):
        args = (PROFILE['id'], PROFILE['repo'], PROFILE['revision'], PROFILE['recipe'], 'multimodal-lm', 'bfloat16', Tokenizer())
        with patch.object(gemma_vision, 'GemmaImageProcessor') as construct:
            with patch.dict(os.environ, {'MODEL_VISION_PROFILE_SPEC': json.dumps(PROFILE)}):
                gemma_vision.load_image_processor(*args)
                self.assertEqual(construct.call_count, 1)
            construct.reset_mock()
            for spec in ('null', '{}', 'invalid', json.dumps({**PROFILE, 'max_image_tokens':1120})):
                with patch.dict(os.environ, {'MODEL_VISION_PROFILE_SPEC':spec}):
                    with self.assertRaises(RuntimeError): gemma_vision.load_image_processor(*args)
            self.assertFalse(construct.called)

    def test_processor_metadata_size_hash_and_pil_configuration_are_enforced(self):
        path = ROOT / 'internal/modelship/testdata/gemma4-processor-config.json'
        class Pil:
            backend = 'pil'
            def __init__(self, **kwargs): self.kwargs = kwargs
        module = types.SimpleNamespace(Gemma4ImageProcessorPil=Pil)
        with patch.object(gemma_vision, 'package_version', return_value='5.22.2'), patch.dict(sys.modules, {
            'huggingface_hub':types.SimpleNamespace(hf_hub_download=lambda *a,**k:str(path)),
            'transformers.models.gemma4.image_processing_pil_gemma4':module,
        }):
            processor = gemma_vision.GemmaImageProcessor(PROFILE, Tokenizer())
            self.assertEqual(processor.processor.kwargs['max_soft_tokens'],280)
            self.assertEqual(processor.processor.kwargs['patch_size'],16)
            with tempfile.TemporaryDirectory() as directory:
                corrupt = Path(directory)/'processor.json';corrupt.write_text('{}')
                with patch.dict(sys.modules, {'huggingface_hub':types.SimpleNamespace(hf_hub_download=lambda *a,**k:str(corrupt))}):
                    with self.assertRaises(RuntimeError):gemma_vision.GemmaImageProcessor(PROFILE,Tokenizer())


class VisionDecoderTests(unittest.TestCase):
    def test_pixel_budget_is_checked_before_full_decode(self):
        data = bytearray(base64.b64decode(data_image().split(',')[1]))
        data[16:24] = struct.pack('>II', 4096, 4096)
        data[29:33] = struct.pack('>I', zlib.crc32(data[12:29]) & 0xffffffff)
        with self.assertRaises(ValueError):
            vision_input.decode_image('data:image/png;base64,' + base64.b64encode(data).decode())

    def test_truncated_noncanonical_and_oversized_images_are_rejected(self):
        valid = data_image()
        for url in (valid[:-16], valid + '\n', 'data:image/png;base64,' + base64.b64encode(b'x' * (vision_input.MAX_IMAGE_BYTES + 1)).decode()):
            with self.assertRaises(ValueError): vision_input.decode_image(url)

    def test_animated_png_is_not_silently_first_framed(self):
        output = BytesIO()
        with Image.new('RGB', (2, 2), 'red') as first, Image.new('RGB', (2, 2), 'blue') as second:
            first.save(output, format='PNG', save_all=True, append_images=[second], duration=20)
        with self.assertRaises(ValueError):
            vision_input.decode_image('data:image/png;base64,' + base64.b64encode(output.getvalue()).decode())


if __name__ == '__main__':
    unittest.main()
