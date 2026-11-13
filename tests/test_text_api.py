"""Native text HTTP contracts using fake model/tokenizer; no CUDA or weights."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "internal/modelship/assets"))
CALLS = []


class Tensor:
    def __init__(self, count=4):
        self.shape = (1, count)

    def to(self, device):
        return self


class Output:
    def __getitem__(self, key):
        return [7, 8]


class Tokenizer:
    eos_token_id = 1
    chat_template = "fake-template"

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        CALLS.append(("tokenizer", model, kwargs))
        return cls()

    def __call__(self, prompt, **kwargs):
        return {"input_ids": Tensor(), "attention_mask": Tensor()}

    def apply_chat_template(self, messages, **kwargs):
        CALLS.append(("chat", messages, kwargs))
        return "chat prompt"

    def decode(self, tokens, **kwargs):
        return "Paris"


class Network:
    @classmethod
    def from_pretrained(cls, model, **kwargs):
        CALLS.append(("load", model, kwargs))
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        CALLS.append(("generate", kwargs))
        return Output()


class OOM(Exception):
    pass


sys.modules["torch"] = types.SimpleNamespace(
    float16="fp16", bfloat16="bf16", inference_mode=nullcontext, OutOfMemoryError=OOM,
    manual_seed=lambda seed: None, cuda=types.SimpleNamespace(empty_cache=lambda: None),
)
sys.modules["transformers"] = types.SimpleNamespace(AutoModelForCausalLM=Network, AutoTokenizer=Tokenizer)
spec = importlib.util.spec_from_file_location("modelship_text_server", ROOT / "internal/modelship/assets/text_server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class NativeTextAPITests(unittest.TestCase):
    def setUp(self):
        CALLS.clear()
        server.MODEL = "org/model"
        server.REVISION = "a" * 40
        server.RUN_ID = "b" * 32
        server.CONTEXT = 4096
        server.RECIPE = ""
        server.TEMPLATE_COMPLETIONS = False
        server.DISABLE_THINKING = False

    def test_completion_and_pinned_loading(self):
        with TestClient(server.app) as client:
            self.assertEqual(client.get("/health").json()["run_id"], "b" * 32)
            self.assertEqual(client.get("/v1/models").json()["data"][0]["id"], "org/model")
            response = client.post("/v1/completions", json={"model": "org/model", "prompt": "Capital?", "max_tokens": 8})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["choices"][0]["text"], "Paris")
            self.assertEqual(result["usage"], {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})
        load = next(call[2] for call in CALLS if call[0] == "load")
        self.assertEqual(load["revision"], "a" * 40)
        self.assertTrue(load["use_safetensors"])
        self.assertFalse(load["trust_remote_code"])
        self.assertEqual(load["attn_implementation"], "eager")

    def test_gemma_multimodal_loader_is_text_only_and_thinking_disabled(self):
        server.RECIPE = "gemma4-text"
        server.TEMPLATE_COMPLETIONS = True
        server.DISABLE_THINKING = True
        with patch.object(sys.modules["transformers"], "AutoModelForMultimodalLM", Network, create=True):
            with patch.dict("os.environ", {"MODEL_NATIVE_LOADER": "multimodal-lm", "MODEL_DTYPE": "bfloat16"}):
                with TestClient(server.app) as client:
                    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}]})
                    self.assertEqual(response.status_code, 200, response.text)
                    completion = client.post("/v1/completions", json={"prompt": "Capital?"})
                    self.assertEqual(completion.status_code, 200, completion.text)
                    image_request = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "https://example.com/x"}]}]})
                    self.assertEqual(image_request.status_code, 422)
        load = next(call[2] for call in CALLS if call[0] == "load")
        self.assertEqual(load["dtype"], "bf16")
        self.assertFalse(load["trust_remote_code"])
        chat = next(call[2] for call in CALLS if call[0] == "chat")
        self.assertFalse(chat["enable_thinking"])
        templated_messages = [call[1] for call in CALLS if call[0] == "chat"]
        self.assertIn([{"role": "user", "content": "Capital?"}], templated_messages)

    def test_chat(self):
        with TestClient(server.app) as client:
            response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}], "temperature": 0.5, "seed": 42})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["choices"][0]["message"], {"role": "assistant", "content": "Paris"})
            self.assertEqual(response.json()["object"], "chat.completion")
        args = next(call[1] for call in CALLS if call[0] == "generate")
        self.assertTrue(args["do_sample"])
        self.assertEqual(args["temperature"], 0.5)

    def test_invalid_requests_and_missing_model(self):
        with TestClient(server.app) as client:
            for body in ({"prompt": "x", "stream": True}, {"prompt": ""}, {"prompt": "x", "max_tokens": 9999}, {"prompt": "x", "unknown": True}):
                self.assertEqual(client.post("/v1/completions", json=body).status_code, 422)
            self.assertEqual(client.post("/v1/completions", json={"prompt": "x", "model": "other"}).status_code, 404)
            self.assertEqual(client.post("/v1/chat/completions", json={"messages": [{"role": "tool", "content": "x"}]}).status_code, 422)

    def test_context_limit(self):
        server.CONTEXT = 8
        with TestClient(server.app) as client:
            response = client.post("/v1/completions", json={"prompt": "x", "max_tokens": 8})
            self.assertEqual(response.status_code, 422)
        self.assertFalse(any(call[0] == "generate" for call in CALLS))

    def test_missing_chat_template(self):
        with TestClient(server.app) as client:
            server.TOKENIZER.chat_template = None
            response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
            self.assertEqual(response.status_code, 422)

    def test_busy_and_oom(self):
        with TestClient(server.app) as client:
            server.BUSY.acquire()
            try:
                self.assertEqual(client.post("/v1/completions", json={"prompt": "x"}).status_code, 503)
            finally:
                server.BUSY.release()
            def fail(**kwargs):
                raise OOM()
            server.NETWORK.generate = fail
            self.assertEqual(client.post("/v1/completions", json={"prompt": "x"}).status_code, 503)
            self.assertFalse(server.BUSY.locked())

    def test_body_limit_and_no_prompt_echo(self):
        with TestClient(server.app) as client:
            response = client.post("/v1/completions", content=b"x" * (server.MAX_BODY + 1))
            self.assertEqual(response.status_code, 413)
            response = client.post("/v1/completions", json={"prompt": "sensitive", "unknown": True})
            self.assertNotIn("sensitive", response.text)
