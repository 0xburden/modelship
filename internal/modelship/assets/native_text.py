"""Reviewed Gemma chat streaming/optional reasoning. No repository Python.

One producer owns the GPU slot until generation and CUDA drain finish. Client
cancellation is cooperative, never a promise to interrupt an in-flight kernel.
"""
import asyncio
from importlib.metadata import version
import json
import os
from pathlib import Path
import queue
import threading
import time

from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

MAX_TEXT_BYTES = 1024 * 1024
MAX_SECONDS = 900
QUEUE_SIZE = 32
TOKEN_IDS = {'<|channel>': 100, '<channel|>': 101, '<|think|>': 98,
             '<|turn>': 105, '<turn|>': 106, '<eos>': 1, '<bos>': 2}


def load_text_profile(profile_id, model, revision, recipe, loader, dtype, tokenizer):
    if not profile_id:
        return None
    path = Path(__file__).with_name('native_text_profiles.json')
    if path.stat().st_size > 16384:
        raise RuntimeError('invalid bundled text profiles')
    catalog = json.loads(path.read_text())
    spec = json.loads(os.environ.get('MODEL_TEXT_PROFILE_SPEC', 'null'))
    profile = next((p for p in catalog.get('profiles', []) if p.get('id') == profile_id), None)
    if catalog.get('schema') != 1 or len(catalog.get('profiles', [])) != 1 or profile is None or profile != spec or profile_id != 'gemma4-e4b-text-controls-v1' or model != profile['repo'] or revision != profile['revision'] or recipe != profile['recipe'] or loader != 'multimodal-lm' or dtype != 'bfloat16' or profile['max_output_tokens'] != 1024 or profile['default_enable_thinking'] is not False or os.environ.get('MODEL_TEMPLATE_COMPLETIONS') != 'true' or os.environ.get('MODEL_DISABLE_THINKING') != 'true' or version('transformers') != '5.22.2':
        raise RuntimeError('text controls require the exact reviewed Gemma profile')
    for token, expected in TOKEN_IDS.items():
        if tokenizer.convert_tokens_to_ids(token) != expected:
            raise RuntimeError('Gemma channel tokens differ from the reviewed profile')
    if (profile['channel_start_id'], profile['channel_end_id'], profile['think_token_id']) != (100, 101, 98):
        raise RuntimeError('invalid Gemma channel profile')
    return profile


def chat_controls(profile):
    if not profile:
        return {}
    return {'streaming': True, 'text_controls': {
        'schema_version': 1, 'profile': profile['id'],
        'enable_thinking': True, 'default_enable_thinking': False,
        'max_output_tokens': 1024,
        'supported_parameters': ['top_p', 'seed', 'chat_template_kwargs.enable_thinking'],
    }}


class OutputStopped(Exception):
    pass


class OutputLayoutError(Exception):
    pass


class GemmaChannelStreamer:
    """Recognize actual channel token IDs, never regex-guess chain-of-thought.

    The pinned TextStreamer handles partial Unicode/word boundaries without
    stdout. Only its finalized text callback enters the bounded output queue.
    """
    def __init__(self, tokenizer, thinking, emit):
        from transformers import TextStreamer
        self.tokenizer, self.thinking, self.emit = tokenizer, thinking, emit
        self.state, self.prompt, self.closed = 'answer', True, False
        self.started, self.think_seen, self.tokens_seen = False, False, False
        self.header, self.header_tokens, self.size = '', 0, 0
        self.header_ids = []
        self.header_complete = False
        owner = self
        class Decoder(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                owner.finalized(text)
        self.decoder_class = Decoder
        self.decoder = Decoder(tokenizer, skip_prompt=False, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
        self.special = set(getattr(tokenizer, 'all_special_ids', [])) | set(TOKEN_IDS.values())

    def finalized(self, text):
        if not text:
            return
        if self.state == 'header':
            self.header += text
            if '\n' not in self.header:
                if len(self.header) > 64:
                    raise OutputLayoutError()
                if not 'thought\n'.startswith(self.header):
                    raise OutputLayoutError()
                return
            prefix, text = self.header.split('\n', 1)
            if prefix != 'thought':
                raise OutputLayoutError()
            self.header, self.state = '', 'reasoning'
        if text:
            self.size += len(text.encode('utf-8'))
            if self.size > MAX_TEXT_BYTES:
                raise OutputLayoutError()
            key = 'reasoning_content' if self.state == 'reasoning' else 'content'
            for offset in range(0, len(text), 2048):
                self.emit({key: text[offset:offset + 2048]})

    def flush(self):
        self.decoder.end()
        self.decoder = self.decoder_class(self.tokenizer, skip_prompt=False,
            skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def put(self, value):
        ids = value.tolist()
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise OutputLayoutError()
            ids = ids[0]
        if self.prompt:
            self.prompt = False
            return
        for token in ids:
            if self.closed:
                raise OutputLayoutError()
            if token == 98:
                if not self.thinking or self.tokens_seen or self.think_seen:
                    raise OutputLayoutError()
                self.think_seen = True
            elif token == 100:
                if not self.thinking or self.tokens_seen or self.started:
                    raise OutputLayoutError()
                self.started, self.state = True, 'header'
            elif token == 101:
                if self.state != 'reasoning':
                    raise OutputLayoutError()
                self.flush()
                self.state = 'answer'
            elif token in (1, 106):
                self.end()
            elif token in self.special:
                raise OutputLayoutError()
            else:
                self.tokens_seen = True
                if self.state == 'header' and not self.header_complete:
                    self.header_tokens += 1
                    self.header_ids.append(token)
                    header = self.tokenizer.decode(self.header_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    self.header_complete = header.startswith('thought\n')
                    if self.header_tokens > 32 and not self.header_complete:
                        raise OutputLayoutError()
                # Preserve the real tensor shape/dtype expected by TextStreamer.
                self.decoder.put(value.new_tensor([token]))

    def end(self):
        if self.closed:
            return
        self.flush()
        if self.state == 'header':
            raise OutputLayoutError()
        self.closed = True


ERRORS = {
    'oom': 'GPU out of memory; reduce the token/context budget or competing workloads.',
    'timeout': 'Generation deadline exceeded. GPU work is draining; do not resubmit blindly.',
    'layout': 'The model returned an unsupported channel/token layout. No retry was made.',
    'failed': 'Generation failed. Inspect the private service log; no retry was made.',
    'drain': 'CUDA drain failed; this worker requires an explicit restart.',
}


class NativeTextJob:
    def __init__(self, tokenizer, thinking, work, cleanup, prompt_tokens, max_tokens, identity, oom_type):
        self.queue, self.stop, self.done = queue.Queue(QUEUE_SIZE), threading.Event(), threading.Event()
        self.deadline = time.monotonic() + MAX_SECONDS
        self.work, self.cleanup, self.oom_type = work, cleanup, oom_type
        self.prompt_tokens, self.max_tokens, self.identity = prompt_tokens, max_tokens, identity
        self.streamer = GemmaChannelStreamer(tokenizer, thinking, self.emit)
        self.thread = threading.Thread(target=self.run, name='modelship-text-generation', daemon=True)

    def put(self, value):
        while not self.stop.is_set():
            if time.monotonic() >= self.deadline:
                self.stop.set()
                break
            try:
                self.queue.put(value, timeout=0.1)
                return
            except queue.Full:
                continue
        raise OutputStopped()

    def emit(self, delta):
        self.put(('delta', delta))

    def cancelled(self):
        return self.stop.is_set() or time.monotonic() >= self.deadline

    def start(self):
        try:
            self.thread.start()
        except BaseException:
            self.cleanup('failed')
            self.done.set()
            raise

    def run(self):
        result, error = None, None
        try:
            count = self.work(self.streamer, self.cancelled)
            self.streamer.end()
            if not isinstance(count, int) or not 0 <= count <= self.max_tokens:
                raise OutputLayoutError()
            if self.cancelled():
                raise OutputStopped()
            result = {'finish_reason': 'length' if count >= self.max_tokens else 'stop',
                      'usage': {'prompt_tokens': self.prompt_tokens, 'completion_tokens': count,
                                'total_tokens': self.prompt_tokens + count}}
        except self.oom_type:
            error = 'oom'
        except OutputLayoutError:
            error = 'layout'
        except OutputStopped:
            error = 'timeout' if time.monotonic() >= self.deadline else 'failed'
        except Exception:
            error = 'failed'
        finally:
            # cleanup drains CUDA, closes images, and only then releases BUSY.
            # Failure poisons readiness rather than risking concurrent GPU work.
            if not self.cleanup(error):
                error = 'drain'
            try:
                self.put(('error', error) if error else ('finish', result))
            except OutputStopped:
                pass
            self.done.set()

    async def events(self):
        heartbeat = time.monotonic()
        while True:
            if time.monotonic() >= self.deadline:
                self.stop.set()
                yield 'error', 'timeout'
                return
            try:
                value = await run_in_threadpool(self.queue.get, True, 0.2)
            except queue.Empty:
                if self.done.is_set():
                    yield 'error', 'failed'
                    return
                if time.monotonic() - heartbeat >= 1:
                    heartbeat = time.monotonic()
                    yield 'heartbeat', None
                continue
            yield value
            if value[0] in ('error', 'finish'):
                return

    def cancel(self):
        self.stop.set()


def sse(data):
    return ('data: ' + json.dumps(data, ensure_ascii=True, separators=(',', ':')) + '\n\n').encode()


class NativeTextResponse(StreamingResponse):
    def __init__(self, job):
        self.job = job
        super().__init__(self.chunks(), media_type='text/event-stream', headers={
            'Cache-Control': 'no-store, no-transform', 'X-Accel-Buffering': 'no'})

    async def chunks(self):
        async for kind, value in self.job.events():
            if kind == 'heartbeat':
                yield b': keep-alive\n\n'
                continue
            if kind == 'error':
                yield sse({'error': {'message': ERRORS[value], 'code': value}})
                return
            frame = dict(self.job.identity, object='chat.completion.chunk')
            if kind == 'delta':
                frame['choices'] = [{'index': 0, 'delta': value, 'finish_reason': None}]
            else:
                frame['choices'] = [{'index': 0, 'delta': {}, 'finish_reason': value['finish_reason']}]
                frame['usage'] = value['usage']
            yield sse(frame)
            if kind == 'finish':
                yield b'data: [DONE]\n\n'
                return

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.job.cancel()
