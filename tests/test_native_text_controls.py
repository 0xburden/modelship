"""CPU-only Gemma channel, API and producer lifetime contracts; no weights/CUDA."""
import asyncio
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import test_vision_api as fixture
import native_text

ASSETS = fixture.ASSETS
PROFILE = json.loads((ASSETS/'native_text_profiles.json').read_text())['profiles'][0]
WORDS = {200:'thought\n',201:'Because ',202:'logic.',203:'Paris',204:' 🌍',205:'!',206:' ',207:'analysis\n',208:'Alpha ',209:'Beta ',210:'literal <|channel> text '}

class Tensor(fixture.Tensor):
    def to(self, device=None, dtype=None):
        fixture.CALLS.append(('move',self.shape,device,dtype))
        return Tensor(self.shape,dtype or self.dtype,self.values)
    def tolist(self):
        return [self.values] if len(self.shape)==2 else self.values
    def new_tensor(self, values):
        return Tensor((len(values),),values=values)

class Tokenizer(fixture.Tokenizer):
    all_special_ids=list(native_text.TOKEN_IDS.values())+[258880,255999,258882]
    def convert_tokens_to_ids(self, token):
        return native_text.TOKEN_IDS.get(token, {'<|image|>':258880,'<|image>':255999,'<image|>':258882}.get(token,-1))
    def decode(self, tokens, **kwargs):
        return ''.join(WORDS.get(t,'') for t in tokens)
    def __call__(self, prompt, **kwargs):
        raw=super().__call__(prompt,**kwargs)
        return {k:Tensor(v.shape,v.dtype,v.values) for k,v in raw.items()}

class TextStreamer:
    """API-compatible CPU decoder. Real pinned decoder has a separate CPU check."""
    def __init__(self, tokenizer, **kwargs):self.tokenizer=tokenizer
    def put(self, value):self.on_finalized_text(self.tokenizer.decode(value.tolist()))
    def end(self):self.on_finalized_text('',True)

class Output:
    def __init__(self, tokens):self.tokens=tokens
    def __getitem__(self,key):return self.tokens

class Network(fixture.Network):
    tokens=[203,206,1]
    wait=None
    entered=None
    def generate(self, **kwargs):
        fixture.CALLS.append(('generate',kwargs))
        streamer=kwargs['streamer'];streamer.put(kwargs['input_ids'])
        result=[]
        for token in self.tokens[:kwargs['max_new_tokens']]:
            if any(stop(None,None) for stop in kwargs['stopping_criteria']):break
            streamer.put(Tensor((1,),values=[token]));result.append(token)
            if self.entered:self.entered.set()
            if self.wait:self.wait.wait(3)
        streamer.end()
        return Output(result)

class NativeTextControlTests(unittest.TestCase):
    def setUp(self):
        fixture.CALLS.clear();Network.tokens=[203,206,1];Network.wait=None;Network.entered=None
        self.drain_gate=None
        self.torch=types.SimpleNamespace(float16='fp16',bfloat16='bf16',long='int64',inference_mode=nullcontext,
            OutOfMemoryError=fixture.OOM,manual_seed=lambda value:None,
            cuda=types.SimpleNamespace(empty_cache=lambda:None,synchronize=self.synchronize))
        self.modules=patch.dict(sys.modules,{'torch':self.torch,'transformers':types.SimpleNamespace(
            AutoModelForCausalLM=Network,AutoModelForMultimodalLM=Network,AutoTokenizer=Tokenizer,
            TextStreamer=TextStreamer,StoppingCriteria=object,StoppingCriteriaList=list)})
        self.modules.start();self.addCleanup(self.modules.stop)
        env={'MODEL_TEXT_PROFILE':PROFILE['id'],'MODEL_TEXT_PROFILE_SPEC':json.dumps(PROFILE),
             'MODEL_ID':PROFILE['repo'],'MODEL_REVISION':PROFILE['revision'],'MODEL_RECIPE':PROFILE['recipe'],
             'MODEL_RUN_ID':'b'*32,'MODEL_NATIVE_LOADER':'multimodal-lm','MODEL_DTYPE':'bfloat16',
             'MODEL_TEMPLATE_COMPLETIONS':'true','MODEL_DISABLE_THINKING':'true','MODEL_VISION_PROFILE':'',
             'MODEL_CONTEXT':'4096'}
        self.env=patch.dict(os.environ,env);self.env.start();self.addCleanup(self.env.stop)
        self.version=patch.object(native_text,'version',return_value='5.22.2');self.version.start();self.addCleanup(self.version.stop)
        spec=importlib.util.spec_from_file_location('modelship_feature_text_fixture',ASSETS/'text_server.py')
        self.server=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.server)
        self.addCleanup(self.drain)
    def synchronize(self):
        if self.drain_gate:self.drain_gate.wait(3)
    def drain(self):
        if Network.wait:Network.wait.set()
        if self.drain_gate:self.drain_gate.set()
        if self.server.ACTIVE_JOB:
            self.server.ACTIVE_JOB.cancel();self.server.ACTIVE_JOB.done.wait(3)
        if self.server.BUSY.locked():self.server.BUSY.release()
    def body(self, stream=False, thinking=None):
        b={'model':PROFILE['repo'],'messages':[{'role':'user','content':'Capital?'}],'max_tokens':64,'stream':stream,'expected_run_id':'b'*32}
        if thinking is not None:b['chat_template_kwargs']={'enable_thinking':thinking}
        return b
    def test_capabilities_default_and_nonstream_thinking(self):
        with TestClient(self.server.app) as client:
            caps=client.get('/v1/capabilities').json()
            self.assertTrue(caps['chat']['streaming']);self.assertFalse(caps['completion']['streaming'])
            self.assertEqual(caps['chat']['text_controls']['profile'],PROFILE['id'])
            self.assertFalse(caps['chat']['text_controls']['default_enable_thinking'])
            r=client.post('/v1/chat/completions',json=self.body())
            self.assertEqual(r.status_code,200,r.text)
            self.assertEqual(r.json()['choices'][0]['message']['content'],'Paris ')
            self.assertFalse(next(c[2]['enable_thinking'] for c in fixture.CALLS if c[0]=='template'))
            Network.tokens=[100,200,201,202,101,203,204,205,106]
            r=client.post('/v1/chat/completions',json=self.body(thinking=True))
            self.assertEqual(r.status_code,200,r.text)
            self.assertEqual(r.json()['choices'][0]['message'],{'role':'assistant','content':'Paris 🌍!','reasoning_content':'Because logic.'})
            self.assertEqual(r.json()['usage']['completion_tokens'],len(Network.tokens))
            self.assertFalse(self.server.BUSY.locked())
    def test_stream_structured_reasoning_usage_and_done(self):
        Network.tokens=[100,200,201,202,101,203,204,205,106]
        with TestClient(self.server.app) as client:
            r=client.post('/v1/chat/completions',json=self.body(True,True))
            self.assertEqual(r.status_code,200,r.text);self.assertIn('text/event-stream',r.headers['content-type'])
            self.assertEqual(r.headers['x-accel-buffering'],'no')
            frames=[x[6:] for x in r.text.splitlines() if x.startswith('data: ')]
            self.assertEqual(frames[-1],'[DONE]')
            values=[json.loads(x) for x in frames[:-1]]
            self.assertEqual(''.join(v['choices'][0]['delta'].get('reasoning_content','') for v in values),'Because logic.')
            self.assertEqual(''.join(v['choices'][0]['delta'].get('content','') for v in values),'Paris 🌍!')
            self.assertEqual(values[-1]['usage']['completion_tokens'],9)
            self.assertNotIn('<|channel>',r.text)
    def test_stale_identity_unknown_fields_and_output_budget_before_generation(self):
        with TestClient(self.server.app) as client:
            for update in ({'expected_run_id':'c'*32},{'expected_run_id':None}):
                b=self.body(True,True);b.update(update)
                self.assertEqual(client.post('/v1/chat/completions',json=b).status_code,409)
            for update in ({'max_tokens':1025},{'chat_template_kwargs':{'enable_thinking':'yes'}},{'chat_template_kwargs':{'thinking_budget':99}},{'chat_template_kwargs':{'chat_template':'source'}}):
                b=self.body(True,True);b.update(update)
                self.assertEqual(client.post('/v1/chat/completions',json=b).status_code,422)
            self.server.CONTEXT=5
            self.assertEqual(client.post('/v1/chat/completions',json=self.body(True,True)).status_code,422)
            self.assertFalse(any(c[0] in ('move','generate') for c in fixture.CALLS))
    def test_plain_completion_stays_nonstreaming(self):
        with TestClient(self.server.app) as client:
            self.assertEqual(client.post('/v1/completions',json={'prompt':'x','stream':True}).status_code,422)
            self.assertEqual(client.post('/v1/completions',json={'prompt':'x','chat_template_kwargs':{'enable_thinking':True}}).status_code,422)
    def test_invalid_channel_layout_is_sanitized_without_done(self):
        with TestClient(self.server.app) as client:
            for tokens in ([100,207,201,101,203],[101,203],[100,200,201,101,101],[100,200,201,101,105],[98,98,203]):
                Network.tokens=tokens
                r=client.post('/v1/chat/completions',json=self.body(True,True))
                self.assertIn('"error"',r.text);self.assertNotIn('[DONE]',r.text)
                self.assertFalse(self.server.BUSY.locked())
            Network.tokens=[100,200,201,101,203]
            r=client.post('/v1/chat/completions',json=self.body(True,False))
            self.assertIn('"error"',r.text)
    def test_token_layout_not_text_regex(self):
        Network.tokens=[210,1]
        with TestClient(self.server.app) as client:
            r=client.post('/v1/chat/completions',json=self.body(thinking=True))
            self.assertEqual(r.json()['choices'][0]['message']['content'],'literal <|channel> text ')
            self.assertNotIn('reasoning_content',r.json()['choices'][0]['message'])
    def test_oom_recovers_and_failed_drain_poisons_readiness(self):
        with TestClient(self.server.app) as client:
            self.server.NETWORK.generate=lambda **kwargs:(_ for _ in ()).throw(fixture.OOM('SECRET_PRIVATE'))
            r=client.post('/v1/chat/completions',json=self.body(True,True))
            self.assertIn('"code":"oom"',r.text);self.assertNotIn('SECRET_PRIVATE',r.text)
            self.assertFalse(self.server.BUSY.locked())
            self.torch.cuda.synchronize=lambda:(_ for _ in ()).throw(RuntimeError('SECRET_PRIVATE'))
            r=client.post('/v1/chat/completions',json=self.body(True,True))
            self.assertIn('"code":"drain"',r.text);self.assertNotIn('SECRET_PRIVATE',r.text)
            self.assertFalse(client.get('/health').json()['ready']);self.assertTrue(self.server.BUSY.locked())
            self.assertTrue(self.server.DRAIN_REFS)
            self.assertTrue(self.server.DRAIN_REFS[-1], 'uncertain GPU inputs must be retained')
            self.assertEqual(client.post('/v1/chat/completions',json=self.body()).status_code,503)
    def test_incremental_output_and_disconnect_keep_busy_until_cuda_drain(self):
        Network.tokens=[208,209,1];Network.wait=threading.Event();Network.entered=threading.Event()
        self.drain_gate=threading.Event()
        with TestClient(self.server.app):
            job=self.server.prepare_feature_job(self.server.Chat.model_validate(self.body(True)))
            self.assertTrue(Network.entered.wait(1))
            kind,value=job.queue.get(timeout=1)
            self.assertEqual((kind,value),('delta',{'content':'Alpha '}));self.assertFalse(job.done.is_set())
            job.cancel()
            self.assertTrue(self.server.BUSY.locked())
            Network.wait.set()
            # A second request cannot enter while CUDA synchronization is held.
            with self.assertRaises(Exception) as caught:self.server.prepare_feature_job(self.server.Chat.model_validate(self.body()))
            self.assertEqual(caught.exception.status_code,503)
            self.drain_gate.set();self.assertTrue(job.done.wait(2));self.assertFalse(self.server.BUSY.locked())
    def test_queue_backpressure_is_bounded_and_cancel_unblocks_producer(self):
        Network.tokens=[208]*200
        with TestClient(self.server.app):
            job=self.server.prepare_feature_job(self.server.Chat.model_validate(self.body(True)))
            # Wait for the producer to signal its 33rd callback, at the queue bound.
            full=threading.Event();original=job.queue.put
            def put(value,*args,**kwargs):
                if job.queue.qsize()==native_text.QUEUE_SIZE:full.set()
                return original(value,*args,**kwargs)
            job.queue.put=put
            self.assertTrue(full.wait(1));self.assertLessEqual(job.queue.qsize(),native_text.QUEUE_SIZE)
            job.cancel();self.assertTrue(job.done.wait(2));self.assertFalse(self.server.BUSY.locked())
    def test_profile_and_token_identity_fail_before_model_loading(self):
        with patch.object(Tokenizer,'convert_tokens_to_ids',return_value=-1):
            with self.assertRaises(RuntimeError):
                with TestClient(self.server.app):pass
        self.assertFalse(any(c[0]=='network-load' for c in fixture.CALLS))
    def test_thinking_budget_includes_reasoning_and_counts_length(self):
        Network.tokens=[100,200,201,202,101,203,1]
        with TestClient(self.server.app) as client:
            b=self.body(thinking=True);b['max_tokens']=4
            r=client.post('/v1/chat/completions',json=b)
            self.assertEqual(r.status_code,200,r.text)
            value=r.json();self.assertEqual(value['choices'][0]['finish_reason'],'length')
            self.assertEqual(value['choices'][0]['message']['content'],'')
            self.assertEqual(value['usage']['completion_tokens'],4)
    def test_vision_stream_preserves_context_and_pixel_integer_dtypes(self):
        self.server.VISION_PROFILE=fixture.PROFILE['id']
        cls=fixture.gemma_vision.GemmaImageProcessor
        def load(profile,tokenizer):
            processor=cls.__new__(cls);processor.max_tokens=280;processor.processor=fixture.processor_pixels;return processor
        with patch.dict(os.environ,{'MODEL_VISION_PROFILE_SPEC':json.dumps(fixture.PROFILE)}),patch.object(fixture.gemma_vision,'GemmaImageProcessor',side_effect=load):
            with TestClient(self.server.app) as client:
                body=self.body(True,True);body['messages']=fixture.chat()['messages']
                r=client.post('/v1/chat/completions',json=body)
                self.assertEqual(r.status_code,200,r.text);self.assertIn('[DONE]',r.text)
                args=next(c[1] for c in fixture.CALLS if c[0]=='generate')
                self.assertEqual(args['pixel_values'].dtype,'bf16');self.assertEqual(args['input_ids'].dtype,'int64')
                self.assertEqual(args['image_position_ids'].dtype,'int64');self.assertEqual(args['mm_token_type_ids'].dtype,'int64')
                self.assertTrue(client.get('/v1/capabilities').json()['chat']['image_input'])
    def test_asgi_disconnect_cancels_producer_without_releasing_gpu_early(self):
        Network.tokens=[208,209,1];Network.wait=threading.Event();self.drain_gate=threading.Event()
        with TestClient(self.server.app):
            job=self.server.prepare_feature_job(self.server.Chat.model_validate(self.body(True)))
            async def disconnected_response():
                disconnect=asyncio.Event()
                async def receive():await disconnect.wait();return {'type':'http.disconnect'}
                async def send(message):
                    if message['type']=='http.response.body' and b'"content"' in message.get('body',b''):disconnect.set()
                scope={'type':'http','asgi':{'spec_version':'2.0'},'method':'POST','path':'/v1/chat/completions','headers':[]}
                await native_text.NativeTextResponse(job)(scope,receive,send)
            asyncio.run(disconnected_response())
            self.assertTrue(job.stop.is_set());self.assertTrue(self.server.BUSY.locked())
            Network.wait.set();self.drain_gate.set();self.assertTrue(job.done.wait(2))
    def test_timeout_has_error_no_done_and_holds_busy_until_work_drains(self):
        Network.wait=threading.Event()
        with patch.object(native_text,'MAX_SECONDS',.01):
            with TestClient(self.server.app):
                job=self.server.prepare_feature_job(self.server.Chat.model_validate(self.body(True)))
                async def collect():return [value async for value in job.events()]
                values=asyncio.run(collect());self.assertEqual(values[-1],('error','timeout'))
                self.assertTrue(job.stop.is_set());self.assertTrue(self.server.BUSY.locked())
                Network.wait.set();self.assertTrue(job.done.wait(2))
    def test_auth_precedes_streaming_and_capabilities(self):
        import hashlib
        key='fixture-native-key-012345678901234567890'
        wrapped=fixture.BearerAuth(self.server.app,fixture.Access('127.0.0.1',hashlib.sha256(key.encode()).digest()))
        with TestClient(wrapped) as client:
            self.assertEqual(client.get('/v1/capabilities').status_code,401)
            self.assertEqual(client.post('/v1/chat/completions',content=b'PRIVATE_PROMPT').status_code,401)
            self.assertFalse(any(c[0]=='generate' for c in fixture.CALLS))
            r=client.post('/v1/chat/completions',json=self.body(True),headers={'Authorization':'Bearer '+key})
            self.assertEqual(r.status_code,200,r.text);self.assertIn('[DONE]',r.text)
