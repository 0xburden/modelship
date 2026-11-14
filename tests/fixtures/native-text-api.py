"""Local CPU-only real-ASGI Gemma stream fixture. Never loads weights or CUDA.
Requires tests/requirements.txt + processor-requirements.txt. Use fixture keys
only. Test fixture functions replace Torch/model/tokenizer, not API/job code.
"""
from importlib.metadata import version
from pathlib import Path
import hashlib
import sys
import time

assert version('transformers') == '5.22.2'
from transformers import TextStreamer as RealTextStreamer
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
import test_native_text_controls as f
import uvicorn
from api_auth import Access, BearerAuth

fixture=f.NativeTextControlTests('test_stream_structured_reasoning_usage_and_done')
fixture.setUp()
sys.modules['transformers'].TextStreamer=RealTextStreamer
f.WORDS.update({300:'CPU-only ',301:'fixture ',302:'answer.',303:'Fixture ',304:'reasoning.'})
def generate(self, **kwargs):
    f.fixture.CALLS.append(('generate',kwargs))
    thinking=next(c[2].get('enable_thinking',False) for c in reversed(f.fixture.CALLS) if c[0]=='template')
    tokens=([100,200,303,304,101] if thinking else [])+[300,301,302,106]
    streamer=kwargs['streamer'];streamer.put(kwargs['input_ids']);result=[]
    for token in tokens[:kwargs['max_new_tokens']]:
        if any(stop(None,None) for stop in kwargs['stopping_criteria']):break
        streamer.put(f.Tensor((1,),values=[token]));result.append(token);time.sleep(.12)
    streamer.end();return f.Output(result)
f.Network.generate=generate
@fixture.server.app.get('/__fixture_metrics')
def metrics():
    return {'generations':sum(c[0]=='generate' for c in f.fixture.CALLS),
            'thinking':[c[2].get('enable_thinking') for c in f.fixture.CALLS if c[0]=='template']}
@fixture.server.app.post('/__fixture_restart')
def restart():
    fixture.server.RUN_ID='c'*32
    return {'run_id':fixture.server.RUN_ID}

key=b'fixture-native-key-012345678901234567890'
try:
    uvicorn.run(BearerAuth(fixture.server.app,Access('127.0.0.1',hashlib.sha256(key).digest())),host='0.0.0.0',port=8125,access_log=False)
finally:
    fixture.doCleanups()
