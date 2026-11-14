"""Actual pinned TextStreamer CPU check: no Torch, Hub, tokenizer/model download."""
from importlib.metadata import version
from pathlib import Path
import sys
import io
from contextlib import redirect_stdout

assert version('transformers') == '5.22.2'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'internal/modelship/assets'))
from native_text import GemmaChannelStreamer, OutputLayoutError

class Tensor:
    def __init__(self, values):self.values=values;self.shape=(len(values),)
    def tolist(self):return self.values
    def new_tensor(self, values):return Tensor(values)

words={200:b'th',201:b'ought\n',202:b'Because ',203:b'logic.',204:b'Paris ',205:b'\xf0\x9f',206:b'\x8c\x8d',207:b'!',208:'汉'.encode(),209:b'literal <|channel> text '}
class Tokenizer:
    all_special_ids=[1,2,98,100,101,105,106]
    def decode(self, tokens, **kwargs):return b''.join(words.get(t,b'') for t in tokens).decode('utf-8',errors='replace')

events=[]
with redirect_stdout(io.StringIO()) as captured:
    s=GemmaChannelStreamer(Tokenizer(),True,events.append)
    s.put(Tensor([2,9,9]))  # prompt is never emitted
    for t in [100,200,201,202,203,101,204,205,206,207,106]:s.put(Tensor([t]))
    s.end()
assert captured.getvalue()==''
assert ''.join(e.get('reasoning_content','') for e in events)=='Because logic.'
assert ''.join(e.get('content','') for e in events)=='Paris 🌍!'
assert all('�' not in str(e) for e in events)
# CJK emits before end(), not a buffered imitation of streaming.
events=[];s=GemmaChannelStreamer(Tokenizer(),False,events.append);s.put(Tensor([2]));s.put(Tensor([208]));assert events==[{'content':'汉'}];s.end()
events=[];s=GemmaChannelStreamer(Tokenizer(),False,events.append);s.put(Tensor([2]));s.put(Tensor([209]));s.end();assert events==[{'content':'literal <|channel> text '}]
print('GEMMA_STREAMER_CPU_PASSED: pinned TextStreamer, channel IDs, split Unicode/CJK, no stdout/Hub/Torch/models')
