"""Multi-image admission/ordering/cleanup CPU fixtures; no model or GPU inference."""
import base64
import json
import os
from unittest.mock import patch
import unittest
from fastapi.testclient import TestClient
import test_vision_api as f
import vision_input

PROFILE = json.loads((f.ASSETS/'pytorch_vision_profiles.json').read_text())['profiles'][1]
POLICY = PROFILE['images']

def body(sizes):
    return {'messages': [{'role':'user','content':[{'type':'text','text':'Compare in order.'}] +
            [{'type':'image_url','image_url':{'url':f.data_image(size=s)}} for s in sizes]}], 'max_tokens':64}

class MultiImageTests(unittest.TestCase):
    def setUp(self):
        f.VisionAPITests.setUp(self)
        os.environ['MODEL_VISION_PROFILE_SPEC']=json.dumps(PROFILE)
        self.server.VISION_PROFILE=PROFILE['id']
        self.torch.cuda.synchronize=lambda:None
    def load(self, profile, tokenizer):
        adapter=self.real_processor_class.__new__(self.real_processor_class)
        adapter.images_policy=profile.get('images');adapter.max_tokens=280
        def pixels(**kw):
            f.CALLS.append(('pixels',kw));n=len(kw['images'])
            return {'pixel_values':f.Tensor((n,2520,768),'fp32'),
                    'image_position_ids':f.Tensor((n,2520,2),'int64'),
                    'num_soft_tokens_per_image':[256]*n}
        adapter.processor=pixels
        return adapter
    def test_four_images_order_followup_and_caps(self):
        sizes=[(2,2),(3,2),(2,3),(4,3)]
        request=body(sizes[:2]);request['messages'] += [{'role':'assistant','content':'Two images.'}]+body(sizes[2:])['messages']
        with TestClient(self.server.app) as client:
            caps=client.get('/v1/capabilities').json()['chat']
            self.assertEqual(caps['max_images'],4);self.assertEqual(caps['max_total_image_pixels'],8388608)
            self.assertEqual(self.server.MAX_BODY,12<<20)
            reply=client.post('/v1/chat/completions',json=request)
            self.assertEqual(reply.status_code,200,reply.text)
            self.assertEqual(reply.json()['usage']['prompt_tokens'],1028)
        images=next(c[1]['images'] for c in f.CALLS if c[0]=='pixels')
        self.assertEqual([im.size for im in images],sizes)
        expanded=next(c[1] for c in f.CALLS if c[0]=='tokenize')
        self.assertEqual(expanded.count('<|image>'),4)
        self.assertEqual(expanded.count('<image|>'),4)
        self.assertEqual(expanded.count('<|image|>'),1024)
        generated=next(c[1] for c in f.CALLS if c[0]=='generate')
        self.assertEqual(generated['pixel_values'].dtype,'bf16')
        self.assertEqual(generated['image_position_ids'].dtype,'int64')
    def test_reject_fifth_context_and_aggregate_before_gpu(self):
        with TestClient(self.server.app) as client:
            for request in (body([(2,2)]*5),body([(2048,1536)]*3)):
                reply=client.post('/v1/chat/completions',json=request)
                self.assertEqual(reply.status_code,422,reply.text)
            self.assertFalse(any(c[0] in ('pixels','generate','move') for c in f.CALLS))
            self.server.CONTEXT=1024
            reply=client.post('/v1/chat/completions',json=body([(2,2)]*4))
            self.assertEqual(reply.status_code,422)
            self.assertFalse(any(c[0]=='generate' or c[0]=='move' and c[2]=='cuda' for c in f.CALLS))
    def test_aggregate_bytes_preflight_before_any_decode(self):
        messages=body([(2,2)]*3)['messages']
        data='data:image/png;base64,'+base64.b64encode(b'x'*(3<<20)).decode()
        for part in messages[0]['content'][1:]:part['image_url']['url']=data
        parsed=[vision_input.VisionMessage.model_validate(m) for m in messages]
        with patch.object(vision_input,'inspect_image') as inspect:
            with self.assertRaises(ValueError):vision_input.normalize_messages(parsed,POLICY)
            inspect.assert_not_called()
    def test_failed_last_image_no_generation_or_partial_preprocessing(self):
        request=body([(2,2)]*4);request['messages'][0]['content'][-1]['image_url']['url']='data:image/png;base64,eA=='
        with TestClient(self.server.app) as client:
            self.assertEqual(client.post('/v1/chat/completions',json=request).status_code,422)
        self.assertFalse(any(c[0] in ('pixels','generate','move') for c in f.CALLS))
    def test_uncertain_sync_keeps_slot_model_and_input_references(self):
        def fail():raise RuntimeError('private CUDA detail')
        self.torch.cuda.synchronize=fail
        with TestClient(self.server.app) as client:
            reply=client.post('/v1/chat/completions',json=body([(2,2)]*2))
            self.assertEqual(reply.status_code,503);self.assertNotIn('private CUDA',reply.text)
            self.assertFalse(client.get('/health').json()['ready'])
            self.assertTrue(self.server.BUSY.locked());self.assertTrue(self.server.DRAIN_REFS[-1])
        self.assertIsNotNone(self.server.NETWORK)
        self.server.BUSY.release()

if __name__=='__main__':unittest.main()
