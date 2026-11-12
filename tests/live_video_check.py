#!/usr/bin/env python3
"""Opt-in localhost WAN GPU/API/MP4 checks. No deployment or cache changes.

Run using the video deployment's Python (imageio-ffmpeg/Pillow are required).
A tiny codec smoke is followed by a bounded quality clip. Job polling is limited
by --timeout; a timeout does not cancel the server job. Stop it explicitly if needed.
"""
import argparse
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

MAX_VIDEO = 64 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--num-frames', type=int, default=17)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.timeout <= 0:
        parser.error('valid localhost port and positive timeout required')
    if not 1 <= args.steps <= 50 or not 5 <= args.num_frames <= 49 or (args.num_frames-1) % 4:
        parser.error('steps must be 1..50; motion checks require frames=4k+1 in 5..49')
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    base = f'http://127.0.0.1:{args.port}'
    checks = []

    def request(path, body=None, raw=None, binary=False):
        data = raw if raw is not None else None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base+path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                if binary:
                    value = response.read(MAX_VIDEO+1)
                    assert 0 < len(value) <= MAX_VIDEO, 'video exceeds artifact byte limit'
                    assert response.headers.get_content_type() == 'video/mp4'
                    assert int(response.headers['Content-Length']) == len(value)
                else:
                    value = json.load(response)
                return response.status, value
        except urllib.error.HTTPError as exc:
            try:
                value = json.load(exc)
            except (ValueError, UnicodeDecodeError):
                value = {'detail': 'non-JSON response; inspect private service log'}
            return exc.code, value

    def check(label, path, body=None, status=200, **kwargs):
        code, result = request(path, body=body, **kwargs)
        checks.append({'check': label, 'status': code, 'expected': status})
        assert code == status, (label, code, result)
        return result

    health = check('health', '/health')
    assert health['ready'] and health['task'] == 'text-to-video'
    assert health['model'] == 'Wan-AI/Wan2.2-T2V-A14B-Diffusers'
    assert re.fullmatch('[a-f0-9]{40}', health['revision'])
    check('unknown-job', '/v1/videos/video-'+'0'*32, status=404)
    invalid = [
        {'prompt': ''}, {'prompt': '   '}, {'prompt': 'x', 'width': 513},
        {'prompt': 'x', 'num_frames': 18}, {'prompt': 'x', 'steps': 51},
        {'prompt': 'x', 'fps': 31}, {'prompt': 'x', 'width': 768, 'height': 768, 'num_frames': 49},
        {'prompt': 'x', 'image_url': 'http://169.254.169.254/'},
    ]
    for i, body in enumerate(invalid):
        check(f'invalid-fields-{i}', '/v1/videos/generations', body, status=422)
    check('body-limit', '/v1/videos/generations', raw=b'x'*(32*1024+1), status=413)

    def probe(path, expected):
        import imageio_ffmpeg
        from PIL import Image, ImageChops, ImageStat
        decoder = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
        frames = []
        try:
            metadata = next(decoder)
            size = tuple(metadata['size'])
            assert size == (expected['width'], expected['height']), metadata
            assert abs(metadata['fps']-expected['fps']) < 0.1, metadata
            for raw in decoder:
                assert len(frames) < expected['num_frames'], 'unexpected extra frames'
                frames.append(Image.frombytes('RGB', size, raw))
        finally:
            decoder.close()
        assert len(frames) == expected['num_frames'], (len(frames), expected)
        selected = [0, len(frames)//2, len(frames)-1]
        for index in selected:
            frames[index].save(args.output/(path.stem+f'-frame-{index:03d}.png'))
        variance = [max(ImageStat.Stat(frame.resize((64, 64))).stddev) for frame in frames]
        assert max(variance) > 2, 'blank/flat clip'
        motion = [sum(ImageStat.Stat(ImageChops.difference(frames[i-1].resize((64, 64)), frames[i].resize((64, 64)))).mean)/3
                  for i in range(1, len(frames))]
        assert max(motion, default=0) > 0.01, 'identical frames; motion not established'
        return {'width': size[0], 'height': size[1], 'frames': len(frames), 'fps': metadata['fps'],
                'duration': metadata.get('duration'), 'codec': metadata.get('codec'),
                'max_frame_stddev': round(max(variance), 3), 'max_adjacent_frame_difference': round(max(motion), 3)}

    clips = []
    for label, body in [
        ('smoke', {'prompt': 'A red paper boat drifting on a quiet pond', 'width': 256, 'height': 256,
                   'num_frames': 5, 'steps': 4, 'fps': 16, 'seed': 40}),
        ('quality', {'prompt': 'A small red paper boat drifts slowly across a calm pond. Gentle ripples move across the water, sunlight reflects softly, natural realistic colors, stationary camera, continuous smooth motion.',
                     'negative_prompt': 'blurry, distorted, text, watermark, low quality',
                     'width': 512, 'height': 512, 'num_frames': args.num_frames, 'steps': args.steps,
                     'fps': 16, 'seed': 42}),
    ]:
        start = time.monotonic()
        job = check(label+'-submit', '/v1/videos/generations', body, status=202)
        job_id = job['id']
        assert re.fullmatch('video-[a-f0-9]{32}', job_id)
        assert job['status'] == 'running'
        check(label+'-busy', '/v1/videos/generations', {'prompt': 'busy probe', 'steps': 1, 'num_frames': 1}, status=503)
        check(label+'-content-pending', f'/v1/videos/{job_id}/content', status=409)
        deadline = start+args.timeout
        while True:
            job = check(label+'-status', f'/v1/videos/{job_id}')
            assert job['model'] == health['model']
            if job['status'] != 'running':
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('bounded video check timed out; server job is not automatically cancelled')
            time.sleep(2)
        (args.output/(label+'-job.json')).write_text(json.dumps(job, indent=2))
        assert job['status'] == 'succeeded', job
        assert job['content_url'] == f'/v1/videos/{job_id}/content'
        data = check(label+'-content', job['content_url'], binary=True)
        assert b'ftyp' in data[:32], 'not an MP4 container'
        path = args.output/(label+'.mp4')
        with path.open('xb') as output:
            output.write(data)
        clip = {'label': label, 'seconds': round(time.monotonic()-start, 3), 'bytes': len(data),
                'job_id': job_id, 'decoded': probe(path, body)}
        clips.append(clip)
        print('VIDEO_CLIP_VALIDATED', json.dumps(clip), flush=True)
    report = {'model': health['model'], 'revision': health['revision'], 'clips': clips, 'checks': checks}
    (args.output/'video-report.json').write_text(json.dumps(report, indent=2))
    print('LIVE_VIDEO_CHECK_PASS', args.output, flush=True)


if __name__ == '__main__':
    main()
