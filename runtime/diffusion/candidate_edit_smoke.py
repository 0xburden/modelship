#!/usr/bin/env python3
"""One offline fixed-fixture GGUF edit. No arbitrary files or service."""

import argparse
import os
import platform
import select
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from candidate_preflight import PreflightError, admit_edit_smoke_gpu, verify_edit_bundle
from candidate_smoke import MAX_RESPONSE, SEED, SIDE, _frame, _gpu_lock, _private_directory, _stop_child, _verified_runtime, _write_png

PROMPT = b"Change the red mug to a blue mug. Keep the yellow lemon, tabletop, and composition unchanged."


def reference_pixels():
    """Synthetic 512-square RGB fixture, independent of any model or user media."""
    pixels = bytearray()
    for y in range(SIDE):
        for x in range(SIDE):
            shade = (x + 2 * y) // 80
            color = (235 - shade, 224 - shade, 207 - shade) if y < 355 else (192 - shade, 170 - shade, 141 - shade)
            if ((x - 228) / 150)**2 + ((y - 395) / 20)**2 < 1:
                color = (164 - shade, 144 - shade, 117 - shade)
            handle = ((x - 313) / 51)**2 + ((y - 285) / 48)**2
            if 0.50 < handle < 1:
                color = (183, 38, 32)
            if 138 <= x < 324 and 222 <= y < 391 and ((x - 231) / 96)**2 + ((y - 315) / 109)**2 < 1.55:
                red = 186 + (x - 138) // 4
                color = (red, 43 + (x - 138) // 10, 37 + (x - 138) // 12)
            if ((x - 231) / 93)**2 + ((y - 222) / 26)**2 < 1:
                color = (231, 70, 61)
            if ((x - 231) / 81)**2 + ((y - 220) / 17)**2 < 1:
                color = (103, 28, 25)
            if ((x - 406) / 49)**2 + ((y - 365) / 35)**2 < 1:
                color = (250, 207 - abs(x - 397) // 2, 56)
            pixels.extend(color)
    return bytes(pixels)


def _send_frame(stream, payload, deadline):
    fd = stream.fileno()
    os.set_blocking(fd, False)
    data = memoryview(struct.pack("!I", len(payload)) + payload)
    poller = select.poll()
    poller.register(fd, select.POLLOUT)
    while data:
        left = deadline - time.monotonic()
        if left <= 0 or not poller.poll(max(1, int(left * 1000))):
            raise PreflightError("Candidate edit request timed out.")
        try:
            written = os.write(fd, data)
        except BlockingIOError:
            continue
        if written <= 0:
            raise PreflightError("Candidate edit request did not complete.")
        data = data[written:]
    stream.close()


def _edit_once(child, reference, load_seconds=600, edit_seconds=900):
    out = child.stdout.fileno()
    if len(reference) != 3 * SIDE * SIDE:
        raise PreflightError("Expected the fixed 512x512 RGB fixture.")
    if _frame(out, time.monotonic() + load_seconds, 2) != b"L\x02":
        raise PreflightError("The candidate did not load the edit context.")
    started = time.monotonic()
    payload = struct.pack("!IH", SEED, len(PROMPT)) + PROMPT + reference
    _send_frame(child.stdin, payload, min(started + 30, started + edit_seconds))
    reply = _frame(out, started + edit_seconds, MAX_RESPONSE)
    if len(reply) < 2 or reply[0] != ord("I") or reply[1] not in (3, 4) or len(reply) != 2 + reply[1] * SIDE * SIDE:
        raise PreflightError("The candidate did not return a drained 512x512 RGB/RGBA edit.")
    return reply[1], reply[2:], time.monotonic() - started


def smoke(args):
    if sys.platform != "linux" or platform.machine() != "x86_64" or os.geteuid() == 0:
        raise PreflightError("Use a non-root account on the dedicated Linux AMD64 test host.")
    _private_directory(args.output.parent)
    if os.path.lexists(args.output):
        raise PreflightError("Choose a new output file; this tool never overwrites.")
    reference = reference_pixels()
    with _gpu_lock(Path.home().resolve() / ".modelship-diffusion-smoke", args.gpu) as lock:
        with _verified_runtime(args.worker, args.worker_sha256) as worker:
            verify_edit_bundle(args.bundle)
            free = admit_edit_smoke_gpu(args.gpu)
            print(f"Preflight passed ({free} MiB free). Loading the edit candidate; one request only.", flush=True)
            with tempfile.TemporaryDirectory(prefix="modelship-diffusion-edit-") as temp:
                env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "HOME": temp, "TMPDIR": temp,
                       "CUDA_VISIBLE_DEVICES": args.gpu, "CUDA_CACHE_DISABLE": "1"}
                child = subprocess.Popen(["modelship-sd-edit-candidate", "--private-edit-stdio", str(args.bundle)],
                                         executable=f"/proc/self/fd/{worker}", env=env, cwd=temp,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         pass_fds=(lock, worker), start_new_session=True)
                try:
                    channels, pixels, elapsed = _edit_once(child, reference)
                    if child.wait(timeout=30) != 0 or child.stdout.read(1):
                        raise PreflightError("The candidate did not exit cleanly after the edit.")
                finally:
                    child.stdout.close()
                    child.stdin.close()
                    _stop_child(child)
                _write_png(args.output, channels, pixels)
    print(f"Candidate returned one 512x512 edit in {elapsed:.2f}s (seed {SEED}). PNG saved; no service started.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="physical NVIDIA GPU UUID, not index")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--experimental", action="store_true")
    args = parser.parse_args()
    if not args.research or not args.experimental:
        parser.error("explicit --research and --experimental acknowledgements are required")

    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        smoke(args)
        return 0
    except PreflightError as exc:
        print(str(exc), file=sys.stderr)
    except (OSError, ValueError, subprocess.SubprocessError):
        print("Candidate edit could not complete. Check private paths and host CUDA userspace. No retry.", file=sys.stderr)
    except KeyboardInterrupt:
        print("Candidate edit interrupted. No retry.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
