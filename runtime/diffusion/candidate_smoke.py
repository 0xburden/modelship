#!/usr/bin/env python3
"""One offline 512x512 CUDA smoke; not an installer or service."""

import argparse
import fcntl
import hashlib
import os
import platform
import re
import select
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

from candidate_preflight import GPU_UUID, PreflightError, admit_smoke_gpu, verify_generation_bundle

SIDE = 512
PROMPT = b"A watercolor painting of a red sailboat on a calm alpine lake, with pine trees and distant mountains."
SEED = 43
MAX_RESPONSE = 2 + 4 * SIDE * SIDE


def _private_directory(path):
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise PreflightError("Use a canonical, private directory.")
    st = path.stat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
        raise PreflightError("Use a private directory owned by this account.")


@contextmanager
def _verified_runtime(path, expected):
    _private_directory(path.parent)
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise PreflightError("An independently verified worker SHA-256 is required.")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid() or before.st_nlink != 1 or
                before.st_mode & (0o7077 | 0o200) or not before.st_mode & 0o100 or
                not 20 <= before.st_size <= 256 * 1024 * 1024):
            raise PreflightError("Worker must be a private, read-only executable (chmod 500).")
        header = os.read(fd, 20)
        if header[:7] != b"\x7fELF\x02\x01\x01" or header[16:20] not in (b"\x02\x00\x3e\x00", b"\x03\x00\x3e\x00"):
            raise PreflightError("Expected a Linux AMD64 worker ELF executable.")
        digest = hashlib.sha256(header)
        remaining = before.st_size - len(header)
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                raise PreflightError("Worker changed during verification.")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if (os.read(fd, 1) or before.st_mtime_ns != after.st_mtime_ns or before.st_ctime_ns != after.st_ctime_ns or
                digest.hexdigest() != expected):
            raise PreflightError("Worker failed SHA-256 verification.")
        # Execute this verified descriptor, not the path we originally opened.
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _gpu_lock(directory, uuid):
    # GPU UUID is validated before using it as a path component.
    if not GPU_UUID.fullmatch(uuid):
        raise PreflightError("An explicit physical GPU UUID is required.")
    directory.mkdir(mode=0o700, exist_ok=True)
    _private_directory(directory)
    fd = os.open(directory / (uuid + ".lock"), os.O_CREAT | os.O_RDWR | os.O_CLOEXEC |
                 os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_nlink != 1 or st.st_mode & 0o077:
            raise PreflightError("Unsafe candidate lock file.")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PreflightError("Another candidate already owns this GPU lock.") from None
        # The worker inherits this descriptor. Even if the parent dies, it
        # keeps the lock until the worker exits. Never unlink an active lock.
        yield fd
    finally:
        os.close(fd)


def _read_exact(fd, size, deadline):
    result = bytearray()
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    while len(result) < size:
        left = deadline - time.monotonic()
        if left <= 0 or not poller.poll(max(1, int(left * 1000))):
            raise PreflightError("Candidate worker timed out.")
        part = os.read(fd, size - len(result))
        if not part:
            raise PreflightError("Candidate worker exited before completing the reply.")
        result.extend(part)
    return result


def _frame(fd, deadline, cap):
    length = struct.unpack("!I", _read_exact(fd, 4, deadline))[0]
    if not 1 <= length <= cap:
        raise PreflightError("Unexpected candidate frame length.")
    return _read_exact(fd, length, deadline)


def _generate_once(child, load_seconds=600, generate_seconds=900):
    out = child.stdout.fileno()
    if _frame(out, time.monotonic() + load_seconds, 2) != b"L\x01":
        raise PreflightError("The candidate did not load the pinned context.")
    payload = struct.pack("!I", SEED) + PROMPT
    started = time.monotonic()
    # One small write to an initially empty pipe; close immediately to forbid
    # a second request. No command arguments or logs contain user input.
    child.stdin.write(struct.pack("!I", len(payload)) + payload)
    child.stdin.close()
    reply = _frame(out, started + generate_seconds, MAX_RESPONSE)
    if len(reply) < 2 or reply[0] != ord("I") or reply[1] not in (3, 4) or len(reply) != 2 + reply[1] * SIDE * SIDE:
        raise PreflightError("The candidate did not return a drained 512x512 RGB/RGBA image.")
    return reply[1], reply[2:], time.monotonic() - started


def _stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def _write_png(path, channels, pixels):
    def chunk(tag, data):
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data))

    stride = SIDE * channels
    scanlines = b"".join(b"\x00" + pixels[start:start + stride] for start in range(0, len(pixels), stride))
    fd, temp = tempfile.mkstemp(prefix=".candidate-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(b"\x89PNG\r\n\x1a\n")
            out.write(chunk(b"IHDR", struct.pack("!2I5B", SIDE, SIDE, 8, {3: 2, 4: 6}[channels], 0, 0, 0)))
            out.write(chunk(b"IDAT", zlib.compress(scanlines)))
            out.write(chunk(b"IEND", b""))
        os.link(temp, path, follow_symlinks=False)  # atomic, refuses overwrite
    finally:
        os.unlink(temp)


def smoke(args):
    if sys.platform != "linux" or platform.machine() != "x86_64" or os.geteuid() == 0:
        raise PreflightError("Use a non-root account on the dedicated Linux AMD64 test host.")
    _private_directory(args.output.parent)
    if os.path.lexists(args.output):
        raise PreflightError("Choose a new output file; this tool never overwrites.")
    with _gpu_lock(Path.home().resolve() / ".modelship-diffusion-smoke", args.gpu) as lock:
        with _verified_runtime(args.worker, args.worker_sha256) as worker:
            verify_generation_bundle(args.bundle)
            free = admit_smoke_gpu(args.gpu)
            print(f"Preflight passed ({free} MiB free). Loading the candidate; one request only.", flush=True)
            with tempfile.TemporaryDirectory(prefix="modelship-diffusion-smoke-") as temp:
                env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "HOME": temp, "TMPDIR": temp,
                       "CUDA_VISIBLE_DEVICES": args.gpu, "CUDA_CACHE_DISABLE": "1"}
                child = subprocess.Popen(["modelship-sd-worker", "--private-stdio", str(args.bundle)],
                                         executable=f"/proc/self/fd/{worker}", env=env, cwd=temp,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         pass_fds=(lock, worker), start_new_session=True)
                try:
                    channels, pixels, elapsed = _generate_once(child)
                    if child.wait(timeout=30) != 0 or child.stdout.read(1):
                        raise PreflightError("The candidate did not exit cleanly after the image.")
                finally:
                    child.stdout.close()
                    child.stdin.close()
                    _stop_child(child)
                _write_png(args.output, channels, pixels)
    print(f"Candidate returned one 512x512 image in {elapsed:.2f}s (seed {SEED}). PNG saved; no service started.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="physical NVIDIA GPU UUID, not index")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--research", action="store_true", help="acknowledge the Qwen Research License")
    parser.add_argument("--experimental", action="store_true", help="acknowledge the unvalidated GPU candidate")
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
        print("Candidate smoke could not complete. Check private paths and host CUDA userspace. No retry.",
              file=sys.stderr)
    except KeyboardInterrupt:
        print("Candidate smoke interrupted. No retry.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
