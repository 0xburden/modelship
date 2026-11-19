"""Offline bundle verification and conservative admission for a future GPU smoke.

Not a deployment command. It never downloads, loads, or starts a model.
"""

import csv
import hashlib
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


@dataclass(frozen=True)
class Component:
    name: str
    bytes: int
    sha256: str


GENERATION_BUNDLE = (
    Component("qwen_image_2.1-Q4_K.gguf", 4197494816,
              "29f9c83c249ff0292fb2943fceddfa2319b446601866c82a4f8be062abea72c2"),
    Component("Qwen3VL-8B-Instruct-Q4_K_M.gguf", 5027784800,
              "67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2"),
    Component("qwen_image_2.1_vae_bf16.safetensors", 675509688,
              "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"),
)
EDIT_BUNDLE = GENERATION_BUNDLE + (
    Component("mmproj-Qwen3VL-8B-Instruct-F16.gguf", 1159029824,
              "ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38"),
)
# First smoke only, intentionally conservative and non-overridable. This is
# not an estimate of minimum residency or the memory requirement for deployment.
SMOKE_FREE_MIB = 40 * 1024
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


class PreflightError(RuntimeError):
    pass


def _version(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mode, st.st_uid, st.st_nlink,
            st.st_mtime_ns, st.st_ctime_ns)


def _verify_components(directory: Path, components: tuple[Component, ...]):
    if not directory.is_absolute() or directory.resolve(strict=True) != directory:
        raise PreflightError("Expected a canonical, private bundle directory.")
    dir_fd = os.open(directory, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        initial = os.fstat(dir_fd)
        if initial.st_uid != os.geteuid() or initial.st_mode & 0o077:
            raise PreflightError("Expected a private bundle directory owned by this account.")
        versions = {}
        for item in components:
            fd = os.open(item.name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
            with os.fdopen(fd, "rb") as file:
                before = os.fstat(file.fileno())
                if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid() or
                        before.st_mode & 0o077 or before.st_nlink != 1 or before.st_size != item.bytes):
                    raise PreflightError("A component does not match the reviewed private-file layout.")
                digest = hashlib.sha256()
                remaining = item.bytes
                while remaining:
                    part = file.read(min(remaining, 1024 * 1024))
                    if not part:
                        raise PreflightError("A component changed during verification.")
                    digest.update(part)
                    remaining -= len(part)
                if digest.hexdigest() != item.sha256:
                    raise PreflightError("A model component failed SHA-256 verification.")
                if file.read(1) or _version(before) != _version(os.fstat(file.fileno())):
                    raise PreflightError("A component changed during verification.")
                versions[item.name] = _version(before)
        for name, version in versions.items():
            if _version(os.stat(name, dir_fd=dir_fd, follow_symlinks=False)) != version:
                raise PreflightError("A component changed during verification.")
        if _version(os.stat(directory, follow_symlinks=False)) != _version(initial):
            raise PreflightError("The bundle directory changed during verification.")
    finally:
        os.close(dir_fd)


def _verify_bundle(directory, components):
    try:
        _verify_components(directory, components)
    except (OSError, RuntimeError) as error:
        if isinstance(error, PreflightError):
            raise
        raise PreflightError("Could not verify the private model bundle.") from None


def verify_generation_bundle(directory: Path):
    """Check exact local weight files. The same account must keep them unchanged through use."""
    _verify_bundle(directory, GENERATION_BUNDLE)


def verify_edit_bundle(directory: Path):
    """Require the exact generation components and the separate matching F16 projector."""
    _verify_bundle(directory, EDIT_BUNDLE)


def _admit_gpu(rows: str, compute_processes: str, uuid: str):
    if not GPU_UUID.fullmatch(uuid):
        raise PreflightError("An explicit physical GPU UUID is required.")
    try:
        entries = list(csv.reader(rows.splitlines(), strict=True))
        if len(entries) != 1 or len(entries[0]) != 5:
            raise ValueError
        actual, total, free, capability, driver = map(str.strip, entries[0])
        if not all(re.fullmatch(r"[0-9]+", memory) for memory in (total, free)):
            raise ValueError
        total, free = int(total), int(free)
        if (actual != uuid or not re.fullmatch(r"\d+\.\d+", capability) or
                not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", driver) or not 0 <= free <= total):
            raise ValueError
    except (ValueError, csv.Error):
        raise PreflightError("The selected physical GPU could not be verified.") from None
    if (tuple(map(int, capability.split("."))) < (8, 0) or
            tuple(map(int, driver.split(".")[:2])) < (570, 26)):
        raise PreflightError("The smoke requires compute capability 8.0+ and NVIDIA driver 570.26+.")
    if compute_processes.strip() or free < SMOKE_FREE_MIB:
        raise PreflightError("The smoke requires a dedicated idle GPU with at least 40 GiB free.")
    return free


def admit_smoke_gpu(uuid: str):
    """Best-effort idle snapshot, not exclusive scheduling. Recheck immediately before launch."""
    if not GPU_UUID.fullmatch(uuid):
        raise PreflightError("An explicit physical GPU UUID is required.")
    env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    try:
        base = ["/usr/bin/nvidia-smi", "--id=" + uuid, "--format=csv,noheader,nounits"]
        rows = subprocess.check_output(base + ["--query-gpu=uuid,memory.total,memory.free,compute_cap,driver_version"],
                                       env=env, timeout=10, text=True, stderr=subprocess.DEVNULL)
        procs = subprocess.check_output(base + ["--query-compute-apps=pid"],
                                        env=env, timeout=10, text=True, stderr=subprocess.DEVNULL)
    except (OSError, UnicodeError, subprocess.SubprocessError):
        raise PreflightError("Could not obtain the selected GPU's memory and process inventory.") from None
    return _admit_gpu(rows, procs, uuid)


def admit_edit_smoke_gpu(uuid: str):
    """Also fail closed if the selected GPU's graphics inventory is not empty."""
    free = admit_smoke_gpu(uuid)
    try:
        xml = subprocess.check_output(["/usr/bin/nvidia-smi", "-i", uuid, "-q", "-x"],
                                      env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"}, timeout=10,
                                      stderr=subprocess.DEVNULL)
        if len(xml) > 1024 * 1024:
            raise ValueError
        root = ElementTree.fromstring(xml)
        gpus = root.findall("gpu")
        if root.tag != "nvidia_smi_log" or len(gpus) != 1 or gpus[0].findtext("uuid", "").strip() != uuid:
            raise ValueError
        inventories = gpus[0].findall("processes")
        if len(inventories) != 1 or len(inventories[0]) != 0 or (inventories[0].text or "").strip():
            raise ValueError
    except (OSError, ValueError, ElementTree.ParseError, subprocess.SubprocessError):
        raise PreflightError("The edit smoke requires a verified empty graphics and compute inventory.") from None
    return free
