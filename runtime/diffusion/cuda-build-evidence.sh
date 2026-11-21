#!/bin/sh
# Invoked inside the pinned CUDA builder, with no driver or GPU.
set -eu
build=$1
out=$2
toolkit=/usr/local/cuda/targets/x86_64-linux
commands=$build/compile_commands.json
# CMake's Unix Makefiles generator records each command on one line. In
# addition to the per-ELF symbol audit, require the flag on *every* .cu TU.
awk '
    /"command":/ && / -c [^ ]+\.cu"/ {
        count++
        if ($0 !~ / -DCCCL_DISABLE_NVTX /) bad++
    }
    END {
        printf "CUDA translation units: %d; missing CCCL_DISABLE_NVTX: %d\n", count, bad
        if (count == 0 || bad > 0) exit 1
    }
' "$commands" > "$out/cuda-compile-guard.txt"
cp "$commands" "$out/compile_commands.json"
# Record versions and hashes, not NVIDIA header bodies, in the evidence.
cat /usr/local/cuda/version.json > "$out/toolkit-version.json"
sha256sum "$toolkit/include/cub/device/device_reduce.cuh" \
    "$toolkit/include/cub/detail/nvtx.cuh" > "$out/toolkit-header-sha256.txt"
grep -Fq CCCL_DISABLE_NVTX "$toolkit/include/cub/detail/nvtx.cuh"
