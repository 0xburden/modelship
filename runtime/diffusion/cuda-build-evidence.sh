#!/bin/sh
# Invoked inside the pinned CUDA builder, with no driver or GPU.
set -eu
build=$1
out=$2
toolkit=/usr/local/cuda/targets/x86_64-linux
commands=$build/compile_commands.json
/src/cmake/bin/cmake -DCOMPILE_DATABASE="$commands" -P /src/check-cuda-compile.cmake > "$out/cuda-compile-guard.txt"
cat "$out/cuda-compile-guard.txt"
cp "$commands" "$out/compile_commands.json"
# Record versions and hashes, not NVIDIA header bodies, in the evidence.
cat /usr/local/cuda/version.json > "$out/toolkit-version.json"
sha256sum "$toolkit/include/cub/device/device_reduce.cuh" \
    "$toolkit/include/cub/detail/nvtx.cuh" > "$out/toolkit-header-sha256.txt"
grep -Fq CCCL_DISABLE_NVTX "$toolkit/include/cub/detail/nvtx.cuh" || {
    echo "Pinned Toolkit CUB headers lack the CCCL_DISABLE_NVTX guard" >&2
    exit 1
}
