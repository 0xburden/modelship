#!/bin/sh
# Invoked inside the pinned CUDA builder, with no driver or GPU.
set -eu
build=$1
out=$2
cuda=/usr/local/cuda
toolkit=$cuda/targets/x86_64-linux
commands=$build/compile_commands.json
/src/cmake/bin/cmake -DCOMPILE_DATABASE="$commands" -P /src/check-cuda-compile.cmake > "$out/cuda-compile-guard.txt"
cat "$out/cuda-compile-guard.txt"
cp "$commands" "$out/compile_commands.json"
# The apt-installed Toolkit in the pinned image has no version.json.
# Record compiler and package versions plus hashes, not header bodies.
"$cuda/bin/nvcc" --version > "$out/toolkit-nvcc-version.txt"
dpkg-query -W -f='${binary:Package}\t${Version}\n' 'cuda-*' > "$out/toolkit-packages.txt"
sha256sum "$toolkit/include/cub/device/device_reduce.cuh" \
    "$toolkit/include/cub/detail/nvtx.cuh" > "$out/toolkit-header-sha256.txt"
cat "$out/toolkit-nvcc-version.txt" "$out/toolkit-header-sha256.txt"
grep -Fq CCCL_DISABLE_NVTX "$toolkit/include/cub/detail/nvtx.cuh" || {
    echo "Pinned Toolkit CUB headers lack the CCCL_DISABLE_NVTX guard" >&2
    exit 1
}
