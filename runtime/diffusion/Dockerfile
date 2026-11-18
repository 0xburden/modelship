# syntax=docker/dockerfile:1.7
# Candidate-only native Linux AMD64 probe. No model weights, deploy, or release.
FROM --platform=linux/amd64 nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7 AS build
ENV SOURCE_DATE_EPOCH=0 LC_ALL=C TZ=UTC
WORKDIR /src
ADD --checksum=sha256:0278dc8e6e180adc4b43b3ace515eeef870dd76533a34fa1edcfa071ce7d7812 https://codeload.github.com/leejet/stable-diffusion.cpp/tar.gz/532d2c15c430a785a712b2d94281f04e733008a4 /src/sd.tar.gz
ADD --checksum=sha256:8bd7adc7eb1366d31670467951d21eb1a5f55e8f6d89b9b91909ffb0246ac5d2 https://codeload.github.com/leejet/ggml/tar.gz/eeda9ec48fc84a3df0bb9dcc19ae9053075a2539 /src/ggml.tar.gz
ADD --checksum=sha256:630615d8e98ac33eba7fbe472626dff5c899c85af3c024585ae109166a6909d0 https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-linux-x86_64.tar.gz /src/cmake.tar.gz
COPY runtime/diffusion/CMakeLists.txt runtime/diffusion/probe.cpp /src/probe/
COPY runtime/llama/audit.sh /src/audit.sh
RUN --network=none mkdir -p /src/sd/ggml /src/cmake && \
    tar -xzf sd.tar.gz --strip-components=1 -C sd && \
    tar -xzf ggml.tar.gz --strip-components=1 -C sd/ggml && \
    tar -xzf cmake.tar.gz --strip-components=1 -C cmake
# The builder provides CUDA headers/libs. Do not fetch GGML, CCCL, CUTLASS or UI dependencies.
# Build one Ampere real target and compute80 PTX. No other accelerator or dynamic backend plugins.
RUN --network=none /src/cmake/bin/cmake -S /src/probe -B /src/build \
    -DSD_SOURCE=/src/sd -DGIT_EXE:FILEPATH= -DSDCPP_BUILD_VERSION=0.0.0 \
    -DSDCPP_BUILD_COMMIT=532d2c15c430a785a712b2d94281f04e733008a4 \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DSD_CUDA=ON -DSD_CUDA_CUTLASS_PATH:PATH= -DSD_BUILD_EXAMPLES=OFF \
    -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_METAL=OFF -DSD_RPC=OFF \
    -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_GRAPHS=OFF -DGGML_BACKEND_DL=OFF -DGGML_NATIVE=OFF \
    -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_OPENMP=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF \
    -DGGML_BMI2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_SSE42=ON \
    -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF && \
    /src/cmake/bin/cmake --build /src/build --target modelship-sd-probe --parallel 2
RUN --network=none mkdir -p /out/bin /out/evidence && \
    cp /src/build/bin/modelship-sd-probe /out/bin/ && \
    cp /src/build/CMakeCache.txt /out/evidence/ && \
    sh /src/audit.sh /out/bin /out/evidence/elf-abi.txt /usr/local/cuda/targets/x86_64-linux/lib 1 && \
    readelf -d /out/bin/modelship-sd-probe > /out/evidence/dynamic.txt && \
    grep -q 'libcudart.so.12' /out/evidence/dynamic.txt && \
    ! grep -Eq 'RPATH|RUNPATH' /out/evidence/dynamic.txt && \
    cd /out && sha256sum bin/modelship-sd-probe > SHA256SUMS
FROM scratch AS candidate
COPY --from=build /out/ /
