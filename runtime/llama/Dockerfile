# syntax=docker/dockerfile:1.7
# Only builds private userspace inference files; never installs onto the GPU host.
FROM --platform=linux/amd64 nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7 AS build
ENV SOURCE_DATE_EPOCH=0 LC_ALL=C TZ=UTC
WORKDIR /src
ADD --checksum=sha256:fbe74146a3cdc00f706773eb233bc4aedb7d8c3c7e2122440e492a99c1ec5a76 https://codeload.github.com/ggml-org/llama.cpp/tar.gz/cd959bae99ff4bed22bef2604f45241c09bbb8c8 /src/llama.tar.gz
ADD --checksum=sha256:2b3eee7b044f528408b45ab85535ce6c7430b063d5bc939599a4e038da7cf6fb https://codeload.github.com/NVIDIA/cccl/tar.gz/refs/tags/v3.4.3 /src/cccl.tar.gz
ADD --checksum=sha256:630615d8e98ac33eba7fbe472626dff5c899c85af3c024585ae109166a6909d0 https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-linux-x86_64.tar.gz /src/cmake.tar.gz
RUN --network=none mkdir llama cccl cmake && \
    tar -xzf llama.tar.gz --strip-components=1 -C llama && \
    tar -xzf cccl.tar.gz --strip-components=1 -C cccl && \
    tar -xzf cmake.tar.gz --strip-components=1 -C cmake && \
    gcc --version && g++ --version && /usr/local/cuda/bin/nvcc --version && \
    /src/cmake/bin/cmake --version
# No apt, pip, fetched UI, BoringSSL, native-CPU tuning or unpinned CMake fetches.
# Real Ampere/Ada cubins + Hopper PTX + architecture-specific Blackwell cubin.
RUN --network=none /src/cmake/bin/cmake -S /src/llama -B /src/build \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
    '-DCMAKE_CUDA_ARCHITECTURES=80-real;86-real;89-real;90-virtual;120a-real' \
    '-DCMAKE_INSTALL_RPATH=$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_SOURCE_DIR_CCCL=/src/cccl \
    -DLLAMA_BUILD_NUMBER=12110 -DLLAMA_BUILD_COMMIT=cd959bae99ff4bed22bef2604f45241c09bbb8c8 \
    -DGGML_CUDA=ON -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DGGML_CUDA_NCCL=OFF \
    -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_OPENMP=OFF \
    -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_BORINGSSL=OFF -DLLAMA_CONNECT=OFF \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_SUBPROCESS=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF && \
    /src/cmake/bin/cmake --build /src/build --target llama-server --parallel 2
COPY runtime/llama/audit.sh /src/audit.sh
RUN --network=none mkdir -p /out/package /out/evidence && \
    cp /src/build/bin/llama-server /out/package/ && \
    cp -P /src/build/bin/lib*.so* /out/package/ && \
    cp /src/llama/LICENSE /out/package/LICENSE.llama.cpp && \
    cp /src/cccl/LICENSE /out/package/LICENSE.CCCL && \
    sh /src/audit.sh /out/package /out/evidence/abi.txt && \
    LD_LIBRARY_PATH=/out/package /out/package/llama-server --version > /out/evidence/version.txt 2>&1 && \
    LD_LIBRARY_PATH=/out/package /out/package/llama-server --help > /out/evidence/help.txt 2>&1 && \
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --hard-dereference \
      -cf - -C /out/package . | gzip -n -9 > /out/llama-b12110-ubuntu22.04-cuda12.8-r1.tar.gz && \
    cd /out && sha256sum llama-b12110-ubuntu22.04-cuda12.8-r1.tar.gz > SHA256SUMS && \
    stat -c '%s' llama-b12110-ubuntu22.04-cuda12.8-r1.tar.gz > /out/evidence/size.txt && \
    cp /src/build/CMakeCache.txt /out/evidence/CMakeCache.txt
FROM scratch AS export
COPY --from=build /out/llama-b12110-ubuntu22.04-cuda12.8-r1.tar.gz /
COPY --from=build /out/SHA256SUMS /
COPY --from=build /out/evidence /evidence
