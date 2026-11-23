# GGUF diffusion runtime redistribution checklist

**Uncleared. Do not publish alpha.17, upload runtime binaries, or merge PR #20 yet.**
The initial local package archives and the Go pins in the draft still refer to the historical GPU-tested executables. They are not the replacement build. `releases/diffusion/THIRD_PARTY_NOTICES.txt` is preliminary; changing it does not clear any executable or preserve its archive hash.

## Replacement candidate

- The source-pinned build removes the AUTOMATIC1111-derived Philox and prompt-attention implementations. It uses the existing C++ standard-library noise generator and a literal unit-weight prompt segment. The worker still refuses control/weight syntax at the protocol boundary. Noise generation changes seed outputs relative to historical binaries, not the CUDA-only neural graph or weight placement policy.
- The private context rejects non-Qwen-Image-2.1 model versions before backend setup. Exclude the AUTOMATIC1111-derived custom CLIP conditioning path and its PhotoMaker caller, plus the unused standalone ControlNet implementation (ambiguous ComfyUI/Apache ControlNet lineage). Disable non-unit token-weight rescaling; the literal Qwen path already supplied unit weights. Verify the excluded implementations are absent from the worker, not just from the small probe.
- No upstream embedded tokenizer tables are compiled in. The worker obtains the exact same ordered Qwen merges from the verified encoder GGUF. There is no extra model artifact, and no tokenizer fallback.
- `CCCL_DISABLE_NVTX` disables the Toolkit NVTX implementation without dropping CUB. Verify the flag on CUDA translation units and check **all** executable symbol tables; a dynamic dependency list on its own does not establish what is compiled in.
- CI is evidence-only. Do not introduce automatic binary uploads or use the historical run artifacts as installer inputs.

## Material to compare against the actual builder

The Dockerfile pins the Ubuntu 22.04 CUDA 12.8.1 builder by digest, and pins both upstream source archives. Use its actual Toolkit version and header hashes from `cuda-build-evidence.sh`, plus compiler dependencies and symbols, for the final inventory. NVIDIA's [12.8.1 manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json) lists the following Linux x86_64 reference archives; these hashes are **not** claims about the files inside the builder:

| Component | Manifest version | Archive SHA-256 |
|---|---|---|
| `cuda_cccl` | 12.8.90 | `0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b` |
| `cuda_nvtx` | 12.8.90 | `7755524026d30e2c211d793392166517e379724d19eb9e8e352693f7d5fa0c95` |
| `cuda_cudart` | 12.8.90 | `8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68` |
| `cuda_nvcc` | 12.8.93 | `9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3` |

The verified CCCL reference archive contains `include/cub/device/device_reduce.cuh` (`b335e3953ac8542eedceb09ae024c34317f094fa0f68c9135d01b72351f02674`) and `include/cub/detail/nvtx.cuh` (`43696060cc5f1db2c97c887328f89e959feeb74b9b293355137f13e53068e070`). [CUDA CI 45272186154](https://github.com/0xburden/modelship/actions/runs/45272186154) (`9f9a7a9b2a3069dcee4a2069c15c673e4e547d03`) matched both header hashes and reported `cuda-cccl-12-8` `12.8.90-1`. It verified `CCCL_DISABLE_NVTX` on 161 CUDA and 146 C++ translation units and passed the four-ELF symbol and ABI checks. The evidence-only ZIP (artifact `14867003431`, SHA-256 `04f67b3cebc3804e653a8481c5a6aab66a079d35ad4e182d4b3556746dcf302c`) contains no executables. The draft notices now include both matching CUB BSD-3-Clause blocks; applicable Thrust/libcu++ notices and exceptions still need review. Check applicable terms for emitted CUDA compiler/runtime header code, device runtime and libdevice against the [12.8.1 Toolkit terms](https://docs.nvidia.com/cuda/archive/12.8.1/eula/index.html). Neither the package-root EULA nor separately downloaded CUDA runtime wheels cover all of that source inventory. The runtime archive does not ship NVIDIA shared libraries.

Expand the final notices using the pinned stable-diffusion.cpp and ggml license files and the linked-source inventory. In particular, include the current Oniguruma source copyrights, the full utf8proc/Unicode text, ZIP/miniz and JSON contributor terms, SageAttention/FlashInfer notices, and attributions for retained ports. Review any remaining upstream citations against linked implementations: a citation alone is not proof of a port, and code unreachable from the worker's API is not proof that it is absent from the binary. The H3 latent-upscaler implementation is a direct port of a separately MIT-licensed [community node](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/f34cadb19f10fee7e36091b690cac39b4eb40573/LICENSE), not of GPL ComfyUI core. The draft includes its full notice (`Copyright (c) 2026 LBH-123-AI`, license SHA-256 `86805fe49f63c8957e4427f7e7f26fba687c6cea05fa9249bf36f9d3316a0734`).

## Before publication

1. Finish the linked-source inventory and notices, resolve any remaining uncertain implementation, and pass the pinned CUDA build and ELF checks. Keep artifact export disabled until redistribution is cleared.
2. On a separately authorized, dedicated A6000, test the exact replacement generation and edit workers end to end: semantic output, capability/readiness, strict request bounds, overlapping request rejection, restart, verified cache reuse, and repeat-seed consistency within this new build. Do not compare hashes against RNG-specific historical PNGs to establish correctness.
3. Package and pin the exact tested pair for each profile, plus the complete notices. Recheck archive extraction, planned consent and offline install, all immutable retained CLI files, and CI before asking for merge/publication approval. Alpha.16 stays public until then.
