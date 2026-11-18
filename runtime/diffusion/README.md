# GGUF diffusion runtime investigation

**Candidate only.** This directory is not a supported deployment profile and does not change the installer, `modelship` CLI, llama.cpp workers, or native Diffusers workers. The probe cannot load a model, open a port, or generate an image. A separate private stdio worker candidate can load the fixed generation profile. It does not open a port and is not wired into the CLI. The stock upstream HTTP server is not part of this build.

## Runtime probe

- [stable-diffusion.cpp `master-1106-532d2c1`](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-1106-532d2c1), source `532d2c15c430a785a712b2d94281f04e733008a4`. Its ggml gitlink is the leejet fork at `eeda9ec48fc84a3df0bb9dcc19ae9053075a2539`. Both sources are MIT; that does not grant rights to model weights or CUDA libraries.
- No standalone CUDA Linux asset is published for that release. The upstream CUDA container is Ubuntu 24.04/CUDA 12.6.3. This probe instead checks a native Linux AMD64 build on Ubuntu 22.04/CUDA 12.8.1, SM80 plus compute80 PTX. The source archives, CMake archive and builder image are independently pinned; configure and compile are offline. CUTLASS, the HTTP server, media output extensions, RPC and dynamic backend plug-ins are excluded.
- The candidate export contains the probe and private worker, checksums, and ELF symbol/dependency reports. It does **not** contain NVIDIA userspace libraries, host drivers, or a redistributable runtime. The build-time ABI check permits only the expected missing host `libcuda.so.1` on a GPU-less runner. It does not run a driver stub.
- `modelship-sd-probe --version` reports the linked source commit. On an authorized machine with reviewed CUDA userspace, `--cuda-check` is intended to verify exactly one visible CUDA device with a non-host 4 KiB allocation; it refuses the ggml unified-memory override. This would not establish inference compatibility, full graph placement, or peak memory.

## First model candidate: Qwen Image 2.1 (non-Turbo)

The [pinned upstream guide](https://github.com/leejet/stable-diffusion.cpp/blob/532d2c15c430a785a712b2d94281f04e733008a4/docs/models/qwen_image_2.1.md#model-files) specifies a separate diffusion transformer, VAE and Qwen3-VL encoder. The fourth file is the matching vision projector **for edit conditioning**, not a llama.cpp chat attachment. The diffusion weights and VAE remain subject to the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE); an explicit research opt-in and separate experimental consent would be required. This is not a commercial-use profile.

These are metadata pins, **not installed or locally whole-file verified**. Only bounded headers were inspected; no tensor payloads were downloaded.

| Role | Repository @ revision | File | Bytes | Published SHA-256 |
| --- | --- | --- | ---: | --- |
| Diffusion, Q4_K | `leejet/Qwen-Image-2.1-GGUF` @ `0bca76da6edd8a18211e9717454615c1bbaaf6ed` | `qwen_image_2.1-Q4_K.gguf` | 4,197,494,816 | `29f9c83c249ff0292fb2943fceddfa2319b446601866c82a4f8be062abea72c2` |
| Encoder, Q4_K_M | `Qwen/Qwen3-VL-8B-Instruct-GGUF` @ `f982a07559d4a2f6c8744d840bf6fccab30eea96` | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | 5,027,784,800 | `67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2` |
| Edit projector, F16 | same Qwen3-VL repository and revision | `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | 1,159,029,824 | `ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38` |
| VAE, BF16 | `Comfy-Org/Qwen-Image-2.1` @ `aacf31a6d48e707f352ecba714cee0680da3479b` | `vae/qwen_image_2.1_vae_bf16.safetensors` | 675,509,688 | `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9` |

Total downloads: **9,900,789,304 bytes** for generation, **11,059,819,128 bytes** with editing. These totals are neither runtime residency estimates nor minimum VRAM requirements. In particular, the diffusion GGUF v3 header is a tensor directory with 265 tensors and **zero KV entries**. The current text GGUF parser's chat architecture/metadata assumptions must not be relaxed to admit it. A diffusion-specific inspection and accounting path is needed, including any runtime dtype expansion and mandatory CUDA workspace.

## Private generation worker (protocol v1)

The worker is a child process, not a service. It uses the pinned C API and patched CUDA-only graph and weight placement. It supports a single non-Turbo image at **512×512**, 40 Euler steps, upstream model-selected schedule, text CFG 6, and an explicit unsigned 32-bit seed. No editing, adapters, sampler overrides, fetches, or request-selected model paths. The worker does not inherit an HTTP interface from upstream.

The future parent must verify the full runtime and three model component hashes, expose only the intended GPU, and pass a curated environment and a private immutable component directory. The worker checks the reviewed local names and byte counts before opening the models; that is not a substitute for hash verification. It requires one visible CUDA device, requests eager loading with no mmap or auto-fit, and checks the loaded model family. The parent needs a hard admission gate, process deadlines and exclusive ownership until the child drains or has been reaped. Do not run it against shared or unverified files.

The private protocol uses a four-byte big-endian body length followed by the body:

| Direction | Body |
| --- | --- |
| Parent → worker | Four-byte big-endian seed, then 1–1,024 UTF-8 prompt bytes (no angle brackets, parentheses, square brackets, backslashes or `BREAK` syntax; reject rather than rewrite). |
| Worker → parent | `L`, byte `1`: context loaded, **not** public or warm-up readiness. |
| Worker → parent | `I`, byte `3` or `4`, then contiguous RGB or RGBA bytes for exactly 512×512 pixels. Only sent after successful CUDA synchronization. |
| Worker → parent | `V`: invalid prompt, no inference. |
| Worker → parent | `F` or premature EOF: fatal; close pipes, reap, and do not retry the request. |

Keep one request outstanding. An invalid frame length or truncated frame closes the worker. The weight-free protocol test uses a simulated engine; it verifies acceptance, rejection, bounded response framing, and that a failed request cannot lead to a second inference. It does not establish CUDA load or generation success.

## Gates before adding `deploy` support

1. **Enforce placement, not just device selection.** The [upstream graph scheduler](https://github.com/leejet/stable-diffusion.cpp/blob/532d2c15c430a785a712b2d94281f04e733008a4/src/core/compute_workspace.cpp#L28-L48) requires CPU registration for allocation and scheduling. `cuda-placement.cmake` patches only the exact pinned ggml files in the disposable build tree and verifies the resulting hashes. It preflights all final scheduler splits before dispatch, guards direct and planned dispatch, and blocks nonempty graphs at the CPU kernel entry point (including nested extension calls). The weight-free test links the real patched CPU backend: allocation still works, rejected compute leaves the output untouched, and a fake CUDA0 followed by an unsupported CPU split is rejected before even the first dispatch. It does not run CUDA. Upstream [`params_buffer_type_for`](https://github.com/leejet/stable-diffusion.cpp/blob/532d2c15c430a785a712b2d94281f04e733008a4/src/model_manager.cpp#L1135-L1169) can choose a host buffer even with CUDA parameters configured. `cuda-residency.cmake` checks the exact pinned model manager: reject split, disk and mismatched compute/parameter backends; reject host or nondefault buffers before allocation; verify attached buffers before reading weight payloads and when reusing loaded parameters. Persistent host mappings and disk demotion are blocked. The weight-free test covers the buffer predicate with real CPU buffers and synthetic device metadata. The private worker must request eager loading, forbid unified memory and treat unsupported graphs or storage as readiness failures. This is enforcement code, **not** a measurement of where the Qwen weights actually load or whether an inference graph runs. Ordinary host tokenization, image decoding and response encoding are not model offload. Don't reuse Diffusers memory or quality claims.
2. **Keep the C API behind a private worker boundary.** Integrate the private worker without exposing upstream model loaders or simply forwarding HTTP. Supervise it under existing Modelship auth and bounded image API, with one in-flight request, readiness after warm-up, and ownership until the GPU drains. A disconnect is not proof of cancellation. Poison on uncertain completion; no automatic replay or downsizing.
3. **Prove the first narrow GPU profile.** With separate authorization and dedicated capacity, measure full placement and peak memory for one 512×512 generation at a time, then validate one bounded edit independently before advertising it. No claim of 1024² fit, output equivalence to Diffusers, or LoRA support. Test error, restart, disconnect and busy paths before making a candidate public.

FLUX.1-schnell is a separate generation-only option if permissive licensing is the priority. It would require its own verified diffusion/encoder/VAE bundle and tests; it is not an edit fallback or a substitute for Qwen.
