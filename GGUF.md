# Experimental GGUF text adapter (v0.3.0-alpha.6)

**GGUF was introduced in alpha.5; use alpha.6 for the Ubuntu 22.04 compatibility repair.** This is a first, deliberately narrow adapter, not general GGUF or prequantized Transformers support. Live loading/inference with the actual model and CUDA release binaries is still unvalidated. Do not treat source support or CPU fixtures as a GPU quality/fit guarantee.

## Explicit target

- Repository: `unsloth/Qwen3.8-27B-GGUF`
- Immutable revision: `4af51b3542f775b0a8e4b310bfb3f17e01c306c9`
- File: **`Qwen3.8-27B-UD-Q4_K_M.gguf`**
- File size: **16,564,259,744 bytes** (15.43 GiB); SHA-256 `1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529`.
- Observed GGUF v3 header: `qwen35`, 866 tensors, **16,553,263,104 encoded tensor bytes**, 10,996,640-byte aligned header, 262144 advertised context, 65 blocks. The tensors have mixed F32/Q8/K/IQ encodings; the filename is not a global compute dtype.

No automatic quant choice, conversion, repository Python, alternate revision, CPU fallback, split-file group, projector, MTP auxiliary file or calibration file is selected. Other Q4/Q5/Q8 files and other GGUF repositories require separately reviewed profiles. **Vision and tools are not enabled.** A malformed/unsupported header fails closed.

## Hardware and runtime

Linux amd64, NVIDIA compute capability >=8.0, driver >=570.26, working `nvidia-smi`, sufficient free VRAM/disk and host RAM, system **glibc >=2.35 / GLIBCXX >=3.4.30**, and SSE4.2 CPU instructions. This path does **not** require, install or replace Python, PyTorch, system CUDA packages or drivers. The existing `doctor --backend native` Python check pertains to Python-based recipes, not this adapter.

The standalone runtime is llama.cpp **b12110**, source reviewed at `cd959bae99ff4bed22bef2604f45241c09bbb8c8`:

| Runtime release asset | Compressed bytes | SHA-256 |
| --- | ---: | --- |
| `llama-b12110-ubuntu22.04-cuda12.8-r1.tar.gz` | 50987396 | `54351967b6120961a76e5a04704797dfdef60ba512f1b210818cf905977dff64` |
| `cudart-llama-b12110-bin-ubuntu-cuda-12.8-x64.tar.gz` | 594371477 | `6f1d562086d84ae85df880d3711079ec225602b51d530608e5001600625067dd` |

The rebuilt core downloads from Modelship’s public `runtime-llama-b12110-u2204-r1` GitHub release; the unchanged CUDA library archive downloads from upstream b12110. No token or `latest` lookup is used. The audited build recipe is `runtime/llama/Dockerfile`: digest-pinned Ubuntu22.04/CUDA12.8.1 builder, checksum-pinned CMake/CCCL/source, offline compilation, SM80 cubin + compute80 PTX baseline, SSE4.2 baseline CPU and no downloaded UI/OpenSSL/subprocess support. Higher GPU SMs can JIT the PTX; optimized per-generation cubins are not selected implicitly. Full size/SHA-256 verification precedes extraction/execution. Extraction rejects path escapes, special/hardlink entries, escaping/unresolved symlinks, duplicates and expansion beyond its budgets. Installed file hashes and link targets are saved and rechecked on startup. Bundled library paths are scoped to the child process, not exported into the shell or host environment. No broad host NVML/library overrides are attempted. Review upstream [llama.cpp licensing](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/LICENSE) and the checkpoint's own terms.

Only the selected GPU UUID is visible to the child: `CUDA0`, all GPU layers, no split, one slot, fixed context, **automatic fit and context shifting disabled**. Essential fixed backend CPU metadata/tensors/operations can still remain on the host; this is not a claim that every byte or operation resides on the GPU. A failed CUDA device probe stops installation before downloading model weights. Actual model warmup/readiness follows loading; device enumeration alone is not inference validation.

## Plan, then deploy

With v0.3.0-alpha.6 or newer:

```sh
# Metadata/header only; no GPU required and no tensor payloads downloaded.
modelship inspect unsloth/Qwen3.8-27B-GGUF \
  --revision 4af51b3542f775b0a8e4b310bfb3f17e01c306c9 \
  --gguf-file Qwen3.8-27B-UD-Q4_K_M.gguf --json

# Hardware/cache checks and plan only.
modelship plan unsloth/Qwen3.8-27B-GGUF --backend native --experimental \
  --revision 4af51b3542f775b0a8e4b310bfb3f17e01c306c9 \
  --gguf-file Qwen3.8-27B-UD-Q4_K_M.gguf \
  --name qwen-gguf --context 4096 --port 8000

# Downloads/install/load happen only after confirmation.
modelship deploy unsloth/Qwen3.8-27B-GGUF --backend native --experimental \
  --revision 4af51b3542f775b0a8e4b310bfb3f17e01c306c9 \
  --gguf-file Qwen3.8-27B-UD-Q4_K_M.gguf \
  --name qwen-gguf --context 4096 --port 8000 --wait 45m
```

Alternatively, `modelship launch` asks for the exact filename, skips Python questions/probes, explains standalone setup and reviews the same plan. It does not choose a quantization implicitly. Empty filename input is not acceptance of a hidden default.

The encoded tensor-byte **placement budget** must fit both total and currently free selected-GPU VRAM. This is conservative for fixed CPU tensors, not a measured resident minimum or peak. The runtime estimate separately budgets F16 KV for all advertised blocks (conservative for this hybrid architecture) plus 8 GiB for recurrent state, temporary allocations and activations. **`--allow-risk` only acknowledges this heuristic; it cannot override the weight-placement budget, disk, metadata, runtime or access policy.** VRAM is refreshed before install/launch/manual starts; no reservation is made. 4096 is a preset; this initial adapter accepts 128–32768, also bounded by the header. Advertised 262144 is not a tested GPU capacity. Host RAM and actual activation peaks remain unvalidated.

The selected missing file plus 5% and 12 GiB must fit the cache filesystem during planning. Installation refreshes disk checks, requires 12 GiB in deployment-state storage for runtime/setup, and accounts for shared versus separate filesystems. Runtime archive expansion is separately bounded. Cache location follows HF cache precedence, but the verified GGUF is stored under `CACHE/modelship-gguf/unsloth--Qwen3.8-27B-GGUF/REVISION/FILENAME`, not a Python snapshot. Runtime identity is saved: the new core is `llama-b12110-cuda12.8-ubuntu22.04-r1`; existing healthy legacy `llama-b12110-cuda12.8` deployments keep their original pins and GLIBC2.38/GLIBCXX3.4.32 policy, never silently swapping binaries. ABI requirements are checked by reading native ELF symbol definitions before runtime downloads, without importing PyTorch or installing packages. Full model SHA-256 and the saved header metadata are checked before each launch; this adds sequential disk reads on cold starts.

Downloads retain private `.partial` files and use validated Range resumes; corrupt final files are never silently replaced. Existing deployment names are not overwritten. If a confirmed install fails, uninstall its stopped deployment before retrying; the shared model cache/partial survives, while the per-deployment runtime cache is removed. No automatic disk cleanup or model substitution occurs.

## Retrying an alpha.5 preflight failure

Alpha.5’s upstream CUDA core requires GLIBC2.38/GLIBCXX3.4.32 and cannot start on the reported glibc2.35/GLIBCXX3.4.30 host. No GPU/model work ran in that failure. Do **not** upgrade glibc or add broad host library overrides.

Install alpha.6 with `sh install.sh version` first (install-only), then inspect `modelship status --backend native --name modelship`. If this is still the **stopped, failed** deployment, explicitly remove only that deployment with `modelship uninstall --backend native --name modelship --yes`, then run `modelship launch` again. The shared model cache and owner API keys are retained; its per-deployment runtime cache is removed. Alternatively, choose a new deployment name and retain the failed installation. No existing name is overwritten or migrated automatically.

Runtime enumeration failures now report bounded, terminal-safe native output and retain a private **`runtime-preflight.log`** with exit details. ABI, missing-library, missing-device and timeout failures are distinguished. Installation metadata is saved before the probe, while model downloads still wait for a real enumerated CUDA0. On a CPU-only CI runner, only `libcuda.so.1` may be unresolved for the static ABI audit; **the CLI never accepts a stub/absent driver or CPU fallback**.

## API and lifecycle

The public listener is Modelship's **Go gateway**, not the general-purpose llama.cpp server. Native access policy is unchanged: loopback by default; `--listen 0.0.0.0` requires a valid owned/private `--api-key-file`. All public routes, including health, discovery and errors, authenticate before dispatch. No-store responses, header-only bearer auth, no CORS/cookie/query credentials. Use trusted HTTPS or a private tunnel outside the host; raw worker HTTP is unencrypted.

The child listens on a separate loopback port with a random, private-file-backed **different** key. The owner's key is not forwarded to it; neither key's value is put in argv or the environment. The gateway does not expose arbitrary proxy paths, model loading, slots, properties, filesystem operations, agent tools, web UI, projector/video/image endpoints or backend error bodies. This is not a hostile-multiuser sandbox; processes run as your user and the host/OS runtime remains trusted.

Implemented routes: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`. Text content only; unknown fields, media, tools, streaming and unloaded model IDs are rejected. One active request, no inference queue/retry. Defaults: 128 output tokens, temperature 0, top-p 1 (top-k/min-p filtering disabled); bounds: 1–1024 output tokens, temperature 0–2, top-p >0–1, seed 0–4294967294 (the backend random-seed sentinel is not accepted as an explicit seed). Requests <=1 MiB, backend JSON <=2 MiB. This is a limited OpenAI-compatible subset, not the complete protocol.

Both completion and chat use the embedded instruct chat template with thinking disabled. Completion is a templated user turn, **not** raw continuation. Request context validation/errors remain delegated to the pinned backend. Client disconnect does not release the busy slot or cancel GPU work early. An uncertain backend/transport failure marks the service unavailable until restart, rather than admitting potentially overlapping work.

`status`, `logs`, `smoke`, `stop`, `start`, `configure` and `uninstall` retain `--backend native --name qwen-gguf`. Access changes require stop/configure/start and do not install Python or rewrite runtime files. New native state is **schema 5**, rejected by older CLIs; this build still reads Python deployments using schemas 1–4. Keep using the updated CLI after migration. Stop signals the managed process group, including the child; workers survive SSH disconnect, not crash/reboot automatically. Uninstall retains shared model caches and owner API keys.

Before calling this GPU-supported, validate the actual pinned archives/model: CUDA load and placement, meaningful completion/chat quality, context/media/auth/busy errors, memory snapshots/peaks, console compatibility, stop/start, cached restart and no host package changes. Vision/projector support is a separate follow-up, not implied by the architecture name.
