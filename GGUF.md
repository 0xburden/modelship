# Experimental GGUF chat and vision (unreleased branch)

**Published v0.3.0-alpha.6 remains text-only with the original single Q4 artifact.** The quantization menus, optional vision and schema 6 below are unreleased feature-branch work; no release/version change is implied. Alpha.6 repaired Ubuntu22.04 runtime compatibility. A user reports working GGUF text inference after that repair; we have not independently measured that deployment. **Actual vision inference, quality and peak memory remain unvalidated.** CPU fixtures, source/ABI inspection and bounded real artifact headers are not GPU validation.

## Explicit target

- Repository: `unsloth/Qwen3.8-27B-GGUF`
- Immutable revision: `4af51b3542f775b0a8e4b310bfb3f17e01c306c9`
- File: **`Qwen3.8-27B-UD-Q4_K_M.gguf`**
- File size: **16,564,259,744 bytes** (15.43 GiB); SHA-256 `1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529`.
- Observed GGUF v3 header: `qwen35`, 866 tensors, **16,553,263,104 encoded tensor bytes**, 10,996,640-byte aligned header, 262144 advertised context, 65 blocks. The tensors have mixed F32/Q8/K/IQ encodings; the filename is not a global compute dtype.

The source-controlled `internal/modelship/assets/gguf_artifacts.json` pins **25 standalone quantizations**, their exact sizes/SHA-256 digests, and one matching projector at this same revision. The wizard lists available reviewed files with GiB download sizes and an explicit **UD-Q4_K_M default**. Numbers, letters or full values select a choice; there is no automatic substitution based on fit. Flags still require `--gguf-file` explicitly. BF16 split groups, MTP, imatrix/calibration, BF16 projector, unknown repositories/revisions and arbitrary files stay unsupported. All 25 selected language headers were parsed using bounded Range reads without tensor downloads; that establishes format/accounting only, not inference quality or compatibility.

### Optional image understanding

Opt in with **both** `--vision --mmproj-file mmproj-F16.gguf`, or choose **vision-images** in the wizard (text-only is the default). The explicit mode pairs:

- `mmproj-F16.gguf`: **927,607,488 bytes**, SHA-256 `cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e`.
- Bounded observed header: `clip`, type `mmproj`, projector `qwen3vl_merger`, **334 tensors / 927,587,264 encoded tensor bytes**, aligned header **20,224 bytes**, projection dimension 5120, patch size 16 / spatial merge 2. Storage is mixed F16/F32, not globally quantized by the language-file selection.
- This first profile requires the Ubuntu22.04 runtime identity below, one PNG/JPEG image **across the conversation request**, accompanying user text, at most **4 MiB encoded-file bytes**, **4,194,304 pixels (2048² total, ~4.19 MP)**, and **4096 pixels per dimension**. It accepts uploaded **data URLs only**: remote/file URLs, WebP/GIF, tools, audio and video are rejected without fetching them. Image generation/editing is not implied.
- Model preprocessing may resize input; the runtime sets at most **1024 image tokens** and encoder batches of 256 tokens. Image tokens consume the same fixed context as prompt/output. 4096 context tokens is a starting recommendation, not a fit guarantee; small contexts or long histories can be rejected by the backend. There is no silent context shifting.
- Both files have independent private cache paths, verified size/full SHA/header before start, and combined disk/physical placement budgets. The heuristic adds **2 GiB** for vision activations on top of the language estimate; this is **not a measured peak**. `--allow-risk` cannot override either file's hard placement/disk checks. CUDA0 is explicitly selected for the projector; a device-init failure cannot silently choose a CPU encoder. Essential backend CPU operations can still exist.

Metadata-only paired inspection:

```sh
modelship inspect unsloth/Qwen3.8-27B-GGUF \
  --revision 4af51b3542f775b0a8e4b310bfb3f17e01c306c9 \
  --gguf-file Qwen3.8-27B-UD-Q4_K_M.gguf --mmproj-file mmproj-F16.gguf --json
```

Add `--vision --mmproj-file mmproj-F16.gguf` to the plan/deploy examples below to opt in. Use a **new deployment name** to preserve an existing text deployment; no automatic in-place conversion or restart occurs. Downloads/loading remain behind final deployment approval. No model/projector tensors or instance changes were used to implement these features.

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

Text-only example (also works on published v0.3.0-alpha.6); vision flags require the unreleased feature build:

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

The feature-branch wizard lists quantizations with a visible default, asks text-only versus vision, skips Python probes and reviews the same plan/reproducible flags. **Esc** goes back to the previous editable answer; `:back` is an alternative. Later selections and approvals are discarded. Read-only metadata may be reused; live GPU/cache/fit checks refresh. Ctrl-C/EOF cancel. Terminal settings are restored on back/cancel/errors; no keys, installs or downloads occur before final approval.

The encoded tensor-byte **placement budget** must fit both total and currently free selected-GPU VRAM. This is conservative for fixed CPU tensors, not a measured resident minimum or peak. The runtime estimate separately budgets F16 KV for all advertised blocks (conservative for this hybrid architecture) plus 8 GiB for recurrent state, temporary allocations and activations. **`--allow-risk` only acknowledges this heuristic; it cannot override the weight-placement budget, disk, metadata, runtime or access policy.** VRAM is refreshed before install/launch/manual starts; no reservation is made. 4096 is a preset; this initial adapter accepts 128–32768, also bounded by the header. Advertised 262144 is not a tested GPU capacity. Host RAM and actual activation peaks remain unvalidated.

The selected missing file(s) plus 5% and 12 GiB must fit the cache filesystem during planning. Installation refreshes disk checks, requires 12 GiB in deployment-state storage for runtime/setup, and accounts for shared versus separate filesystems. Runtime archive expansion is separately bounded. Cache location follows HF cache precedence, but the verified GGUF is stored under `CACHE/modelship-gguf/unsloth--Qwen3.8-27B-GGUF/REVISION/FILENAME`, not a Python snapshot. Runtime identity is saved: the new core is `llama-b12110-cuda12.8-ubuntu22.04-r1`; existing healthy legacy `llama-b12110-cuda12.8` deployments keep their original pins and GLIBC2.38/GLIBCXX3.4.32 policy, never silently swapping binaries. ABI requirements are checked by reading native ELF symbol definitions before runtime downloads, without importing PyTorch or installing packages. Full model SHA-256 and the saved header metadata are checked before each launch; this adds sequential disk reads on cold starts.

Downloads retain private `.partial` files and use validated Range resumes; corrupt final files are never silently replaced. Existing deployment names are not overwritten. If a confirmed install fails, uninstall its stopped deployment before retrying; the shared model cache/partial survives, while the per-deployment runtime cache is removed. No automatic disk cleanup or model substitution occurs.

## Retrying an alpha.5 preflight failure

Alpha.5’s upstream CUDA core requires GLIBC2.38/GLIBCXX3.4.32 and cannot start on the reported glibc2.35/GLIBCXX3.4.30 host. No GPU/model work ran in that failure. Do **not** upgrade glibc or add broad host library overrides.

Install alpha.6 with `sh install.sh version` first (install-only), then inspect `modelship status --backend native --name modelship`. If this is still the **stopped, failed** deployment, explicitly remove only that deployment with `modelship uninstall --backend native --name modelship --yes`, then run `modelship launch` again. The shared model cache and owner API keys are retained; its per-deployment runtime cache is removed. Alternatively, choose a new deployment name and retain the failed installation. No existing name is overwritten or migrated automatically.

Runtime enumeration failures now report bounded, terminal-safe native output and retain a private **`runtime-preflight.log`** with exit details. ABI, missing-library, missing-device and timeout failures are distinguished. Installation metadata is saved before the probe, while model downloads still wait for a real enumerated CUDA0. On a CPU-only CI runner, only `libcuda.so.1` may be unresolved for the static ABI audit; **the CLI never accepts a stub/absent driver or CPU fallback**.

## API and lifecycle

The public listener is Modelship's **Go gateway**, not the general-purpose llama.cpp server. Native access policy is unchanged: loopback by default; `--listen 0.0.0.0` requires a valid owned/private `--api-key-file`. All public routes, including health, discovery and errors, authenticate before dispatch. No-store responses, header-only bearer auth, no CORS/cookie/query credentials. Use trusted HTTPS or a private tunnel outside the host; raw worker HTTP is unencrypted.

The child listens on a separate loopback port with a random, private-file-backed **different** key. The owner's key is not forwarded to it; neither key's value is put in argv or the environment. The gateway does not expose arbitrary proxy paths, model loading, slots, properties, filesystem operations, agent tools, web UI, arbitrary projector/video/image-generation endpoints or backend error bodies. This is not a hostile-multiuser sandbox; processes run as your user and the host/OS runtime remains trusted.

Implemented routes: `GET /health`, `GET /v1/models`, **`GET /v1/capabilities`**, `POST /v1/chat/completions`, `POST /v1/completions`. Capabilities are authenticated, versioned (`schema_version: 1`) and identify the model/revision, input/output modalities and chat image limits. They describe configured features, not measured memory safety. Legacy text-only endpoints can return 404; consumers must not infer vision from model names.

Text-only deployments reject image parts; vision chat accepts string messages or typed user `text`/`image_url` parts with one bounded `data:image/png;base64,...` or JPEG. Images are size/header/pixel checked **before full decoding**, then decoded for validity before a fixed normalized request reaches the private child. Completion remains text-only. Unknown fields, tools, streaming and unloaded model IDs are rejected. One active request, no inference queue/retry. Defaults: 128 output tokens, temperature 0, top-p 1 (top-k/min-p disabled); bounds: 1–1024 output tokens, temperature 0–2, top-p >0–1, seed 0–4294967294. Request bodies are <=1 MiB text-only / <=8 MiB vision; backend JSON <=2 MiB. This is a limited OpenAI-compatible subset, not the complete protocol.

The separate Console change discovers capabilities before enabling Chat uploads, retains the image for follow-up questions in tab memory, and requires clearing the conversation before a second image. It does not turn the Console into a URL/image fetch proxy. Image generation/editing/video remain separate tasks.

Both completion and chat use the embedded instruct chat template with thinking disabled. Completion is a templated user turn, **not** raw continuation. Request context validation/errors remain delegated to the pinned backend. Client disconnect does not release the busy slot or cancel GPU work early. An uncertain backend/transport failure marks the service unavailable until restart, rather than admitting potentially overlapping work.

`status`, `logs`, `smoke`, `stop`, `start`, `configure` and `uninstall` retain `--backend native --name qwen-gguf`. Access changes require stop/configure/start and do not install Python or rewrite runtime files. New native state is **schema 6**, rejected by older CLIs so vision execution constraints cannot be ignored; Python schemas 1–5 and existing text-only GGUF schema 5 remain readable. Keep using the updated CLI after migration. Stop signals the managed process group, including the child; workers survive SSH disconnect, not crash/reboot automatically. Uninstall retains shared model caches and owner API keys.

Before calling this GPU-supported, validate the actual pinned archives/model: CUDA load and placement, meaningful completion/chat quality, context/media/auth/busy errors, memory snapshots/peaks, console compatibility, stop/start, cached restart and no host package changes. Vision/projector code is experimental branch work, never inferred from the architecture name; real CUDA image understanding/quality/load and restart validation still require explicit authorization.
