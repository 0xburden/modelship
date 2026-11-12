# Model recipes and requested-model support

These recipes ship in `v0.3.0-alpha.1`. They are **not** included in `v0.2.0-alpha.1`; that earlier release remains byte-for-byte unchanged. The recipe registry is embedded in `internal/modelship/assets/recipes.json`.

## Status semantics

- **supported:** the adapter is implemented. This does not mean every matching repository has passed GPU inference; check recipe notes and `VALIDATION.md`.
- **experimental:** an implemented, isolated path that requires `--experimental`. GPU validation may be incomplete; opt-in does not remove memory/hardware checks.
- **planned:** metadata/layout is recognized, but deployment is explicitly blocked. `--experimental` and `--allow-risk` cannot enable an unimplemented adapter.
- **unsupported:** no explicit compatible recipe is known.

Proven GPU support requires architecture/layout recognition, compatible dependencies, correct input/output handling, resource planning, and a real smoke test. An experimental adapter is implemented code, not a claim of GPU validation. See the explicit validation status below.

## Host-independent discovery

These commands read metadata/configuration only. They require neither an NVIDIA GPU nor Docker and do not download weights, execute repository code, or install dependencies:

```sh
modelship catalog
modelship catalog --json
modelship inspect https://huggingface.co/google/gemma-4-E4B-it
modelship inspect Qwen/Qwen-Image-2.1 --json
```

Use `--revision COMMIT_OR_BRANCH` with `inspect` to select a revision; output resolves it to a commit SHA. Public metadata can still be reported when a gated configuration cannot be read; the warning remains explicit and normal deployment inspection still fails without access. `HF_TOKEN` is used only for Hugging Face requests and is not printed.

Artifact size totals count the weight files listed by the repository, including variants/support files. They are **not** VRAM estimates or necessarily the download set a future adapter will select. Parameter metadata may cover only one component. Image/video/text components must be counted separately.

## The requested repositories

Metadata/configuration was read at the revisions below. Sizes use decimal GB; these are listed artifacts, not measured inference memory. All five repository metadata responses were public and ungated at inspection time.

| Repository | Listed weight artifacts | Recipe / status | A6000 48 GiB considerations |
| --- | ---: | --- | --- |
| `google/gemma-4-E4B-it` | 15.99 GB | `gemma4-text` / experimental | GPU-tested text path. Full checkpoint has ~8B parameters despite the E4B effective-size name; text-only API initially. |
| `Qwen/Qwen3.8-27B` | 55.56 GB | `qwen3.8-multimodal` / experimental | BF16 weights alone exceed 48 GiB. The full modern multimodal loader is implemented for text input on larger hardware. No quantization/offload; A6000 is hard-blocked. Full text-only native inference/API/restart tested on A100 80 GB. |
| `Qwen/Qwen-Image-2.1` | 33.12 GB | `qwen-image21` / experimental | Includes 17.53 GB text encoder, 14.23 GB image transformer, 1.35 GB VAE. Native generation (512/1024) and image-conditioned editing (512) GPU-tested; non-commercial `--research` acknowledgment required. |
| `OpenMOSS-Team/MOSS-Image-SFT` | 27.20 GB | `moss-image` / experimental | Implemented with explicit `--trust-model-code`, one reviewed revision/three source hashes, BF16/FP32 components and CPU noise; native 512/1024 generation GPU-tested after approval. |
| `Wan-AI/Wan2.2-T2V-A14B` | 126.18 GB | `wan22-t2v-native` / planned | Original PTH layout stays blocked. Explicitly select the official Diffusers conversion below; full resident weights still exceed A6000. |

The official **`Wan-AI/Wan2.2-T2V-A14B-Diffusers`** conversion has an experimental safetensors-only video adapter (`wan22-t2v-diffusers`). Its source files total **126.18 GB** too, but the FP32 expert tensors load as BF16; disk bytes are not resident VRAM bytes. Both experts, the text encoder and FP32 VAE are counted. Native dual-expert A100 80 GB inference and H.264 export have passed bounded 256²/5-frame and 512²/17-frame tests. The tested 80 GB configuration needs explicit `--allow-risk` for the conservative activation heuristic, not for weights fit. The original and converted repositories are never silently interchanged.

Pinned inspection sources:

- [Qwen3.8-27B @ 3756181](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0)
- [Qwen Image 2.1 @ d26bb61](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3)
- [Gemma 4 E4B @ 4ecc2ba](https://huggingface.co/google/gemma-4-E4B-it/tree/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c)
- [MOSS Image SFT @ f972bd4](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/tree/f972bd4b4b9e3852133cc5da954d87bc32cd8fb9)
- [WAN 2.2 T2V A14B @ c8c270b](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B/tree/c8c270b13ee05bfa474194ac9fb07a5868a97cea)
- [Official WAN Diffusers conversion @ 5be7df9](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers/tree/5be7df9619b54f4e2667b2755bc6a756675b5cd7)

Minimal public metadata/config/weight-size fixtures are checked into `internal/modelship/testdata/models/`; they contain no actual weights or authentication data. Full inspection captures remain in ignored `validation-artifacts/model-intake/`.

### License caveat: Qwen Image 2.1

The [Qwen Research License at the inspected revision](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE) explicitly grants use **for non-commercial research/evaluation only**, unless a separate commercial license is obtained. Do not treat its `license: other` metadata as an unrestricted open-source license. The other four metadata responses declare Apache-2.0; users still need to review their exact repository terms.

## Gemma 4 text implementation

This is a native-only, explicitly experimental recipe:

```sh
modelship plan google/gemma-4-E4B-it \
  --backend native --python python3 --experimental \
  --revision 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c \
  --name gemma4-test --port 8003

modelship deploy google/gemma-4-E4B-it \
  --backend native --python python3 --experimental \
  --revision 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c \
  --name gemma4-test --port 8003
```

Implemented differences from the established text recipe:

- `Gemma4ForConditionalGeneration` / `AutoModelForMultimodalLM`, BF16, compute capability >=8.0.
- Text generation only despite the upstream `any-to-any` task. The HTTP schema still rejects image/audio/video payloads and tool calls.
- Nested `text_config` supplies context/KV-cache dimensions for planning.
- Recipe-driven chat templates are called with `enable_thinking=False` for the initial bounded-output contract. Because this is an instruction checkpoint, Gemma's `/v1/completions` also frames its prompt as a single user turn; it is not a raw base-model continuation API. Live testing found repetitive output for untemplated prefixes, so HTTP 200/non-empty output alone was not treated as a quality pass. Established Qwen/raw causal-LM completion behavior is unchanged.
- Python >=3.11 is checked before pip. Pinned modern requirements: Transformers 5.22.2, Hugging Face Hub 2.4.0, safetensors 0.8.0, accelerate 1.15.0, plus pinned web dependencies.
- Existing CUDA-enabled host PyTorch is inherited, constrained, and checked before/after installation; old Transformers 4.56 deployments keep their existing private environments.
- Model recipe ID, dtype, loader, and dependency pins are recorded with the deployment. New native state now uses schema 4 to preserve research/code consent, component dtypes and cache location; older CLIs safely refuse it. The feature CLI reads schema-1 and schema-2 deployments too. The historical Gemma validation below used schema 2.

**Validation:** actual checkpoint loading and inference now pass on the user's fresh Thunder Base/A6000: Ubuntu 22.04.5, Python 3.12.13, driver 610.43.02, host PyTorch 2.12.1+cu130. The pinned Gemma revision passes completions/chat, a semantic Paris check, busy handling, context/parameter errors, media-input rejection, stop/start, and cache reuse. A simultaneous Qwen2.5-3B regression passes in the separate legacy Transformers environment. Runtime environment inspection confirms Gemma BF16/multimodal loader/templating versus Qwen float16/causal loader/raw completions. The published alpha CLI correctly rejects schema-2 state. Keep the recipe experimental: only this checkpoint, text-only input, 4096 context, and this hardware/runtime combination have been tested; this is not validation of all Gemma 4 variants or full multimodal support.

Individual post-fix request timings were 4.680 s for the completion and 5.431 s for chat, not a controlled benchmark. Both workers together used 21370 MiB of device memory after requests (a snapshot, not peak allocated memory). Local reports are ignored under `validation-artifacts/gemma4/templated/` and `validation-artifacts/qwen-regression/`.

On that development server, the feature binary is installed separately as `~/.local/bin/modelship-dev`. Gemma is retained as `gemma4-e4b-test` on localhost port 8003; Qwen as `qwen-regression` on 8000. Use the development CLI for their schema-2 lifecycle, not the published alpha CLI. Neither service has public HTTP forwarding configured.

## Weights-alone VRAM and disk preflight

`plan` / `deploy` inspect **only bounded safetensors headers** using HTTP Range at the immutable commit, plus shard manifests. Missing shards/sizes, malformed shapes/offsets/dtypes, a server ignoring Range, or failed metadata access block deployment. No complete tensor file is downloaded by planning. Hub parameter counts are not used as proof: they can omit the image text encoder or a video expert.

The plan separates:

- `weight_download_bytes`: selected source files on disk.
- `weight_resident_bytes`: a **minimum** from tensor shapes at the implemented load dtype, with integer/bool buffers preserved and explicit FP32 VAE overrides.
- `weights_fit_total` / `weights_fit_free`: that minimum versus total and currently free memory of the selected GPU.
- `estimated_vram_mib`: a separate, heuristic runtime/cache/activation budget.

Real A6000 header checks measured minimums of **14.894 GiB** (Gemma), **51.747 GiB** (Qwen3.8), **30.841 GiB** (Qwen Image), **25.332 GiB** (MOSS), and **64.283 GiB** (official WAN conversion). These are not measured peak inference memory.

**Weights exceeding either total or free VRAM are hard blockers. `--allow-risk` overrides only the runtime heuristic, never weights, metadata safety, licensing, code consent, compatibility or native disk capacity.** There is no automatic quantization, offload or model substitution. Fit flags are not an OOM guarantee; retained-FP32 modules, buffers and activations need more memory. VRAM is refreshed before installation/launch and on manual starts with saved budgets; no GPU-memory reservation is made. Older saved deployments lack this budget. Docker's own automatic restart is outside CLI preflight.

Native planning inspects the actual HF cache filesystem and correctly-sized selected files at the exact commit. Missing downloads plus 5% and 4 GiB headroom must fit available disk. `HF_HUB_CACHE` selects a different filesystem; the absolute cache path is saved for restarts. RAM remains advisory; direct CUDA placement in the modern adapters avoids constructing a full anonymous CPU copy of large converted weights. Dependency/state files may be on another filesystem and also need space.

## Experimental native commands

All new paths require Python >=3.11 and compute capability >=8.0. Qwen3.8, Qwen Image, MOSS and WAN require existing PyTorch >=2.9; it is checked, never replaced. Modern text/image/video dependencies are isolated from established deployments. Workers explicitly set `USE_HUB_KERNELS=NO`, preventing modern Transformers from implicitly fetching Hub kernels. Installed host packages are still part of the trusted runtime.

```sh
# Full BF16 text weights: a larger GPU is required. Text input only.
modelship plan Qwen/Qwen3.8-27B --backend native --experimental \
  --revision 37561816e66b5b9153b1b64c10968c18c1e6f4a0

# Research-only Qwen Image. Use deploy instead of plan once all blockers resolve.
modelship plan Qwen/Qwen-Image-2.1 --backend native --experimental --research \
  --revision d26bb61231c349cf6b7896fa83353113880e1ba3 \
  --task text-to-image --name qwen-images --port 8004

# This flag permits repository Python as your user: review the source first.
modelship plan OpenMOSS-Team/MOSS-Image-SFT --backend native --experimental \
  --trust-model-code --revision f972bd4b4b9e3852133cc5da954d87bc32cd8fb9 \
  --name moss-images --port 8005

# Explicit official conversion, not original PTH deserialization or substitution.
# --revision applies to the SELECTED weights repository.
modelship plan Wan-AI/Wan2.2-T2V-A14B --backend native --experimental \
  --weights-repo Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7 \
  --task text-to-video --name wan-video --port 8006
```

### Qwen Image 2.1

`QwenImage21Pipeline` serves generation and image-conditioned editing; do not use the SDXL AutoPipeline/strength contract. A separate deployment chooses `--task image-to-image` and `/v1/images/edits` with inline `image_b64` (RGBA preserved). Both tasks use the same pinned model weights/cache.

Dimensions are 256–1024 and divisible by **32**. `guidance_scale` defaults to **1.0**, mapped to `true_cfg_scale`; negative prompts require a scale >1. `strength` is rejected, rather than ignored. Input conditioning resolution is explicitly bounded to the request area. The dedicated adapter bridges BF16 conditioning pixels into the FP32 VAE, then casts encoded latents back to BF16; this avoids the pinned upstream mixed-precision editing mismatch. Modern image prompts have a 4096-character limit. VAE is FP32, transformer/text encoder BF16; generation defaults to 28 steps (API cap 100), one request at a time. These bounds are not proof of peak-memory fit.

### MOSS source-execution boundary

Research acknowledgment is **not** code approval. Only `OpenMOSS-Team/MOSS-Image-SFT` at the reviewed revision above is executable, and only with separate `--trust-model-code` and `--experimental` flags. These three identical source files are verified before weights download and again before local import:

- `pipeline_moss_image.py`
- `scheduler/pipeline_moss_image.py`
- `transformer/pipeline_moss_image.py`

Each SHA-256: `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`.

Source review found standard Torch/Diffusers/Transformers inference classes, no shell/exec or direct network helpers; the unused component-loading convenience method can call standard HF loaders. The runtime verifies the component index, downloads only selected safetensors/config/tokenizer files and these approved Python paths, then loads locally with no additional fetching. This is **not a security audit or sandbox**. Do not approve code on an untrusted/shared account. Different code revisions require a new review/manifest, not merely `--allow-risk`.

Text-to-image only, multiples of **16**, guidance default **3.0**, BF16 transformer/text encoder and FP32 VAE, CPU FP32 initial noise. Text encoding is capped at **512 tokens** (long prompts are truncated by the upstream encoder); use compact prompts. The pinned checkpoint passed native 512/1024 generation and busy/validation checks on A6000 after explicit code approval.

### WAN video API

Only the official converted repository is accepted by the runtime. Original PTH files, arbitrary community conversions, image-to-video and repository Python are not loaded. Both expert networks stay on the selected GPU; no offload/quantization.

```sh
curl -sS http://127.0.0.1:8006/v1/videos/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A paper boat drifting on a quiet pond","width":512,"height":512,"num_frames":17,"steps":30,"fps":16,"seed":42}'
# HTTP 202 -> id. Poll status, then download content after status=succeeded:
curl -sS http://127.0.0.1:8006/v1/videos/VIDEO_ID
curl -f http://127.0.0.1:8006/v1/videos/VIDEO_ID/content -o clip.mp4
```

A reusable opt-in API/codec client is also available from the repository. Run it using the deployment's Python so its pinned codec dependencies are available (it generates a small smoke plus a bounded 512²/17-frame clip):

```sh
~/.local/state/modelship/native/wan-video/venv/bin/python tests/live_video_check.py \
  --port 8006 --output validation-artifacts/wan-live
```

Statuses: `running`, `succeeded`, `failed`. One active job, **no backlog**; busy returns 503. Bounds: 256–768 dimensions divisible by 16, frames `4k+1` <=49, <=`512*512*49` pixel-frames, 1–50 steps, 1–30 fps, 32 KiB request body. Defaults are 512²/17 frames/30 steps/16 fps. No user URLs or arbitrary fields are accepted.

Exports are private MP4s, <=64 MiB each; retention <=8 jobs/256 MiB and one-hour expiry. At most two concurrent downloads. IDs/exports are ephemeral and removed on restart/shutdown. Graceful shutdown drains the worker; the CLI's forced stop can terminate a long job, with partial files cleaned at next startup. Failed-job responses are sanitized; they do not echo prompts, tokens or tracebacks. CPU fakes test the API and export boundary; **real A100 80 GB inference and H.264 encoding passed the bounded tests above**. Sampled quality frames were coherent with visible ripples/motion, but the requested red boat appeared white; this is not a general quality/prompt-adherence guarantee. Longer clips/maximal API bounds remain unvalidated.

## Remaining validation and optimizations

- **Completed with explicit user approval:** Qwen Image 512/1024 generation, 512 image-conditioned editing, busy/validation and cached restart; MOSS 512/1024 generation and busy/validation. PNGs were visually inspected. Both retained demos were restored and smoke-tested afterward. MOSS code approval remains required for each new deployment.
- **Qwen3.8 completed on a separate A100 80 GB instance:** full BF16 loading, semantic/API checks and cached restart. It still cannot fit A6000. WAN generation/MP4 checks also passed on that GPU with a separate cache and explicit runtime-heuristic override; both checkpoints cannot coexist on the 150 GB test disk. Do not attempt full downloads on the previous 100 GB instance.
- All five requested families now have pinned native inference checks: Gemma/Qwen text, Qwen Image generation/editing, MOSS generation and explicit official WAN conversion/video export. This is bounded model/hardware-specific validation, not general compatibility.
- Controlled peak-memory/performance measurements, longer/maximal video clips, broader prompts/contexts/hardware, failure/cancellation and crash/reboot behavior remain outstanding.
- Explicitly versioned quantization/offload and LoRA composition remain later work; no automatic smaller-model fallback.

## LoRA tooling: subsequent phase, not implemented yet

Start with startup-time adapters rather than unauthenticated hot-swapping APIs:

- Inspect adapter metadata, identify required base architecture, and report base-model dependencies instead of treating an adapter as standalone weights.
- Pin both base and adapter repository revisions; select exact safetensors filenames and an explicit scale.
- Validate target modules, rank, tensor shapes, and family compatibility. A text PEFT adapter is not an image/video LoRA.
- Text: versioned PEFT profiles. Images: model-specific Diffusers `load_lora_weights` / adapter configuration. Video: explicit high/low-noise expert targeting where applicable.
- Account for adapter memory and combined licenses; preserve the unmodified cached base.
- Add CPU shape/metadata tests and real one-family-at-a-time GPU tests. No merging, dynamic public adapter loading, training, or arbitrary remote code by default.

Publish a new version only after the intended scope is tested. The tracked `releases/current` now selects `v0.3.0-alpha.3`; the legacy Dokploy `RELEASE_VERSION` build argument is ignored with a warning. Release-site builds retain checksum-pinned prior assets and fail closed if the current public version already has different checksums. Bump the tracked version and extend `releases/retained.sha256` before the next release.
