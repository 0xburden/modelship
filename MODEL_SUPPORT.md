# Model recipes and requested-model roadmap

This iteration lives on `feat/model-recipes`. It is **not** included in the published `v0.2.0-alpha.1`, and must not overwrite that version's files. The recipe registry is embedded in `internal/modelship/assets/recipes.json`.

## Status semantics

- **supported:** the adapter is implemented. This does not mean every matching repository has passed GPU inference; check recipe notes and `VALIDATION.md`.
- **experimental:** an implemented, isolated path that requires `--experimental`. GPU validation may be incomplete; opt-in does not remove memory/hardware checks.
- **planned:** metadata/layout is recognized, but deployment is explicitly blocked. `--experimental` and `--allow-risk` cannot enable an unimplemented adapter.
- **unsupported:** no explicit compatible recipe is known.

Model support requires architecture/layout recognition, compatible dependency versions, correct input/output handling, resource planning, and a real smoke test—not just an allowlist entry.

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
| `google/gemma-4-E4B-it` | 15.99 GB | `gemma4-text` / experimental | First implementation target. Full checkpoint has ~8B parameters despite the E4B effective-size name; text-only API initially. |
| `Qwen/Qwen3.8-27B` | 55.56 GB | `qwen3.8-multimodal` / planned | BF16 weights alone exceed 48 GiB. Needs explicit quantization/offload or larger hardware; hybrid attention + multimodal loader, not the legacy causal-LM recipe. |
| `Qwen/Qwen-Image-2.1` | 33.12 GB | `qwen-image21` / planned | Includes 17.53 GB text encoder, 14.23 GB image transformer, 1.35 GB VAE. Candidate for bounded-resolution tests, not a fit guarantee. Different generation/edit contract. |
| `OpenMOSS-Team/MOSS-Image-SFT` | 27.20 GB | `moss-image` / planned | Requires repository Python code, mixed BF16/FP32 components, and a CPU noise generator. No automatic remote-code execution. |
| `Wan-AI/Wan2.2-T2V-A14B` | 126.18 GB | `wan22-t2v-native` / planned | High/low-noise expert artifacts plus PTH support weights exceed the current 100 GB server disk. Upstream single-GPU example specifies >=80 GB VRAM; optimized 48 GiB use needs separate validation. |

Pinned inspection sources:

- [Qwen3.8-27B @ 3756181](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0)
- [Qwen Image 2.1 @ d26bb61](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3)
- [Gemma 4 E4B @ 4ecc2ba](https://huggingface.co/google/gemma-4-E4B-it/tree/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c)
- [MOSS Image SFT @ f972bd4](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/tree/f972bd4b4b9e3852133cc5da954d87bc32cd8fb9)
- [WAN 2.2 T2V A14B @ c8c270b](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B/tree/c8c270b13ee05bfa474194ac9fb07a5868a97cea)

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
- Model recipe ID, dtype, loader, and dependency pins are recorded with the deployment. New native state uses schema 2 so an old alpha CLI refuses it rather than launching with the wrong legacy loader/dtype. The feature CLI still accepts legacy schema-1 saved plans with fallback defaults.

**Validation:** actual checkpoint loading and inference now pass on the user's fresh Thunder Base/A6000: Ubuntu 22.04.5, Python 3.12.13, driver 610.43.02, host PyTorch 2.12.1+cu130. The pinned Gemma revision passes completions/chat, a semantic Paris check, busy handling, context/parameter errors, media-input rejection, stop/start, and cache reuse. A simultaneous Qwen2.5-3B regression passes in the separate legacy Transformers environment. Runtime environment inspection confirms Gemma BF16/multimodal loader/templating versus Qwen float16/causal loader/raw completions. The published alpha CLI correctly rejects schema-2 state. Keep the recipe experimental: only this checkpoint, text-only input, 4096 context, and this hardware/runtime combination have been tested; this is not validation of all Gemma 4 variants or full multimodal support.

Individual post-fix request timings were 4.680 s for the completion and 5.431 s for chat, not a controlled benchmark. Both workers together used 21370 MiB of device memory after requests (a snapshot, not peak allocated memory). Local reports are ignored under `validation-artifacts/gemma4/templated/` and `validation-artifacts/qwen-regression/`.

On that development server, the feature binary is installed separately as `~/.local/bin/modelship-dev`. Gemma is retained as `gemma4-e4b-test` on localhost port 8003; Qwen as `qwen-regression` on 8000. Use the development CLI for their schema-2 lifecycle, not the published alpha CLI. Neither service has public HTTP forwarding configured.

## Next implementation slices

1. **Gemma GPU validation:** initial text-only A6000 deployment/API/lifecycle and simultaneous Qwen regression are complete. Next: controlled performance/peak-memory measurements, broader prompts/hardware, cancellation, and longer contexts. Never stop the user's running service to make room without approval.
2. **Qwen Image 2.1:** pin a released Diffusers build containing `QwenImage21Pipeline`; map generation and editing through the same loader; do not assume SDXL img2img `strength` semantics. Bound resolution and validate memory/offload behavior. Resolve intended-use licensing first.
3. **Qwen3.8 quantization/offload:** isolate modern runtime/bitsandbytes or another supported format. Record actual quantization, base revision, and memory estimates. Do not silently switch to a community quantized repository.
4. **MOSS custom-code adapter:** explicit consent boundary, exact code revision/hashes/source review, component dtype rules, CPU generator, bounded output, and real smoke tests. Do not simply set `trust_remote_code=True` globally.
5. **WAN video adapter:** async bounded job API, frame/duration/resolution limits, artifact retention/export, checkpoint safety, and accurate disk/host-RAM/GPU planning. Converted/quantized/smaller repositories require explicit user selection, not automatic substitution.

Read-only inspection also found `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (~34.18 GB listed artifacts) as a possible smaller future video test target. It was **not selected or downloaded**; it is not the requested A14B model, and has no implemented adapter here.

## LoRA tooling: subsequent phase, not implemented yet

Start with startup-time adapters rather than unauthenticated hot-swapping APIs:

- Inspect adapter metadata, identify required base architecture, and report base-model dependencies instead of treating an adapter as standalone weights.
- Pin both base and adapter repository revisions; select exact safetensors filenames and an explicit scale.
- Validate target modules, rank, tensor shapes, and family compatibility. A text PEFT adapter is not an image/video LoRA.
- Text: versioned PEFT profiles. Images: model-specific Diffusers `load_lora_weights` / adapter configuration. Video: explicit high/low-noise expert targeting where applicable.
- Account for adapter memory and combined licenses; preserve the unmodified cached base.
- Add CPU shape/metadata tests and real one-family-at-a-time GPU tests. No merging, dynamic public adapter loading, training, or arbitrary remote code by default.

Publish a new version (for example `v0.3.0-alpha.1`) only after the intended scope is tested. Change Dokploy's explicit `RELEASE_VERSION` build argument before deploying new code; changing the Dockerfile default alone does not override an existing build argument.
