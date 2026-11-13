# Validation record

## Alpha.5 GGUF text: header and CPU-adapter evidence only

- Exact target and runtime pins: [GGUF.md](GGUF.md). No instance changes or model weight downloads were made. Full model/asset SHA-256 pins were checked against immutable HF LFS and named GitHub release metadata; actual archive/model bytes have not been downloaded or independently hash-verified in this run.
- The new Go inspector parsed the real selected file using exact HTTP 206 Range responses, bounded header-only read-ahead: GGUF v3, `qwen35`, 866 non-overlapping tensors, 10,996,640-byte aligned header, 16,553,263,104 encoded tensor bytes, 262144 advertised context, 65 blocks, 4 KV heads, key/value dimensions 256. Mixed tensor types: F32, Q8_0, Q3_K/Q4_K/Q5_K/Q6_K, IQ4_NL/IQ3_S/IQ4_XS. No tensor payload was fetched.
- Pinned llama.cpp source contains qwen35 support and the exact configured device/fitting/auth/template/server flags. GGML block-size formulas were checked against that source. This is source compatibility evidence, **not runtime/inference proof**.
- Synthetic tests cover strict header/range parsing, quant-block bytes/overflow/truncation, explicit one-file selection, hard free/total VRAM and disk gates, cache accounting, verified/resumed downloads, extraction/link traversal defense, runtime manifest checks, schema boundary, sanitized/authenticated text-only gateway, templating, unknown/media/tool/stream fields, busy behavior and retaining the slot across client disconnect. A 15-second fuzz run completed 378,963 executions; race/vet checks passed.
- A real Go gateway spawned a separate CPU-only synthetic private HTTP child with an independent key; readiness, owner authentication, a fixture response and clean shutdown passed. The word “Paris” in these fixtures is **not output from the actual GGUF model**. The public CLI model/GPU gates were not bypassed. Existing Linux Python/real-CLI regression suite: 84 tests passed.
- Still required: actual pinned runtime extraction/device execution, selected model download/load, meaningful text/chat evaluation, measured memory, real GPU busy/errors/auth, stop/start/cache/console testing, host-package preservation. No claim of deployed GGUF support or vision/projector support yet. Published alpha.4 and retained releases are unchanged.

## User-run native GPU baseline: Thunder Compute

Hardware/environment reported from the server:

- Ubuntu 22.04.5 LTS, Linux x86_64.
- NVIDIA RTX A6000, approximately 48 GiB VRAM, driver 610.43.02.
- PyTorch `2.12.1+cu130`.
- Transformers `4.56.2`, Diffusers `0.35.1`.
- Basic native CUDA tensor computation produced the expected result (`4.0`).

### Text

- Model: `Qwen/Qwen2.5-3B-Instruct`.
- Revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Native Transformers, float16, eager attention, greedy generation, max 80 new tokens.
- Model loaded and produced a coherent explanation of a GPU.
- Reported peak **allocated** VRAM: **5.87 GiB**.

### Text-to-image and image-to-image

- Model: `stabilityai/stable-diffusion-xl-base-1.0`.
- Revision: `462165984030d82259a11f4367a4eed129e94a7b`.
- Native Diffusers, float16, VAE slicing/tiling.
- Text-to-image: 512×512, 20 steps, guidance 7.0, seed 42.
- Image-to-image: generated image as input, strength 0.65, 20 requested steps (13 effective), guidance 7.0, seed 43; model components shared via `from_pipe`.
- Both requests completed and wrote `sdxl-original.png` / `sdxl-edited.png`.
- Reported peak allocated VRAM across both runs: **7.65 GiB**.
- Progress output suggests roughly 3 seconds / 1.5 seconds respectively, **not a controlled latency benchmark**.
- Visual quality has not been independently inspected by the agent. 1024×1024 behavior has not yet been reported.

These are direct Python tests performed by the user, **not end-to-end validation of Modelship's native deployment/lifecycle adapter**. Allocated VRAM excludes other runtime/driver allocations and is not total GPU usage.

## Agent-run native CLI validation: new Thunder Base/A6000 instance

The agent connected to the newly created instance using the saved CLI login with `TNR_API_TOKEN` omitted for individual commands. The environment token returned an empty instance list; saved credentials could see instance `0`. No stored credentials were changed.

Environment: Ubuntu 22.04.5, Python 3.12.13, A6000 (49140 MiB), driver 610.43.02, PyTorch 2.12.1+cu130. Installed CLI: `v0.2.0-dev.1`. Torch source/version preservation and actual CUDA preflights passed during CLI dependency installation.

### Text API and lifecycle — passed

Model/revision match the Qwen baseline above. A real `modelship deploy --backend native` created the environment, loaded weights, and reached readiness.

- `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`: HTTP 200 with valid output/usage.
- One 16-token completion request: 0.846 s; one 37-token chat response: 2.042 s. These are individual request timings, not throughput benchmarks.
- Unknown model: 404. Unsupported streaming, invalid token budget, context overflow: 422.
- Two simultaneous requests: one 200 and one 503 (busy), as intended.
- CLI text smoke passed before and after stop/start.
- Worker remained usable after disconnecting SSH and reconnecting in a new session.
- `ss` showed `127.0.0.1:8000`; `tnr ports list` showed no HTTP forwarding configured for this instance. This is not an external network penetration test.
- The restart reused the existing environment/cache; no weight reinstallation was requested.

Machine-readable reports and responses were downloaded to ignored `validation-artifacts/thunder-native/`. The reusable, opt-in API client is `tests/live_api_check.py`.

### Image API adapter

SDXL native text-to-image and separate image-to-image deployments passed actual dependency installation, PyTorch-preservation checks, model loading, and API inference at the pinned revision above.

| Request | Resolution / steps | HTTP | Individual request time |
| --- | --- | --- | --- |
| Text-to-image | 512×512 / 20 | 200 | 5.385 s |
| Text-to-image | 1024×1024 / 20 | 200 | 4.901 s |
| Image-to-image | 512×512 / 20 requested, strength 0.65 | 200 | 4.318 s |

These are single sequential requests, not comparative performance benchmarks; the first request includes warm-up effects. Seeds: 42 for generation, 43 for editing.

- PNG signatures and exact dimensions were checked programmatically. All three images were downloaded and visually inspected by the agent: recognizable watercolor observatory/mountain scenes, not blank/corrupted. The edit retained the source composition while changing the observatory and sky; it did not fully turn the sunset into a dark night, which is a model-quality limitation rather than an API failure.
- Wrong task endpoints: 409. Invalid dimensions: 422. Missing image input and image URL supplied as base64: 422.
- The edit worker remained running across an SSH session closure during readiness, and its API worked after reconnection.
- Observed image listeners were `127.0.0.1:8001` / `127.0.0.1:8002` while active.
- Image-to-image loaded the existing SDXL cache; no second set of model weights was needed. Python service environments remain separate per deployment.
- Reports/PNGs are in ignored local `validation-artifacts/thunder-native/` and remote `~/modelship-validation/`.

Final cleanup **completed**: all three test workers report `stopped` and PID 0. `nvidia-smi` reported 0 MiB used / 49140 MiB free after cleanup; the host's original PyTorch remained `2.12.1+cu130` and CUDA computation still returned 4.0. Before stopping the final edit worker, device memory usage was 9174 MiB (a post-request device snapshot, **not** a peak allocated-memory measurement). Disk usage was approximately 24 GiB / 96 GiB, with 73 GiB free. State, environments, and cached weights remain for restart; the cloud instance was left running and was not resized/deleted. Native services do not auto-start after reboot.

Retained deployment names:

```sh
~/.local/bin/modelship start --backend native --name native-chat-smoke
~/.local/bin/modelship start --backend native --name native-images-smoke
~/.local/bin/modelship start --backend native --name native-edits-smoke
```

Choose the desired service rather than blindly starting every test worker. The local artifact directory includes `final-state.json` with the stopped-state/PyTorch checks.

## Recipe-branch Gemma 4 validation on a fresh A6000

This work is on `feat/model-recipes`, **not the published v0.2.0-alpha.1**. New development CLI: `v0.3.0-recipes-dev.3`, installed separately on the test server as `~/.local/bin/modelship-dev`.

- Same Ubuntu 22.04.5 / Python 3.12.13 / A6000 48 GiB / driver 610.43.02 / host PyTorch 2.12.1+cu130 environment family as the baseline.
- `google/gemma-4-E4B-it` pinned to `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`, ~15.99 GB of checkpoint artifacts. Full model loads in BF16; context 4096; text-only API; eager attention.
- Isolated Transformers 5.22.2 / Hugging Face Hub 2.4.0 / safetensors 0.8.0 / accelerate 1.15.0 profile installed without replacing host PyTorch. Pre/post CUDA probes passed.
- Initial untemplated completion output repeated text despite HTTP 200. This was treated as a quality failure, not a passing model test. The recipe now frames completion prompts as a user turn and disables thinking for bounded text responses. Legacy causal-model completion framing is unchanged.
- Post-fix semantic completion: `The capital of France is **Paris**.` Chat: `Gemma is working.`
- Nine live API checks passed, including model listing, completions/chat, unknown model 404, unsupported streaming/parameter/context errors 422, and simultaneous 200/503 busy handling. A separate image-message-content request returned 422, as required by the text-only scope.
- Post-fix single request times: 4.680 s completion / 5.431 s chat. These are not throughput benchmarks.
- Stop/start, cached loading, and schema-2 metadata/loader/dtype flags passed. The actual published alpha binary refuses schema-2 state, preventing accidental legacy loader/dtype fallback.
- A newly created Qwen2.5-3B regression deployment passed the same API and Paris semantic checks while Gemma stayed healthy. Worker configuration confirms Qwen uses its legacy float16/causal/raw-completion profile and Gemma uses BF16/multimodal/templated-completion settings. No existing deployment was stopped to make room.
- Listeners: `127.0.0.1:8000` (Qwen), `127.0.0.1:8003` (Gemma). No Thunder HTTP forwarding is configured. Combined post-request device memory was 21370 MiB used / 27770 MiB free; this is not a peak allocated-memory measurement.
- Both demonstrations are retained running for comparison under names `qwen-regression` and `gemma4-e4b-test`. Use `modelship-dev` to manage them. The instance remains running/billable and was not resized by the agent.
- Machine-readable reports are ignored under local `validation-artifacts/gemma4/templated/`, `validation-artifacts/qwen-regression/`, and remote `~/modelship-validation/`.

All five newly requested repositories were also inspected at pinned revisions using the host-independent CLI. The next batch below implements the remaining experimental adapters and hard weight checks; GPU-inference validation remains model-specific. LoRA is still future work. See `MODEL_SUPPORT.md` for precise format/license/code constraints.

## All-model branch: tensor-header and current-server preflight

All changes remain on `feat/model-recipes` / draft PR #1; no merge or published release change. Native experimental paths now exist for Qwen3.8 text, Qwen Image 2.1 generation/editing, reviewed/opt-in MOSS images, and explicitly selected official WAN Diffusers video. Original WAN PTH loading remains blocked; there is no silent repository substitution.

Real CLI `plan --backend native --experimental --research --allow-risk --json` ran on the A6000 with **49140 MiB total / 27770 MiB free**, while both retained demonstrations remained running. Only shard manifests and bounded safetensors headers were fetched, not complete tensor files. Minimum requested-dtype resident bytes were computed from every selected component, preserving integer buffers and FP32 VAE overrides:

| Model / selected format | Download bytes (GB) | Minimum weight VRAM (GiB) | Fits total / free | Observed decision |
| --- | ---: | ---: | --- | --- |
| Qwen3.8-27B | 55.563 | 51.747 | no / no | Hard total/free blockers despite `--allow-risk` |
| Qwen Image 2.1 | 33.116 | 30.841 | yes / no | Hard free-VRAM blocker despite `--allow-risk` |
| Gemma 4 E4B | 15.993 | 14.894 | yes / yes | Plan accepted |
| MOSS Image SFT | 27.200 | 25.332 | yes / yes | Separate repository-code approval still required; research did not imply it |
| WAN 2.2 official Diffusers | 126.178 | 64.283 | no / no | Hard VRAM blockers plus cache disk shortfall |

WAN's FP32 experts explain why on-disk bytes differ from BF16 resident weight bytes; both expert networks, text encoder and FP32 VAE were included. The native cache had 72425 MiB free versus a 130445 MiB missing-download/headroom budget, so a full WAN download was blocked independently of VRAM.

- Local Go race tests/vet pass; Linux cross-compiled Go tests also pass on the actual server, including native subprocess/state tests. Fake native tests use ephemeral ports rather than colliding with existing localhost services.
- All **54 Python tests** pass on Linux, including three real CLI/fake-worker lifecycle integrations and 25 fake-WAN API/lifecycle tests. No fake test claims real GPU/video-codec validation.
- Tests cover malformed/duplicate/overflowing header data, ignored/oversized/partial HTTP ranges, missing shards/manifests/weight sizes, transport-error token redaction, source-FP32 load conversion, mixed component dtypes, integer buffers, both WAN experts, selected/free GPU checks, unchanged research/code boundaries, code SHA checks, pinned-cache disk accounting, and restart metadata compatibility.
- Modern native/container environments explicitly disable implicit Hub-kernel fetching with `USE_HUB_KERNELS=NO` in addition to rejecting ordinary model repository Python. The only repository-code exception is the explicitly approved, hash-pinned MOSS path.
- Qwen3.8/WAN GPU inference needs larger hardware; Qwen Image/MOSS live image generation remains pending. Existing services were not stopped to make room, and MOSS source code was not executed.
- Ignored evidence: local `validation-artifacts/batch-dev4/`, remote `~/modelship-validation/recipes-batch-dev4/`. No source code, credentials or weights were copied out of the shared HF cache.

### Current-loader Gemma regression — passed

The updated `v0.3.0-recipes-dev.5` loader was also exercised on real hardware, not just the earlier Gemma implementation. A temporary `gemma4-dev5-regression` deployment on **127.0.0.1:18013** coexisted with both retained services without stopping either.

- Direct CUDA placement, BF16/eager attention, separate modern dependency environment, and pre/post-install host-PyTorch preservation passed.
- All live text checks passed, including semantic completion `The capital of France is **Paris**.`, chat, model listing, validation/context errors and concurrent busy handling.
- Individual completion/chat times: 3.839 s / 5.333 s; not a benchmark.
- Actual worker settings verified native **schema 3**, BF16/multimodal loader, templated completions, thinking disabled, explicit research acknowledgment, persisted absolute `HF_HUB_CACHE`, and **`USE_HUB_KERNELS=NO`**.
- Stop/start, cached loading, current-memory recheck and post-restart CLI smoke passed.
- The temporary deployment was stopped and uninstalled. The original `qwen-regression` / `gemma4-e4b-test` workers remained running; GPU memory returned to **21370 MiB used / 27770 MiB free**. Shared model cache and host PyTorch `2.12.1+cu130` were preserved.
- The development test binary remains under the private validation directory, not installed over `~/.local/bin/modelship-dev`, and the published alpha release was not changed.

## Approved image GPU validation: Qwen Image 2.1

The user explicitly approved temporarily pausing/restoring `qwen-regression` and `gemma4-e4b-test`, and executing the reviewed pinned MOSS code. All tests remain on `feat/model-recipes`; no published release changes.

Qwen Image uses revision `d26bb61231c349cf6b7896fa83353113880e1ba3`, BF16 transformer/text encoder and FP32 VAE, direct CUDA placement, research acknowledgment, and Diffusers 0.42.1 / Transformers 5.22.2. No host PyTorch replacement or model substitution.

| Request | Shape / steps | Result | Single-request time |
| --- | --- | --- | --- |
| Text-to-image | 512² / 20 | HTTP 200 / valid PNG | 7.618 s |
| Text-to-image | 1024² / 20 | HTTP 200 / valid PNG | 18.191 s |
| Image-conditioned edit | 512² / 20 | HTTP 200 / valid PNG | 9.591 s |

These are individual request timings, not benchmarks. Both generated images were downloaded and visually inspected: coherent watercolor observatories on mountains at sunset. The edit preserved the observatory/rock composition and changed the sunset to a dark starry sky, matching the prompt. Seed 42 for generation, 43 for editing.

**A real editing bug was found and fixed**, rather than counted as an HTTP-only success. Upstream Diffusers 0.42.1 casts conditioning pixels to BF16 before calling a separately FP32 VAE, causing a convolution input/bias dtype mismatch. The dedicated local pipeline subclass now encodes/normalizes in VAE precision and casts encoded latents back to the incoming BF16 precision before transformer concatenation. A regression test covers both cast boundaries. The failed run stopped its test worker and successfully restored both demos before retrying. After the fix, both generation PNGs were byte-identical to the earlier successful outputs.

Additional checks passed: wrong task 409; invalid dimensions/steps/long prompts/URL fields/unsupported strength/negative prompt at CFG 1 returned 422; concurrent generation 200/503; required inline image input; URL-as-base64 rejection; cached image-to-image deployment and stop/start readiness. Observed fixed-run device usage was 31626 MiB loaded / 32808 MiB after generation / 33320 MiB after editing, **snapshots, not peak memory**. No offload/quantization was used.

Ignored evidence: `validation-artifacts/image-live-20261112-072826/` (initial/failing edit run) and `validation-artifacts/image-live-20261112-074424/` (fixed run), plus the matching directories under remote `~/modelship-validation/recipes-batch-dev4/`.

### MOSS Image SFT and final restoration — passed

After explicit user approval, `OpenMOSS-Team/MOSS-Image-SFT` at `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9` passed real native inference using the reviewed code gate, BF16 transformer/text encoder, FP32 VAE, CPU noise generator, 512-token text cap, and guidance 3.0.

- **512² / 20 steps:** HTTP 200, 11.671 s. **1024² / 20 steps:** HTTP 200, 29.205 s. Single-request timings, not benchmarks.
- Both PNGs were downloaded and visually inspected: clear watercolor observatory/mountain/sunset scenes matching the prompt, not blank or corrupted.
- Wrong endpoint/dimensions/steps/oversized prompt/URL fields rejected; concurrent requests returned 200/503 as intended.
- The exact three approved Python files were SHA-256-verified before loading, and their cached bytes were independently checked afterward. Recorded approval/code revision matched the pinned checkpoint.
- Pre-test source inspection found that the restricted snapshot whitelist omitted `tokenizer/chat_template.jinja`. It was added as tokenizer data (not a broad Python allowlist); a regression test verifies its inclusion. Real prompt encoding now passes with the bundled chat template.
- Device-memory snapshots: **25976 MiB loaded / 32562 MiB after generation**, not peak memory. Existing host PyTorch remained **2.12.1+cu130**, same import path/package version, with pre/post CUDA checks passing.
- All temporary image workers were stopped and uninstalled. **Both original demos were restored and CLI smoke-tested.** Final device snapshot: **21364 MiB used / 27776 MiB free**. No restore/cleanup errors.
- Shared HF weights remain cached intentionally; no automatic cache eviction. Disk ended at **14549 MiB free** on the 100 GB instance. More large model downloads need additional space or deliberate cache management.
- All **55 Python tests passed on Linux with the actual CLI binary**, including three fake-worker lifecycle integrations; the extra regression test covers the Qwen edit precision bridge. Go race tests/vet and Linux Go subprocess tests also pass. At that stage, Qwen3.8/full WAN GPU inference and actual WAN encoding still required larger hardware; both were subsequently exercised on the A100 instance in the sections below.

## Larger-instance Qwen3.8 validation — passed

The user supplied instance `1` (`pjj91chs`) specifically for the remaining large models. Observed hardware: **NVIDIA A100-SXM4-80GB, 81920 MiB total / 81176 MiB initially free**, compute8.0; 96 GB host RAM; 150 GB disk (143688 MiB initially available). Ubuntu22.04.5/Python3.12.13/driver610.43.02/PyTorch2.12.1+cu130. Actual CUDA preflight returned4.0. No driver or host-Torch changes.

`Qwen/Qwen3.8-27B` at `37561816e66b5b9153b1b64c10968c18c1e6f4a0` passed native deployment, full BF16/conditional-generation loading, eager text-only inference with context4096, and **no quantization/offload/heuristic override**. The header minimum was51.747GiB; runtime heuristic64010MiB, fitting this GPU.

- Semantic completion: `The capital of France is **Paris**.` Arithmetic chat returned`56` for7×8. General chat produced a coherent GPU explanation.
- All live text API/model listing/error/context and busy checks passed; concurrent requests returned200/503. Media payload rejected422, preserving text-only scope.
- Individual completion/chat times2.047s/2.246s, not a benchmark.
- Cached stop/start and CLI smoke passed. Host PyTorch import/version/package metadata were preserved.
- Device-memory snapshot52733MiB loaded (not a peak). Provider counters reserve memory separately; total/free/used may not sum exactly.
- Testing is sequential because both source checkpoints exceed the150GB disk together. Qwen used a newly created, marked, private test cache. After stopping/uninstalling only its test deployment, the harness verified that no saved deployment referenced that cache and deleted **only that owned cache** before WAN. Global/user model caches and credentials were not removed.
- Evidence: ignored `validation-artifacts/a100-large-models/` and remote `~/modelship-validation/large-models/run-20261112-082534/`. No operations were issued against instance0.

### WAN 2.2 dual-expert video/codec validation — passed

On the same dedicated A10080GB/96GBRAM/150GB-disk instance, explicitly selected official `Wan-AI/Wan2.2-T2V-A14B-Diffusers` at `5be7df9619b54f4e2667b2755bc6a756675b5cd7`. Original PTH loading was not enabled. Both experts/text encoder loaded in BF16, VAEFP32, directly on CUDA; no quantization/offload or repository Python.

- Header estimate64.283GiB resident weights fits total and free VRAM. The conservative runtime estimate82313MiB exceeded90% of81176MiB free. **Explicit `--allow-risk` was used only for this runtime heuristic**; hard tensor-header/weight/disk checks stayed enforced. Small smoke preceded the larger clip.
- The owned Qwen cache was removed before WAN, restoring143641MiB available disk. Full source checkpoint126.178GB downloaded; WAN loaded with22892MiB disk free.
- **Codec smoke:**256²/5frames/4steps/16fps,4.023s including async polling/download. MP4H.264 decoded exactly5 frames,0.31s duration,29598bytes.
- **Bounded quality sample:**512²/17frames/30steps/16fps,50.076s including polling/download. MP4H.264 decoded exactly17 frames,1.06s duration,105755bytes. These are individual end-to-end request times, not benchmarks.
- All frames decoded; dimensions/FPS/count, non-flat color variance and nonidentical adjacent frames passed. First/middle/last quality frames were downloaded and visually inspected: coherent paper boat, water/ripples and reflections. **Boat color was white rather than requested red**, so prompt adherence is not perfect. Only a short bounded sample was tested, not longer/maximal clips or production quality.
- Async202/job polling/content retrieval, busy503/no backlog, content-before-ready409, unknown job404, parameter/shape/frame/body limits, and URL-field rejection passed.
- Device snapshots66485MiB loaded /69538MiB after generation (not peaks). Provider total/free/used counters reserve memory separately.
- Cached stop/start passed; pre-restart job IDs returned404 as designed, and a fresh post-restart256²/5-frame/4-step job succeeded.
- Test workers were stopped successfully; after CUDA teardown, observed device use returned to0MiB (81176MiB free). WAN cache/state/venv are retained stopped as`wan22-a14b-native`, localhost18032; Qwen's private test deployment/cache were retired to fit WAN on disk. Final disk22892MiB free. Instance1 remains running/billable; no instance resize/delete or operation on instance0.
- Reusable opt-in client: `tests/live_video_check.py`; run using a video deployment's Python for its pinned imageio-ffmpeg/Pillow. Three CPU/fake-codec client tests bring the suite to58 tests; all58 passed on the actual Linux host with the CLI binary, alongside Go race/vet and Linux Go lifecycle checks. Fakes do not establish GPU/codec correctness by themselves.

## Thunder Docker blocker

Provider Docker rejects `--gpus` / `--runtime=nvidia`; its documented GPU flag is `--device nvidia.com/gpu=all`.

The CUDA 12.8.1 Ubuntu 22.04 diagnostic container failed to load NVML. Directory inspection showed the injected alias:

```text
/run/thundercompute/host-libs/usr-lib-x86_64-linux-gnu/libnvidia-ml.so.1
  -> /run/thundercompute/host-libs/usr-lib-x86_64-linux-gnu/libnvidia-ml.so.1
```

That alias is self-referential. `ldd` could not open the exposed container NVML path. Prepending the entire host-library directory to `LD_LIBRARY_PATH` caused a GLIBC_PRIVATE conflict in the provider's proot wrapper; **do not use that workaround**.

Docker also warned that IPv4 forwarding was disabled. Thunder documents non-isolated experimental nested Docker networking; ordinary localhost port publishing is not a sufficient exposure guarantee there. No driver/runtime reconfiguration was performed by the agent.

## Automated checks

- Go unit/race tests: URL/revision validation, metadata inspection, planning, Docker/CDI argument generation and fallback guards, native environment generation, state permissions/locks, Linux process identity, stop, readiness identity, cache preservation.
- Python HTTP contract tests with fake GPU pipelines/models: image and native text API validation, outputs, context limits, busy/OOM responses, size limits, revision/safetensors settings.
- Real CLI subprocess integration with a fake CPU-only native worker: start, smoke, status, stop, restart, refusal to overwrite a running worker, explicit uninstall, failed startup/log retention, occupied-port rejection.
- Installer checksum validation/corruption rejection, Linux amd64 release packaging, static-release image build.
- Live Hugging Face metadata planning for the Qwen/SDXL revisions above; no GPU required for these metadata tests.

Fake-model tests do not establish CUDA/runtime compatibility. Native text and SDXL generation/edit API deployments have now passed agent-run real-server tests above. Fake-model tests alone do not establish GPU compatibility for other models, formats, runtimes, or hardware.

## Remaining initial checks

1. **Completed:** accessed the new Base/A6000 instance using `env -u TNR_API_TOKEN tnr connect 0`. The environment token could not see the instance; the existing saved CLI login could. No credentials were overwritten and no instance was created/deleted/modified by the agent.
2. **Completed:** transferred `v0.2.0-dev.1`, ran native `doctor`, `plan`, `deploy`, text smoke/API checks, and stop/start. The text worker also survived SSH disconnect/reconnect. Detailed results are recorded below.
3. **Completed:** validated both image API tasks using the native adapter and inspected the PNG artifacts.
4. **Partially completed:** 512/1024 SDXL request timings and input validation passed; text busy handling passed. Controlled performance/peak-memory measurements, image busy/OOM tests, and image-to-image at 1024 remain future checks.
5. **Completed for tested paths:** text stop/start and SSH reconnection, cache reuse, loopback listeners, and no configured Thunder HTTP forwarding. External network exposure testing has not been performed.
6. Test graceful termination during loading/generation and retained logs after failures. Native auto-restart/reboot persistence is not implemented.
7. Test FLUX separately on suitable memory/hardware; it remains unvalidated.
8. Publish a versioned release via Dokploy and test the curl bootstrap on another fresh supported host.

## Versioned release host: v0.3.0-alpha.1

Release preparation adds a source-controlled version, verified retention of all three original v0.2.0-alpha.1 assets, and an existing-version checksum guard. Four offline regression cases cover stale Dokploy build arguments, retained-file corruption, differing existing versions and identical rebuilds. The release Docker image was built locally with a deliberately stale `RELEASE_VERSION=v0.2.0-alpha.1`; it selected the tracked new version and retained checksum-pinned old files instead of relabeling or removing them. Publication/live endpoint verification is a separate release step.

## Native API access boundary (v0.3.0-alpha.2 preparation)

The feature branch adds opt-in, header-only bearer authentication for every native HTTP route, including health/docs/model discovery/video status/content, before routing/body parsing. Network listeners require a valid private key. Tests cover mode/ownership/type/size/content checks, symlink/FIFO rejection, malformed/duplicate/query/cookie credentials, constant-size digest comparisons, no-store responses, websocket denial and fail-closed validation before inference imports. Go readiness/status/smoke use the file-backed key without redirects or environment proxies; keys do not enter metadata/argv/output.

Linux CPU-only subprocess integration exercises generated keys, authenticated readiness/smoke, missing-key status/stop behavior, refusal to configure a running worker, and stopped migration using the actual native entrypoint/auth module/Uvicorn with a tiny fake inference module. No CUDA, weights or GPU service modifications are involved. Existing inference modules, venv/cache and model policy remain unchanged by configure. This is targeted security regression coverage, not an external security audit; live Thunder authenticated forwarding and new GPU inference are not yet validated.

Final local checks: Go race tests/vet passed. Python74 tests passed on Linux with the actual cross-compiled CLI (including real ASGI/Uvicorn migration) and no skips; macOS ran the same suite with five Linux-only lifecycle skips. The release Docker image built for LinuxAMD64 with a deliberately stale version argument and verified all six retained assets across v0.2.0-alpha.1 and v0.3.0-alpha.1. The current branch prepares v0.3.0-alpha.2 but is not merged/published by this validation step.

## Interactive launch preparation (v0.3.0-alpha.3)

Guided setup uses the shared Hub/weight inspector, recipe policy/fit planner, native CUDA check and deployer. Unit cases cover exact pinned-plan deployment, canceled/EOF/oversized input, deferred private key creation, existing-key secrecy and shell quoting, unbypassable hard blockers, separate heuristic consent, explicit WAN conversion and repository-code refusal. Bootstrap tests cover checksum/atomic-install behavior, TTY→launch, non-TTY install-only behavior, explicit argument forwarding, pre-exec cleanup, baked immutable defaults and environment overrides. Packaging tests check URL/version validation and checksum the actual generated installer. Actual Linux CLI PTY tests check interactive default/explicit launch and prompt cancellation before any GPU/Hub work. No live model was deployed for wizard validation.

Local verification passed: Go race/vet checks, Python84 tests on Linux with the actual CLI and no skips (macOS skips seven Linux-only cases), real PTY cancellation/default routing, and LinuxAMD64 release Docker packaging with all nine retained files checksum-verified. No live GPU model deployment, provider forwarding change or existing worker modification was performed. The branch prepares v0.3.0-alpha.3; publication is a separate approval step.

An additional bootstrap regression checks explicit overrides to older CLIs: the versioned installer warns about an override, feature-detects launch support, and falls back to install-only rather than invoking a nonexistent command.

## Wizard default-clarity follow-up

Prompt help now distinguishes upstream branch defaults, reviewed recipe pins, detected metadata/GPU suggestions, recipe-controlled precision, recommended presets, computed key paths and safety choices. Enter/bracket/No conventions are explicit; the number/order of questions and the actual defaults remain unchanged. Context is described in tokens with prompt/output sharing, CLI bounds and advertised-vs-validated capacity; startup wait is distinguished from generation duration and package/image-build time. Port, Python and loopback/HTTPS-ingress/key semantics are explained before selection and summarized during review. Tests assert unchanged accepted defaults, nested context metadata, safety/secret handling and synthetic Gemma-family matching.

User report (not independent validation): `llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic` was reported working via guided setup after the alpha.3 rollout. The actual pinned revision, hardware, options and inference checks are unknown; no repository files, weights, model code or live instance were accessed to record it. See MODEL_SUPPORT's separate user-reported compatibility section.

Clarity checks passed locally: Go race/vet and84 Python tests on Linux with the actual CLI, including existing PTY/cancellation/lifecycle tests, no skips. The new explanatory text is presentation-only: regression assertions retain native/first GPU/modelship/8000/4096/python3/local/no-key/30m and pin the same reviewed SHA. No live user instance, model repository or deployment was accessed; the community checkpoint row remains user-reported, not independently validated. The branch prepares v0.3.0-alpha.4 with all prior release assets pinned for retention; no release/merge is performed by this change.
