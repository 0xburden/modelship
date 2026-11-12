# Validation record

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
