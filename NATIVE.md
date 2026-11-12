# Native deployments

Use this backend when host CUDA PyTorch works but Docker GPU passthrough does not (for example, the tested Thunder Compute server). The default backend remains Docker; select native explicitly for **every** lifecycle command.

## Requirements

- Linux amd64 with an NVIDIA GPU and working `nvidia-smi`.
- Python 3.10+ with **already installed, functioning CUDA-enabled PyTorch** for established recipes; experimental recipes require Python >=3.11; new Qwen/Image/MOSS/WAN profiles also check PyTorch >=2.9 without replacing it.
- The selected Python's PyTorch must be available to a new `venv --system-site-packages`. A PyTorch installed only inside a parent virtual environment may not be inherited; use the working system Python in that case. The installer checks inheritance before invoking pip.
- Python's `venv`/pip support (on Ubuntu, the matching `python3-venv` package may be required).
- Network access for metadata, service dependencies, and any uncached weights.
- No Docker, Container Toolkit, root privileges, or driver changes are required by the native backend itself.

## First deployment

```sh
modelship doctor --backend native

modelship plan Qwen/Qwen2.5-3B-Instruct \
  --backend native --python python3 --name chat

modelship deploy Qwen/Qwen2.5-3B-Instruct \
  --backend native --python python3 --name chat --wait 45m

modelship smoke --backend native --name chat
modelship status --backend native --name chat
```

`--python` names the **base interpreter** with working CUDA PyTorch. Do not resolve a venv interpreter symlink to the system interpreter yourself; that would change its import environment. Default: `python3`.

Native text uses **Transformers**, not vLLM. The initial implementation is serial, non-streaming, and eager attention. Established recipes use float16; the experimental Gemma 4 recipe uses BF16, a modern conditional-generation loader, and its own dependency pins. See `MODEL_SUPPORT.md`; it is text-only and requires `--experimental`. It supports a limited `/v1/completions`, `/v1/chat/completions`, and `/v1/models` interface—not the complete OpenAI protocol. Request limits are documented in `/docs`. Chat requires a tokenizer chat template. Streaming/tool calls are not implemented; unsupported fields are rejected.

## Images

Stop text first if you do not intend simultaneous GPU use:

```sh
modelship stop --backend native --name chat

modelship deploy stabilityai/stable-diffusion-xl-base-1.0 \
  --backend native --python python3 --task text-to-image \
  --name images --port 8001 --wait 45m
```

Use the JSON/base64 image API examples in `README.md`. For image-to-image, create a separate deployment with `--task image-to-image` and a distinct name/port; stop competing deployments as appropriate. One task is served per deployment. The image runtime is shared with the Docker adapter; the native entrypoint defaults to localhost; a network listener requires explicit bearer authentication.

## Experimental image/video families

[MODEL_SUPPORT.md](MODEL_SUPPORT.md) has exact pinned commands and contracts. Qwen Image needs separate `--research` acknowledgment, a dedicated same-pipeline editing contract (no `strength`), and enough **free** VRAM. MOSS additionally requires explicit `--trust-model-code` approval of one reviewed revision/three source hashes; this is not implied by research use. WAN requires explicit official converted safetensors selection, a much larger GPU/cache disk, and serves bounded asynchronous jobs, not the image endpoint. These adapters remain experimental. Pinned Qwen3.8 text and bounded WAN H.264 generation have now passed on A10080GB; Qwen Image/MOSS passed on A6000. This is not validation of every model variant or maximum request shape. The tested A100 WAN run needs explicit `--allow-risk` only for the conservative runtime heuristic; hard weights/disk checks remain enforced.

All modern loaders place components directly on the selected CUDA device; no CPU offload/quantization is inferred. The planner counts all loaded components, including both WAN experts. `--allow-risk` never bypasses weights-alone fit, missing/invalid header metadata, disk limits or policy guards.

## API access

See [Native API authentication](README.md#native-api-authentication-v030-alpha2) for private key generation, explicit `--listen 0.0.0.0` opt-in and stopped-only migration of an existing deployment. Health/status/smoke requests authenticate automatically. Every native HTTP route is protected when enabled. Non-loopback without a valid private key is refused, never inferred from the environment. This does not add authentication to Docker services.

## Installation and PyTorch preservation

After confirmation, the CLI:

1. Selects one physical GPU by UUID using `CUDA_VISIBLE_DEVICES`.
2. Runs actual CUDA computation using the base Python and records the imported PyTorch file/version/package version.
3. Creates a private per-deployment `--system-site-packages` virtual environment.
4. Verifies that the new environment imports the **same PyTorch before pip runs**. If not, installation fails rather than trying to obtain another CUDA wheel.
5. Installs pinned service dependencies with a constraint matching the existing torch package version. It never explicitly installs torch.
6. Runs CUDA computation again and refuses startup if PyTorch's imported file/version changed. Recipe ID, loader, component dtypes, dependency pins, research/code approval, reviewed source hashes and absolute cache path are retained. New native state uses **schema 4** (older CLIs reject it safely); this CLI still reads schema-1, schema-2 and schema-3 deployments.
7. Refreshes selected-GPU total/free VRAM against the saved weight minimum, then starts one detached worker, waits for health reporting the exact model revision and unique run identity, and prints its localhost endpoint.

Workers set `USE_HUB_KERNELS=NO` to prevent implicit Transformers Hub-kernel downloads; standard loaders also reject model repository Python. MOSS is the sole explicit, reviewed-code exception.

The host Python environment is not modified. Native dependencies are pinned at their direct versions; this is not a fully locked dependency graph. Shared host packages remain part of the trusted runtime. Native mode is **not a sandbox**: it runs as your user.

## Lifecycle

```sh
modelship logs --backend native --name images
modelship status --backend native --name images
modelship stop --backend native --name images
modelship start --backend native --name images
modelship uninstall --backend native --name images --yes
```

- Workers detach into their own Linux session/process group and survive SSH disconnection.
- **No automatic restart after a crash or reboot** in this version. Run `start` explicitly; a systemd adapter is future work.
- `start` waits up to 30 minutes for readiness and refreshes the weight budget if present. Older saved plans lack this check. Deploy supports `--wait 0` or another timeout.
- Readiness failure leaves logs/config/worker state for inspection; the process may still be downloading/loading. Do not blindly start a second copy.
- `stop` verifies PID start ticks and unique command-line identity before signaling the process group, avoiding stale PID reuse. Graceful termination is attempted before a forced stop.
- Failed installations retain state. Use `uninstall --backend native --name NAME --yes`, then redeploy after correcting the cause.
- Names are unique within each backend. A native deployment and Docker container can have the same name; choose clear names and specify the backend to avoid confusion.
- Logs follow until interrupted, and inference access logs are disabled. Error libraries may still log details; treat logs as private.

## State, cache, and credentials

State is stored at:

```text
${XDG_STATE_HOME:-$HOME/.local/state}/modelship/native/NAME/
  config.json          Pinned plan + PyTorch metadata (no HF_TOKEN)
  run.json             PID, Linux start ticks, unique run identity
  service.log          Runtime output
  venv/                Per-deployment packages
  requirements.txt     Direct service dependency pins
  constraints.txt      Existing torch package version constraint
  native_entry.py      Localhost-only entrypoint
  server.py            Image API
  text_server.py       Text API
  video_server.py      Bounded WAN jobs/MP4 API
  model_code.py        Explicit reviewed-source loading gate
  artifacts/           Private ephemeral video exports (video deployments)
```

Directories are private (0700); metadata/assets/log files are 0600. `XDG_STATE_HOME` must be absolute. Lifecycle operations use a per-deployment lock.

Weights use the usual Hugging Face host cache (`HF_HUB_CACHE`, legacy `HUGGINGFACE_HUB_CACHE`, then `HF_HOME` / `XDG_CACHE_HOME` defaults). The resolved absolute hub-cache path is saved for new deployments so restarts do not switch it silently. Planning inspects free disk and correctly-sized selected files at the pinned snapshot, without modifying them. Missing downloads plus 5% and 4 GiB headroom must fit; dependency/state files also need space on their own filesystem. **Uninstall removes deployment state/venv/video exports, not the shared model cache.** No cache eviction runs automatically.

`HF_TOKEN` is inherited from the launching CLI, never written to config/arguments by Modelship. Supply it again on `start` if private/gated downloads need it; the CLI does not persist arbitrary environment variables across starts. Processes owned by the same user or root can inspect environments. Use a scoped read-only token and a trusted host.

## Network exposure

The managed native entrypoint defaults to `127.0.0.1`, regardless of Docker/provider port publishing. From v0.3.0-alpha.2 onward, an explicit `--listen 0.0.0.0` requires a valid private `--api-key-file`; every HTTP route then requires its bearer key. Keep the default loopback/tunnel setup unless you intend network access. Use trusted HTTPS ingress (such as Thunder's generated HTTPS URL) and verify both unauthenticated401 and authenticated200 responses before sharing. Bearer auth does not encrypt raw HTTP. Docker image APIs are unchanged and remain unauthenticated.

## Validation status

See `VALIDATION.md`. The native CLI adapter has now been tested end-to-end by the agent on a fresh Thunder Base/A6000 instance: Qwen text completions/chat, SDXL text-to-image at 512/1024, SDXL image-to-image at 512, validation/busy responses, text stop/start, and survival across SSH disconnect. PNGs were downloaded and visually inspected. The later dedicated A100 instance also passed full Qwen3.8 text and bounded WAN MP4/H.264 generation/restart checks. Host PyTorch was preserved. This does not establish compatibility for every supported architecture or production-grade resilience; FLUX, sustained load, OOM handling, and interrupted loading/generation remain further validation work.
