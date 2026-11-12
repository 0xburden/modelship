# modelship

A GPU-first model deployment CLI: inspect a Hugging Face repository and the local NVIDIA server, review a plan, then deploy a pinned model revision through Docker or a native Python worker.

**Early MVP, not a universal model converter or production control plane.** Go handles orchestration; vLLM (Docker), Transformers (native text), and a small Python/Diffusers service perform inference. No agent API or backend is required.

## Supported scope

- Linux **amd64**, NVIDIA; one selected GPU per deployment.
- **Docker backend (default):** local daemon with standard NVIDIA Container Toolkit or provider-managed CDI, conservative CUDA 12.8 driver minimum **570.26**. CDI mode currently requires exactly one visible GPU and is not fully validated on Thunder.
- **Native backend:** working host CUDA-enabled PyTorch plus Python/venv. No Docker/toolkit installation or hard-coded CUDA driver minimum; actual CUDA computation is checked. See **[NATIVE.md](NATIVE.md)**. This is the recommended path for the current Thunder server.
- Text: unquantized **safetensors** repositories with explicitly recognized architectures: Llama, Mistral, Qwen2, Qwen3, Gemma, Gemma2, Phi3 causal LMs. Exact architecture names and dependency profiles are in `internal/modelship/assets/recipes.json`.
- Images: standard Diffusers repositories whose `model_index.json` declares **StableDiffusionPipeline**, **StableDiffusionXLPipeline**, or **FluxPipeline**. Both text-to-image and image-to-image are selectable; a deployment serves **one task**.
- Text requires compute capability >=7.5 in Docker / >=7.0 natively; SD/SDXL >=7.0; FLUX >=8.0 (bfloat16).

Not yet supported: GGUF, single-file image checkpoints, variant-only image repositories, adapters/LoRAs, AWQ/GPTQ/bitsandbytes, custom remote model code, FLUX.2, video, multi-GPU sharding, automatic quantization/offload, or arbitrary model architectures. Matching an architecture is a compatibility filter, **not** proof that every repository variant will load.

## Build

Go 1.23+:

```sh
go test ./...
go build -o bin/modelship ./cmd/modelship
```

Or without a host Go installation:

```sh
docker run --rm -v "$PWD:/src" -w /src golang:1.24.4 \
  sh -c 'go test ./... && go build -o bin/modelship ./cmd/modelship'
```

The Docker command builds for the container's platform. Use `scripts/release.sh` for a Linux amd64 release from any supported Go build host.

## Recipe discovery (feature branch)

The `feat/model-recipes` iteration adds host-independent `modelship catalog` and `modelship inspect MODEL [--json]`, versioned dependency profiles, and an opt-in Gemma 4 text-only recipe. Qwen3.8, Qwen Image 2.1, MOSS Image, and WAN are recognized with explicit implementation/resource/license blockers, **not silently enabled**. See **[MODEL_SUPPORT.md](MODEL_SUPPORT.md)** for the requested-model matrix and LoRA roadmap. These additions are not in the published `v0.2.0-alpha.1`.

## Native quickstart (Thunder / working host PyTorch)

```sh
modelship doctor --backend native
modelship plan Qwen/Qwen2.5-3B-Instruct --backend native --python python3 --name chat
modelship deploy Qwen/Qwen2.5-3B-Instruct --backend native --python python3 --name chat
modelship smoke --backend native --name chat
modelship stop --backend native --name chat
```

Native workers bind directly to `127.0.0.1`, reuse the base PyTorch through a private environment, and survive SSH disconnection. They do **not** auto-restart after crash/reboot. Specify `--backend native` on lifecycle commands too. Complete installation/lifecycle/API/security details: **[NATIVE.md](NATIVE.md)**. Test results and remaining GPU checks: **[VALIDATION.md](VALIDATION.md)**.

## Docker use on a GPU server

Install the CLI using the release workflow below, or build it on the server.

```sh
modelship setup                # prerequisite docs and commands; does not replace drivers
modelship doctor               # JSON: GPU UUIDs, VRAM, driver, compute, RAM, Docker/disk

# Optional; never pass secrets as command-line arguments (Bash example).
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN

modelship plan https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
modelship deploy Qwen/Qwen2.5-3B-Instruct --name chat --port 8000

modelship plan stabilityai/stable-diffusion-xl-base-1.0 --task text-to-image
modelship deploy stabilityai/stable-diffusion-xl-base-1.0 \
  --task text-to-image --name images --port 8001

# Image-to-image is a separate deployment; avoid overlapping GPU allocations.
modelship deploy stabilityai/stable-diffusion-xl-base-1.0 \
  --task image-to-image --name edits --port 8002 --gpu 1
```

Examples are real repository IDs, **not** claims of hardware validation. Check licenses and gating on Hugging Face. FLUX models often require substantially more VRAM than SDXL; this MVP does not CPU-offload them.

Common flags:

```sh
modelship plan org/model --revision COMMIT_OR_BRANCH --gpu 0 --context 4096 --json
modelship deploy org/model --yes --wait 45m
```

- Model URLs must reference the repository root. Select branches/tags/commits with `--revision`; plans resolve them to an immutable commit SHA.
- Plans list default safetensors weight candidates and their sizes. Runtimes also download tokenizer/config/supporting files. Alternate variant weights are excluded; no model substitution occurs.
- Missing weight sizes, unsupported architectures/formats, old drivers, and incompatible GPUs fail clearly.
- VRAM estimates include a text KV-cache allowance or image-pipeline overhead. Available VRAM, host RAM, and Docker disk are snapshots; competing workloads can invalidate them.
- vLLM is budgeted at 90% of the selected GPU's currently free VRAM, expressed as a fraction of total VRAM. Other workloads changing after inspection can still cause startup failures.
- `--allow-risk` overrides **only the VRAM estimate**, not driver/runtime compatibility.
- Disk/RAM budgets are advisory: loading method, filesystem, existing caches, build cache, and container image sizes affect real requirements.
- Plan output containing blockers has a nonzero exit status, including with `--json`.
- Deploy confirms before runtime installation/download/start. `--yes` is explicit noninteractive consent.
- Deploy verifies GPU access using a CUDA container, checks that exactly the selected GPU is visible, builds the image service if needed, then starts the model.
- Readiness defaults to 30 minutes. `--wait 0` returns after container creation. On timeout/interruption, the container is retained; use `status` and `logs`.
- Reusing an existing container name fails instead of replacing it.
- No Docker or driver installation runs automatically. `setup` points to official instructions and prints toolkit configuration commands.

### Thunder Compute / CDI-managed Docker servers

Prefer the **native backend above** for the current Thunder server. This section documents the experimental Docker adapter, not a recommended workaround for the observed broken library mapping.

Thunder Compute's Docker rejects `--gpus` and `--runtime=nvidia`; it requires `--device nvidia.com/gpu=all`. Do not install/reconfigure the NVIDIA runtime on that provider unless their support instructs you to.

**Experimental / not yet validated:** [Thunder's Docker documentation](https://www.thundercompute.com/docs/guides/using-docker-on-thundercompute) says nested Docker networking is not isolated and ports may be accessible even without `-p`. The current inference containers bind internally to `0.0.0.0`; publishing to `127.0.0.1` does **not** establish localhost-only access on this provider. Do not deploy the unauthenticated service there until network exposure is verified and protected or provider-specific binding is implemented. The CDI option implements GPU argument selection, not full Thunder platform compatibility. Initial server testing currently fails with missing `libnvidia-ml.so` inside the CUDA diagnostic container; host GPU discovery succeeds. GPU library injection and actual inference remain unverified.

```sh
# Verify the provider-managed GPU path first:
docker run --rm --device nvidia.com/gpu=all \
  nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi

modelship plan Qwen/Qwen2.5-3B-Instruct --gpu-mode cdi --name text-test
modelship deploy Qwen/Qwen2.5-3B-Instruct --gpu-mode cdi --name text-test
modelship smoke --name text-test
```

`--gpu-mode auto` (default) tries standard NVIDIA access first. It retries CDI **only** on the recognized Thunder Compute rejection, and only when one GPU is visible. `--gpu-mode cdi` skips the rejected attempt. `--gpu-mode nvidia` forces the standard path. Both text and image deployments use the resolved mode.

CDI's `all` alias must not grant unintended GPU access, so the MVP rejects CDI mode on hosts with multiple visible GPUs. Selecting individual CDI devices is follow-up work. Generic Docker/permission/driver failures do not trigger fallback.

### Manage

```sh
modelship status --name chat
modelship logs --name chat
modelship smoke --name chat          # real short /v1/completions request
modelship stop --name chat
modelship start --name chat
modelship uninstall --name chat --yes
```

Management only operates on containers labeled as modelship-managed. Start does not repeat planning/readiness checks. Uninstall removes the container, not downloaded model caches or runtime images.

Cache volumes: `modelship-hf-cache` (text), `modelship-diffusers-cache` (images). Different volumes prevent root/non-root ownership conflicts; Docker initializes the image cache directory with the image user's ownership. Explicitly delete these volumes only when no deployment needs them.

## APIs

On standard, isolated Docker networking, the published ports bind to **127.0.0.1** on the server. **This is not guaranteed on provider-managed nested Docker such as Thunder Compute; see its warning above.** The runtime process listens on `0.0.0.0` inside the container. These APIs have **no authentication** in this MVP. Do not expose them directly; use an SSH tunnel or add a properly authenticated TLS reverse proxy, and ensure no direct unprotected route exists. Do not trust Docker localhost publishing as your only isolation boundary on old Docker releases, shared hosts, or non-isolated provider networks.

### Text — vLLM / native Transformers

Both expose the following basic request shape. Native supports only a non-streaming subset; see `NATIVE.md`.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

Chat requires the model tokenizer to supply a compatible chat template; otherwise use `/v1/completions`. Text defaults to float16, eager execution, 4096 context, and one concurrent sequence. These favor a predictable starting point, **not optimal throughput**.

### Text-to-image — JSON/base64

```sh
curl --fail http://127.0.0.1:8001/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a tiny observatory in the mountains","width":768,"height":768,"steps":28,"seed":42}' \
  > result.json

python3 - <<'PY'
import base64, json
from pathlib import Path
data = json.loads(Path("result.json").read_text())
Path("image.png").write_bytes(base64.b64decode(data["data"][0]["b64_json"]))
PY
```

### Image-to-image — JSON/base64 (not OpenAI multipart format)

```sh
python3 - <<'PY'
import base64, json, urllib.request
from pathlib import Path
body = {
    "prompt": "paint this scene in watercolor",
    "image_b64": base64.b64encode(Path("input.png").read_bytes()).decode(),
    "strength": 0.65, "steps": 30, "width": 768, "height": 768,
}
request = urllib.request.Request(
    "http://127.0.0.1:8002/v1/images/edits",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=600) as response:
    result = json.load(response)
Path("edited.png").write_bytes(base64.b64decode(result["data"][0]["b64_json"]))
PY
```

Images: one generation at a time (503 if busy), one output, PNG base64, explicit seed returned. Width/height: 256–1024, multiples of 8; steps: 1–100; guidance: 0–20. Input images are resized to the requested dimensions. Request limit 24 MiB, decoded image limit 16 MiB / ~16 MP. The service never fetches input image URLs. Request validation errors do not echo prompts or base64 data.

FLUX: standard FluxPipeline only, bfloat16, 256-token maximum sequence setting. Use model-appropriate guidance and step values (for example, schnell differs from dev). Negative prompts are rejected by this wrapper for FLUX.

Image service: `/health`, `/docs`. Image generation is GPU-intensive; smoke-test it with the requests above, not just a health check.

## Publish a curlable release (Dokploy or GitHub)

Build immutable release assets:

```sh
VERSION=v0.2.0 sh scripts/release.sh
# dist/modelship_linux_amd64, dist/install.sh, dist/SHA256SUMS
```

Host all three over HTTPS at a versioned path, e.g. **your own** `https://deploy.example.com/v0.2.0/`, using a static web server on Dokploy, or attach them to a GitHub release. Do not publish a moving "latest" directory as if it were immutable. This repository does not yet have a public release URL.

On the fresh server:

```sh
export MODELSHIP_RELEASE_BASE=https://YOUR_HOST/v0.2.0
curl -fsSL "$MODELSHIP_RELEASE_BASE/install.sh" -o install.sh
# Inspect the script from your trusted release before running it.
sh install.sh deploy https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
```

For explicitly unattended operation:

```sh
curl -fsSL "$MODELSHIP_RELEASE_BASE/install.sh" |
  sh -s -- deploy Qwen/Qwen2.5-3B-Instruct --yes
```

For Thunder or another working host-PyTorch environment, append `--backend native --python python3` to the installer-forwarded `deploy` command; the default remains Docker.

Piping consumes stdin, so use `--yes` with that form; the inspect-then-run form retains interactive confirmation. The installer verifies the binary against SHA256SUMS, installs atomically into `~/.local/bin`, and forwards arguments. Override `MODELSHIP_INSTALL_DIR` if needed.

Checksums from the same release host detect corruption, **not a compromised host**. TLS and your release host are part of the trust boundary; signed releases and digest-locked runtime images are follow-up work. Runtime image tags/direct Python versions are pinned, but transitive packages and OS packages are not fully locked.

### Optional Dokploy static image

Deploy `Dockerfile.releases` from this Git repository in Dokploy and route an HTTPS domain to container port 80. The multi-stage Dockerfile builds the Linux binary, installer, and checksums itself; `dist/` does not need to be committed or uploaded. Its build argument `RELEASE_VERSION` defaults to `v0.2.0`; set it to a new version for each published release and deploy a fixed Git commit/tag. The service only serves static assets, not model requests or credentials. See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the walkthrough.

## Security and operational limits

- No shell evaluation of model metadata; subprocesses receive argument arrays.
- No custom Hugging Face remote code; safetensors only for model loading. This reduces, but does not eliminate, supply-chain risk.
- `HF_TOKEN` is forwarded by environment name, not included in CLI argument values or plans. Docker administrators can inspect container environments; use read-only, scoped tokens and a trusted server. Runtime libraries may write token-related state in their cache; treat caches as private.
- Standard NVIDIA mode selects one GPU UUID. CDI mode uses the provider's all-device alias only when exactly one GPU is visible, and preflight verifies visibility. Containers are not `--privileged`; capabilities are dropped and privilege escalation is disabled. Diffusers runs non-root; upstream vLLM retains its image's user.
- Docker daemon access is effectively root-equivalent. Model and runtime/container content must be trusted.
- The planner is deterministic. A future agent backend should propose structured recommendations that pass the same validator, never privileged shell commands.
- No auto-upgrade, rollback snapshot, fleet management, benchmark optimization, API auth/rate limiting, native auto-restart, or comprehensive GPU integration validation yet.

## Tests

```sh
go test -race ./...
go vet ./...
python3 -m pip install -r tests/requirements.txt  # preferably in a venv
go build -o bin/modelship-test ./cmd/modelship
MODELSHIP_TEST_BINARY="$PWD/bin/modelship-test" python3 -m unittest discover -s tests -v
sh -n install.sh scripts/release.sh
```

Go tests cover model URL validation, pinned metadata fetching, gated access, architecture selection, resource checks, generated Docker arguments, deployment preflights, and lifecycle protections. Python tests exercise installer integrity and the image HTTP contract using fake GPU pipelines. **These do not validate real CUDA inference or actual checkpoint loading.**

Optional public Hugging Face checks (metadata/configuration only, no weights):

```sh
MODELSHIP_LIVE_TEST=1 go test ./internal/modelship -run TestLiveHubMetadata -v
```

Those checks build plans for Qwen2.5-3B and SDXL against simulated GPU capacity; they are not hardware benchmarks.

Native end-to-end CLI tests have passed on Thunder/A6000 for Qwen text and SDXL generation/editing; see **[VALIDATION.md](VALIDATION.md)** for exact scope, timings, and remaining checks. Before production, validate your own models/hardware, sustained load and failures, resource usage, and exposure controls. FLUX and Docker GPU inference remain unvalidated in this run.

## Layout

```text
cmd/modelship/                 Go entrypoint
internal/modelship/
  hub.go                       Hugging Face metadata/config inspection
  hardware.go                  NVIDIA, driver, RAM, Docker/disk discovery
  plan.go                      Compatibility and resource planning
  recipes.go                   Versioned embedded recipe validation/matching
  inspect.go                   Hardware-independent metadata inspection/catalog
  deploy.go                    Containers, cache, readiness, lifecycle, text smoke
  gpu_access.go                Standard NVIDIA / Thunder Compute CDI GPU access
  native.go                    Native environment, CUDA preflight, worker lifecycle
  cli.go                       CLI/confirmation
  assets/                      Embedded image Dockerfile + FastAPI service
install.sh                     HTTPS/checksum/bootstrap
scripts/release.sh             Linux release packaging
Dockerfile.releases            Source-built static release hosting
DEPLOYMENT.md                  GitHub/Dokploy publishing and curl test walkthrough
tests/                         Bootstrap, image/text API, native CLI integration tests
NATIVE.md                      Native deployment guide
VALIDATION.md                  GPU baseline results and outstanding checks
MODEL_SUPPORT.md               New-model profiles, size/license blockers, LoRA roadmap
```
