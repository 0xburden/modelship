# modelship

A GPU-first model deployment CLI: inspect a Hugging Face repository and the local NVIDIA server, review a plan, then deploy a pinned model revision through Docker or a native Python worker.

**Early MVP, not a universal model converter or production control plane.** Go handles orchestration; vLLM (Docker), Transformers (native text), and a small Python/Diffusers service perform inference. No agent API or backend is required.

## Guided setup (v0.3.0-alpha.3+)

On a supported NVIDIA server with the prerequisites already installed:

```sh
curl -fsSL https://modelship.bubuntu.xyz/v0.3.0-alpha.3/install.sh -o install.sh
# Inspect the downloaded script from your trusted release before running it.
sh install.sh
```

The published installer knows its immutable release URL; **no environment variable or model flags are needed**. It verifies/installs the CLI first, then launches guided setup when stdin/stdout are terminals. You can also run `~/.local/bin/modelship launch`, or simply `modelship` from a terminal after adding `~/.local/bin` to PATH.

The wizard asks for the HF model page/revision, resolves it to an immutable commit, reads bounded headers, and offers the recipe's supported backend/task choices. Single-choice menus select themselves. It inspects GPUs, uses sensible ports (text8000/images8001/video18032), and collects Python/context/API-access/key-file settings. Model precision/loading policy stays fixed by the reviewed recipe: no inferred quantization/offload. Native mode checks existing CUDA PyTorch without installing or replacing it.

You then review the exact shared deployment plan and a shell-quoted equivalent command. **No model tensor payload download, pip installation, deployment state or API key generation happens before final confirmation.** A tiny CUDA preflight and metadata/cache inspection are read-only setup checks. New keys are planned file references until approval, generated privately afterward, and never printed. Before final confirmation, cancellation/EOF/Ctrl-C does not deploy; existing keys/deployments are never overwritten/stopped implicitly. WAN conversion, experimental use, research-only licensing, custom-code consent and runtime-memory heuristic override each require their own explicit approval where applicable. Physical weights-fit/disk/policy blockers cannot be overridden.

The wizard is just a UI over the existing inspector/planner/deployer; flag-based commands remain the automation interface. It deploys exactly the pinned plan shown, while launch still refreshes GPU-fit checks. After confirmation, leaving/interruption does not automatically stop a retained worker. Readiness, logs and deployment failures retain their existing lifecycle behavior; if a confirmed deployment fails, a newly generated key is retained privately rather than silently deleted. Downloading/installing the CLI itself is separate from deployment confirmation.

**Terminal requirement:** download-and-run is the supported guided path. `curl | sh`, CI, redirected input/output and SSH without a TTY do not start interactive prompts. With no installer arguments those environments install only, print the version and show how to launch later. An explicit `launch` without terminal input/output fails fast; use `deploy MODEL ... --yes` or request an SSH TTY for interactive use. `--help` and `version` never prompt. No secrets are requested as terminal text; use `HF_TOKEN` in the environment for gated repositories and private file paths for API auth.

This first iteration does not add new model adapters, capability-driven generation forms or soft generation-limit changes; those remain separate work.

## Supported scope

- Linux **amd64**, NVIDIA; one selected GPU per deployment.
- **Docker backend (default):** local daemon with standard NVIDIA Container Toolkit or provider-managed CDI, conservative CUDA 12.8 driver minimum **570.26**. CDI mode currently requires exactly one visible GPU and is not fully validated on Thunder.
- **Native backend:** working host CUDA-enabled PyTorch plus Python/venv. No Docker/toolkit installation or hard-coded CUDA driver minimum; actual CUDA computation is checked. See **[NATIVE.md](NATIVE.md)**. This is the recommended path for the current Thunder server.
- Text: unquantized **safetensors** repositories with explicitly recognized architectures: Llama, Mistral, Qwen2, Qwen3, Gemma, Gemma2, Phi3 causal LMs. Exact architecture names and dependency profiles are in `internal/modelship/assets/recipes.json`.
- Images: standard Diffusers repositories whose `model_index.json` declares **StableDiffusionPipeline**, **StableDiffusionXLPipeline**, or **FluxPipeline**. Both text-to-image and image-to-image are selectable; a deployment serves **one task**.
- Text requires compute capability >=7.5 in Docker / >=7.0 natively; SD/SDXL >=7.0; FLUX >=8.0 (bfloat16).
- **Experimental/native:** Gemma 4 and Qwen3.8 text-only; Qwen Image 2.1 generation/editing; explicitly approved/hash-pinned MOSS images; bounded WAN video via an explicitly selected official Diffusers conversion. These need `--experimental` and model-specific resource/license/code prerequisites. See [MODEL_SUPPORT.md](MODEL_SUPPORT.md) for validation status; unvalidated adapters are not GPU-support guarantees.

Not yet supported: GGUF, single-file image checkpoints, variant-only image repositories, adapters/LoRAs, AWQ/GPTQ/bitsandbytes, arbitrary remote model code, original WAN PTH checkpoints, FLUX.2, multi-GPU sharding, automatic quantization/offload, or arbitrary model architectures. Matching an architecture is a compatibility filter, **not** proof that every repository variant will load.

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

## Recipe discovery

The `v0.3.0-alpha.1` release adds host-independent `modelship catalog` and `modelship inspect MODEL [--json]`, isolated dependency profiles, and experimental adapters for the requested model families. Weights-alone VRAM is checked from bounded safetensors headers, separately from download bytes and runtime heuristics. Research-only licensing, reviewed repository-code consent, explicit WAN format selection, and hardware limits remain enforced; nothing is silently substituted or quantized. See **[MODEL_SUPPORT.md](MODEL_SUPPORT.md)** for the requested-model matrix and LoRA roadmap. These additions are not in the published `v0.2.0-alpha.1`.

## Native quickstart (Thunder / working host PyTorch)

```sh
modelship doctor --backend native
modelship plan Qwen/Qwen2.5-3B-Instruct --backend native --python python3 --name chat
modelship deploy Qwen/Qwen2.5-3B-Instruct --backend native --python python3 --name chat
modelship smoke --backend native --name chat
modelship stop --backend native --name chat
```

Native workers default to `127.0.0.1` (network listeners require explicit bearer auth), reuse the base PyTorch through a private environment, and survive SSH disconnection. They do **not** auto-restart after crash/reboot. Specify `--backend native` on lifecycle commands too. Complete installation/lifecycle/API/security details: **[NATIVE.md](NATIVE.md)**. Test results and remaining GPU checks: **[VALIDATION.md](VALIDATION.md)**.

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
- Plans list default safetensors weight candidates and their sizes, validate bounded tensor headers/shard manifests at the pinned SHA, and compare requested-dtype minimum residency with total **and free** selected-GPU VRAM. Runtimes also download tokenizer/config/supporting files. Alternate variant weights are excluded; no model substitution occurs.
- Missing weight sizes, unsupported architectures/formats, old drivers, and incompatible GPUs fail clearly.
- VRAM estimates include a text KV-cache allowance or image-pipeline overhead. Available VRAM, host RAM, and Docker disk are snapshots; competing workloads can invalidate them.
- vLLM is budgeted at 90% of the selected GPU's currently free VRAM, expressed as a fraction of total VRAM. Other workloads changing after inspection can still cause startup failures.
- `--allow-risk` overrides **only the runtime VRAM heuristic**, never weights-alone total/free fit, missing/invalid metadata, license/code consent, native cache disk or driver/runtime compatibility.
- Native missing-weight downloads plus headroom must fit the inspected HF cache filesystem; exact pinned cached weights are accounted for and the cache path is persisted. Docker disk and host-RAM budgets remain advisory. Loading method, dependencies/state filesystem, build cache and supporting files affect real requirements.
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

On standard, isolated Docker networking, the published ports bind to **127.0.0.1** on the server. **This is not guaranteed on provider-managed nested Docker such as Thunder Compute; see its warning above.** The runtime process listens on `0.0.0.0` inside the container. These **Docker APIs have no authentication**; the native backend has a separate opt-in bearer-auth boundary below. Do not expose them directly; use an SSH tunnel or add a properly authenticated TLS reverse proxy, and ensure no direct unprotected route exists. Do not trust Docker localhost publishing as your only isolation boundary on old Docker releases, shared hosts, or non-isolated provider networks.

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
VERSION=v0.3.0-alpha.3 sh scripts/release.sh
# dist/modelship_linux_amd64, dist/install.sh, dist/SHA256SUMS
```

Release packaging embeds `MODELSHIP_PUBLISH_ORIGIN/VERSION` in the served installer (default origin `https://modelship.bubuntu.xyz`); set `MODELSHIP_PUBLISH_ORIGIN=https://YOUR_HOST` for another HTTPS origin. `MODELSHIP_RELEASE_BASE` remains an optional install-time override. A differing explicit override is announced without echoing its value; older selected CLIs fall back to install-only if they lack guided launch. The source `install.sh` is a template and still requires that override if run directly before packaging. Host all three over HTTPS at a versioned path, e.g. **your own** `https://deploy.example.com/v0.2.0/`, using a static web server on Dokploy, or attach them to a GitHub release. Do not publish a moving "latest" directory as if it were immutable. Public releases are hosted at `https://modelship.bubuntu.xyz/`; the current source-controlled version is in `releases/current`.

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

Deploy `Dockerfile.releases` from this Git repository in Dokploy and route an HTTPS domain to container port 80. The multi-stage Dockerfile builds the Linux binary, installer, and checksums itself; `dist/` does not need to be committed or uploaded. The tracked `releases/current` file selects the version (`v0.3.0-alpha.3`); the old `RELEASE_VERSION` build argument is ignored with a warning. The build preserves checksum-pinned prior releases and refuses to overwrite an existing version with different bytes. Deploy a reviewed commit/tag. The service only serves static assets, not model requests or credentials. See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the walkthrough.

## Install WAN from the versioned release

On Linux x86_64 with an idle A10080GB-class GPU, Python>=3.11 and existing working CUDA PyTorch>=2.9, plus sufficient free disk (roughly150GB recommended):

```sh
export MODELSHIP_RELEASE_BASE=https://modelship.bubuntu.xyz/v0.3.0-alpha.3
curl -fsSL "$MODELSHIP_RELEASE_BASE/install.sh" -o /tmp/modelship-install.sh
# Inspect this trusted script before running it.
sh /tmp/modelship-install.sh deploy Wan-AI/Wan2.2-T2V-A14B \
  --backend native --python python3 --experimental \
  --weights-repo Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7 \
  --task text-to-video --name wan-video --port 18032 \
  --allow-risk --wait 90m --yes
```

`--allow-risk` acknowledges the conservative activation estimate on the tested80GB configuration. It cannot bypass hard weights/free-VRAM/disk checks. No host PyTorch/driver replacement, quantization or offload is enabled. The API binds to localhost; use an SSH/private tunnel for the console, never publicly expose this unauthenticated model service. The new binary replaces `~/.local/bin/modelship` unless `MODELSHIP_INSTALL_DIR` is set.

## Native API authentication (v0.3.0-alpha.2+)

Native listeners remain **127.0.0.1 by default**. To use Thunder's generated HTTPS URL, explicitly enable a network listener with bearer authentication. This is native-only; these flags are rejected for Docker deployments. API keys are independent of `HF_TOKEN` and the console login password.

For an **existing deployment**, using the updated CLI:

```sh
# Creates a random 256-bit key in a private file; never prints its value.
modelship auth-keygen --file "$HOME/.config/modelship/wan.key"

# Download any videos you want to keep first; stopping invalidates job IDs.
modelship stop --backend native --name wan-video
modelship configure --backend native --name wan-video \
  --listen 0.0.0.0 --api-key-file "$HOME/.config/modelship/wan.key"
modelship start --backend native --name wan-video
modelship status --backend native --name wan-video
```

`configure` refuses to touch a running worker. It updates only access metadata, the entrypoint and the tiny auth module: **no model download, pip install, host PyTorch/driver change, inference-module rewrite or automatic restart**. The same pinned model/cache/venv, GPU and memory checks are retained. `start` still loads weights and refreshes VRAM fit. New deployments can pass the same `--listen`/`--api-key-file` flags to `plan`/`deploy`.

Forward the selected port with Thunder as usual and add the resulting `https://INSTANCE-18032.thundercompute.net` base URL to Modelship Console. Put the file's **key value** in the console's existing **Upstream bearer key** field using a secure local/clipboard workflow; never paste secrets into a chat or commit them. The console retains this value only in tab memory and sends it as `Authorization: Bearer <key>`.

All native HTTP paths are protected when enabled: health, model discovery, inference, video status/content, docs/OpenAPI, and unknown routes. Missing/malformed/wrong credentials return401 before body parsing or inference. Query-string/cookie keys are not accepted. Authenticated responses use `Cache-Control: no-store`; CLI readiness/status/text smoke authenticate automatically and refuse redirects. An ordinary browser/curl request without a header returning401 is **expected**.

Keys must be owned, regular **0600** files with32–128 base64url characters; the generator emits43 characters plus a newline and refuses overwrites. Generation needs a private parent directory (created0700 if absent). Symlinks, FIFOs, insecure permissions and invalid keys fail closed before weight loading. Only the absolute path—not the key value—is stored in schema4 metadata or the worker environment/argv. Older CLIs reject schema4; keep using the updated CLI after migration. Legacy schemas1–3 without access settings still load and remain loopback-only.

**Rotation requires a stop/configure/start** with a newly generated key file; a running worker holds the startup key digest, so changing/deleting the file does not revoke that active key immediately. A missing key prevents a new start, but does not prevent status/stop/uninstall. Uninstall does not remove the default external key file.

Use **HTTPS ingress or a private tunnel**—the native worker serves HTTP and bearer auth does not encrypt it. This is one shared API key, not accounts/roles, per-user quotas or distributed rate limiting. No running deployment is exposed merely by installing the new CLI.

## Security and operational limits

- No shell evaluation of model metadata; subprocesses receive argument arrays.
- Safetensors-only model loading; no automatic repository-code execution. The one reviewed MOSS revision requires separate explicit `--trust-model-code` approval and three SHA-256 checks before local loading. This runs as your user, not in a sandbox. All other loaders keep `trust_remote_code=False`. This reduces, but does not eliminate, supply-chain risk.
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

Native end-to-end CLI tests have passed on Thunder/A6000 for Qwen text, Gemma 4 text, SDXL generation/editing, Qwen Image 2.1 generation/editing, explicitly approved pinned MOSS generation, and bounded WAN video/H.264 export on a separate A100 80 GB; see **[VALIDATION.md](VALIDATION.md)** for exact scope, timings, and remaining checks. Before production, validate your own models/hardware, sustained load and failures, resource usage, and exposure controls. FLUX and Docker GPU inference remain unvalidated in this run.

## Layout

```text
cmd/modelship/                 Go entrypoint
internal/modelship/
  hub.go                       Hugging Face metadata/config inspection
  hardware.go                  NVIDIA, driver, RAM, Docker/disk discovery
  plan.go                      Compatibility and resource planning
  weight_metadata.go           Bounded safetensors header/shard validation
  weight_runtime.go            Fresh total/free VRAM checks before launch
  cache.go                     Native cache location, cached bytes, disk space
  recipe_policy.go             Research/code consent and explicit WAN selection
  recipes.go                   Versioned embedded recipe validation/matching
  inspect.go                   Hardware-independent metadata inspection/catalog
  deploy.go                    Containers, cache, readiness, lifecycle, text smoke
  gpu_access.go                Standard NVIDIA / Thunder Compute CDI GPU access
  native.go                    Native environment, CUDA preflight, worker lifecycle
  cli.go                       CLI/confirmation
  assets/                      Embedded recipes, text/image/video APIs, reviewed-code gate
install.sh                     HTTPS/checksum/bootstrap
scripts/release.sh             Linux release packaging
Dockerfile.releases            Source-built static release hosting
DEPLOYMENT.md                  GitHub/Dokploy publishing and curl test walkthrough
tests/                         Bootstrap, image/text API, native CLI integration tests
NATIVE.md                      Native deployment guide
VALIDATION.md                  GPU baseline results and outstanding checks
MODEL_SUPPORT.md               New-model profiles, size/license blockers, LoRA roadmap
```
