# Publish and use Modelship

Source repository: **https://github.com/0xburden/modelship** (private initially).

There are two separate servers:

```text
GitHub source → Dokploy release host (ordinary CPU server, HTTPS)
                              ↓ installer / binary / SHA256SUMS
                      NVIDIA inference server
                              ↓
                  Hugging Face model downloads
```

Dokploy only serves release files. Model weights, Hugging Face tokens, and inference stay on the GPU server. No agent API backend is required.

## 1. Check GitHub CI

Open the repository's **Actions** tab and confirm the CI workflow is green. It runs Go tests/race checks/vet, Python API and native subprocess-lifecycle tests, shell syntax checks, and Linux release packaging. Its downloadable build artifact is for testing, not a public curl endpoint.

The Git repository excludes `bin/`, `dist/`, `.env`/credential files, virtual environments, Python caches, and `validation-artifacts/`. Public model names and sanitized validation summaries are tracked; generated images and machine-readable run artifacts are local only.

The repository starts private. No open-source license has been selected. Choose an appropriate license before deliberately publishing the source. The standalone installer/binary can be served over HTTPS from Dokploy without exposing the private GitHub repo or requiring a GitHub token on inference servers.

## 2. Create the release-host application in Dokploy

1. Create an **Application** for Modelship releases.
2. Connect your GitHub account/app and grant it access to `0xburden/modelship`.
3. Select the repository and initially the `main` branch. For an immutable published release, pin the deployment to the intended Git tag/commit using your Dokploy version's source-ref controls.
4. Choose the **Dockerfile** build method.
5. Build context: repository root (`.`).
6. Dockerfile path: `Dockerfile.releases`.
7. Set the Docker **build argument** `RELEASE_VERSION` to a unique version, for example `v0.2.0-alpha.1`. It must be a build argument, not just a runtime environment variable. Default if omitted: `v0.2.0`.
8. Configure a domain, for example `models.YOUR_DOMAIN`, routed to container port **80**.
9. Point the domain's DNS record at the Dokploy host, enable HTTPS/certificate issuance, then deploy.

The multi-stage Dockerfile compiles the CLI and creates `install.sh` plus `SHA256SUMS`; you do **not** upload or commit `dist/`. The final image is a small static nginx server, with no GPU/runtime installation.

Do not publish the same version path with different content. Disable automatic rebuilds of an already published immutable release, or bump `RELEASE_VERSION` for every source change you intend to distribute. The minimal image serves the version built into that image; it does not automatically retain previous version directories across upgrades. Use durable static storage or separate retained deployments if old URLs must stay available.

### Pinning a release

After CI is green and you select a version, tag the reviewed commit, for example:

```sh
git switch main
git pull --ff-only
git tag -a v0.2.0-alpha.1 -m 'Modelship v0.2.0-alpha.1'
git push origin v0.2.0-alpha.1
```

Select that tag/commit for the Dokploy build and use the same `RELEASE_VERSION`. This is source versioning; GitHub Release assets are optional and are not automatically published by the current CI workflow.

## 3. Verify the public release files

Replace the host and version with your real values:

```sh
export MODELSHIP_RELEASE_BASE=https://models.YOUR_DOMAIN/v0.2.0-alpha.1
curl -fI "$MODELSHIP_RELEASE_BASE/install.sh"
curl -fI "$MODELSHIP_RELEASE_BASE/modelship_linux_amd64"
curl -fsSL "$MODELSHIP_RELEASE_BASE/SHA256SUMS"
```

All three should be reachable without a GitHub login or provider API key. Source can remain private; the release host endpoints are intentionally public. Only expose this static application, not unauthenticated inference APIs.

Checksums detect corruption, not a compromised release host. Signed releases and digest-locked build dependencies are future hardening work.

## 4. Test the curl installer on the GPU server

On the existing Thunder account, the locally supplied `TNR_API_TOKEN` could not see the test instance while saved CLI login could. If that mismatch is still present, connect without the override for that one command:

```sh
env -u TNR_API_TOKEN tnr connect 0
```

Inside the GPU server:

```sh
export MODELSHIP_RELEASE_BASE=https://models.YOUR_DOMAIN/v0.2.0-alpha.1
curl -fsSL "$MODELSHIP_RELEASE_BASE/install.sh" -o /tmp/modelship-install.sh
# Review the script from your trusted release before running it.
sh /tmp/modelship-install.sh version
export PATH="$HOME/.local/bin:$PATH"
modelship version
modelship doctor --backend native
```

`version` should match `RELEASE_VERSION`. The installer verifies SHA256SUMS, atomically installs the binary, and forwards arguments. Use `MODELSHIP_INSTALL_DIR` to test in a different directory if you do not want to replace an existing CLI yet.

Thunder Base already provides working PyTorch. Native mode requires existing CUDA-enabled PyTorch; the bootstrap does not install or replace GPU drivers/PyTorch. Other providers may instead use the Docker backend after installing their supported prerequisites.

## 5. Resume an existing validated service

The three test deployments on the validated Thunder instance were stopped, not removed. Environments/cache remain:

```sh
modelship start --backend native --name native-chat-smoke
modelship smoke --backend native --name native-chat-smoke
```

Other retained names: `native-images-smoke` (text-to-image, port 8001), `native-edits-smoke` (image-to-image, port 8002). Text uses port 8000. Start only the service(s) you intend to use and account for combined GPU memory.

To deploy a new model instead:

```sh
modelship plan Qwen/Qwen2.5-3B-Instruct \
  --backend native --python python3 --name my-chat --port 8000
modelship deploy Qwen/Qwen2.5-3B-Instruct \
  --backend native --python python3 --name my-chat --port 8000
```

Stop any existing service on that port first. Model repository URLs and IDs are both accepted; gated repositories need an appropriately scoped `HF_TOKEN` and accepted terms on Hugging Face. Never put tokens in the Git repo, release host, or shell command argument values.

Native inference binds directly to `127.0.0.1` and currently has **no API authentication**. Use an SSH tunnel for personal access. For shared/public serving, add a correctly authenticated TLS reverse proxy and verify there is no bypass path. Do not add public Thunder HTTP forwarding to unauthenticated service ports.

## 6. Remaining milestones

- Validate the public HTTPS/curl path above, then repeat on a fresh supported GPU host.
- Add native restart-on-crash/reboot management (e.g. a user-service/systemd adapter).
- Add API authentication/rate limiting and formal exposure tests before public inference.
- Run sustained-load, failure/cancellation, peak-memory, FLUX, and broader hardware/model tests.
- Sign releases and lock container/transitive dependency versions more tightly.

Model compatibility and exact live test scope: `README.md`, `NATIVE.md`, `VALIDATION.md`.
