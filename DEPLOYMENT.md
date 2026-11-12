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
7. The tracked **`releases/current`** selects `v0.3.0-alpha.4`. No version build argument is needed. Existing `RELEASE_VERSION` build arguments are ignored with a warning, so an old Dokploy value cannot accidentally relabel the new binary as the previous alpha.
8. Configure a domain, for example `models.YOUR_DOMAIN`, routed to container port **80**.
9. Point the domain's DNS record at the Dokploy host, enable HTTPS/certificate issuance, then deploy.

The optional Docker build argument `MODELSHIP_PUBLISH_ORIGIN` selects the HTTPS origin embedded in the installer (default `https://modelship.bubuntu.xyz`). It also selects the existing-version immutability-check origin; the checksum-pinned archive origin remains explicit and separate. Ensure the publish domain is reachable during builds (an absent version must return404; network errors fail closed). The release installer needs no environment override on that intended host.

The multi-stage Dockerfile compiles the CLI and creates `install.sh` plus `SHA256SUMS`; you do **not** upload or commit `dist/`. The final image is a small static nginx server, with no GPU/runtime installation.

Do not publish the same version path with different content. `scripts/build-release-site.sh` checks the existing public version's `SHA256SUMS` and refuses a differing rebuild. Unavailable/unexpected HTTP responses also fail closed. Bump `releases/current` for a new binary or installer; identical rebuilds are allowed.

The image also retains every asset listed in `releases/retained.sha256`, including the old checksum file itself. The retained versions are `v0.2.0-alpha.1`, `v0.3.0-alpha.1`, `v0.3.0-alpha.2` and `v0.3.0-alpha.3`; all three original files for each are downloaded from the live HTTPS host and verified against committed SHA-256 pins before the new image is created. Failed downloads/hash checks fail the build rather than silently dropping an old release. Old URLs remain available after the deployment.

**Operational dependency:** the default archive/immutability-check origin is `https://modelship.bubuntu.xyz`; that host must remain reachable during image builds. Keep independent release backups (the matching GitHub Release assets are one option). For a fresh host/domain migration, provide a trusted HTTPS archive and update this explicit origin deliberately. This is a small release host, not a durable object-store archive. Automatic main deployments may remain enabled, but every subsequent change to binary/installer bytes needs a new tracked release version and retention pins for the preceding version.

### Pinning a release

After CI is green and you select a version, tag the reviewed commit, for example:

```sh
git switch main
git pull --ff-only
git tag -a v0.3.0-alpha.4 -m 'Modelship v0.3.0-alpha.4'
git push origin v0.3.0-alpha.4
```

Select that tag/commit for the Dokploy build; its `releases/current` must match the tag. This is source versioning; GitHub Release assets are optional and are not automatically published by the current CI workflow.

## 3. Verify the public release files

Replace the host and version with your real values:

```sh
export MODELSHIP_RELEASE_BASE=https://modelship.bubuntu.xyz/v0.3.0-alpha.4
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
export MODELSHIP_RELEASE_BASE=https://modelship.bubuntu.xyz/v0.3.0-alpha.4
curl -fsSL "$MODELSHIP_RELEASE_BASE/install.sh" -o /tmp/modelship-install.sh
# Review the script from your trusted release before running it.
# No arguments launches guided setup in a terminal. Use version for install-only.
sh /tmp/modelship-install.sh version
export PATH="$HOME/.local/bin:$PATH"
modelship version
modelship doctor --backend native
```

`version` should match `releases/current` (`v0.3.0-alpha.4`). The installer verifies SHA256SUMS, atomically installs the binary, and forwards arguments. Use `MODELSHIP_INSTALL_DIR` to test in a different directory if you do not want to replace an existing CLI yet.

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

Native inference defaults to `127.0.0.1`; use an SSH tunnel for private access. v0.3.0-alpha.4 adds explicit, file-backed bearer auth and `--listen 0.0.0.0` for trusted HTTPS ingress. See README's native authentication section for key generation and stopped-only migration. Never enable public forwarding for an unauthenticated service. These native auth flags are not supported for Docker image/vLLM deployments.

## 6. Remaining milestones

- Validate the public HTTPS/curl path above, then repeat on a fresh supported GPU host.
- Add native restart-on-crash/reboot management (e.g. a user-service/systemd adapter).
- Add API authentication/rate limiting and formal exposure tests before public inference.
- Run sustained-load, failure/cancellation, peak-memory, FLUX, and broader hardware/model tests.
- Sign releases and lock container/transitive dependency versions more tightly.

Model compatibility and exact live test scope: `README.md`, `NATIVE.md`, `VALIDATION.md`.
