# Publish and use Modelship

Source repository: **https://github.com/0xburden/modelship** (public; source license has not yet been selected).

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

The repository is public after a review of reachable Git history and GitHub descriptions/comments. No open-source license has been selected; visibility does not grant additional usage/redistribution rights. Standalone installer/binary and reviewed runtime release assets are downloadable without a GitHub token on inference servers.

## 2. Create the release-host application in Dokploy

1. Create an **Application** for Modelship releases.
2. Connect your GitHub account/app and grant it access to `0xburden/modelship`.
3. Select the repository and initially the `main` branch. For an immutable published release, pin the deployment to the intended Git tag/commit using your Dokploy version's source-ref controls.
4. Choose the **Dockerfile** build method.
5. Build context: repository root (`.`).
6. Dockerfile path: `Dockerfile.releases`.
7. The tracked **`releases/current`** selects `v0.3.0-alpha.7`. No version build argument is needed. Existing `RELEASE_VERSION` build arguments are ignored with a warning, so an old Dokploy value cannot accidentally relabel the new binary as the previous alpha.
8. Configure a domain, for example `models.YOUR_DOMAIN`, routed to container port **80**.
9. Point the domain's DNS record at the Dokploy host, enable HTTPS/certificate issuance, then deploy.

The optional Docker build argument `MODELSHIP_PUBLISH_ORIGIN` selects the HTTPS origin embedded in the installer (default `https://modelship.bubuntu.xyz`). It also selects the existing-version immutability-check origin; the checksum-pinned archive origin remains explicit and separate. Ensure the publish domain is reachable during builds (an absent version must return404; network errors fail closed). The release installer needs no environment override on that intended host.

The multi-stage Dockerfile compiles the CLI and creates `install.sh` plus `SHA256SUMS`; you do **not** upload or commit `dist/`. The final image is a small static nginx server, with no GPU/runtime installation.

Do not publish the same version path with different content. `scripts/build-release-site.sh` checks the existing public version's `SHA256SUMS` and refuses a differing rebuild. Unavailable/unexpected HTTP responses also fail closed. Bump `releases/current` for a new binary or installer; identical rebuilds are allowed.

The image also retains every asset listed in `releases/retained.sha256`, including the old checksum file itself. The retained versions are `v0.2.0-alpha.1`, `v0.3.0-alpha.1`, `v0.3.0-alpha.2`, `v0.3.0-alpha.3`, `v0.3.0-alpha.4` and `v0.3.0-alpha.5`; all three original files for each are downloaded from the live HTTPS host and verified against committed SHA-256 pins before the new image is created. Failed downloads/hash checks fail the build rather than silently dropping an old release. Old URLs remain available after the deployment.

**Operational dependency:** the default archive/immutability-check origin is `https://modelship.bubuntu.xyz`; that host must remain reachable during image builds. Keep independent release backups (the matching GitHub Release assets are one option). For a fresh host/domain migration, provide a trusted HTTPS archive and update this explicit origin deliberately. This is a small release host, not a durable object-store archive. Automatic main deployments may remain enabled, but every subsequent change to binary/installer bytes needs a new tracked release version and retention pins for the preceding version.

### Stable installer entry point

`https://modelship.bubuntu.xyz/latest/install.sh` tracks the version selected by `releases/current`. It is a byte-for-byte copy of that version's installer, served with `Cache-Control: no-store, max-age=0` (including errors). It is not included in immutable historical pins. Only the entry point moves: the downloaded script's baked binary/checksum URLs remain versioned, avoiding cross-release races. All `/v.../` assets retain their existing bytes and behavior. This alias-only change does not change the CLI, installer bytes or release version; the normal identical-rebuild immutability guard still applies.

```sh
curl -fsSL https://modelship.bubuntu.xyz/latest/install.sh -o install.sh
# Review it, then launch guided setup from a terminal:
sh install.sh
# Or: sh install.sh version  # install/update the CLI only; no GPU deployment
```

Use explicit version URLs for reproducible deployment scripts. The nginx configuration is source-controlled in `releases/nginx.conf`; do not cache/override `/latest/install.sh` at an ingress/CDN. After deploying the alias, verify its body matches the current versioned installer and that its response includes `Cache-Control: no-store`. An explicit `MODELSHIP_RELEASE_BASE` environment override still takes precedence; unset an old override if you want the downloaded installer's selected release.

For optional real-nginx regression checks against a built local container (no GPU work):

```sh
MODELSHIP_RELEASE_SITE_URL=http://127.0.0.1:18107 \
  python3 -m unittest discover -s tests -p test_release_http.py -v
```

### Pinning a release

After CI is green and you select a version, tag the reviewed commit, for example:

```sh
git switch main
git pull --ff-only
git tag -a v0.3.0-alpha.6 -m 'Modelship v0.3.0-alpha.6'
git push origin v0.3.0-alpha.6
```

Select that tag/commit for the Dokploy build; its `releases/current` must match the tag. This is source versioning; GitHub Release assets are optional and are not automatically published by the current CI workflow.

## 3. Verify the public release files

Replace the host and version with your real values:

```sh
export MODELSHIP_RELEASE_BASE=https://modelship.bubuntu.xyz/v0.3.0-alpha.6
curl -fI "$MODELSHIP_RELEASE_BASE/install.sh"
curl -fI "$MODELSHIP_RELEASE_BASE/modelship_linux_amd64"
curl -fsSL "$MODELSHIP_RELEASE_BASE/SHA256SUMS"
```

All three should be reachable without a GitHub login or provider API key. The source repository and release assets are public; no GitHub token is required for these downloads. Only expose this static application, not unauthenticated inference APIs.

Checksums detect corruption, not a compromised release host. Signed releases and digest-locked build dependencies are future hardening work.

## 4. Test the curl installer on the GPU server

On the existing Thunder account, the locally supplied `TNR_API_TOKEN` could not see the test instance while saved CLI login could. If that mismatch is still present, connect without the override for that one command:

```sh
env -u TNR_API_TOKEN tnr connect 0
```

Inside the GPU server:

```sh
export MODELSHIP_RELEASE_BASE=https://modelship.bubuntu.xyz/v0.3.0-alpha.6
curl -fsSL "$MODELSHIP_RELEASE_BASE/install.sh" -o /tmp/modelship-install.sh
# Review the script from your trusted release before running it.
# No arguments launches guided setup in a terminal. Use version for install-only.
sh /tmp/modelship-install.sh version
export PATH="$HOME/.local/bin:$PATH"
modelship version
modelship doctor --backend native
```

`version` should match `releases/current` (`v0.3.0-alpha.6`). The installer verifies SHA256SUMS, atomically installs the binary, and forwards arguments. Use `MODELSHIP_INSTALL_DIR` to test in a different directory if you do not want to replace an existing CLI yet.

Thunder Base already provides working PyTorch. Python-based native recipes require existing CUDA-enabled PyTorch; the experimental GGUF/llama.cpp adapter does not (see [GGUF.md](GGUF.md)); the bootstrap does not install or replace GPU drivers/PyTorch. Other providers may instead use the Docker backend after installing their supported prerequisites.

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

Native inference defaults to `127.0.0.1`; use an SSH tunnel for private access. v0.3.0-alpha.6 adds explicit, file-backed bearer auth and `--listen 0.0.0.0` for trusted HTTPS ingress. See README's native authentication section for key generation and stopped-only migration. Never enable public forwarding for an unauthenticated service. These native auth flags are not supported for Docker image/vLLM deployments.

## 6. Remaining milestones

- Validate the public HTTPS/curl path above, then repeat on a fresh supported GPU host.
- Add native restart-on-crash/reboot management (e.g. a user-service/systemd adapter).
- Add API authentication/rate limiting and formal exposure tests before public inference.
- Run sustained-load, failure/cancellation, peak-memory, FLUX, and broader hardware/model tests.
- Sign releases and lock container/transitive dependency versions more tightly.

Model compatibility and exact live test scope: `README.md`, `NATIVE.md`, `VALIDATION.md`.
