# GGUF image deployment (experimental)

**Alpha.17 candidate — do not publish or merge.** The public alpha.16 installer does not include this backend. The previous CUDA build passed separate generation and editing service tests. The replacement build removes the ported Philox RNG and the embedded tokenizer tables. Public redistribution remains blocked pending notice review and GPU validation of the replacement. It does not change existing llama.cpp or Diffusers workers. The private stdio worker never opens a port and the stock upstream HTTP server is not part of this build.

## Runtime probe

- [stable-diffusion.cpp `master-1106-532d2c1`](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-1106-532d2c1), source `532d2c15c430a785a712b2d94281f04e733008a4`. Its ggml gitlink is the leejet fork at `eeda9ec48fc84a3df0bb9dcc19ae9053075a2539`. Both sources are MIT; that does not grant rights to model weights or CUDA libraries.
- No standalone CUDA Linux asset is published for that release. The upstream CUDA container is Ubuntu 24.04/CUDA 12.6.3. This probe instead checks a native Linux AMD64 build on Ubuntu 22.04/CUDA 12.8.1, SM80 plus compute80 PTX. The source archives, CMake archive and builder image are independently pinned; configure and compile are offline. CUTLASS, the HTTP server, media output extensions, RPC and dynamic backend plug-ins are excluded.
- The candidate export contains the probe, private generation worker, separate edit smoke and resident edit executables, checksums, and ELF symbol/dependency reports. It does **not** contain NVIDIA userspace libraries, host drivers, or a redistributable runtime. The build-time ABI check permits only the expected missing host `libcuda.so.1` on a GPU-less runner. It does not run a driver stub.
- `modelship-sd-probe --version` reports the linked source commit. On an authorized machine with reviewed CUDA userspace, `--cuda-check` is intended to verify exactly one visible CUDA device with a non-host 4 KiB allocation; it refuses the ggml unified-memory override. This would not establish inference compatibility, full graph placement, or peak memory.

## First model candidate: Qwen Image 2.1 (non-Turbo)

The [pinned upstream guide](https://github.com/leejet/stable-diffusion.cpp/blob/532d2c15c430a785a712b2d94281f04e733008a4/docs/models/qwen_image_2.1.md#model-files) specifies a separate diffusion transformer, VAE and Qwen3-VL encoder. The fourth file is the matching vision projector **for edit conditioning**, not a llama.cpp chat attachment. The diffusion weights and VAE remain subject to the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE); an explicit research opt-in and separate experimental consent would be required. This is not a commercial-use profile.

The three generation files and matching edit projector were downloaded, verified against these whole-file hashes, and exercised in the dedicated GPU tests described below.

| Role | Repository @ revision | File | Bytes | Published SHA-256 |
| --- | --- | --- | ---: | --- |
| Diffusion, Q4_K | `leejet/Qwen-Image-2.1-GGUF` @ `0bca76da6edd8a18211e9717454615c1bbaaf6ed` | `qwen_image_2.1-Q4_K.gguf` | 4,197,494,816 | `29f9c83c249ff0292fb2943fceddfa2319b446601866c82a4f8be062abea72c2` |
| Encoder, Q4_K_M | `Qwen/Qwen3-VL-8B-Instruct-GGUF` @ `f982a07559d4a2f6c8744d840bf6fccab30eea96` | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | 5,027,784,800 | `67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2` |
| Edit projector, F16 | same Qwen3-VL repository and revision | `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | 1,159,029,824 | `ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38` |
| VAE, BF16 | `Comfy-Org/Qwen-Image-2.1` @ `aacf31a6d48e707f352ecba714cee0680da3479b` | `vae/qwen_image_2.1_vae_bf16.safetensors` | 675,509,688 | `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9` |

Total downloads: **9,900,789,304 bytes** for generation, **11,059,819,128 bytes** with editing. These totals are neither runtime residency estimates nor minimum VRAM requirements. In particular, the diffusion GGUF v3 header is a tensor directory with 265 tensors and **zero KV entries**. The current text GGUF parser's chat architecture/metadata assumptions must not be relaxed to admit it. The draft uses the exact full-file pins above rather than the chat GGUF parser. Encoded bytes are not a prediction of runtime expansion or CUDA workspace.

## Private generation worker (protocol v1)

The worker is a child process, not a service. It uses the pinned C API and patched CUDA-only graph and weight placement. It supports a single non-Turbo image at **512×512**, 40 Euler steps, upstream model-selected schedule, text CFG 6, and an explicit unsigned 32-bit seed. No editing, adapters, sampler overrides, fetches, or request-selected model paths. The worker does not inherit an HTTP interface from upstream.

The Go supervisor verifies the full runtime and three model component hashes, exposes only the intended GPU, and passes a curated environment and a private component directory. The worker checks the reviewed local names and byte counts before opening the models; that is not a substitute for hash verification. It requires one visible CUDA device, requests eager loading with no mmap or auto-fit, and checks the loaded model family. The supervisor enforces process deadlines and holds a same-account GPU lock until the child has been reaped. Do not run it against shared or unverified files.

`candidate_preflight.py` implements the offline component checks and the NVIDIA inventory gate for `candidate_smoke.py`. The smoke launcher requires an idle, dedicated physical GPU with at least **40 GiB free** (compute capability 8.0+, driver 570.26+). This is a conservative, non-overridable test admission threshold, not a measured fit limit. A snapshot is not a reservation. The launcher does not download models or install any CUDA libraries. The weight-free tests run with `python3 -m unittest discover -s runtime/diffusion -p 'test_candidate_*.py'`.

The private protocol uses a four-byte big-endian body length followed by the body:

| Direction | Body |
| --- | --- |
| Parent → worker | Four-byte big-endian seed, then 1–1,024 UTF-8 prompt bytes (no angle brackets, parentheses, square brackets, backslashes or `BREAK` syntax; reject rather than rewrite). |
| Worker → parent | `L`, byte `1`: context loaded, **not** public or warm-up readiness. |
| Worker → parent | `I`, byte `3` or `4`, then contiguous RGB or RGBA bytes for exactly 512×512 pixels. Only sent after successful CUDA synchronization. |
| Worker → parent | `V`: invalid prompt, no inference. |
| Worker → parent | `F` or premature EOF: fatal; close pipes, reap, and do not retry the request. |

Keep one request outstanding. An invalid frame length or truncated frame closes the worker. The weight-free protocol test uses a simulated engine; it verifies acceptance, rejection, bounded response framing, and that a failed request cannot lead to a second inference. It does not establish CUDA load or generation success.

## One-shot smoke command (not a deployment)

Run only on a separately authorized, dedicated Linux AMD64 test host, using a regular user account, Python 3.10+, and compatible CUDA 12.8 userspace already available through the system loader. The candidate export does not supply those libraries. Do not modify the host drivers or copy libraries from an existing worker to make this pass.

Prepare a canonical private directory (mode `0700`) for the verified CI worker executable (mode `0500`), and a private bundle with just the three generation components listed above (mode `0600` per file; use the file basenames, no projector). Provide the expected worker SHA-256 from the matching trusted CI run, not a hash calculated from an unverified download. Choose a new output path in a private directory. With explicit research and experimental consent:

```sh
python3 runtime/diffusion/candidate_smoke.py --research --experimental \
  --worker "$WORKER" --worker-sha256 "$WORKER_SHA256" \
  --bundle "$BUNDLE" --gpu "$GPU_UUID" --output "$OUTPUT_PNG"
```

The command checks the executable and all three components, checks free VRAM and compute processes immediately before load, and sends one built-in synthetic prompt at seed 43. It checks the fixed-size reply, waits for the child to exit successfully, and atomically writes a private 512×512 PNG. A per-account GPU lock stays with the child until it exits; this is not a reservation against unrelated work. Varying the prompt, connecting to Console, installation and editing are out of scope for this one-shot command.

### First GPU result

One offline request passed on a dedicated Linux AMD64 Ubuntu 22.04 host with an NVIDIA RTX A6000 (49,140 MiB available before launch; driver 610.43.02). The worker from [CUDA build 44424659799](https://github.com/0xburden/modelship/actions/runs/44424659799) used the three pinned generation files and private, hash-verified CUDA 12.8.90 runtime and cuBLAS 12.8.4.1 libraries. No host libraries or drivers were modified.

The fixed 512×512, 40-step, seed-43 request took **25.07 seconds after context load** and produced a verified PNG of the requested red sailboat in watercolor. Whole-device VRAM usage reached **10,280 MiB in 250 ms samples** across load and generation, then returned to zero after exit. Sampling can miss short peaks. This one result does not establish a minimum VRAM requirement, and it does not lower the 40 GiB experimental admission threshold. The test files were removed from the host. This one-shot did not cover editing, larger sizes, repeated requests, or service lifecycle.

## Separate edit smoke candidate (protocol v2)

`modelship-sd-edit-candidate` is a different executable, not an upgrade to the installed generation worker. It requires the same three local model files **and** the matching F16 projector above, using the same patched CUDA placement. The launcher verifies all four complete hashes and requires an empty graphics and compute inventory and at least 40 GiB free on the selected physical GPU. No CLI deployment, public route, combined mode or LoRA is enabled by this candidate.

Use the trusted CUDA CI checksum for the edit executable and the same private host and file layout as for the generation smoke. Run on an authorized dedicated host:

```sh
python3 runtime/diffusion/candidate_edit_smoke.py --research --experimental \
  --worker "$EDIT_WORKER" --worker-sha256 "$EDIT_WORKER_SHA256" \
  --bundle "$FOUR_COMPONENT_BUNDLE" --gpu "$GPU_UUID" --output "$EDIT_PNG"
```

The program generates a fixed synthetic reference drawing and sends one instruction to turn its red mug blue while preserving the yellow lemon and composition. The length-prefixed request contains a 32-bit seed, 16-bit prompt byte count, UTF-8 prompt, and exactly 512×512 raw RGB pixels. Both integers are big-endian. EOF is required before inference; a second frame, a truncated frame and a generation-v1 handshake are rejected. The private worker passes a single typed reference tensor to the pinned C API and only returns a 512×512 image after CUDA synchronization. It accepts no input path, remote URL, encoded attachment, or resize options. A loaded context is not proof that edit conditioning works: pinned Qwen edit conditioning rejects missing vision and an empty condition aborts the request. An edit is not retried as generation.

The worker uses seed 43, 40 Euler steps and text CFG 6. Weight-free tests cover framing, the fixed reference, bundle selection, and pipe/admission failures.

The executable from [CUDA build 44464799948](https://github.com/0xburden/modelship/actions/runs/44464799948), commit `410fd61fa7462852096cadbc514251d96a4052bd`, passed this one-shot on Ubuntu 22.04 AMD64 with an RTX A6000 (49,140 MiB initially free, driver 610.43.02). Its SHA-256 was `f49c38425b6c694b1ad700b8d6a5a2725409b73211c6bcc54b20f46d6f1f5769`. With the four exact pinned files and private CUDA libraries, the 512×512 edit took **27.95 seconds after context load**. Visual inspection confirmed that the mug turned blue while the yellow lemon and composition were retained. The PNG SHA-256 was `6f0db22a45175eb226bfd26265589028c4e0451555d8a10a87efe397ce1b941a`. Whole-device VRAM usage reached **12,372 MiB across 304 samples at 250 ms** during load and editing, and returned to zero after exit and test-stage cleanup. This is not a true peak or a minimum fit measurement; the 40 GiB admission gate is unchanged. No service was started, and the test does not cover other reference images, repeated edits, or switching modes.

## Native deployment (generation only)

Use the alpha.17 CLI once released. `--diffusion` selects this exact profile (flags only, not the guided wizard). After confirmation, deployment fetches the generation or edit archive from the versioned `runtime-sd-532d2c1-u2204-r1` GitHub release. The CLI checks archive size and SHA-256, checks each executable against the original GPU-tested size and SHA-256, and installs only those two executables and their third-party notices. No CI artifact lookup, mirrors, model weights, NVIDIA shared libraries or host libraries are included in the archive.

For an offline runtime source, use `--diffusion-runtime-dir DIR` instead of the archive download. It must contain the exact reviewed worker/probe pair in a canonical, user-owned directory (`0700`, files `0500`, no links; protected ancestors). This option skips only the runtime archive download, not any missing CUDA wheels or model files. There is no fallback from a failed download or rejected local pair.

With a private bearer key file, preview on the target machine:

```sh
modelship plan leejet/Qwen-Image-2.1-GGUF \
  --revision 0bca76da6edd8a18211e9717454615c1bbaaf6ed \
  --backend native --task text-to-image --name qwen-gguf-image --port 8001 \
  --diffusion --api-key-file "$KEY_FILE" \
  --experimental --research
```

`plan` does not fetch dependencies or weights. On a separately authorized, dedicated host, `deploy` with the same options asks for confirmation. It installs the pinned runtime and downloads and verifies two pinned NVIDIA wheels into private deployment state. It extracts only their three reviewed CUDA libraries, probes CUDA0 before fetching weights, and uses a separate verified model cache. It never installs drivers, changes global CUDA or Python packages, or edits an existing worker. Requires Linux AMD64, a compatible glibc/libstdc++ userspace, driver 570.26+, compute capability 8.0+, and an idle GPU with 40 GiB free and no graphics or compute processes, with no risk override. The idle check is a snapshot and the lock coordinates only Modelship diffusion processes under the same account. Start rechecks local files, CUDA and fresh GPU admission without re-downloading. Use `status`, `logs`, `start`, `stop` or `uninstall` with `--backend native --name qwen-gguf-image`.

The default listener is loopback, and bearer auth is required even there. Use a trusted TLS ingress or SSH tunnel for remote traffic. Authenticated `GET /health` becomes ready only after a fixed startup generation synchronizes, unlike the child's earlier context-loaded signal. `GET /v1/models` and `GET /v1/capabilities` describe the fixed profile. `POST /v1/images/generations` takes a literal `prompt` (up to 1,024 UTF-8 bytes) and optional unsigned 32-bit `seed`. Values for `width`, `height`, `steps`, and `guidance_scale`, when supplied, must match `512`, `512`, `40`, and `6` respectively. Returns the seed and base64 PNG. Unsupported options are rejected; editing returns 409. Only one request runs at a time. Disconnecting does not release the slot early. Protocol failure or timeout causes the private child to be terminated and reaped without retry. A Linux parent-death signal and an inherited deployment lock prevent cleanup while an orphaned child still holds the lock.

### Integrated GPU result

The draft CLI at `615e4edc8e10ebb05d2e11c510c587234c9e3465` passed a generation lifecycle test on a separate, fresh Ubuntu 22.04 AMD64 host with an RTX A6000 (49,140 MiB initially free, driver 610.43.02). `plan` left the private weight cache empty. `deploy` used the pinned runtime pair and components, and the authenticated loopback service reached readiness after warm-up. Unauthenticated health returned 401, authenticated discovery described the fixed profile, editing returned 409, an unsupported output size returned 422, and an overlapping request returned 503 without preventing the active generation from completing.

Two distinct prompts returned valid 512×512 PNGs in **24.29 s** and **25.25 s**. The images visibly matched the requested subjects: a red sailboat in watercolor and a yellow lemon on a blue plate. After `stop` and `start`, the supervisor PID and run ID changed, the three cached model files were unchanged, and readiness returned in **106.04 s**. Repeating the first request at seed 43 returned a byte-identical PNG in **24.19 s** on this host. Whole-device VRAM reached **10,280 MiB in 331 samples** taken at roughly 250 ms intervals during the API and restart checks, not including the first deployment. The GPU returned to 0 MiB used after uninstall. These are scoped observations, not a minimum fit claim or a cross-host determinism guarantee. Live disconnect and forced-failure paths have weight-free tests only.

## Edit-only deployment (resident protocol v3)

The one-shot protocol v2 executable is **not** the service binary. The separate resident `modelship-sd-edit-worker` accepts multiple sequential frames with the same layout, without waiting for EOF after each frame. The supervisor requires a v3 handshake and a completed, synchronized edit of the synthetic red-mug fixture in that same child before reporting ready. It never treats an edit as generation or restarts a failed request.

Select `--task image-to-image` explicitly with the same repository, immutable revision, bearer key and separate consent flags as above. Use a new deployment name and an available port. The edit profile fetches its own archive with the exact edit worker/probe pair from [CUDA build 44470677936](https://github.com/0xburden/modelship/actions/runs/44470677936); the generation profile still uses its original pair. The edit plan uses a separate private cache for all four artifacts, including the F16 projector. The default `auto` task still selects generation. The 40 GiB free and idle GPU admission gate is unchanged.

| Edit runtime executable | Bytes | SHA-256 |
| --- | ---: | --- |
| `modelship-sd-edit-worker` | 115,614,808 | `519b0f987681f72da95d75df2d40e801cb23490b5a840d417510399fedabdaa2` |
| `modelship-sd-probe` | 72,731,640 | `e5f7357f4e920d336d04215c71cd605e37459de62f23ae5735a5724701352c3e` |

Once ready, authenticated discovery reports `image-to-image` and `image_edit` limits only. `POST /v1/images/edits` accepts the same fixed output settings and optional seed as generation, plus `image_b64`: one canonical base64-encoded, opaque, static 512×512 PNG or JPEG of at most 2 MiB before base64. No input resizing, transparency flattening, external image URLs, masks or adapters. The worker receives only RGB pixels and the prompt/seed. The result format is the same base64 PNG with the actual seed. The generation route returns 409. A second edit while the slot is held returns 503; disconnect does not release it before synchronization. Unsupported requests do not run inference.

### Integrated GPU result

The draft CLI at `1f50ce010a4dc523b85fdfc519439bc08f140be4` passed the edit lifecycle on Ubuntu 22.04 AMD64 with an RTX A6000 (49,140 MiB initially free, driver 610.43.02), using the pinned resident runtime pair above. Preview left the private weight cache empty. The loopback worker reached authenticated readiness after the startup edit. The test verified unauthenticated requests return 401, discovery advertises edit only, generation returns 409, invalid sizes and reference images return 422, and overlapping edits return 503 without preventing the first edit from completing.

Two successive 512×512 edits took **27.12 s** each, using a synthetic red mug and yellow lemon and a horizontally flipped version of that reference. In both results, the mug turned blue while its orientation, the lemon, and the composition were preserved. Restart took **105.92 s** with unchanged cache files and a new PID and run ID. Repeating the first request at seed 43 returned a byte-identical PNG in **27.20 s** on this host. Whole-device VRAM reached **12,468 MiB across 893 samples** taken at roughly 250 ms intervals during deployment, edits, and restart. The test deployment and its isolated cache were removed; the GPU returned to **0 MiB used**. These results are not a peak or minimum fit measurement, do not cover arbitrary reference images, and do not change the 40 GiB admission gate. Disconnect and forced-failure behavior remain covered by weight-free tests only.

## Placement and remaining validation gates

1. **Enforce placement, not just device selection.** The [upstream graph scheduler](https://github.com/leejet/stable-diffusion.cpp/blob/532d2c15c430a785a712b2d94281f04e733008a4/src/core/compute_workspace.cpp#L28-L48) requires CPU registration for allocation and scheduling. `cuda-placement.cmake` patches only the exact pinned ggml files in the disposable build tree and verifies the resulting hashes. It preflights all final scheduler splits before dispatch, guards direct and planned dispatch, and blocks nonempty graphs at the CPU kernel entry point (including nested extension calls). The weight-free test links the real patched CPU backend: allocation still works, rejected compute leaves the output untouched, and a fake CUDA0 followed by an unsupported CPU split is rejected before even the first dispatch. It does not run CUDA. Upstream [`params_buffer_type_for`](https://github.com/leejet/stable-diffusion.cpp/blob/532d2c15c430a785a712b2d94281f04e733008a4/src/model_manager.cpp#L1135-L1169) can choose a host buffer even with CUDA parameters configured. `cuda-residency.cmake` checks the exact pinned model manager: reject split, disk and mismatched compute/parameter backends; reject host or nondefault buffers before allocation; verify attached buffers before reading weight payloads and when reusing loaded parameters. Persistent host mappings and disk demotion are blocked. The weight-free test covers the buffer predicate with real CPU buffers and synthetic device metadata. The private worker must request eager loading, forbid unified memory and treat unsupported graphs or storage as readiness failures. This is enforcement code, **not** a measurement of where the Qwen weights actually load or whether an inference graph runs. Ordinary host tokenization, image decoding and response encoding are not model offload. Don't reuse Diffusers memory or quality claims.
2. **Keep the C API behind a private worker boundary.** The draft Go gateway uses the private worker and does not forward HTTP or expose upstream loaders. Weight-free tests exercise warm-up, failure, disconnect and process shutdown. Dedicated A6000 service tests covered readiness, repeated requests, busy rejection, and restart independently for generation and editing.
3. **Keep promotion and additional capabilities separate.** Durable runtime distribution has not been published. The fixed-size results do not establish peak memory, larger-size fit, or output equivalence to Diffusers. Combined generation/editing and LoRA remain unimplemented for this backend.

FLUX.1-schnell is a separate generation-only option if permissive licensing is the priority. It would require its own verified diffusion/encoder/VAE bundle and tests; it is not an edit fallback or a substitute for Qwen.

## Release operator checklist

1. **Stop until cleared:** the current Go pins and local archive tool still describe the historical binaries below, not the replacement. After auditing and validating the replacement on GPU, update the pins and package only the exact newly tested pairs using Python 3.11+:
   `python3 scripts/package-diffusion-runtime.py generation /path/to/generation-pair /path/to/output` and
   `python3 scripts/package-diffusion-runtime.py edit /path/to/edit-pair /path/to/output`.
2. Compare archive hashes and byte lengths with `internal/modelship/diffusion_runtime_release.go`. Existing archives must never be replaced. After release approval, publish the staged runtime release **before** merging the CLI PR. Verify both public artifact URLs and full pinned hashes before merging, so the auto-deployed CLI never points to unpublished dependencies.
3. After the CLI site deploys, verify all retained/current assets and the uncached latest installer. Do not deploy a user model as part of the installer smoke.

### Runtime redistribution hold

Do not upload the historical binaries to a runtime release. The pinned source credits `src/core/rng_philox.hpp` as a port from [AUTOMATIC1111 `5ef669de`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/5ef669de080814067961f28357256e8fe27544f4/modules/rng_philox.py), an AGPLv3 project. Both historically tested workers contained that implementation. The top-level MIT notices do not resolve its terms.

The replacement candidate applies source-digest-checked CMake transforms. It removes that header from the disposable tree; the private worker explicitly selects `std_default` noise for the model and sampler. The build checks all ELF symbol tables for the removed implementation. This is host-side noise generation only; CUDA0 tensor residency and graph-dispatch guards are unchanged. Seeds are reproducible only within a validated runtime: the same seed will generally give different output from the historical binaries.

`external-vocab.cmake` replaces the exact built-in vocabulary translation unit with code that reads `tokenizer.ggml.merges` from the existing full-file-verified Qwen3-VL encoder. No extra file or tensor download is required. A metadata-only range check on the pinned encoder confirmed all 151,387 merge lines match the prior embedded data in order, byte-for-byte with the final LF (1,671,839 bytes, SHA-256 `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3`). The Qwen BPE implementation and special-token order are unchanged. Missing or invalid data fails startup; other embedded vocabularies are not compiled in and have no fallback. The ELF audit checks for all removed table symbols.

The replacement also removes the upstream AUTOMATIC1111-derived prompt-attention parser: the library passes through one unit-weight segment, while the worker retains its existing refusal of control/weight syntax. CCCL builds with `CCCL_DISABLE_NVTX` so CUB stays enabled without compiling the Toolkit NVTX implementation. CI checks compiler flags, the exact builder's CUB header hashes, and all four ELF symbol tables.

The CUDA Toolkit's compiled-in headers and other linked code still need a complete notice/provenance inventory; separately downloaded NVIDIA wheels do not cover code in the executable. `releases/diffusion/THIRD_PARTY_NOTICES.txt` remains preliminary. The build workflow retains only hashes and audit evidence, not executable artifacts, while redistribution is on hold. After clearance, independently verify the exact replacement CUDA binaries on an authorized A6000 (generation, editing, readiness, repeat seed and restart/cache reuse), then repin the binaries and archives. No runtime release has been created. Alpha.16 remains public.
