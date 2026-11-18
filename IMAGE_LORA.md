# Qwen Image 2.1 image LoRA

Modelship supports **one reviewed, optional adapter** on the native BF16
`Qwen/Qwen-Image-2.1` checkpoint at `d26bb61231c349cf6b7896fa83353113880e1ba3`,
with Diffusers 0.42.1. Generation-only, editing-only and combined image workers
are supported, provided the saved native configuration is schema 9 or 10 and
matches the reviewed recipe. This does not add editing to a generation-only
worker or vice versa. Docker, SDXL, other checkpoints/adapters and stacking are
not supported by this command.

| Adapter | Reviewed artifact |
| --- | --- |
| Repository | `SEEDRAAI/Qwen-Image-2.1-LoRA-Polaroid` |
| Revision | `6bc9554265bd638a1167682c04a1802eb3c57ac0` |
| File | `p0laroid_qwen_v1.safetensors` (167,830,408 bytes) |
| SHA-256 | `d0832a8ecd06b6d3c4292427e6901c0e7684720518e3a4774267e84a8a39d49c` |
| Trigger | `p0laroid`; the adapter card suggests 0.6–0.8 as a starting range |
| License | Qwen non-commercial research/evaluation; requires explicit adapter acknowledgment |

## Update an existing deployment

First upgrade the CLI. Preview does not download anything or touch a worker:

```sh
modelship lora add --name modelship https://huggingface.co/SEEDRAAI/Qwen-Image-2.1-LoRA-Polaroid
```

When you're ready for downtime, run these **in order**. Do not continue past a
failed command without checking the error:

```sh
modelship stop --backend native --name modelship
modelship lora add --name modelship --research --yes https://huggingface.co/SEEDRAAI/Qwen-Image-2.1-LoRA-Polaroid
modelship start --backend native --name modelship
```

`lora add` refuses to interrupt a running worker. It validates the stopped
worker's saved recipe/GPU, checks free and total weight placement and disk,
then downloads only the pinned adapter file (without requesting credentials).
It verifies the SHA-256 and stages separate worker assets before atomically
updating the configuration to schema 11. The original runtime files, base
cache, venv, port and authentication are retained. The exact artifact size is
added to the hard resident-memory budget; the additional 1024 MiB of temporary
workspace is an **unmeasured heuristic**. `--allow-risk` acknowledges only the
heuristic, never insufficient physical memory/disk or incompatible metadata.
Loaded adapter weights occupy VRAM even at strength 0.

If the adapter isn't needed or start fails, stop the worker if necessary,
remove the adapter configuration and start the original base runtime:

```sh
modelship stop --backend native --name modelship
modelship lora remove --name modelship --yes
modelship start --backend native --name modelship
```

This rollback retains the private staged artifact for reuse. It does not
retry automatically or change the base checkpoint. An older CLI must not be
used on schema 11; remove the adapter with a LoRA-aware CLI before downgrading.

## Generation and edit API

An authenticated `GET /v1/capabilities` on a ready adapter worker returns its
base model, revision, `run_id`, and `image_lora` (adapter identity, trigger,
supported tasks and numeric bounds). Keep using `/v1/images/generations` or
`/v1/images/edits`. Add these optional fields together to an existing request:

```json
{"lora_strength": 0.75, "expected_run_id": "<run_id from capabilities>"}
```

Omit both to use the base model. An explicit `0` also uses the base model, but
still requires a matching run. A stale run replies 409 before inference, never
falling back to the base model. `lora_strength` is a number from 0 to 1 and is
**not** img2img denoise `strength` (which this Qwen pipeline does not support).
For example, prefix your prompt with `p0laroid` and try 0.75. Prompt text is never
rewritten; normal dimensions (256–1024, multiples of 32) and edit requirements
stay the same. In Console, the optional field appears only for matching,
ready native image capabilities and the selected operation.

The bounded runtime verifies all 448 BF16 tensors (224 rank-32 pairs) and
linear targets, applies unfused deltas in the existing exclusive GPU job, and
reverts to zero on success or failure. It never modifies base model weights,
loads repository Python or installs new packages. The full-image GPU quality,
latency and peak VRAM of this adapter have not been measured.
