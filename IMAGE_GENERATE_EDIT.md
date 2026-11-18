# One native worker for image generation and editing

`image-generate-and-edit` is an opt-in task introduced in `v0.3.0-alpha.14`.
The worker loads one checkpoint, exposes both image endpoints, and runs one
GPU request at a time across them. It does not run two workers or schedule jobs.

| Model | Exact reviewed revision | Native runtime |
| --- | --- | --- |
| `Qwen/Qwen-Image-2.1` | `d26bb61231c349cf6b7896fa83353113880e1ba3` | Diffusers 0.42.1, BF16 transformer, FP32 VAE; `--experimental --research` required |
| `stabilityai/stable-diffusion-xl-base-1.0` | `462165984030d82259a11f4367a4eed129e94a7b` | Diffusers 0.35.1, FP16; edit wrapper shares weights with generation and owns a separate scheduler |

Other revisions, forks, Docker, FLUX, and Stable Diffusion 1.x/2.x are not
approved for this task. Their existing single-task options are unchanged. In the
wizard, supply the **exact revision above**, choose **native**, and select
`image-generate-and-edit` under Task. Modelship will not substitute a revision
or make dual mode the default. Qwen remains restricted to non-commercial
research under its upstream license.

Inspect a plan before deploying:

```sh
modelship plan --backend native --name qwen-dual \
  --revision d26bb61231c349cf6b7896fa83353113880e1ba3 \
  --task image-generate-and-edit --experimental --research Qwen/Qwen-Image-2.1
```

The usual disk, physical VRAM, exact metadata, runtime and license checks still
apply. The plan adds **2,048 MiB** to its runtime heuristic, not to the hard
weight lower bound. This is not a measured peak or a fit guarantee.

Both routes accept the same bounded JSON as their single-task equivalents:

- `POST /v1/images/generations`: prompt and generation settings, no input image.
- `POST /v1/images/edits`: prompt, settings, and one inline `image_b64` input.
  Qwen uses image conditioning, dimensions divisible by 32, and no `strength`.
  Reserved Qwen vision markers are rejected in prompts and negative prompts.
  SDXL uses img2img `strength`. Neither worker fetches input URLs.

New Console versions discover both modes from `/health` and offer **Generate
from text** and **Edit an input image** on the same endpoint. The legacy `task`
field still advertises generation; `image_tasks` advertises both only while the
worker is ready. Discovery never sends an inference request. A competing request
receives 503 rather than waiting in a queue. The worker synchronizes CUDA before
releasing the shared lock; a pipeline or CUDA drain failure makes the worker
unavailable until the operator restarts it. Stopping the client is not a GPU
cancellation guarantee.

A **new native deployment** is required. Updating the CLI, stopping/starting an
old worker, or changing its access settings does not add a route. Do not replace
or stop a working service just to try dual mode; use a distinct name and port
and check available VRAM first. Dual deployments use native schema 10. Existing
single-task deployments retain their execution behavior. Combined-mode tests
use CPU fixtures; GPU behavior and peak VRAM are not yet verified for this mode.
