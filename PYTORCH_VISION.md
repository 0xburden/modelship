# Gemma PyTorch image understanding — v0.3.0-alpha.8

**Published alpha.7 still has text-only PyTorch chat.** Alpha.8 adds explicit image input for the already text-validated Gemma checkpoint, with alpha.7 assets retained byte-for-byte. The working GGUF vision path and its user-confirmed real-GPU validation are separate and unchanged.

## Initial reviewed profile

- Profile: `gemma4-e4b-image-v1`, native Transformers only.
- Repository: **`google/gemma-4-E4B-it`**.
- Revision: **`4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`**, the previously validated text checkpoint.
- Architecture: `Gemma4ForConditionalGeneration`; full conditional-generation model, BF16, eager attention, one request at a time, no CPU weight offload or automatic quantization.
- Processor: Transformers **5.22.2** `Gemma4ImageProcessorPil` (Pillow/NumPy preprocessing). The image-only composition follows the same release's Gemma placeholder expansion and modality-ID contract, without constructing unused audio/video processors or installing torchvision.
- `processor_config.json`: **1,689 bytes**, SHA-256 **`32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c`** at that exact revision. Size/digest and reviewed image settings are checked before model weight loading. Only whitelisted image settings are passed to the known built-in processor; no repository Python executes.
- Preset: **280 image soft tokens**, upstream processor default; patch size 16, spatial pooling 3, padded maximum **2,520 patches**. Preprocessing preserves aspect ratio and may resize/upsample input. Soft tokens plus wrapper/text/generated tokens all consume the fixed deployment context.

This first profile does **not** infer vision support for every Gemma architecture match or repository variant. Existing text-only family matching remains; unknown repositories/revisions and other PyTorch multimodal families are rejected by `--vision` until separately reviewed. No substitute model/revision is selected automatically.

## Explicit opt-in, same deployment plan

With v0.3.0-alpha.8 or newer:

```sh
modelship plan google/gemma-4-E4B-it \
  --backend native --experimental --vision \
  --revision 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c \
  --name gemma4-vision --context 4096 --port 8004

modelship deploy google/gemma-4-E4B-it \
  --backend native --experimental --vision \
  --revision 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c \
  --name gemma4-vision --context 4096 --port 8004 --wait 30m
```

**Do not add `--mmproj-file`:** that belongs to GGUF. The PyTorch checkpoint already includes the vision weights; this path uses a pinned small processor configuration, not a separate GGUF projector.

The wizard defaults the official Gemma repository to this visible reviewed revision and asks **text-only** (default) versus **vision-images**. Numeric/letter choices and Esc-back still work. Final review shows the exact profile, processor pin, context and memory heuristic. No deployment, new keys or tensor downloads before final approval. Use a new deployment name to preserve any existing worker. `configure` still changes access only, not input modality or model/runtime files.

## Resources and compatibility

E4B is an effective-parameter label, not a 4B stored checkpoint. Previous text validation established approximately **14.894 GiB** requested-BF16 minimum weight residency for the complete checkpoint. All selected safetensors/components remain counted; enabling image input neither drops weights nor pretends a smaller language-only checkpoint is being loaded.

The vision plan adds a visible **2,048 MiB / 2 GiB** activation allowance to the existing runtime estimate. This is a conservative **preset heuristic, not a measured peak or a fit guarantee**. Actual peak memory is still unmeasured. Total/free physical placement, metadata, disk, dtype, revision and access checks remain hard; only the independent runtime heuristic may use separate `--allow-risk` consent. A 24 GB GPU is not declared validated by these estimates. Context stays fixed; input is checked after image expansion and before CUDA transfer, with no silent truncation.

The worker reuses the selected host's working CUDA-enabled Torch through the existing isolated system-site-packages venv. It never installs/replaces Torch, drivers or host libraries. Image preprocessing uses the built-in PIL backend, not a newly installed torchvision wheel; an incompatible existing environment still fails rather than modifying it. Pixel tensors alone transfer in BF16; token IDs, attention masks, modality IDs and image positions preserve integer dtypes.

New native metadata is **schema 7**, so older CLIs cannot ignore the image-processing execution contract. Schemas 1–6 remain readable, including schema-5 GGUF text and schema-6 GGUF vision. Text-only deployments clear inherited vision environment settings and do not instantiate an image processor. Existing deployments are not rewritten/restarted automatically.

## API and Console

- Output/task remains **text / `text-generation`**: image recognition/interpretation, not image generation, editing, audio, video or tools.
- One PNG/JPEG **across the conversation request**, accompanying user text, **4 MiB file bytes**, **4,194,304 total pixels (2048²)**, at most **4096 per side**. APNG/animated, unsupported formats, MIME mismatches and malformed images are rejected. Dimensions are checked before full decoding.
- Only bounded canonical `data:image/png;base64,...` / JPEG `image_url` content parts; **remote/file URLs and request-supplied paths are never fetched**. The template receives only an image marker and decoded in-memory image objects, not a URI/path.
- Non-streaming chat accepts typed user text/image parts; other roles stay text-only. Completions remain text-only. Body bounds: 1 MiB text-only, 8 MiB opt-in image deployments. One busy slot, no inference retry/queue; decoding/preprocessing and GPU work share the slot.
- `GET /v1/capabilities` uses schema version 1, matching model/revision, input/output modalities and the same image bounds expected by **Console v0.2.0**. The existing Console automatically shows image upload only when this feature is advertised. No Console source change is required. Text-only workers advertise no image input.
- Existing native bearer authentication still protects every route before routing/body parsing, including capabilities, docs and errors. Loopback default; network listeners require a valid private key. Use trusted HTTPS/private tunnels; no secrets in URLs/argv/logs.

## What has actually been tested

**CPU only — not real model inference:**

- Go profile/flag planning, complete unchanged checkpoint budgets plus separate activation heuristic, hard-gate enforcement, exact revision/repository/format checks, persisted schema/profile identity, inherited-env clearing and wizard reproduction.
- Synthetic Torch/tokenizer/model HTTP fixtures: PNG/JPEG, canonical/bounded decoding, no remote/file inputs, one image, strict roles/unknown fields, context includes image tokens before CUDA transfer, BF16 pixel versus integer ID handling, capability auth, busy/OOM release and sanitized errors.
- **Actual pinned Transformers PIL processor, no Torch/torchvision/model weights/Hub network:** synthetic 224×224 -> 256 soft tokens; 480×640 and 640×480 -> 266; 4096×1 -> 280. All produce bounded `(1,2520,768)` FP32 pixel patches and `(1,2520,2)` int64 positions. Placeholder expansion was compared with the real package's Gemma processor method. This validates preprocessing shapes/accounting only, not recognition quality.

Real CUDA image understanding, quality, peak memory, startup/cache/restart behavior on the image path and live Console-to-GPU requests remain **unvalidated**. Prior Gemma **text** validation and the user's **GGUF** image-understanding confirmation do not establish those facts. No instance was created/recreated/changed, and no model tensors were downloaded for implementation.

For the eventual user GPU test: start with one small image and a short factual question; confirm both image and follow-up text responses, then stop/start and repeat from cache. Record the exact model/revision, GPU, request/image dimensions, response, memory observations and CLI version. Do not recreate the user's deleted instance or begin GPU work without authorization.
