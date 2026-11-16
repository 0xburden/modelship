# Reviewed multi-image ingestion — alpha.11 candidate

Image understanding only: multiple still images + user text → text. This is not image generation/editing, video, audio or tools. Text-only remains the deployment default; explicitly select `--vision` (and the reviewed `--mmproj-file` for GGUF). Existing saved workers remain unchanged when updating the CLI or using access-only configure.

## Exact execution contracts

| Path | Saved vision profile | Model/revision | Processor/runtime |
| --- | --- | --- | --- |
| Gemma native PyTorch | `gemma4-e4b-images-v2` | `google/gemma-4-E4B-it` @ `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` | Transformers5.22.2 built-in Gemma PIL, ≤280 soft tokens/image |
| Qwen GGUF | reviewed projector + saved `images` policy | `unsloth/Qwen3.8-27B-GGUF` @ `4af51b3542f775b0a8e4b310bfb3f17e01c306c9` | same reviewed Jammy llama.cpp b12110 core, quant/projector manifest; ≤1024 tokens/image |
| Qwen native PyTorch | `qwen38-27b-images-v1` | `Qwen/Qwen3.8-27B` @ `37561816e66b5b9153b1b64c10968c18c1e6f4a0` | Transformers5.22.2 built-in Qwen2VL PIL image-only composition, ≤1024 merged tokens/image |

All require native **schema9**. Earlier schemas1–8 remain readable without acquiring multi-image behavior. The old exact `gemma4-e4b-image-v1` contract stays in the catalog; nil GGUF image policy stays one-image, even if access metadata is subsequently saved at schema9. New policy/processor/budgets cannot be smuggled into old schemas or mixed with the wrong runtime. No automatic migration, runtime install/worker replacement, revision/model/quantization substitution or checkpoint conversion occurs.

Qwen native vision is **new**, not an implication of its earlier A100 text validation or the user's GGUF vision success. Its exact 390-byte `preprocessor_config.json` SHA-256 is `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516`. Config bytes/settings, special token IDs, built-in PIL backend and patch layout are checked before loading model weights. No AutoProcessor/unused video processor, torchvision installation or repository Python. Qwen image aspect ratio must be ≤200; very narrow images otherwise allowed by generic pixel bounds may still be rejected by this reviewed processor. M-RoPE positions are computed by the pinned model from per-image grids/modality IDs, not fabricated one-dimensional positions. See the small processor fixture's source-review notes.

## Bounded conversation contract

- **1–4 images across the entire conversation request**, including earlier image turns. Order is message order, then content-part order. Each image-bearing user message needs nonblank accompanying text; system/assistant roles stay text-only.
- Each image: static PNG/JPEG, canonical data URL, **≤4 MiB file bytes**, **≤4,194,304 pixels**, **≤4096 per side**.
- All images combined: **≤8 MiB PNG/JPEG file bytes** (not base64 bytes), **≤8,388,608 pixels**. Limits are checked before full pixel preprocessing/model work. APNG/animation, malformed encodings, unsupported MIME, excessive dimensions, wrong-role parts and worker-requested remote/file URLs are rejected.
- New multi-image worker request body: **12 MiB**; old one-image workers remain8MiB; text-only1MiB. Console's independent bounded JSON envelope remains in effect. This does not expand prompt/output/context or general provider limits.
- Image/template/text/generated tokens share the **fixed deployment context**; four allowed images is not a guarantee that four maximum-resolution images plus output fit the selected context. Native PyTorch rejects expanded input plus the requested output budget before CUDA transfer. The pinned GGUF child expands image input and rejects over-context prompts before language generation; --no-context-shift also prevents sliding the context during output. This is not a claim that image preprocessing never uses its GPU, nor a guarantee of the requested maximum output if the context is exhausted. Gemma defaults4096 context; four worst-case Qwen images may need a larger explicitly planned context. No automatic context increase/paid retry or image removal/resizing fallback.
- Worker health/capabilities advertise actual saved image count and aggregate limits. **Console0.5+ is required** for multi-image discovery; older Consoles may hide image/advanced controls when a new worker advertises max_images4, while basic text remains compatible. Console must use matching successful discovery, not a model-name guess. Uploads and guarded URL imports become ordered data-image parts. No prompt/media/history/key persistence; URL fetch remains at the authenticated separately allowlisted Console media proxy, never at the worker.

## Resources, compatibility and controls

Full checkpoint/projector encoded residency and physical total/free VRAM, disk, exact metadata/security/license/access gates remain hard and non-overridable. The existing dtype and host CUDA Torch are preserved; only pixel values use BF16, IDs/masks/positions/grids remain integer. No quantization, CPU weight offload, host library or driver replacement. Native Qwen BF16 has >51GiB minimum weights alone and does not fit a48GiB A6000; select an appropriately provisioned GPU explicitly.

New multi-image plans add **4,096MiB** activation allowance, replacing the old single-image2,048MiB allowance, not reducing/removing weight budgets. This is an **unmeasured heuristic**, not a measured peak/fit guarantee. All input decoding/preprocessing and inference share one exclusive GPU slot. New native multi-image non-streaming work also synchronizes before releasing ownership; uncertain drain poisons readiness, retains model/input references and requires explicit restart. No optimistic retry or concurrent GPU reuse.

Gemma's separately saved text controls still provide default-off chat SSE/thinking, with ≤1024 total output tokens including reasoning. Images do not bypass run-identity, context or token-ID channel checks. Completion remains text-only/non-streaming, Qwen's thinking stays recipe-disabled, and GGUF streaming is not enabled by multi-image work. Stop receiving is not guaranteed immediate CUDA/kernel cancellation.

## Validation and rollout

CPU validation proceeds Gemma → Qwen GGUF → Qwen PyTorch: real pinned Gemma/PIL batches against separate reference calls; synthetic API/child admission/lifetime tests; real pinned Qwen/PIL preprocessing/placeholder/modality/grid reference checks without Torch/Hub/weights. These are not real model recognition/inference or peak-memory measurements. Browser/production proxy checks use explicit local fixtures, not a real GPU.

User previously reported working **single-image E4B vision**, **Qwen GGUF vision**, and **Gemma4 streaming and Thinking through Console**. Exact requests/hardware/peak measurements were not provided for those reports; they do not validate multi-image behavior or native Qwen vision.

**Real multi-image GPU confirmation remains pending**, in order: Gemma, Qwen GGUF, Qwen native BF16. Requires an authorized test server/new worker plan; do not replace the healthy worker or recreate the deleted GGUF instance automatically. Use distinct labeled images (e.g. red square, blue triangle, numbered object), ask for ordered cross-image comparison, reverse the order, add images in a follow-up, test count/aggregate/context rejection without generation, and test authenticated Console uploads/URL import. For Gemma also check default-off and selected SSE/thinking. Record model/revision/quant/runtime/profile/context, GPU total/free/peak observations, request dimensions and actual output; a loaded snapshot is not peak memory. Stop/start/cache tests and deployed proxy buffering are separate facts, not inferred from CPU fixtures or a successful request.
