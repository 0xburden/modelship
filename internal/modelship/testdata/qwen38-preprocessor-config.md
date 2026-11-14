# Qwen3.8 reviewed image-only processor

## Immutable metadata evidence

Model: `Qwen/Qwen3.8-27B` at `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.
All reads were bounded public metadata/source reads, not tensors or repository Python execution.

- [preprocessor_config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/preprocessor_config.json): the complete checked-in `qwen38-preprocessor-config.json`, **390 bytes**, SHA-256 `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516`. No final newline; its exact bytes are pinned. There is no `processor_config.json` at this revision.
- [tokenizer_config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/tokenizer_config.json): 9,984 bytes, SHA-256 `207c03c41a06a9c3b0cc527d8a56115159c1eac6b59a4ffae85b646e773d074e`. Reviewed added tokens: `<|vision_start|>` **248053**, `<|vision_end|>` **248054**, `<|image_pad|>` **248056**, `<|video_pad|>` **248057**. Adapter verifies these against the existing loaded tokenizer; this fixture does not replace/load the real tokenizer.
- [config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/config.json): architecture `Qwen3_5ForConditionalGeneration`, `image_token_id=248056`, `vision_start_token_id=248053`, `vision_end_token_id=248054`, `video_token_id=248057`; vision patch16, temporal2, spatial merge2, three channels.
- [chat_template.jinja](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/chat_template.jinja): 9,004 bytes, SHA-256 `af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375`. Lines 8–18 render each image as `<|vision_start|><|image_pad|><|vision_end|>`, with optional picture numbering outside the markers. Existing pinned tokenizer chat-template handling is retained; the adapter only expands those complete markers in occurrence order.

## Reviewed Transformers 5.22.2 source

Built-in source only, never model-repository Python:

- [Qwen2VLImageProcessorPil](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/qwen2_vl/image_processing_pil_qwen2_vl.py): explicit public PIL backend, bicubic resize with factor32, RGB conversion, rescale1/255, normalize mean/std0.5. Patch rows are merge-major, temporal duplicated still images; output `pixel_values` float32 `[sum(t*h*w),1536]`, `image_grid_thw` int64 `[N,3]`, each grid temporal dimension1. Source aspect ratio maximum200.
- [Qwen3VLProcessor.replace_image_token](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/qwen3_vl/processing_qwen3_vl.py): each original image placeholder expands to `image_pad * (prod(image_grid_thw[i]) // merge_size**2)`; the surrounding vision-start/end markers remain untouched.
- [ProcessorMixin](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/processing_utils.py): `get_text_with_replacements` consumes original placeholder occurrences sequentially; `create_mm_token_type_ids` assigns 1 to image-pad IDs and 0 to ordinary text/start/end markers. Adapter comparisons call these actual pinned upstream methods, not reimplemented references.
- [Qwen3_5Model.get_rope_index / compute_3d_position_ids](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py): image grids are consumed in image-run occurrence order using `mm_token_type_ids`. Multimodal data without modality IDs raises an error. The adapter forwards grids/modality IDs and **does not supply ordinary 1D position_ids**; pinned model code computes M-RoPE/rope deltas. This source review is not an execution test of the model.

## Conservative profile for leader integration

```json
{
  "id": "qwen38-27b-images-v1",
  "repo": "Qwen/Qwen3.8-27B",
  "revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
  "recipe": "qwen3.8-multimodal",
  "architecture": "Qwen3_5ForConditionalGeneration",
  "processor": {
    "name": "preprocessor_config.json",
    "size": 390,
    "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
  },
  "max_image_tokens": 1024,
  "activation_allowance_mib": 4096,
  "images": {
    "max_images": 4,
    "max_total_image_bytes": 8388608,
    "max_total_image_pixels": 8388608
  }
}
```

Upstream minimum resized area65536 is retained; maximum resized area is explicitly reduced from16777216 to1048576 (1024 merged image tokens). Encoded input limits remain leader-validator-owned: each4MiB, aggregate8MiB; decoded source each4MP, aggregate8MP, maximum4096 per edge. Qwen additionally rejects aspect ratios>200. Supports1–4 images, with one marker per image, preserving same-turn and historical-turn order. Result bounds: each `t=1`, even `2<=h,w<=1024`, `h*w<=4096`, at most16384 aggregate patch rows /4096 image tokens; each patch1536 float32 values. 4096MiB activation allowance is conservative planning policy, **not measured GPU peak memory**. Text tokens plus actual expanded image tokens must still pass the existing context budget before generation.

Runtime requirements: existing `multimodal-lm` full conditional-generation loader, recipe `qwen3.8-multimodal`, exact revision, BF16; `transformers==5.22.2`, `numpy==2.3.5`, `pillow==11.3.0`, and existing pinned Hub/safetensors dependencies. No torchvision, AutoProcessor, remote code, audio/video processors or new Torch installation. CPU verification uses `tests/processor-requirements.txt` and forbids Torch/torchvision. Serving converts prepared NumPy arrays to the runtime's **already existing Torch** tensors; model inference still needs the existing GPU runtime and is not validated by this check.

Integration: bundle `qwen_vision.py`; add this exact profile to the catalog; dispatch its ID to `qwen_vision.load_image_processor` (same signature as Gemma). Alternatively use `QwenImageProcessor(profile, tokenizer)` after existing catalog/approved-profile validation. Adapter exposes `.images_policy` and `.max_tokens`; `prepare(prompt, images, tokenizer)` returns PT tensors by default, or NumPy via keyword-only `return_tensors="np"` for the CPU check. Do not send explicit `position_ids`, truncate expanded placeholders, reorder images, or reconstruct images from user paths/URLs. The leader owns shared routing, metadata download allowlisting, budget/capability validation and existing Gemma changes.

## CPU evidence boundary

`tests/qwen_processor_check.py` executes the **real pinned PIL image processor** on synthetic images at1–4 counts, several source shapes/modes, maximum4MP source and8MP aggregate; compares concatenated patch/grid order, placeholder expansion and modality IDs with pinned upstream methods; tests malformed markers, token mismatch, digest/profile tampering, count/pixel/shape bounds. Hub download is redirected only to the digest-pinned metadata fixture, offline flags are set and socket connects are blocked. The lightweight tokenizer fixture is explicitly not actual tokenizer/model inference. No weights, GPU, serving endpoint or user instance is involved.

Executed successfully on Python3.13.11 with the exact CPU requirements and no installed Torch/torchvision: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /tmp/modelship-qwen-processor/bin/python tests/qwen_processor_check.py`. Observed ordered1–4-image token counts `[64]`, `[64,300]`, `[64,300,300]`, `[64,300,300,1024]`; 2048x2048 source yields1024 tokens, two such sources at8MP aggregate yield`[1024,1024]`; extreme accepted4096x32 /32x4096 yield128 tokens each. Terminal marker: `QWEN_PROCESSOR_CPU_VALIDATED`. All negative assertions passed. No model/GPU inference claim.
