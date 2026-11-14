# Reviewed native Gemma chat controls — alpha.10

The initial `gemma4-e4b-text-controls-v1` profile supports **real chat SSE** and optional `chat_template_kwargs.enable_thinking` for `google/gemma-4-E4B-it` at `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`, native BF16 Transformers5.22.2 only. It applies to new text-only or reviewed image-enabled Gemma deployments. Other revisions, variants, Qwen, causal workers and GGUF keep their existing execution policies. No source license or host Torch/driver/library change is introduced.

## Defaults and API

Both streaming and thinking remain **off by default**. Ordinary existing chat JSON remains valid. Completions keep their existing non-streaming, recipe-controlled behavior; these new controls are chat-only. No arbitrary template, thinking-budget, reasoning-effort, tools, audio/video or generation kwargs are accepted.

An advanced client first fetches authenticated `GET /v1/capabilities`. Its schema1 response identifies the exact model/revision/run, `chat.streaming`, and `chat.text_controls` schema1/profile/defaults/supported parameters. The Console0.4 automatically enables controls only for matching capabilities, then revalidates them server-side before sending **one** inference POST. It never retries an inference after a discovery error. Missing/stale/failed discovery keeps controls unavailable, not an automatic fallback.

Example body (the run ID is non-secret metadata, **not** an API key):

```json
{
  "model": "google/gemma-4-E4B-it",
  "expected_run_id": "RUN_ID_FROM_CAPABILITIES",
  "messages": [{"role": "user", "content": "Explain your answer, then give the result."}],
  "max_tokens": 512,
  "temperature": 0,
  "stream": true,
  "chat_template_kwargs": {"enable_thinking": true}
}
```

`expected_run_id` must match the active 32-hex worker run for streaming or any thinking override (including explicit false); otherwise409 occurs before GPU submission. Use header-only bearer auth via a client that reads private credential files/memory; never put keys in URLs, command arguments or saved connections. Default-off requests need no new fields. `--vision` and the original image restrictions remain separate.

SSE uses `chat.completion.chunk`, `delta.content` for final text, separate `delta.reasoning_content`, a finish/usage frame and `[DONE]` **only after successful generation and CUDA drain**. Non-streaming chat uses the same channel decoder and returns separate answer/reasoning fields. The parser recognizes pinned actual Gemma control-token IDs (98/100/101 and EOS/turn IDs), not regex guesses over answer prose. It uses the exact built-in TextStreamer without printing generated content. Split Unicode/CJK are covered by a Torch-free test of the actual pinned decoder. Unknown/malformed channels fail closed rather than exposing control text as an answer.

## Budgets and cancellation

- **1–1024 generated tokens total**, including thinking, channel/control tokens and final text. This is not 1024 tokens per channel. A model may spend the whole budget reasoning and produce no final answer; Console displays the reasoning without inventing or appending an assistant answer/history turn. It never silently changes the budget or disables thinking and retries.
- Prompt/template/image expansion plus the entire requested output budget must fit the fixed deployment context **before CUDA transfer**. Only pixels become BF16; integer IDs/positions/masks stay integer. Existing full-checkpoint weight, free/total VRAM, disk, processor, revision, security and consent gates remain mandatory. No quantization/offload/model substitution is introduced. The existing image activation allowance is still an unmeasured heuristic; streaming/thinking peak memory is not measured.
- One GPU producer; no request queue. Internal **output** queue≤32 chunks, chunks≤2048 characters, decoded text+reasoning≤1MiB and generation/receipt deadline900s. Idle heartbeats maintain the transport without exposing upstream comments. No reconnect/resubmit or fake buffered streaming.
- Stop receiving/disconnect/deadline requests cooperative cancellation. A CUDA kernel/prefill may still run; stopping receipt does **not** promise stopped compute/billing. BUSY stays held through actual generation and CUDA synchronization. Shutdown drains the producer before releasing the model. A failed CUDA drain poisons readiness and keeps the worker unavailable until an explicit operator restart; no new inference is admitted optimistically.
- Midstream failures send a sanitized error, never a success `[DONE]`. Partial output is not appended as a completed assistant turn. Auth/body/image/context checks still precede generation. No prompt/image/reasoning/API key is logged by the new streamer.

## Deployment compatibility

New exact-profile Gemma plans include the full text-control contract and use native **schema8**; schemas1–7 remain readable without acquiring controls. Saved profile/spec and worker catalog must agree. Old CLIs cannot ignore schema8 constraints. Existing request defaults remain off and compatible.

**Updating the CLI alone does not upgrade a running or saved worker.** Its private inference assets/spec remain unchanged, including after access-only `configure`. A new explicitly approved deployment is needed for these capabilities. Choose a new name/port and reuse the verified cache; do not overwrite an existing deployment. If free-VRAM gates block another worker, decide explicitly whether to stop the old one—Modelship does not interrupt it automatically. Old deployments remain available for rollback. Console0.3 can still use default non-streaming chat; Console0.4 is required for native control discovery/admission.

## Evidence, not GPU claims

CPU fixtures cover profile/schema/environment binding, default-off/legacy rejection, token channels, Unicode, vision/context/dtypes, auth, stale run IDs, real incremental ASGI output, bounded backpressure, timeout/disconnect/drain/OOM/poisoned readiness and unchanged text/image/video behavior. A local real-HTTP fixture uses this API/job code plus the actual pinned TextStreamer, but replaces Torch/model/tokenizer with explicit CPU fixtures. No weights or user GPU/instance activity is involved. Actual Gemma CUDA streaming/thinking quality, latency, peak memory, image+thinking behavior and live deployed Console routing still require operator validation.
