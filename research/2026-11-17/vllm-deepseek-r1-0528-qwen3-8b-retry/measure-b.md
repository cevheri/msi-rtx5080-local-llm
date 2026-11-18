# DeepSeek-R1-0528-Qwen3-8B on vLLM v0.33.0: the LD11 tokenizer retry

Last verified: 2026-11-17.

Result: fixed, with a condition: pointing vLLM at a tokenizer folder whose `tokenizer_config.json` names `PreTrainedTokenizerFast` instead of `LlamaTokenizerFast` gave llama.cpp's exact token counts (8, 8,078 and 30,254) and replies that read correctly, at 73.1 tok/s (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, retry 2).
The condition: today's vLLM recipe cannot mount that folder, so the fix needs llmctl to build a tokenizer overlay for the recipe, as the design's D7 builds the SGLang model overlay; the draft `$SCRATCH/sglang/overrides/vllm/deepseek-r1-0528-qwen3-8b.toml` says so in its header.
A fix with arguments only also exists: `--tokenizer-mode deepseek_v32` loads the byte-level tokenizer and replies read correctly, but vLLM's DeepSeek V3.2 encoder replaces the model's chat template, so every prompt is one token longer than llama.cpp's and multi-turn prompts gain a `</think>` per earlier reply (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, retry 1, and offline).
Two of the three allowed starts were used.

## The defect, and what the sources suggest

- The 2026-11-17 record (`research/2026-11-17/vllm-deepseek-r1-0528-qwen3-8b/`): the model's `tokenizer_config.json` names `LlamaTokenizerFast` over a byte-level BPE `tokenizer.json`, and transformers 5.20.2 in the vLLM image builds a Metaspace `LlamaTokenizer` from it, so prompts lose their spaces and replies carry `Ġ` and `Ċ` markers.
- In transformers 5.20.2, `AutoTokenizer` overrides a hub class only for model types in `MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS` (which lists `qwen2`, `qwen3_5` and `deepseek_v3` among others, but not `qwen3`) or model ids in `MODEL_IDS_TO_TOKENIZERS_BACKEND` (`deepseek-ai/deepseek-r1-distill-llama-*` but not this model); this model's type is `qwen3`, so the hub class wins ([offline/vllm-transformers-autotokenizer.txt](../sglang-runtime/measure-b/offline/vllm-transformers-autotokenizer.txt), [offline/vllm-transformers-incorrect-class.txt](../sglang-runtime/measure-b/offline/vllm-transformers-incorrect-class.txt)) (inferred from the image's `transformers/models/auto/tokenization_auto.py`).
- vLLM v0.33.0 has its own list for this case, `_MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS` (internlm2, step3_vl and a few others, not qwen3), and no command-line option adds a type to it (`../offline/vllm-src/registry.py`, lines 31 to 44) (inferred from the source).
- Its tokenizer modes are `auto`, `hf`, `alpamayo2_super`, `mistral`, `deepseek_v32`, `deepseek_v4`, `inkling`, `kimi_k3` and `cohere`; there is no slow mode (`ValueError: No tokenizer registered for tokenizer_mode='slow'.`), and the `fastokens` backend is not installed in the image (`../offline/vllm-src/config_model.py`, lines 89 to 99; [logs offline](offline-out/tok-offline.json)) (measured offline in the image).
- The same symptom on `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` with vLLM and transformers 5.7.0 is reported in that model's Hugging Face discussion #22, whose workaround is a copy of the tokenizer files with the class changed to `Qwen2TokenizerFast`, passed with `--tokenizer` (third-party report, https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/discussions/22, read on 2026-11-17); transformers PR #45741 fixed the Qwen2 mapping for those distills (vendor-reported, https://github.com/huggingface/transformers/pull/45741, read on 2026-11-17), which is why `qwen2` is in the list and `qwen3` is not.
- The tokenizer of another repository was not an option: no other snapshot with this vocabulary is cached (only the 6e8885a snapshot of this model and the llama.cpp recipe's GGUF), and this lane downloads nothing.

## Offline checks in the vLLM image (CPU only, `--network none`)

| Candidate | Class | "Say hello world." | Chat tokens: OK / probe 8k / probe 30k | Against llama.cpp (8 / 8,078 / 30,254) |
|---|---|---|---|---|
| `auto` (shipped) | `CachedLlamaTokenizer`, Metaspace | `Sayhelloworld.` | 9 / 6,903 / 25,845 | +1 / -1,175 / -4,409 |
| `--tokenizer-mode deepseek_v32` | `CachedDSV32TokenizersBackend` | `Say hello world.` | 9 / 8,079 / 30,255 | +1 each (`</think>`, or `<think>` in thinking mode) |
| `--tokenizer-mode deepseek_v4` | `CachedDSV4TokenizersBackend` | `Say hello world.` | 91 / 8,161 / 30,337 | +83 each (a "Reasoning Effort" system text) |
| `--tokenizer-mode slow` | none | none | none | `ValueError: No tokenizer registered for tokenizer_mode='slow'.` |
| `--tokenizer` = the llama.cpp GGUF | none | none | none | killed at an 8 GiB cap (exit 137) while transformers read the 8.1 GiB file |
| `--tokenizer` = overlay with `PreTrainedTokenizerFast` | `CachedTokenizersBackend` | `Say hello world.` | 8 / 8,078 / 30,254 | equal |

All rows are measured on the MSI Raider 18 Max HX, vLLM v0.33.0 image, offline, with vLLM's own `get_tokenizer` ([scripts/tok_offline.py](scripts/tok_offline.py), [scripts/tok_overlay_check.py](scripts/tok_overlay_check.py), [offline-out/](offline-out/)).
The V3.2 encoder against the model's own template, rendered offline for four conversations ([scripts/template_compare.py](scripts/template_compare.py), [offline-out/template-compare.json](offline-out/template-compare.json)): single-turn prompts equal the model's prompt plus `<think>`, and each earlier assistant reply gains a `</think>` before it, so a two-turn prompt is 2 tokens longer (measured offline).

## The two GPU starts

Both mirror `recipes/vllm/deepseek-r1-0528-qwen3-8b.toml` as `launcher/runtimes/vllm.py` `container_spec` builds it: image `vllm/vllm-openai:v0.33.0`, the runtime's `[env]`, the Hugging Face cache read-only, the runtime's `vllm-cache` and `flashinfer-cache` volumes (warm from the earlier runs), `--ipc=host`, `--memory 32g --memory-swap 32g`, 127.0.0.1:8000, the default `--gpu-memory-utilization` 0.92, and the recipe's arguments, plus the change named.

| Start | Change | Ready | Chat tokens (OK / 8k / 30k) | Replies | Decode (probe) | First token, 8k / 30k |
|---|---|---|---|---|---|---|
| ld11-r1 ([cmds](cmds/ld11-r1.sh)) | `--tokenizer-mode deepseek_v32 --default-chat-template-kwargs '{"thinking":true}'` | 24.7 s | 9 / 8,079 / 30,255 | Correct, no byte markers | 73.1 tok/s | 1.393 / 8.521 s |
| ld11-r2 ([cmds](cmds/ld11-r2.sh)) | `--tokenizer /llmctl/tokenizer` with the overlay mounted read-only ([tokenizer-overlay/](tokenizer-overlay/)) | 25.3 s | 8 / 8,078 / 30,254 | Correct, no byte markers | 73.1 tok/s | 1.404 / 8.540 s |

Both rows are measured on the MSI Raider 18 Max HX, vLLM v0.33.0 (`shapes/ld11-r*-seq.jsonl`, `shapes/ld11-r*-probe.json`, `shapes/ld11-r*-api.json`, `logs/ld11-r*.full.log`).
- Both logged `Model loading took 9.12 GiB`, `Available KV cache memory: 4.8 GiB` and `GPU KV cache size: 69,936 tokens`, so the default 0.92 started with the compile cache warm, as the e2e runs found; the GPU peaked at 15,038 MiB with 1,103 MiB free after ready (measured).
- The engine configuration line names the tokenizer each used: `tokenizer=.../snapshots/6e8885a..., tokenizer_mode=deepseek_v32` and `tokenizer='/llmctl/tokenizer', tokenizer_mode=auto` (measured).
- Retry 2's replies: "Say hello world." gave `\nHello, World! This is a simple program ...` with clean reasoning in `message.reasoning`, "What is 17 + 25?" gave `\n17 + 25 = 42`, the probe's `Reply with exactly: OK` gave `\nOK` (llama.cpp's reply is `\nOK` too), and every thinking request answered `\n391` with reasoning (measured).
- Raw text counts without the template were 5, 8,075 and 30,251 in both retries, llama.cpp's chat counts minus its three template tokens, so the tokenizer itself matched in both; only retry 1's template differed (measured).
- Decode is the earlier record's 72.1 tok/s within noise; the first-token times are now for the right prompt sizes (the garbled tokenizer had made the 8k prompt 6,903 tokens) (measured).
- A request with tools got HTTP 400 `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`, `"tools": []` got 400, and a forced `tool_choice` got 400, as a chat-only vLLM recipe should; `json_schema` returned `{"capital": "Paris"}` (measured).

## What ships, and what the lead decides

- By the brief's rule (the same token count as llama.cpp and replies that read correctly), retry 2 counts and retry 1 does not.
- Retry 2 needs an llmctl change (a tokenizer overlay built per recipe and mounted at `/llmctl/tokenizer`), which today's vLLM recipe schema cannot express; LD15's rule sends anything larger than one environment variable to the backlog, so the lead decides between that change, retry 1's arguments with its template difference, and keeping the recipe recorded as failed.
- The draft `$SCRATCH/sglang/overrides/vllm/deepseek-r1-0528-qwen3-8b.toml` carries retry 2's arguments and names retry 1's as the arguments-only alternative.

## Files

- [cmds/](cmds/): the two `docker run` files; they read `$HOME` and `$SCRATCH`.
- [tokenizer-overlay/](tokenizer-overlay/): the folder of retry 2 (two links to container paths and the edited `tokenizer_config.json`).
- [scripts/](scripts/) and [offline-out/](offline-out/): the offline checks and their output.
- `logs/ld11-r*` and `shapes/ld11-r*`: full logs, traces, readiness, sequence records (`tokcheck` counts and replies), probe, api probe and summaries.
