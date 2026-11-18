# DeepSeek-R1-0528-Qwen3-8B on vLLM, the tokenizer retry, 2026-11-17

Last verified: 2026-11-17.

Result: measured: retry 2, a read-only tokenizer folder whose `tokenizer_config.json` names `PreTrainedTokenizerFast` instead of `LlamaTokenizerFast`, passed as `--tokenizer /llmctl/tokenizer`, gave llama.cpp's chat token counts (8, 8,078 and 30,254), replies that read correctly and 73.1 tok/s decode (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
Retry 1, `--tokenizer-mode deepseek_v32 --default-chat-template-kwargs '{"thinking":true}'`, is the alternative with arguments only: its replies read correctly too, but vLLM's DeepSeek V3.2 encoder replaces the model's chat template, so every prompt is one token longer than llama.cpp's and each earlier assistant reply gains a `</think>` (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
The shipped recipe failed on 2026-11-17 because the image's transformers loads the tokenizer as a Metaspace `LlamaTokenizer` ([vllm-deepseek-r1-0528-qwen3-8b/](../vllm-deepseek-r1-0528-qwen3-8b/README.md)); these are measurement B's two retries of it, started by hand with `docker run`, not by `./llmctl up`.

Image: `vllm/vllm-openai:v0.33.0`.
Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` at revision `6e8885a6ff5c1dc5201574c8fd700323f23c25fa`, served as `deepseek-r1-8b`.
Both starts mirror `recipes/vllm/deepseek-r1-0528-qwen3-8b.toml` as `launcher/runtimes/vllm.py` `container_spec` builds it: the runtime's `[env]`, the Hugging Face cache read-only, the runtime's `vllm-cache` and `flashinfer-cache` volumes (warm from earlier runs), `--ipc=host`, `--memory 32g --memory-swap 32g`, 127.0.0.1:8000, the default `--gpu-memory-utilization` 0.92, and `--max-model-len 65536 --quantization fp8_per_tensor --max-num-seqs 1 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN --reasoning-parser deepseek_r1 --override-generation-config {"temperature":0.6,"top_p":0.95}`, plus the change of each retry as the last arguments ([cmds/ld11-r1.sh](cmds/ld11-r1.sh), [cmds/ld11-r2.sh](cmds/ld11-r2.sh)).
Measurement B's own report on both retries, with the offline checks that led to them, is [measure-b.md](measure-b.md), where every figure below comes from.

## Retry 2's tokenizer folder

Measurement B mounted the folder read-only with `-v <folder>:/llmctl/tokenizer:ro` and appended `--tokenizer /llmctl/tokenizer` after the recipe's arguments ([cmds/ld11-r2.sh](cmds/ld11-r2.sh)).
It held exactly three entries:

| Entry | Kind | Content |
|---|---|---|
| `config.json` | Link | `/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528-Qwen3-8B/snapshots/6e8885a6ff5c1dc5201574c8fd700323f23c25fa/config.json`, a container path |
| `tokenizer.json` | Link | `/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528-Qwen3-8B/snapshots/6e8885a6ff5c1dc5201574c8fd700323f23c25fa/tokenizer.json`, a container path |
| [tokenizer-overlay/tokenizer_config.json](tokenizer-overlay/tokenizer_config.json) | Real file, 3,963 bytes | The snapshot's `tokenizer_config.json` with `tokenizer_class` set to `PreTrainedTokenizerFast` |

- The two links resolve only inside the container, so the archive keeps the edited file alone in `tokenizer-overlay/`.
- Compared with the snapshot's `tokenizer_config.json` when it was archived, the file has the same keys in the same order and differs only in `tokenizer_class` (`LlamaTokenizerFast` there); it is written with a two-space indent, its non-ASCII characters kept as they are, and no newline at the end (inferred from a comparison of the two files with Python's `json` module).
- The engine configuration line of retry 2 reads `tokenizer='/llmctl/tokenizer', tokenizer_mode=auto` (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
- Offline in the image, vLLM's own `get_tokenizer` loaded the folder as `CachedTokenizersBackend` and gave `Say hello world.` and the counts 8, 8,078 and 30,254, equal to llama.cpp's (measured on the MSI Raider 18 Max HX, vLLM v0.33.0 image, offline, 2026-11-17; [offline-out/tok-overlay-check.json](offline-out/tok-overlay-check.json)).

## The two GPU starts

| Start | Change | Ready | Chat tokens (OK / 8k / 30k) | Replies | Decode (probe) | First token, 8k / 30k |
|---|---|---|---|---|---|---|
| ld11-r1 | `--tokenizer-mode deepseek_v32 --default-chat-template-kwargs '{"thinking":true}'` | 24.7 s | 9 / 8,079 / 30,255 | Correct, no byte markers | 73.1 tok/s | 1.393 / 8.521 s |
| ld11-r2 | `--tokenizer /llmctl/tokenizer` with the folder above | 25.3 s | 8 / 8,078 / 30,254 | Correct, no byte markers | 73.1 tok/s | 1.404 / 8.540 s |

Both rows are measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; llama.cpp's counts for the same three prompts are 8, 8,078 and 30,254.

- Both logged `Model loading took 9.12 GiB`, `Available KV cache memory: 4.8 GiB` and `GPU KV cache size: 69,936 tokens`; the GPU peaked at 15,038 MiB with 1,103 MiB free after ready, with the compile cache warm (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
- Retry 2's replies: "Say hello world." gave `\nHello, World! This is a simple program ...` with its reasoning in `message.reasoning`, "What is 17 + 25?" gave `\n17 + 25 = 42`, `Reply with exactly: OK` gave `\nOK` as on llama.cpp, and every thinking request answered `\n391` with reasoning (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
- Without the template, the raw text counts were 5, 8,075 and 30,251 in both retries, so the tokenizer matched in both and only retry 1's template differed (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
- Decode equals the 2026-11-17 record's 72.1 tok/s within noise (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
- A request with tools, `"tools": []` and a forced `tool_choice` each got HTTP 400, as a chat-only vLLM recipe should, and `json_schema` returned `{"capital": "Paris"}` (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
- Retry 2 needs llmctl to build and mount the folder, which the recipe format of 2026-11-17 cannot express; the lead's ruling on LD11 takes that route.

## Files

| Path | Holds |
|---|---|
| [measure-b.md](measure-b.md) | Measurement B's report on the retries, the offline checks and the sources, links rewritten |
| [links.diff](links.diff) | The link rewrite of that report against the scratch copy |
| [cmds/](cmds/) | The `docker run` of ld11-r1 and ld11-r2; they read `$HOME` and `$SCRATCH` |
| [logs/](logs/) | Per start the full log, the `docker run` output, the readiness poll, the sequence output, the 1 s trace, the process memory, the state and the metadata |
| [shapes/](shapes/) | Per start the readiness record, the sequence records with the token counts and replies (`-seq.jsonl`), the repository probe, the api probe and a summary |
| [scripts/](scripts/) | The offline checks: `tok_offline.py`, `tok_overlay_check.py` and `template_compare.py` |
| [offline-out/](offline-out/) | Their output |
| [tokenizer-overlay/](tokenizer-overlay/) | Retry 2's edited `tokenizer_config.json` |
