# DeepSeek-R1-0528-Qwen3-8B on vLLM, the cold start with the tokenizer folder, 2026-11-18

Last verified: 2026-11-18.

Result: measured: with the tokenizer folder that llmctl will build and `--kv-cache-memory-bytes 5153960755` (4.8 GiB) added to the shipped arguments, a cold start (empty compile cache) and a warm start both served 64k with 69,904 KV tokens, a 15,036 MiB GPU peak and 1,105 MiB free after ready, and the warm start decoded 72.5 tok/s with llama.cpp's token counts (8, 8,078 and 30,254), replies that read correctly and HTTP 400 for every request with tools (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18).
Without the flag the cold start at the default `--gpu-memory-utilization` 0.92 was refused: vLLM left 3.83 GiB for the KV cache, short of the 4.5 GiB that 64k needs (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18).
These are measurement C's three starts, started by hand with `docker run` as llmctl's `container_spec` builds the recipe, not by `./llmctl up`.

Image: `vllm/vllm-openai:v0.33.0`.
Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` at revision `6e8885a6ff5c1dc5201574c8fd700323f23c25fa`, served as `deepseek-r1-8b`.
Measurement C's own report on the starts is [measure-c.md](measure-c.md), and its lane README is [sglang-runtime/measure-c/README.md](../../2026-11-17/sglang-runtime/measure-c/README.md); every figure below comes from them.
The tokenizer evidence that this folder builds on, measurement B's retry 2 with a three-entry folder at a warm compile cache, is in [vllm-deepseek-r1-0528-qwen3-8b-retry/](../../2026-11-17/vllm-deepseek-r1-0528-qwen3-8b-retry/README.md).

## The settings C chose for a cold and a warm start

- Both the cold start cold-r2-kvbytes and the warm start warm-r3-kvbytes ran the same command ([cmds/cold-r2-kvbytes.sh](cmds/cold-r2-kvbytes.sh), [cmds/warm-r3-kvbytes.sh](cmds/warm-r3-kvbytes.sh)): `recipes/vllm/deepseek-r1-0528-qwen3-8b.toml` as `launcher/runtimes/vllm.py` `container_spec` builds it with the runtime's `[env]`, the Hugging Face cache read-only, `--ipc=host`, `--memory 32g --memory-swap 32g` and 127.0.0.1:8000, plus the tokenizer folder below and one argument.
- The arguments: `--served-model-name deepseek-r1-8b --max-model-len 65536 --tokenizer /llmctl/tokenizer --quantization fp8_per_tensor --max-num-seqs 1 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN --reasoning-parser deepseek_r1 --override-generation-config {"temperature":0.6,"top_p":0.95} --kv-cache-memory-bytes 5153960755`.
- `--kv-cache-memory-bytes 5153960755` is the 4.8 GiB KV cache that warm starts at 0.92 got; with it vLLM still runs its profile run and the first torch.compile but reserves the given bytes instead of sizing the cache from the peak it measures (measured; the mechanism inferred from vLLM's `gpu_worker.py`).
- The context stays 65,536 and the output cap 16,384, as shipped; the default `--gpu-memory-utilization` 0.92 is kept.
- The cold start used a new empty volume in place of `vllm-cache` (the torch.compile cache) and became ready in 44.2 s; the warm start reused that volume and became ready in 20.3 s (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18).
- Smaller changes would not do, by measurement C's estimate from the 3.83 GiB cold figure: `--gpu-memory-utilization` 0.97 would leave about 4.6 GiB cold and take about 0.75 GiB more of the GPU warm, and a context of 55,744 or less would cut the 65,536 that the Kimi test pins (estimate).

## The tokenizer folder

The folder was built as llmctl will ship it (plan ruling 35), not as measurement B's three-entry folder, and was mounted read-only at `/llmctl/tokenizer`, with `--tokenizer /llmctl/tokenizer` right after `--max-model-len` ([tokenizer-folder.make.json](tokenizer-folder.make.json)).

| Entry | Kind | Content |
|---|---|---|
| `.gitattributes`, `LICENSE`, `README.md`, `config.json`, `figures/benchmark.png`, `model-00001-of-000002.safetensors`, `model-00002-of-000002.safetensors`, `model.safetensors.index.json`, `tokenizer.json` | Nine links, `figures/` a real folder | Each to its container path under `/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528-Qwen3-8B/snapshots/6e8885a6ff5c1dc5201574c8fd700323f23c25fa/` |
| [tokenizer-folder/tokenizer_config.json](tokenizer-folder/tokenizer_config.json) | Real file, 3,964 bytes, mode 0644 | The snapshot's JSON object with `tokenizer_class` set to `PreTrainedTokenizerFast` instead of `LlamaTokenizerFast`, written as `json.dumps(config, indent=2, ensure_ascii=False) + "\n"` |

- The nine links resolve only inside the container, so the archive keeps the real file alone in `tokenizer-folder/`; `tokenizer-folder.make.json` lists every link and its target.
- Measurement C reports that the file differs from measurement B's text-edited file only by that serialization, one byte longer, and is equal to it as JSON.
- vLLM's engine line read `tokenizer='/llmctl/tokenizer', ..., tokenizer_mode=auto` (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18).

## The three starts

| Start | Change | Compile cache | Outcome | KV cache | Ready | GPU peak; free after ready |
|---|---|---|---|---|---|---|
| cold-r1 | none: the default 0.92 | empty | Refused: 4.5 GiB of KV cache needed, 3.83 GiB available | 3.83 GiB | none (exit 1) | 12,072 MiB |
| cold-r2-kvbytes | `--kv-cache-memory-bytes 5153960755` | empty | Started | 69,904 tokens | 44.2 s | 15,036 MiB; 1,105 MiB |
| warm-r3-kvbytes | the same | warm, cold-r2's volume | Started | 69,904 tokens | 20.3 s | 15,036 MiB; 1,105 MiB |

All rows are measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18.

- warm-r3-kvbytes: chat token counts 8, 8,078 and 30,254 for `Reply with exactly: OK` and the probe's 8k and 30k prompts, equal to llama.cpp's; `Say hello world.` answered `\nHello, World! This is a simple program ...` with its reasoning in `message.reasoning` and no byte markers; `What is 17 + 25? Answer briefly.` answered `\n17 + 25 = 42` (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18).
- Decode 72.5 tok/s in the chat-only probe (72.3, 72.6 and 72.5 over three 1,500-token runs), and 1.410 s and 8.580 s to the first token for 8,078 and 30,254 tokens, against retry 2's 73.1 tok/s, 1.404 s and 8.540 s (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18 and 2026-11-17).
- A request with tools, `"tools": []` and a forced `tool_choice` each got HTTP 400, as a chat-only vLLM recipe should, and `json_schema` returned `{"capital": "Paris"}`; every request reasoned and answered `\n391`, and neither `reasoning_effort` nor `enable_thinking` false turned thinking off (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18).
- The cold start cold-r2-kvbytes ran the same sequence with the same counts, replies and HTTP 400s, at 72.5 tok/s and 1.406 s and 8.564 s to the first token (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18).
- llmctl's preflight still computes this recipe's GPU need from 0.92, as for every vLLM recipe without `--gpu-memory-utilization`, which is above the 15,036 MiB the server then used (inferred from `launcher/runtimes/vllm.py` `gpu_need_mib`).

## Files

| Path | Holds |
|---|---|
| [measure-c.md](measure-c.md) | Measurement C's report on the three starts, links rewritten |
| [links.diff](links.diff) | The link rewrite of that report against the scratch copy |
| [cmds/](cmds/) | The `docker run` of cold-r1, cold-r2-kvbytes and warm-r3-kvbytes; they read `$HOME` and `$SCRATCH` |
| [logs/](logs/) | Per start the full log, the `docker run` output, the readiness poll, the 1 s GPU and cgroup trace, the state, the metadata, the compile-cache volume listing before and after, and the `/dev/shm` readings; for the two starts that ran, the sequence output, the process memory and the mappings at ready |
| [shapes/](shapes/) | Per start the readiness record; for the two starts that ran, the sequence records with the token counts and replies (`-seq.jsonl`), the chat-only repository probe and the api probe |
| [tokenizer-folder/](tokenizer-folder/) | The folder's real `tokenizer_config.json` |
| [tokenizer-folder.make.json](tokenizer-folder.make.json) | The folder's link list and targets |
