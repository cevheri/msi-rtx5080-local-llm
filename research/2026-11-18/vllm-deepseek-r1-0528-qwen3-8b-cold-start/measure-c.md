# DeepSeek-R1-0528-Qwen3-8B on vLLM v0.33.0: the cold start with the tokenizer folder (step 4)

Last verified: 2026-11-18.

Result: the cold-start failure still holds with the tokenizer folder: with an empty compile cache and the default `--gpu-memory-utilization` 0.92, vLLM left 3.83 GiB for the KV cache and refused 64k, which needs 4.5 GiB (measured on the MSI Raider 18 Max HX, vLLM v0.33.0).
One flag fixes it: `--kv-cache-memory-bytes 5153960755` (4.8 GiB, the KV cache that warm starts at 0.92 get) started both a cold start and a warm start at 64k with 69,904 KV tokens, the same 15,036 MiB GPU peak and 1,105 MiB free after ready, and the warm start measured 72.5 tok/s with llama.cpp's token counts (8, 8,078 and 30,254) and replies that read correctly (measured on the MSI Raider 18 Max HX, vLLM v0.33.0).
Context and output cap stay as shipped (65,536 and 16,384), and the draft `$SCRATCH/sglang/overrides/vllm/deepseek-r1-0528-qwen3-8b.toml` carries the flag, the evidence, and the tokenizer flag in a comment; three starts were used, the fix being the first one tried.

## The tokenizer folder, as llmctl will ship it

Per the lead's message of 2026-11-18 (plan ruling 35), the folder replaced measurement B's three-entry folder before any start of this lane, so every start here used it, and B's folder was not used ([tokenizer-folder.make.json](tokenizer-folder.make.json), built by `$SCRATCH/sglang/measure-c/scripts/make_tokenizer_folder.py`):
- nine symbolic links, one per file of snapshot `6e8885a6ff5c1dc5201574c8fd700323f23c25fa`, each to its container path `/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528-Qwen3-8B/snapshots/6e8885a6ff5c1dc5201574c8fd700323f23c25fa/<relative path>`: `.gitattributes`, `LICENSE`, `README.md`, `config.json`, `figures/benchmark.png` (in a real `figures/` folder), `model-00001-of-000002.safetensors`, `model-00002-of-000002.safetensors`, `model.safetensors.index.json` and `tokenizer.json`;
- one real file, `tokenizer_config.json` (3,964 bytes, mode 0644): the snapshot's JSON object with `"tokenizer_class": "PreTrainedTokenizerFast"` in place of `"LlamaTokenizerFast"`, written as `json.dumps(config, indent=2, ensure_ascii=False) + "\n"`; it differs from B's text-edited file only by that serialization (one byte longer) and is equal to it as JSON.

It was mounted read-only at `/llmctl/tokenizer` with `--tokenizer /llmctl/tokenizer` right after `--max-model-len` in the argv, and vLLM's engine line read `tokenizer='/llmctl/tokenizer', ..., tokenizer_mode=auto` (measured).

## The starts

Each mirrors `recipes/vllm/deepseek-r1-0528-qwen3-8b.toml` as `launcher/runtimes/vllm.py` `container_spec` builds it with `recipes/vllm/_runtime.toml`: `vllm/vllm-openai:v0.33.0`, the runtime's `[env]` (`HF_HUB_OFFLINE=1`, `VLLM_NO_USAGE_STATS=1`, `MAX_JOBS=2`, `HF_MODULES_CACHE=/tmp/hf_modules`), the Hugging Face cache read-only, `--ipc=host`, `--memory 32g --memory-swap 32g`, 127.0.0.1:8000, the snapshot path, `--served-model-name deepseek-r1-8b --max-model-len 65536`, the folder above, the recipe's arguments, the runtime's `flashinfer-cache` volume, and a new empty volume `vllm-measure-cold` in place of `vllm-cache` (the torch.compile cache) ([cmds/](cmds/)).

| Start | Change | Compile cache | Outcome | KV cache | Ready (poll) | GPU peak; free after ready |
|---|---|---|---|---|---|---|
| cold-r1 (21:46 UTC) | none: the default 0.92 | empty: `torch.compile took 18.08 s in total` | Refused: `ValueError: To serve at least one request with the model's max seq len (65536), (4.5 GiB KV cache is needed, which is larger than the available KV cache memory (3.83 GiB). Based on the available memory, the estimated maximum model length is 55744. ...` | `Available KV cache memory: 3.83 GiB` | none (exit 1) | 12,072 MiB |
| cold-r2-kvbytes (21:51 UTC) | `--kv-cache-memory-bytes 5153960755` | empty: `torch.compile took 17.77 s in total` | Started: `Initial free memory 15.08 GiB, reserved 4.8 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling.` | 69,904 tokens, `Maximum concurrency for 65,536 tokens per request: 1.07x` | 44.2 s | 15,036 MiB; 1,105 MiB |
| warm-r3-kvbytes (21:57 UTC) | the same, on cold-r2's volume | warm: `torch.compile took 0.42 s in total` | Started | 69,904 tokens | 20.3 s | 15,036 MiB; 1,105 MiB |

All rows are measured on the MSI Raider 18 Max HX, vLLM v0.33.0 (`logs/<run>.full.log`, `logs/<run>-trace.csv`, `shapes/<run>-ready.json`); the volume held 355 files and 47 MB after cold-r1 and 435 files and 48 MB after warm-r3, and was removed after cold-r1 and after warm-r3 (`logs/<run>.volume-vllm-measure-cold.txt`).
- Why 3.83 GiB cold and 4.8 GiB warm: vLLM sizes the KV cache from a profile run that, with an empty cache, includes the first torch.compile, and takes the peak it measures from the 0.92 budget (`../offline/vllm-src/gpu_worker.py`, lines 641 to 760); with `kv_cache_memory_bytes` set it still runs the profile run (and the compile) but skips the measurement and reserves the given bytes (lines 656 to 681) (inferred from the source; the two KV figures are measured).
- Smaller changes would not do: raising `--gpu-memory-utilization` to 0.97 would leave about 4.6 GiB cold, barely 64k, and warm starts would take about 0.75 GiB more of the GPU; lowering `--max-model-len` to 55,744 or less would cut the context that the Kimi test pins (estimate from the 3.83 GiB cold figure).
- llmctl's preflight for this recipe still computes its GPU need from 0.92, as for every vLLM recipe without `--gpu-memory-utilization`, which is above the 15,036 MiB the server then uses (inferred from `launcher/runtimes/vllm.py` `gpu_need_mib`).

## The warm measurement (warm-r3-kvbytes), as B measured retry 2

| Item | warm-r3-kvbytes (this lane) | B's retry 2 (warm cache, 0.92) |
|---|---|---|
| Chat token counts: "Reply with exactly: OK", the probe's 8k and 30k prompts | 8, 8,078, 30,254 (llama.cpp: 8, 8,078, 30,254) | 8, 8,078, 30,254 |
| Raw text counts | 5, 8,075, 30,251 | 5, 8,075, 30,251 |
| "Say hello world." | `\nHello, World! This is a simple program ...`, reasoning in `message.reasoning`, no byte markers | the same |
| "What is 17 + 25? Answer briefly." | `\n17 + 25 = 42` | the same |
| Decode, repository probe (chat-only copy, three 1,500-token runs) | 72.5 tok/s (72.3, 72.6, 72.5) | 73.1 tok/s |
| First token, 8,078 and 30,254 tokens | 1.410 s and 8.580 s; repeats 0.049 and 0.091 s | 1.404 and 8.540 s |
| Request with tools, `"tools": []`, forced `tool_choice` | HTTP 400 each (`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`; `` `tools` must not be an empty array``; `tool_choice="required" requires --tool-call-parser`) | HTTP 400 each |
| `json_schema` | `{"capital": "Paris"}` | `{"capital": "Paris"}` |
| Thinking (always) | every request reasoned (1,375 to 5,279 characters) and answered `\n391`; `reasoning_effort` and `enable_thinking` false did not turn it off | the same shape |
| KV cache; GPU peak | 69,904 tokens; 15,036 MiB | 69,936 tokens; 15,038 MiB |

This lane's column is measured on the MSI Raider 18 Max HX, vLLM v0.33.0 ([shapes/warm-r3-kvbytes-seq.jsonl](shapes/warm-r3-kvbytes-seq.jsonl) events `tokcheck_counts`, `tokcheck_reply` and `m9_tools`, [shapes/warm-r3-kvbytes-probe.json](shapes/warm-r3-kvbytes-probe.json), [shapes/warm-r3-kvbytes-api.json](shapes/warm-r3-kvbytes-api.json)); B's column is from [its record](../../2026-11-17/vllm-deepseek-r1-0528-qwen3-8b-retry/measure-b.md).
The cold start cold-r2-kvbytes ran the same sequence with the same counts (8, 8,078, 30,254), replies and HTTP 400s, at 72.5 tok/s and 1.406 and 8.564 s (measured; `shapes/cold-r2-kvbytes-*`).
So the full folder behaves as B's three-entry folder did, and the recipe needs the tokenizer folder from llmctl plus the one `--kv-cache-memory-bytes` argument.

## Files

- [cmds/](cmds/): `cold-r1.sh`, `cold-r2-kvbytes.sh`, `warm-r3-kvbytes.sh`, reading `$HOME` and `$SCRATCH`.
- `tokenizer-folder/` and [tokenizer-folder.make.json](tokenizer-folder.make.json): the folder (links and the real file) and its file list.
- `logs/<run>.*` and `shapes/<run>-*`: full logs, traces, readiness, the sequence records (`tokcheck` counts and replies), probe, api probe, the volume listing before and after.
