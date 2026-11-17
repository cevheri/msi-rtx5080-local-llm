# Gemma 4 E4B IT QAT Q4_0 on llama.cpp b12407 (M16)

Last verified: 2026-11-17.

Result: the downloads lane's draft runs unchanged at 128k context with f16 KV, at 151.3 tok/s and a 5,422 MiB GPU peak, and passed `llmctl test`, a tool call with its second turn, Hermes' title request with a strict `json_schema` and a forced tool request (measured on the MSI Raider 18 Max HX, llama.cpp b12407).
b12407 loads E4B's per-layer embeddings without a warning, and the whole server fits in 5.3 GiB of GPU memory, far below the draft's 9,216 MiB estimate (measured; the estimate was the draft's upper bound).
The draft `$SCRATCH/sglang/overrides/llamacpp/gemma-4-e4b.toml` is finalized in place: `gpu_mib = 5310`, `ram_gib = 7.9`, `[evidence] kind = "measured"`, args, context and `memory_gib` (20) unchanged.

The start ran on 2026-11-17 at 20:30 UTC as [cmds/m16-a1.sh](cmds/m16-a1.sh): `LLMCTL_CLIENTS= LLMCTL_RECIPES_DIR=$SCRATCH/sglang/measure-c/overrides $SCRATCH/llmctl-main/llmctl up gemma-4-e4b --runtime llamacpp`, the copy of main at 5ad8883, so the container was llmctl's own `llamacpp` (`ghcr.io/ggml-org/llama.cpp:server-cuda-b12407`, `--memory 20g --memory-swap 20g`, 127.0.0.1:8080, `GGML_CUDA_DISABLE_GRAPHS=1`), and no client file was read or written ([logs/m16-a1.show.txt](logs/m16-a1.show.txt), [logs/m16-a1.up.txt](logs/m16-a1.up.txt)).
Server flags: `-ngl all -fa on -np 1 --jinja -ctxcp 16 --reasoning off --temp 1.0 --top-p 0.95 --top-k 64`, `-c 131072` (the model's 131,072 positions), KV f16.
The enforced GPU power limit read 140 W before the start; the K2 Horizon MoVA download of this lane ran beside it ([logs/m16-a1.meta](logs/m16-a1.meta)).

| Item | Value |
|---|---|
| `llmctl up` to `ready` | 2.4 s (the GGUF was in the page cache) |
| GPU memory right after ready | 5,366 MiB used |
| GPU peak (1 s trace) | 5,422 MiB, so `gpu_mib` 5,310 |
| Container RAM peak (`docker stats`) | 7.86 GiB: 5.00 GiB page cache of the mmapped GGUF and 2.84 GiB anonymous memory |
| Decode (repository probe, 3 runs of up to 1,500 tokens) | 151.3 tok/s median (151.3, 151.6, 150.5 over 1,079, 1,120 and 1,204 tokens) |
| First token, 9,932-token fresh prompt | 0.866 s, 11,772 tok/s prefill; repeated 0.024 s |
| First token, 37,190 tokens with 9,915 cached | 3.120 s, 8,834 tok/s for the 27,275 new tokens; repeated 0.052 s |
| First token, 124,156 tokens with 37,173 cached | 17.668 s, 4,949 tok/s for the 86,983 new tokens; decode there 84.5 tok/s; repeated 0.125 s |
| `llmctl test` | reply ok (`OK`), tool call ok (`get_weather {"city":"Paris"}`), decode 150.8 tok/s over 363 tokens |
| Code generation (LRU module with tests, 3,000-token cap) | 1,673 tokens at 148.6 tok/s, finished, no reasoning |
| Power and temperature | at most 149.4 W and 70 C |

All figures are measured on the MSI Raider 18 Max HX, llama.cpp b12407 ([shapes/m16-a1-probe.json](shapes/m16-a1-probe.json), [shapes/m16-a1-summary.json](shapes/m16-a1-summary.json), [logs/m16-a1-trace.csv](logs/m16-a1-trace.csv), [logs/m16-a1.docker-stats.log](logs/m16-a1.docker-stats.log), [logs/m16-a1.test.txt](logs/m16-a1.test.txt)).

Requests ([shapes/m16-a1-seq.jsonl](shapes/m16-a1-seq.jsonl), [shapes/m16-a1-api.json](shapes/m16-a1-api.json)), all HTTP 200 (measured on the MSI Raider 18 Max HX, llama.cpp b12407):
- Hermes' title request: `{"title": "Python Addition Basics"}`, which parses; a forced `tool_choice` request returned `get_weather {"city":"Paris"}`.
- A plain reply `OK`, streamed and not; a tool call `get_weather {"city":"Paris"}`, streamed in 7 fragments; the second turn answered "The weather in Paris is sunny and the temperature is 18°C."
- Thinking (toggle, off by default): `enable_thinking` true gave 259 characters of `reasoning_content` and 391; no field, `enable_thinking` false and every `reasoning_effort` gave no reasoning.
- `json_schema` returned `{"capital": "Paris"}`.

Against the other engines: vLLM v0.33.0 runs E4B only in eager mode at 64k, 45.2 tok/s (the shipped vLLM recipe), and SGLang v0.5.23 at 64k at 141.3 tok/s (measurement B), so llama.cpp holds twice the context of either at the highest decode speed (measured on the MSI Raider 18 Max HX; the other two figures from their records).
Not run: an OpenCode session (M14 covers it) and the vision projector, which the draft does not load, as no shipped llama.cpp recipe of the family does.

## Files

As in [../llamacpp-gemma-4-12b/README.md](../llamacpp-gemma-4-12b/measure-c.md#files): `cmds/m16-a1.sh`, `logs/m16-a1.*` and `shapes/m16-a1-*`.
