# Qwen3.5-9B Q8_0 on llama.cpp b12407 (M16)

Last verified: 2026-11-17.

Result: the downloads lane's draft runs unchanged at 128k context with f16 KV, at 66.0 tok/s and a 12,782 MiB GPU peak, and passed `llmctl test`, a tool call with its second turn, Hermes' title request with a strict `json_schema` and a forced tool request (measured on the MSI Raider 18 Max HX, llama.cpp b12407).
The draft `$SCRATCH/sglang/overrides/llamacpp/qwen3.5-9b.toml` is finalized in place: `gpu_mib = 12670`, `ram_gib = 15.0`, `[evidence] kind = "measured"`, args, context and `memory_gib` (16) unchanged.
The decode is about 28% slower than on vLLM (90.8 tok/s) and SGLang (93.7 tok/s), which run the NVFP4 checkpoint (measured); that Q8_0 stores the linear weights at 8.5 bits against NVFP4's 4.5 bits, so each token reads more weight bytes, is the likely cause (inferred from the formats, not measured).

The start ran on 2026-11-17 at 20:34 UTC as [cmds/m16-a1.sh](cmds/m16-a1.sh): `LLMCTL_CLIENTS= LLMCTL_RECIPES_DIR=$SCRATCH/sglang/measure-c/overrides $SCRATCH/llmctl-main/llmctl up qwen3.5-9b --runtime llamacpp`, the copy of main at 5ad8883, so the container was llmctl's own `llamacpp` (`--memory 16g --memory-swap 16g`, 127.0.0.1:8080, `GGML_CUDA_DISABLE_GRAPHS=1`), and no client file was read or written ([logs/m16-a1.show.txt](logs/m16-a1.show.txt), [logs/m16-a1.up.txt](logs/m16-a1.up.txt)).
Server flags: `-ngl all -fa on -np 1 --jinja --reasoning off --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0`, `-c 131072`, KV f16; the model has 16 query and 4 KV heads of 256 (a GQA ratio of 4), below the ratio above 4 at which CLAUDE.md's q8_0 note applies, and f16 holds 128k anyway, so q8_0 was not tried.
The enforced GPU power limit read 145 W before the start; the K2 Horizon MoVA download of this lane ran beside it ([logs/m16-a1.meta](logs/m16-a1.meta)).

| Item | Value |
|---|---|
| `llmctl up` to `ready` | 3.4 s (the GGUF was in the page cache) |
| GPU memory right after ready | 12,770 MiB used |
| GPU peak (1 s trace) | 12,782 MiB, so `gpu_mib` 12,670 (the draft estimated 13,312) |
| Container RAM peak (`docker stats`) | 14.95 GiB: 9.07 GiB page cache of the mmapped GGUF and 5.87 GiB anonymous memory (the host prompt cache and the input embedding that llama.cpp keeps in RAM), under the 16 GiB cap |
| Decode (repository probe, 3 runs of up to 1,500 tokens) | 66.0 tok/s median (66.2, 66.0, 65.4 over 861, 836 and 817 tokens) |
| First token, 8,591-token fresh prompt | 1.942 s, 4,495 tok/s prefill; repeated 0.046 s |
| First token, 32,153 tokens with 8,075 cached | 6.305 s, 3,834 tok/s for the 24,078 new tokens; repeated 0.065 s |
| First token, 107,327 tokens with 31,637 cached | 29.520 s, 2,571 tok/s for the 75,690 new tokens; decode there 45.1 tok/s; repeated 0.125 s |
| `llmctl test` | reply ok (`OK`), tool call ok (`get_weather {"city":"Paris"}`), decode 67.2 tok/s over 424 tokens |
| Code generation (LRU module with tests, 3,000-token cap) | 1,525 tokens at 66.2 tok/s, finished, no reasoning |
| Power and temperature | at most 146.0 W and 71 C |

All figures are measured on the MSI Raider 18 Max HX, llama.cpp b12407 ([shapes/m16-a1-probe.json](shapes/m16-a1-probe.json), [shapes/m16-a1-summary.json](shapes/m16-a1-summary.json), [logs/m16-a1-trace.csv](logs/m16-a1-trace.csv), [logs/m16-a1.docker-stats.log](logs/m16-a1.docker-stats.log), [logs/m16-a1.test.txt](logs/m16-a1.test.txt)).

Requests ([shapes/m16-a1-seq.jsonl](shapes/m16-a1-seq.jsonl), [shapes/m16-a1-api.json](shapes/m16-a1-api.json)), all HTTP 200 (measured on the MSI Raider 18 Max HX, llama.cpp b12407):
- Hermes' title request: `{"title": "Python Addition"}`, which parses; a forced `tool_choice` request returned `get_weather {"city":"Paris"}`.
- A tool call `get_weather {"city":"Paris"}`, streamed in 5 fragments, and the second turn "It is currently sunny in Paris with a temperature of 18°C."
- Thinking (toggle, off by default): `enable_thinking` true gave 735 characters of `reasoning_content`; no field, `enable_thinking` false and every `reasoning_effort` gave none, so on llama.cpp an effort does not turn thinking on, unlike on SGLang (report section 3.6).
- `json_schema` returned `{"capital": "Paris"}`.

Not run: an OpenCode session (M14 covers it).
The prompt counts match the Qwen tokenizer of the other Qwen-family records (8,591, 32,153 and 107,327 for the probe's targets, as in the Groot and Qwen3.6 rows of llamacpp/benchmarking.md) (measured).

## Files

As in [../llamacpp-gemma-4-12b/README.md](../llamacpp-gemma-4-12b/measure-c.md#files): `cmds/m16-a1.sh`, `logs/m16-a1.*` and `shapes/m16-a1-*`.
