# Mellum2.1 12B-A2.5B Thinking Q6_K on llama.cpp b12407 (M16)

Last verified: 2026-11-17.

Result: the downloads lane's draft runs unchanged at 128k context with f16 KV, at 188.2 tok/s with thinking on (the server default), and a 12,674 MiB GPU peak; b12407 parses Mellum's tool calls and reasoning, and `llmctl test`, a tool call with its second turn, Hermes' title request with a strict `json_schema` and a forced tool request passed (measured on the MSI Raider 18 Max HX, llama.cpp b12407).
The draft `$SCRATCH/sglang/overrides/llamacpp/mellum2.1-12b.toml` is finalized in place: `gpu_mib = 12562`, `ram_gib = 13.2`, `[evidence] kind = "measured"`, args, context and `memory_gib` (16) unchanged.

The start ran on 2026-11-17 at 20:32 UTC as [cmds/m16-a1.sh](cmds/m16-a1.sh): `LLMCTL_CLIENTS= LLMCTL_RECIPES_DIR=$SCRATCH/sglang/measure-c/overrides $SCRATCH/llmctl-main/llmctl up mellum2.1-12b --runtime llamacpp`, the copy of main at 5ad8883, so the container was llmctl's own `llamacpp` (`--memory 16g --memory-swap 16g`, 127.0.0.1:8080, `GGML_CUDA_DISABLE_GRAPHS=1`), and no client file was read or written ([logs/m16-a1.show.txt](logs/m16-a1.show.txt), [logs/m16-a1.up.txt](logs/m16-a1.up.txt)).
Server flags: `-ngl all -fa on -np 1 --jinja --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0`, `-c 131072`, KV f16, no `--reasoning off`, so the template's default (thinking on) applies.
The enforced GPU power limit read 140 W before the start; the K2 Horizon MoVA download of this lane ran beside it ([logs/m16-a1.meta](logs/m16-a1.meta)).

| Item | Value |
|---|---|
| `llmctl up` to `ready` | 4.5 s (the GGUF was in the page cache) |
| GPU memory right after ready | 12,620 MiB used |
| GPU peak (1 s trace) | 12,674 MiB, so `gpu_mib` 12,562 (the draft estimated 13,312) |
| Container RAM peak (`docker stats`) | 13.16 GiB: 10.32 GiB page cache of the mmapped GGUF and 2.83 GiB anonymous memory, under the 16 GiB cap |
| Decode (repository probe, 3 runs of up to 1,500 tokens, thinking on) | 188.2 tok/s median (187.6, 189.6, 188.2 over 1,387, 1,167 and 1,237 tokens, reasoning included) |
| First token, 9,932-token fresh prompt | 1.553 s, 6,594 tok/s prefill; repeated 0.039 s |
| First token, 37,190 tokens with 9,915 cached | 5.073 s, 5,422 tok/s for the 27,275 new tokens; repeated 0.061 s |
| First token, 124,156 tokens with 37,173 cached | 27.247 s, 3,204 tok/s for the 86,983 new tokens; decode there 121.1 tok/s; repeated 0.123 s |
| `llmctl test` | reply ok (`OK`, with 66 characters of reasoning), tool call ok (`get_weather {"city": "Paris"}`), decode 190.4 tok/s over 512 tokens |
| Code generation (LRU module with tests, 3,000-token cap) | 3,000 tokens at 180.4 tok/s, all reasoning (10,449 characters), no answer within the cap |
| Power and temperature | at most 145.1 W and 70 C |

All figures are measured on the MSI Raider 18 Max HX, llama.cpp b12407 ([shapes/m16-a1-probe.json](shapes/m16-a1-probe.json), [shapes/m16-a1-summary.json](shapes/m16-a1-summary.json), [logs/m16-a1-trace.csv](logs/m16-a1-trace.csv), [logs/m16-a1.docker-stats.log](logs/m16-a1.docker-stats.log), [logs/m16-a1.test.txt](logs/m16-a1.test.txt)).

Requests ([shapes/m16-a1-seq.jsonl](shapes/m16-a1-seq.jsonl), [shapes/m16-a1-api.json](shapes/m16-a1-api.json)), all HTTP 200 (measured on the MSI Raider 18 Max HX, llama.cpp b12407):
- Hermes' title request (`reasoning_effort` `none`): `{"title": "Adding Two Numbers in Python"}`, which parses, with no reasoning; a forced `tool_choice` request returned `get_weather {"city": "Paris"}` after 377 characters of reasoning.
- A tool call `get_weather {"city": "Paris"}`, streamed in 7 fragments, and the second turn "The current weather in Paris is 18°C and sunny."
- Thinking (toggle, on by default): no field gave 116 characters of `reasoning_content`, `enable_thinking` true 128, `enable_thinking` false none, `reasoning_effort` `none` none, `low` 179 and `high` 117; every answer was 391.
- `json_schema` returned `{"capital": "Paris"}`; the api probe's forced `tool_choice` on "Reply with exactly: OK" with `enable_thinking` false and a 256-token cap ended with `finish_reason` `length` and its thinking as text in `content` (668 characters), no tool call; the forced tool request above, with a weather question and 2,048 tokens, called the tool.

Against the other engines: vLLM v0.33.0 decodes the AWQ-INT4 build at 257.2 tok/s with thinking on at 128k (the shipped vLLM recipe), and SGLang v0.5.23 does not load it (measurement B), so llama.cpp is the second engine for this model (measured on the MSI Raider 18 Max HX; the other figures from their records).
As with K2 Horizon on llama.cpp, a thinking-on code request needs a large output budget: 3,000 tokens went to reasoning alone.
Not run: an OpenCode session (M14 covers it).

## Files

As in [../llamacpp-gemma-4-12b/README.md](../llamacpp-gemma-4-12b/measure-c.md#files): `cmds/m16-a1.sh`, `logs/m16-a1.*` and `shapes/m16-a1-*`.
