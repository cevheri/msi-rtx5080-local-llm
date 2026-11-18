# Gemma 4 12B IT QAT Q4_0 on llama.cpp b12407 (M16)

Last verified: 2026-11-17.

Result: the downloads lane's draft runs unchanged at 128k context with f16 KV, at 77.1 tok/s and a 9,772 MiB GPU peak, and passed `llmctl test`, a tool call with its second turn, Hermes' title request with a strict `json_schema` and a forced tool request (measured on the MSI Raider 18 Max HX, llama.cpp b12407).
The draft `$SCRATCH/sglang/overrides/llamacpp/gemma-4-12b.toml` is finalized in place: `gpu_mib = 9660`, `ram_gib = 12.9`, `[evidence] kind = "measured"`, args, context and `memory_gib` (20) unchanged.

The start ran on 2026-11-17 at 20:26 UTC as [cmds/m16-a1.sh](cmds/m16-a1.sh): `LLMCTL_CLIENTS= LLMCTL_RECIPES_DIR=$SCRATCH/sglang/measure-c/overrides $SCRATCH/llmctl-main/llmctl up gemma-4-12b --runtime llamacpp`, the copy of main at 5ad8883, so the container was llmctl's own `llamacpp` (`ghcr.io/ggml-org/llama.cpp:server-cuda-b12407`, `--memory 20g --memory-swap 20g`, 127.0.0.1:8080, `GGML_CUDA_DISABLE_GRAPHS=1`), and no client file was read or written ([logs/m16-a1.show.txt](logs/m16-a1.show.txt) holds the `docker run` line, [logs/m16-a1.up.txt](logs/m16-a1.up.txt) the output).
Server flags: `-ngl all -fa on -np 1 --jinja -ctxcp 16 --reasoning off --temp 1.0 --top-p 0.95 --top-k 64`, `-c 131072`, KV f16 (the default, as the Gemma 4 26B recipe keeps it for the 512-wide global heads).
The enforced GPU power limit read 140 W before the start; the K2 Horizon MoVA download of this lane ran beside it on the network and disk ([logs/m16-a1.meta](logs/m16-a1.meta)).

| Item | Value |
|---|---|
| `llmctl up` to `ready` | 3.5 s (the GGUF was in the page cache) |
| GPU memory right after ready | 9,758 MiB used |
| GPU peak (1 s trace) | 9,772 MiB, so `gpu_mib` 9,660 by the llama.cpp rule (peak minus the desktop's 112 MiB) |
| Container RAM peak (`docker stats`) | 12.94 GiB: 6.68 GiB page cache of the mmapped GGUF and 6.25 GiB anonymous memory, which grew from 0.29 GiB at ready as llama-server's host prompt cache (`--cache-ram`, 8 GiB by default) kept the long prompts |
| Decode (repository probe, 3 runs of up to 1,500 tokens) | 77.1 tok/s median (77.3, 77.1, 76.5 over 894, 854 and 952 tokens) |
| First token, 9,936-token fresh prompt (the 8k target in Gemma tokens) | 3.416 s, 2,986 tok/s prefill; repeated 0.092 s |
| First token, 37,194 tokens with 9,915 cached (the 30k target) | 11.617 s, 2,355 tok/s for the 27,279 new tokens; repeated 0.113 s |
| First token, 124,160 tokens with 37,173 cached (the 100k target) | 66.394 s, 1,312 tok/s for the 86,987 new tokens; decode there 58.6 tok/s; repeated 0.191 s |
| `llmctl test` | reply ok (`OK`), tool call ok (`get_weather {"city":"Paris"}`), decode 75.9 tok/s over 360 tokens |
| Code generation (LRU module with tests, 3,000-token cap) | 1,628 tokens at 72.9 tok/s, finished, no reasoning |
| Power and temperature | at most 146.1 W and 71 C |

All figures are measured on the MSI Raider 18 Max HX, llama.cpp b12407 ([shapes/m16-a1-probe.json](shapes/m16-a1-probe.json), [shapes/m16-a1-summary.json](shapes/m16-a1-summary.json), [logs/m16-a1-trace.csv](logs/m16-a1-trace.csv), [logs/m16-a1.docker-stats.log](logs/m16-a1.docker-stats.log), [logs/m16-a1.test.txt](logs/m16-a1.test.txt), [shapes/m16-a1-code-decode.json](shapes/m16-a1-code-decode.json)).

Requests ([shapes/m16-a1-seq.jsonl](shapes/m16-a1-seq.jsonl), [shapes/m16-a1-api.json](shapes/m16-a1-api.json)), all HTTP 200 (measured on the MSI Raider 18 Max HX, llama.cpp b12407):
- Hermes' title request (`json_schema`, strict, `reasoning_effort` `none`): `{"title": "Python Addition Tutorial"}`, which parses; a forced `tool_choice` request returned `get_weather {"city":"Paris"}` with `finish_reason` `tool_calls`.
- A plain reply `OK`, streamed and not, with a `[DONE]` marker; a tool call `get_weather {"city":"Paris"}`, streamed in 7 fragments; the second turn with the tool result answered "The weather in Paris is currently sunny with a temperature of 18°C."
- Thinking (toggle, off by default): `chat_template_kwargs.enable_thinking` true gave 233 characters of `reasoning_content` and the answer 391; no field, `enable_thinking` false and `reasoning_effort` `none`, `low` and `high` gave no reasoning, so on this template only `enable_thinking` turns thinking on.
- `json_schema` (`{"capital": "Paris"}`) and a forced `tool_choice` on an unrelated prompt (a `get_weather` call) both held their format; `"tools": []`, `tool_choice` `auto` without tools and a wrong model name got 200.
- The log holds no error; its warnings are the `<|tool_response>` control-token note and the prompt-cache `n_past` notes ([logs/m16-a1.server-timings.log](logs/m16-a1.server-timings.log)).

Against the other engines: vLLM v0.33.0 decodes the QAT W4A16 build at 69.5 tok/s with 128k FP8 KV (the shipped vLLM recipe), and SGLang v0.5.23 at 70.4 tok/s (measurement B), so llama.cpp is the fastest of the three here (measured on the MSI Raider 18 Max HX; the other two figures from their records).
Not run: an OpenCode session (M14, the e2e, covers it once the recipe ships with an OpenCode entry) and a q8_0 KV variant, because f16 holds 128k with 6 GiB of the GPU left.

## Files

- [cmds/m16-a1.sh](cmds/m16-a1.sh): the llmctl command, reading `$SCRATCH`.
- `logs/m16-a1.*`: the recipe as it ran (`.recipe.toml`), `llmctl show` and `up` output, the full server log and its timing lines, the 1 s GPU and cgroup trace (`-trace.csv`), `docker stats` every 2 s, process memory at ready and at the end, the container state, `llmctl test`, `llmctl down --remove` and the metadata.
- `shapes/m16-a1-*`: the repository probe (`-probe.json`, `-probe-summary.json`), the grammar requests (`-seq.jsonl`), the api probe (`-api.json`), the code decode and the summary (`-summary.json`, written by `$SCRATCH/sglang/measure-c/scripts/m16_summary.py`).
