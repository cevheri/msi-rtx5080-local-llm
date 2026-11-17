# Gemma 4 12B IT QAT Q4_0 on llama.cpp, 2026-11-17

Last verified: 2026-11-17.

Result: measured in measurement C (M16): the draft llama.cpp recipe runs unchanged at 131072 context with f16 KV, at 77.1 tok/s decode, a 9,772 MiB GPU peak (`gpu_mib` 9,660) and 12.9 GiB of container RAM, and passed `llmctl test`, a tool call with its second turn, Hermes' title request with a strict `json_schema` and a forced tool request (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
An OpenCode session was not run.

Image: `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407`.
Model: `google/gemma-4-12B-it-qat-q4_0-gguf` at revision `29d097773436b69ff9feafd636ab4cf873786537`, file `gemma-4-12b-it-qat-q4_0.gguf`.
Settings ([logs/m16-a1.recipe.toml](logs/m16-a1.recipe.toml)): `-c 131072 -ngl all -fa on -np 1 --jinja -ctxcp 16 --reasoning off --temp 1.0 --top-p 0.95 --top-k 64`, f16 KV, `--memory 20g --memory-swap 20g`, started at 20:26 UTC (23:26 local time) with `LLMCTL_CLIENTS= LLMCTL_RECIPES_DIR=$SCRATCH/sglang/measure-c/overrides $SCRATCH/llmctl-main/llmctl up gemma-4-12b --runtime llamacpp`, a copy of `main` at 5ad8883, so no client file changed ([cmds/m16-a1.sh](cmds/m16-a1.sh)).
The reports are [measurement C's report on this model](measure-c.md) and [measurement C's lane README](../sglang-runtime/measure-c/README.md), where every figure below comes from.

## Measurement C

- `llmctl up` reached `ready` in 3.5 s with the GGUF in the page cache; the GPU held 9,758 MiB right after ready and peaked at 9,772 MiB (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- Container RAM peaked at 12.94 GiB, of which 6.68 GiB was page cache of the mmapped GGUF and 6.25 GiB anonymous memory, mostly llama-server's host prompt cache (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- Decode 77.1 tok/s median in the repository probe; first token 3.416 s for a fresh 9,936-token prompt, 11.617 s for 37,194 tokens with 9,915 cached, and 66.394 s for 124,160 tokens with 37,173 cached, where decode was 58.6 tok/s (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- Requests: `OK`, a `get_weather {"city":"Paris"}` call streamed and not, its second turn, `json_schema`, the title request and a forced `tool_choice` all held their format; thinking turns on only with `enable_thinking` true, not with a `reasoning_effort` (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- The same model decodes 69.5 tok/s on vLLM v0.33.0 ([vllm-gemma-4-12b/](../vllm-gemma-4-12b/README.md)) and 70.4 tok/s on SGLang v0.5.23 ([sglang-gemma-4-12b/](../sglang-gemma-4-12b/README.md)), so llama.cpp is the fastest of the three here (measured on the MSI Raider 18 Max HX, 2026-11-17).

## Files

| Path | Holds |
|---|---|
| [measure-c.md](measure-c.md) | Measurement C's report on this model |
| [cmds/m16-a1.sh](cmds/m16-a1.sh) | The `llmctl up` command; it reads `$SCRATCH` |
| [logs/](logs/) | The recipe as it ran, `llmctl show`, `up`, `test` and `down --remove` output, the full server log and its timing lines, the 1 s GPU and cgroup trace, `docker stats` every 2 s, process memory, the container state and the metadata |
| [shapes/](shapes/) | The repository probe, the grammar requests (`-seq.jsonl`), the api probe, the code decode and the summary |
