# Gemma 4 E4B IT QAT Q4_0 on llama.cpp, 2026-11-17

Last verified: 2026-11-17.

Result: measured in measurement C (M16): the draft llama.cpp recipe runs unchanged at 131072 context with f16 KV, at 151.3 tok/s decode, a 5,422 MiB GPU peak (`gpu_mib` 5,310) and 7.9 GiB of container RAM, and passed `llmctl test`, a tool call with its second turn, Hermes' title request with a strict `json_schema` and a forced tool request (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
An OpenCode session was not run, and the vision projector is not loaded.

Image: `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407`.
Model: `google/gemma-4-E4B-it-qat-q4_0-gguf` at revision `4b4a2c1d584be7264f87aac328a1bc739ce81b6c`, file `gemma-4-E4B_q4_0-it.gguf`.
Settings ([logs/m16-a1.recipe.toml](logs/m16-a1.recipe.toml)): `-c 131072 -ngl all -fa on -np 1 --jinja -ctxcp 16 --reasoning off --temp 1.0 --top-p 0.95 --top-k 64`, f16 KV, `--memory 20g --memory-swap 20g`, started at 20:30 UTC (23:30 local time) with `LLMCTL_CLIENTS= LLMCTL_RECIPES_DIR=$SCRATCH/sglang/measure-c/overrides $SCRATCH/llmctl-main/llmctl up gemma-4-e4b --runtime llamacpp`, a copy of `main` at 5ad8883, so no client file changed ([cmds/m16-a1.sh](cmds/m16-a1.sh)).
The reports are [measurement C's report on this model](measure-c.md) and [measurement C's lane README](../sglang-runtime/measure-c/README.md), where every figure below comes from.

## Measurement C

- `llmctl up` reached `ready` in 2.4 s with the GGUF in the page cache; the GPU held 5,366 MiB right after ready and peaked at 5,422 MiB, far below the draft's 9,216 MiB estimate, and b12407 loaded E4B's per-layer embeddings without a warning (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- Container RAM peaked at 7.86 GiB, of which 5.00 GiB was page cache of the mmapped GGUF and 2.84 GiB anonymous memory (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- Decode 151.3 tok/s median in the repository probe; first token 0.866 s for a fresh 9,932-token prompt, 3.120 s for 37,190 tokens with 9,915 cached, and 17.668 s for 124,156 tokens with 37,173 cached, where decode was 84.5 tok/s (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- Requests: `OK`, a `get_weather {"city":"Paris"}` call and its second turn, `json_schema`, the title request and a forced `tool_choice` passed; thinking turns on only with `enable_thinking` true (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- The same model runs on vLLM v0.33.0 only in eager mode at 64k, 45.2 tok/s ([vllm-gemma-4-e4b/](../vllm-gemma-4-e4b/README.md)), and on SGLang v0.5.23 at 64k at 141.3 tok/s ([sglang-gemma-4-e4b/](../sglang-gemma-4-e4b/README.md)), so llama.cpp holds twice their context at the highest decode (measured on the MSI Raider 18 Max HX, 2026-11-17).

## Files

| Path | Holds |
|---|---|
| [measure-c.md](measure-c.md) | Measurement C's report on this model, links rewritten |
| [links.diff](links.diff) | The link rewrite of that report against the scratch copy |
| [cmds/m16-a1.sh](cmds/m16-a1.sh) | The `llmctl up` command; it reads `$SCRATCH` |
| [logs/](logs/) | The recipe as it ran, `llmctl show`, `up`, `test` and `down --remove` output, the full server log and its timing lines, the 1 s GPU and cgroup trace, `docker stats` every 2 s, process memory, the container state and the metadata |
| [shapes/](shapes/) | The repository probe, the grammar requests (`-seq.jsonl`), the api probe, the code decode and the summary |
