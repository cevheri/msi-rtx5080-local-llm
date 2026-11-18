# Qwen3.5-9B Q8_0 on llama.cpp, 2026-11-17

Last verified: 2026-11-17.

Result: measured in measurement C (M16): the draft llama.cpp recipe runs unchanged at 131072 context with f16 KV, at 66.0 tok/s decode, a 12,782 MiB GPU peak (`gpu_mib` 12,670) and 15.0 GiB of container RAM, and passed `llmctl test`, a tool call with its second turn, Hermes' title request with a strict `json_schema` and a forced tool request (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
An OpenCode session was not run.

Image: `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407`.
Model: `unsloth/Qwen3.5-9B-GGUF` at revision `3885219b6810b007914f3a7950a8d1b469d598a5`, file `Qwen3.5-9B-Q8_0.gguf`.
Settings ([logs/m16-a1.recipe.toml](logs/m16-a1.recipe.toml)): `-c 131072 -ngl all -fa on -np 1 --jinja --reasoning off --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0`, f16 KV, `--memory 16g --memory-swap 16g`, started at 20:34 UTC (23:34 local time) with `LLMCTL_CLIENTS= LLMCTL_RECIPES_DIR=$SCRATCH/sglang/measure-c/overrides $SCRATCH/llmctl-main/llmctl up qwen3.5-9b --runtime llamacpp`, a copy of `main` at 5ad8883, so no client file changed ([cmds/m16-a1.sh](cmds/m16-a1.sh)).
The reports are [measurement C's report on this model](measure-c.md) and [measurement C's lane README](../sglang-runtime/measure-c/README.md), where every figure below comes from.

## Measurement C

- `llmctl up` reached `ready` in 3.4 s with the GGUF in the page cache; the GPU held 12,770 MiB right after ready and peaked at 12,782 MiB, below the draft's 13,312 MiB estimate (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- Container RAM peaked at 14.95 GiB under the 16 GiB cap, of which 9.07 GiB was page cache of the mmapped GGUF and 5.87 GiB anonymous memory (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- Decode 66.0 tok/s median in the repository probe; first token 1.942 s for a fresh 8,591-token prompt, 6.305 s for 32,153 tokens with 8,075 cached, and 29.520 s for 107,327 tokens with 31,637 cached, where decode was 45.1 tok/s (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- Requests: `OK`, a `get_weather {"city":"Paris"}` call and its second turn, `json_schema`, the title request and a forced `tool_choice` passed; thinking turns on only with `enable_thinking` true, not with a `reasoning_effort` (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- The decode is about 28% below the NVFP4 build's 90.8 tok/s on vLLM v0.33.0 ([vllm-qwen3.5-9b/](../vllm-qwen3.5-9b/README.md)) and 93.7 tok/s on SGLang v0.5.23 ([sglang-qwen3.5-9b/](../sglang-qwen3.5-9b/README.md)) (measured on the MSI Raider 18 Max HX, 2026-11-17); that Q8_0 reads more weight bytes per token than NVFP4 is the likely cause (inferred from the formats, not measured).

## Files

| Path | Holds |
|---|---|
| [measure-c.md](measure-c.md) | Measurement C's report on this model, links rewritten |
| [links.diff](links.diff) | The link rewrite of that report against the scratch copy |
| [cmds/m16-a1.sh](cmds/m16-a1.sh) | The `llmctl up` command; it reads `$SCRATCH` |
| [logs/](logs/) | The recipe as it ran, `llmctl show`, `up`, `test` and `down --remove` output, the full server log and its timing lines, the 1 s GPU and cgroup trace, `docker stats` every 2 s, process memory, the container state and the metadata |
| [shapes/](shapes/) | The repository probe, the grammar requests (`-seq.jsonl`), the api probe, the code decode and the summary |
