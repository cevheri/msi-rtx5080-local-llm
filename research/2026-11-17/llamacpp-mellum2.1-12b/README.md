# Mellum2.1 12B-A2.5B Thinking Q6_K on llama.cpp, 2026-11-17

Last verified: 2026-11-17.

Result: measured in measurement C (M16): the draft llama.cpp recipe runs unchanged at 131072 context with f16 KV, at 188.2 tok/s decode with thinking on (the template's default), a 12,674 MiB GPU peak (`gpu_mib` 12,562) and 13.2 GiB of container RAM; b12407 parses Mellum's tool calls and reasoning, and `llmctl test`, a tool call with its second turn, Hermes' title request with a strict `json_schema` and a forced tool request passed (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
An OpenCode session was not run.

Image: `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407`.
Model: `JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF` at revision `4159d684ffdb09ba89dc2a58feb3e75c54eef036`, file `Mellum2.1-12B-A2.5B-Thinking-Q6_K.gguf`.
Settings ([logs/m16-a1.recipe.toml](logs/m16-a1.recipe.toml)): `-c 131072 -ngl all -fa on -np 1 --jinja --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0`, f16 KV, no `--reasoning off`, `--memory 16g --memory-swap 16g`, started at 20:32 UTC (23:32 local time) with `LLMCTL_CLIENTS= LLMCTL_RECIPES_DIR=$SCRATCH/sglang/measure-c/overrides $SCRATCH/llmctl-main/llmctl up mellum2.1-12b --runtime llamacpp`, a copy of `main` at 5ad8883, so no client file changed ([cmds/m16-a1.sh](cmds/m16-a1.sh)).
The reports are [measurement C's report on this model](measure-c.md) and [measurement C's lane README](../sglang-runtime/measure-c/README.md), where every figure below comes from.

## Measurement C

- `llmctl up` reached `ready` in 4.5 s with the GGUF in the page cache; the GPU held 12,620 MiB right after ready and peaked at 12,674 MiB, below the draft's 13,312 MiB estimate (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- Container RAM peaked at 13.16 GiB under the 16 GiB cap, of which 10.32 GiB was page cache of the mmapped GGUF and 2.83 GiB anonymous memory (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- Decode 188.2 tok/s median in the repository probe with thinking on, reasoning included; first token 1.553 s for a fresh 9,932-token prompt, 5.073 s for 37,190 tokens with 9,915 cached, and 27.247 s for 124,156 tokens with 37,173 cached, where decode was 121.1 tok/s (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- Requests: a `get_weather {"city": "Paris"}` call and its second turn, `json_schema`, the title request with `reasoning_effort` `none` and a forced tool request passed; thinking is on without a field, off with `enable_thinking` false or `reasoning_effort` `none`, and a code request spent its whole 3,000-token budget on reasoning (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17).
- The same model's AWQ-INT4 build decodes 257.2 tok/s on vLLM v0.33.0 ([vllm-mellum2.1-12b/](../vllm-mellum2.1-12b/README.md)) and does not load on SGLang v0.5.23 ([sglang-mellum2.1-12b/](../sglang-mellum2.1-12b/README.md)) (measured on the MSI Raider 18 Max HX, 2026-11-17).

## Files

| Path | Holds |
|---|---|
| [measure-c.md](measure-c.md) | Measurement C's report on this model, links rewritten |
| [links.diff](links.diff) | The link rewrite of that report against the scratch copy |
| [cmds/m16-a1.sh](cmds/m16-a1.sh) | The `llmctl up` command; it reads `$SCRATCH` |
| [logs/](logs/) | The recipe as it ran, `llmctl show`, `up`, `test` and `down --remove` output, the full server log and its timing lines, the 1 s GPU and cgroup trace, `docker stats` every 2 s, process memory, the container state and the metadata |
| [shapes/](shapes/) | The repository probe, the grammar requests (`-seq.jsonl`), the api probe, the code decode and the summary |
