# Measurements, 2026-11-18

Last verified: 2026-11-18.

These are the records of measurement C whose latest runs fall on 2026-11-18 local time (+03:00), from 21:00 UTC on 2026-11-17: wave 3 of the SGLang v0.5.23 work and the vLLM DeepSeek-R1-0528-Qwen3-8B cold start with the tokenizer folder that llmctl will build.
One earlier start sits here too: m11-k64 in sglang-qwen3.6-35b-a3b/, which began at 20:56 UTC, still 2026-11-17 local time.
Measurement C's earlier runs of the same evening, the four llama.cpp GGUF recipes and the gpt-oss-20b pinned-store check, are dated 2026-11-17 and sit in [research/2026-11-17/](../2026-11-17/README.md), together with C's lane report in [sglang-runtime/measure-c/](../2026-11-17/sglang-runtime/measure-c/README.md), its scripts, lane logs, estimates and download records, and the earlier SGLang records of the hands-on run and measurements A and B.
Qwen3.6-35B-A3B runs on SGLang with Paged Experts at 151.3 tok/s; Nemotron 3.5 Lightning, Gemma 4 26B-A4B and K2 Horizon MoVA 36B-A4B fail on SGLang v0.5.23; vLLM DeepSeek-R1-0528-Qwen3-8B starts cold and warm at 64k with one more argument.
Do not edit these records by hand.
A script replaced local paths in these records with `$HOME` and `$SCRATCH`, and the user and host names with `<user>` and `<host>`, before they were copied into the repository.

## Setup

- Machine: MSI Raider 18 Max HX A2WI, RTX 5080 Laptop GPU (16,303 MiB per `nvidia-smi`), 61 GiB of RAM; the enforced power limit read 140 W before most starts and 145 W before the rest, as each record's `.meta` file says.
- SGLang records: `lmsysorg/sglang:v0.5.23-runtime` started by hand with `docker run`, not by `./llmctl up`, one container at a time, with measurement C's model overlay at `/llmctl/model`, its environment including `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` in every start except Nemotron's diag-k72-f088-noround, Paged Experts with the Marlin MoE runner, and GPU and cgroup memory traced every 1 s ([sglang-runtime/measure-c/README.md](../2026-11-17/sglang-runtime/measure-c/README.md)).
- vLLM record: `vllm/vllm-openai:v0.33.0` started by hand with `docker run` as llmctl's `container_spec` builds the shipped recipe, with the tokenizer folder and the changes the record names, GPU and cgroup memory traced every 1 s.
- Full logs keep the request lines from Docker's bridge 172.17.0.1, which identifies no machine or person.

## Folders

| Folder | Runtime and launch | Main result |
|---|---|---|
| [sglang-qwen3.6-35b-a3b/](sglang-qwen3.6-35b-a3b/README.md) | SGLang v0.5.23 by hand in measurement C, 131072 context, BF16 KV, 8 Mamba slots, Paged Experts with 88 of 256 experts resident, fraction 0.88, plus the M11 sweep, a 27 GiB cap run and two offline cold starts | Measured: 151.3 tok/s decode, a 161,736-token pool, 951 MiB free after ready and at least 465 MiB through the whole protocol (a 486 MiB drop, which raises D8's threshold to 768 MiB), 300 s of load without error, a 16.97 GiB pinned expert store, offline cold start 345.5 s |
| [sglang-nemotron-3.5-lightning/](sglang-nemotron-3.5-lightning/README.md) | SGLang v0.5.23 by hand in measurement C, 131072 context, BF16 KV, Paged Experts, the M11 sweep and two diagnostic starts at K 72 and fraction 0.88 | Failed the smoke gate: loads and holds 128k at 133.3 tok/s, but no tool call in any thinking mode and M13 10 of 20, against tool calls on llama.cpp |
| [sglang-gemma-4-26b-a4b/](sglang-gemma-4-26b-a4b/README.md) | SGLang v0.5.23 by hand in measurement C, `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`, 131072 context, FP8 KV, Paged Experts, four starts | Failed: the vision tower stopped the first three starts while loading, and the fourth, with a `quantization_config` override, died on `Only gated SiLU/SiTU is supported, got gelu` in the Marlin MoE runner |
| [sglang-k2-horizon-mova-36b/](sglang-k2-horizon-mova-36b/README.md) | SGLang v0.5.23 by hand in measurement C, `cyankiwi/K2-Horizon-MoVA-36B-A4B-AWQ-INT4`, 65536 context, Paged Experts, two starts | Failed before loading: `K2 Horizon MoVA supports unquantized bf16/fp16 weights only` |
| [vllm-deepseek-r1-0528-qwen3-8b-cold-start/](vllm-deepseek-r1-0528-qwen3-8b-cold-start/README.md) | The shipped vLLM recipe by hand in measurement C, with llmctl's tokenizer folder and `--tokenizer /llmctl/tokenizer`, cold and warm, three starts | Measured: the default 0.92 leaves 3.83 GiB of KV cache cold, short of 64k; with `--kv-cache-memory-bytes 5153960755` cold and warm starts got 69,904 tokens, 1,105 MiB free after ready, llama.cpp's token counts, correct replies and 72.5 tok/s |

Each `sglang-<id>/` folder holds a `README.md` with its result line and, under `measure-c/`, measurement C's report on the model, the `docker run` files, the logs and traces, the request and reply shapes and the overlay's generation config.
The vLLM folder holds a `README.md` with the result line, measurement C's report as `measure-c.md`, the `docker run` files, the logs, the shapes and the tokenizer folder's real `tokenizer_config.json` with its link list.
