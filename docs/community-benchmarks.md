# Community benchmarks and calculators for the RTX 5080 Laptop

Last verified: 2026-11-16.

None of the third-party sources reviewed for this knowledge base contains a measurement on an RTX 5080 Laptop GPU; the only measurements on this laptop so far are the gpt-oss-20b runs of 2026-11-16 ([research/2026-11-16/](../research/2026-11-16/README.md)).
Every throughput number below is either a desktop RTX 5080 measurement or a calculator estimate, and each row says which.
The most useful anchors are the llama.cpp gpt-oss-20b run on a desktop RTX 5080 (tg128 204.85 tok/s), the TechFuelHQ Ollama dataset, and a handful of llama.cpp issue-thread measurements of our picks on other cards.
Calculators (CanIRun, ModelFit, Twoody, Periscale) mislead on this machine because most of them do not model llama.cpp MoE expert offload, model-specific quants or sliding-window KV, so use them only as sanity checks.
The user's own LibreDB Studio agent harness is a separate, first-hand evidence class for tool reliability, measured on an Apple M5 Max; see [the harness section](#libredb-studio-agent-harness-first-hand-m5-max) and [agent-harness-evidence.md](agent-harness-evidence.md).

## How to read third-party numbers for this laptop

The desktop and laptop RTX 5080 share a name but not a power budget.

| Property | Desktop RTX 5080 | RTX 5080 Laptop (this machine) |
|---|---|---|
| CUDA cores | 10752 ([NVIDIA](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/)) | 7680 ([NVIDIA](https://www.nvidia.com/en-us/geforce/laptops/50-series/)) |
| Boost clock | 2.62 GHz ([NVIDIA](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/)) | 1500-2287 MHz ([NVIDIA](https://www.nvidia.com/en-us/geforce/laptops/50-series/)) |
| Memory | 16 GB GDDR7, 256-bit | 16 GB GDDR7 (16303 MiB observed), 256-bit |
| Bandwidth | 960 GB/s (as modelled by [ModelFit](https://modelfit.io/gpu/rtx-5080/) and [Twoody](https://www.twoody.com/local-llm-calculator/data.json); NVIDIA's desktop page does not print it) | 896 GB/s ([NVIDIA](https://www.nvidia.com/en-us/geforce/laptops/50-series/)) |
| Power | 360 W total graphics power ([NVIDIA](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/)) | 140 W limit set here (80 W default, 175 W max, observed with nvidia-smi) |

Four rules follow from this.

1. Decode at batch 1 is mostly bandwidth-bound, so scale desktop decode by 896/960 = 0.933 as an upper bound (estimate), and expect less if the 140 W limit lowers clocks; the reason is that a single stream spends its time loading weights ([Dettmers and Zettlemoyer, arXiv 2212.09720](https://arxiv.org/abs/2212.09720)).
2. Prefill is compute-bound, so it scales with cores and clock rather than bandwidth: 7680 x 2.287 GHz against 10752 x 2.62 GHz gives about 0.62 of the desktop at full boost (estimate), and less if the laptop does not hold boost.
3. Runtime matters: on the same desktop card, llama.cpp b10507 decoded 2-6% faster than Ollama 0.32.1 on dense models, and Ollama prefilled 4-7% faster at about 420-token prompts ([TechFuelHQ, Ollama vs llama.cpp on RTX 5080](https://techfuelhq.com/self-hosted/ollama-vs-llamacpp-rtx-5080-2026/), measured 2026-08-20).
   The gpt-oss gap there (196 vs 222-238 tok/s) comes mostly from Ollama shipping a different conversion of the model, not from wrapper overhead.
4. Depth matters: most published numbers are at empty or short context, while OpenCode runs at 64k-128k (see [opencode.md](opencode.md)).
   The table below shows what depth did in the few measurements that report it.

| Measurement | Short context | Long context | Change | Source |
|---|---|---|---|---|
| gpt-oss-20b MXFP4, desktop RTX 5080, llama.cpp build 6210, prefill | pp2048 7477 tok/s | pp32768 5531 tok/s | -26% | [ggml-org gpt-oss guide #15396](https://github.com/ggml-org/llama.cpp/discussions/15396) |
| Gemma 4 26B-A4B UD-Q6_K_XL, --n-cpu-moe 23, q8_0 KV, desktop RTX 5080 (OC), decode | tg128 40.92 tok/s at depth 0 | 30.89 tok/s at depth 100000 | -25% | [u/grumd llama-benchy table](https://www.reddit.com/r/LocalLLaMA/comments/1sdu8oz/bartowski_vs_unsloth_for_gemma_4/oem5j5k/) |
| Same run, prefill | pp4096 2590.02 tok/s | 2104.35 tok/s at depth 100000 (TTFT 49.6 s) | -19% | same |
| Qwen3.8-27B UD-Q3_K_XL, all on GPU, RTX 4080 SUPER 16 GB, llama-server, q8_0 KV, no FA, decode | about 38-39 tok/s at zero context | 33.1 tok/s at position 68,642 | about -14% | [llama.cpp #27623](https://github.com/ggml-org/llama.cpp/issues/27623) |
| Qwen3.8-27B UD-Q4_K_XL, 2x RTX 3090 (layer split, 250 W limit), llama-bench, decode | tg32 40.92 tok/s at depth 0 | 32.49 tok/s at depth 81920 | -21% | [#27623 comment](https://github.com/ggml-org/llama.cpp/issues/27623#issuecomment-5398821159) |
| Qwen3.8-27B UD-Q4_K_M, RTX 5090, decode | tg32 76.5 tok/s at depth 0 | 59.3 tok/s at 98K | -22% | [llama.cpp #27623](https://github.com/ggml-org/llama.cpp/issues/27623) |

Two more traps recur across sources.
Warm repeats of the same prompt hit the prompt cache: TechFuelHQ saw about 60,000 tok/s "prefill" that measured nothing until it added a per-run nonce ([dataset README](https://huggingface.co/datasets/iBlessi/rtx-5080-llm-throughput)).
Very short prompts make prefill rates measure per-request overhead: a 40-token prompt showed Ollama 32-55% ahead on prefill, which collapsed to 4-7% at about 420 tokens ([TechFuelHQ](https://techfuelhq.com/self-hosted/ollama-vs-llamacpp-rtx-5080-2026/)).
Our own method avoids both; see [llamacpp/benchmarking.md](../llamacpp/benchmarking.md) and [vllm/benchmarking.md](../vllm/benchmarking.md).

## Measured throughput data

All rows in this section are third-party measurements on other hardware.
None of them is a laptop number.

### llama.cpp reference runs (desktop)

| Model and quant | Hardware | Runtime | Prefill tok/s | Decode tok/s | Source |
|---|---|---|---|---|---|
| gpt-oss-20b MXFP4 (ggml-org), all on GPU | Desktop RTX 5080 16 GB | llama.cpp build 6210 (a094f381), CUDA 12.8, -fa 1 | pp2048 7477, pp32768 5531 | tg128 204.85 | [ggml-org guide #15396](https://github.com/ggml-org/llama.cpp/discussions/15396) (measured) |
| gpt-oss-20b, ggml-org GGUF | Desktop RTX 5080 | llama-server b10507, 4096 ctx, 256-token output | not run | 222-238 (published median 222 includes a cold run at 186; varied prompts 175-243, median 235.3) | [TechFuelHQ](https://techfuelhq.com/self-hosted/ollama-vs-llamacpp-rtx-5080-2026/) (measured) |
| Llama 3.2 3B | Desktop RTX 5080 | llama-bench / llama-server b10507 / Ollama 0.32.1 | 16,809 (llama-server), 17,444 (Ollama) | 367 / 338 / 319 | [TechFuelHQ](https://techfuelhq.com/self-hosted/ollama-vs-llamacpp-rtx-5080-2026/) (measured) |
| Qwen 2.5 14B | Desktop RTX 5080 | llama-server b10507 / Ollama 0.32.1 | 4,697 / 5,004 | 101.4 / 99.4 | [TechFuelHQ](https://techfuelhq.com/self-hosted/ollama-vs-llamacpp-rtx-5080-2026/) (measured) |
| Llama 2 7B Q4_0 | Desktop RTX 5080 | llama.cpp CUDA, commit 8a4280c, submitted by @Hedede | pp512 9487.70 | tg128 184.68 | [llama.cpp #15013](https://github.com/ggml-org/llama.cpp/discussions/15013) (measured) |
| Llama 2 7B Q4_0 | Desktop RTX 5070 Ti (896 GB/s, the laptop's bandwidth) | llama.cpp CUDA, commit 933414c | pp512 8419.56 | tg128 182.43 | [llama.cpp #15013](https://github.com/ggml-org/llama.cpp/discussions/15013) (measured) |
| Llama 2 7B Q4_0 | DGX Spark (273 GB/s) | llama.cpp CUDA | pp512 3661.37 | tg128 56.74 | [llama.cpp #15013](https://github.com/ggml-org/llama.cpp/discussions/15013) (measured) |
| Llama 2 7B Q4_0 | Desktop RTX 5080 | llama.cpp, "fastest backend", build and date not given | 9,305 | 195 | [Tech Centurion](https://www.techcenturion.com/local-llm-gpu-rankings) (measured, republished scoreboard) |

The 5070 Ti row is the closest public proxy for this laptop's memory system: at 896 GB/s it decodes Llama 2 7B within about 1% of the desktop 5080, while its prefill is about 11% lower.
That suggests laptop decode can approach desktop decode if clocks hold at 140 W, and that prefill is where the laptop will fall furthest behind (estimate).
Tech Centurion's 195 / 9,305 row does not match the #15013 CUDA row (184.68 / 9487.70); the page does not say which backend or build produced it.

### TechFuelHQ RTX 5080 dataset (Ollama 0.32.1, desktop)

[TechFuelHQ RTX 5080 LLM Throughput v1.1.0](https://huggingface.co/datasets/iBlessi/rtx-5080-llm-throughput) (CC BY 4.0, mirror of [techfuelhq.com](https://techfuelhq.com/data/rtx-5080-llm-throughput/)) has 34 measured rows on a retail RTX 5080 (desktop by inference: Ryzen 7 7800X3D host, Windows 11, driver 610.88).
Each cell is the median of 3 runs with a fresh prefill (nonce), temperature 0, seed 42, a 256-token output, num_ctx 4096 or 16384, and prompts of 467-2597 tokens.
Captures are from 2026-08-12 and 2026-08-26; the 2026-11-04 page update added no new GPU runs.
Power limit, clocks, KV type and flash-attention setting are not recorded, and the VRAM column is a whole-GPU reading that can include a leftover model (llama3.2:3b reads 15682 MiB at 4k and 5495 MiB at 16k).

| Model (Ollama tag) | Quant | Context | Prefill tok/s | Decode tok/s | Whole-GPU VRAM MiB |
|---|---|---|---|---|---|
| gpt-oss:20b | MXFP4 | 4k and 16k, 503-1658 prompt tokens | 9078-11001 | 178.34-190.14 (flat across 4k and 16k) | 13381-14355 |
| qwen2.5:14b | Q4_K_M | 4k and 16k | 4864-5226 | 94.48-97.02 | 10841-13159 |
| qwen2.5-coder:14b | Q4_K_M | 4k and 16k | 4842-5272 | 93.41-94.89 | 10169-12487 |
| qwen2.5:7b | Q4_K_M | 4k and 16k | 9672-10152 | 176.61-180.02 | 6190-7034 |
| qwen2.5-coder:7b | Q4_K_M | 4k and 16k | 9472-9825 | 171.41-174.58 | 5671-6515 |
| llama3.2:3b | Q4_K_M | 4k and 16k | 17681-19397 | 298.21-312.89 | see caveat above |
| qwen3-coder:30b (Qwen3-Coder-30B-A3B), partly offloaded | Q4_K_M, 19 GB file | 4k | 1744-2369 | 61.55-61.89 | 15101 |
| qwen3-coder:30b, partly offloaded | Q4_K_M | 16k | 1525-1928 | 51.62-51.65 | 15003 |
| devstral:24b | Q4_K_M, 14 GB file | 4k only | 5335-9377 (9377 looks anomalous) | 34.9-36.32 | 14997 |

What the dataset says about offload, with the check's corrections applied:

- qwen3-coder:30b was 74% GPU-resident under Ollama's whole-layer offload, measured at 8k context, not at the 4k or 16k rows ([TechFuelHQ](https://techfuelhq.com/data/rtx-5080-llm-throughput/)).
- Its prefill was 4-6x below the fully resident models, which supports llama.cpp `--n-cpu-moe` over Ollama for MoE models that spill (see [llamacpp/README.md](../llamacpp/README.md)).
- The author saw 0.7, 62 and 83 tok/s for the same model under three host states, and attributes the spread to load-time memory state and the resulting GPU/CPU split, not only to free RAM ([TechFuelHQ](https://techfuelhq.com/data/rtx-5080-llm-throughput/)).
- devstral:24b fit only at 4k; at 8k its KV spilled and decode fell below 1 tok/s.
- Measured on this laptop on 2026-11-16 with Ollama 0.40.16 and a 32k context, qwen2.5:7b decoded at 119.4 tok/s and prefilled 8k tokens at 5,044 tok/s, about 67% and 50% of the dataset's desktop rows, which used Ollama 0.32.1 at 4k-16k with 467-2597-token prompts ([agent-harness-evidence.md](agent-harness-evidence.md#try-here-first)).

Laptop estimate for gpt-oss:20b under Ollama: 178.34-190.14 x 0.933 = about 166-177 tok/s (estimate, Ollama's own gpt-oss blob only, power limit not included).
For llama.cpp with the upstream ggml-org GGUF, scale the llama.cpp rows instead: 204.85 x 0.933 = about 191 tok/s tg128 (estimate).

### Issue-thread and model-card measurements of our picks on other GPUs

These were collected while verifying each pick; see [model-selection.md](model-selection.md) for the picks themselves.

| Model | Hardware | Runtime and setup | Result | Source |
|---|---|---|---|---|
| Qwen3.8-27B UD-Q3_K_XL | RTX 4080 SUPER 16 GB (sm_89) | llama-server, all on GPU | about 38-39 tok/s at zero context, 33.1 at a 68K position (q8_0 KV, no FA); prefill about 1300 tok/s at a 91K fill; a later comment retracted the ">80K cliff" as a metric error and measured 32.4 tok/s at a 91K position (q4_0 KV, FA) | [llama.cpp #27623](https://github.com/ggml-org/llama.cpp/issues/27623) (measured) |
| Qwen3.8-27B UD-Q4_K_XL (16.34 GiB) | 2x RTX 3090, layer split, 250 W limit | llama-bench b10603 | tg32 40.92 at depth 0, 32.49 at 81920, 28.68 at 131072 | [#27623 comment](https://github.com/ggml-org/llama.cpp/issues/27623#issuecomment-5398821159) (measured) |
| Qwen3.8-27B UD-Q4_K_M | RTX 5090 | llama.cpp b10615 | tg32 76.5 at depth 0, 59.3 at 98K; llama-server 60 decode and 2342 prefill at 90K | [#27623 comment](https://github.com/ggml-org/llama.cpp/issues/27623#issuecomment-5406012488) (measured) |
| Qwen3.8-27B UD-IQ4_XS | RTX 2000 Ada (sm_89) | llama.cpp | q8_0 KV 5.9 tok/s vs f16 KV 13.3 tok/s | [llama.cpp #27623](https://github.com/ggml-org/llama.cpp/issues/27623) (measured, conflicting with the 4080 SUPER report) |
| Qwen3.6-35B-A3B Q4_K_M | RTX 3060, PCIe 4.0 | llama.cpp, -ncmoe 26 | pp2048 about 1143 tok/s (1880 with a prototype patch) | [llama.cpp #25859](https://github.com/ggml-org/llama.cpp/issues/25859) (measured) |
| GLM-4.7-Flash Q4_K | RTX 4090 | llama.cpp, all on GPU | pp2048 about 8,440-8,590 tok/s | [llama.cpp PR #30489](https://github.com/ggml-org/llama.cpp/pull/30489) (measured) |
| GLM-4.7-Flash Q8_0 | RTX PRO 6000 Blackwell (sm_120, 96 GB) | llama.cpp with MTP speculative decoding | about 128-130 tok/s mean decode on coding turns | [llama.cpp PR #30196](https://github.com/ggml-org/llama.cpp/pull/30196) (measured) |
| Ornith-1.5-9B community W4A4 NVFP4 | RTX PRO 6000 Blackwell (1792 GB/s, shared box) | vLLM | 151 tok/s decode, 192.5 with MTP K=1 | [protoLabsAI model card](https://huggingface.co/protoLabsAI/Ornith-1.5-9B-NVFP4) (community-reported) |

The q8_0 KV question for Qwen3.8-27B is unresolved: the [RTX 2000 Ada report](https://github.com/ggml-org/llama.cpp/issues/27623#issuecomment-5706129096) shows a large q8_0 decode penalty and an [RTX 4090 commenter](https://github.com/ggml-org/llama.cpp/issues/27623#issuecomment-5752465933) reports the same, the [4080 SUPER reporter](https://github.com/ggml-org/llama.cpp/issues/27623#issuecomment-5421513201) saw no difference between q4 and q8 KV, and the fix PR [#31802](https://github.com/ggml-org/llama.cpp/pull/31802) is still open.
Measure both KV types on this laptop before trusting either.

### u/grumd: 18 models on a desktop RTX 5080 (quality and speed)

[u/grumd on r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1s9mkm1/benchmarked_18_models_that_i_can_run_on_my_rtx/) (2026-04-01) ran Nick Lothian's 25-task SQL tool-calling benchmark on a desktop RTX 5080 16 GB (overclocked, the author claims about 1000 GB/s), Ryzen 7 9800X3D, 96 GB DDR5-6000, PCIe 5.0 x16, self-built llama.cpp.
Each model ran once at temperature 0.1, and the author says identical runs vary by 1-3 tasks, so differences of one or two points mean nothing.
The runs date from April 2026, so the Qwen entries are the Qwen3.5 generation, older than most of our picks.

| Model and quant | Score /25 |
|---|---|
| Qwen3.5-122B-A10B UD-Q4_K_XL (experts in about 68 GB RAM) | 24 |
| Qwen3.5-27B IQ4_XS (bartowski) | 23 |
| Qwen3.5-122B-A10B UD-IQ3_XXS (about 33 GB RAM) | 23 |
| Qwen3.5-35B-A3B UD-Q6_K_XL | 22 |
| Nemotron-3-Super-120B-A12B UD-Q4_K_S | 21 |
| Qwen3-Coder-Next UD-Q5_K_XL | 20 |
| GLM-4.7-Flash UD-Q6_K_XL | 19 |
| Qwen3-Coder-Next UD-Q4_K_XL | 16 |
| Devstral-Small-2-24B IQ3_S (byteshape) | 16 |
| Qwen3.5-9B UD-Q6_K_XL (mostly malformed tool calls) | 5 |

Later comments add Qwen3.8-27B UD-Q6_K_XL with q8_0 KV at 23/25 (2026-08-24), but by then the author had moved to 2x RTX 3080 20 GB, and from mid-June to early July he ran a 5080 + 3080 20 GB pair, so the Ornith-1.0-35B Q8_0 result ("5 errors") and the "Qwen3.6-27B is very quant-sensitive" remark are probably not single 16 GB card results.
The author also judged a dense 27B squeezed to Q3/IQ4_XS on 16 GB weaker than the 35B-A3B MoE at Q6/Q8 with experts in RAM on an Aider suite (opinion, Qwen3.5 generation), which is a reason to A/B pick 2 (Qwen3.6-35B-A3B) against pick 1 (Qwen3.8-27B) locally.

Speed figures from the same author are recalled ranges from comments, not tables, except the Gemma 4 llama-benchy table shown earlier:

| Model and quant | Setup | Prefill tok/s | Decode tok/s | Label |
|---|---|---|---|---|
| Qwen3.5-122B-A10B UD-Q4_K_XL | Experts in RAM, -ub 2048, about 131k ctx | 1000-1500 | 15-19 | measured, recalled |
| Qwen3.5-122B-A10B UD-IQ3_XXS | --fit on, -b/-ub 2048, 131k ctx | 1500-2000 | 25-30 | measured, recalled |
| Qwen3.5-122B-A10B IQ3 | -ub 512 vs -ub 2048-4096 | under 300 vs about 1500 | n/a | measured, recalled |
| Qwen3.6-35B-A3B UD-Q6_K_XL | --fit on, f16 KV, 262k ctx, 15 GB VRAM + 29 GB RAM | n/a | 35 | measured |
| Qwen3.5-35B-A3B UD-Q6_K_XL | Some experts in RAM, short test context | 2000+ | about 70 | measured, recalled |

The tuning lessons transfer better than the numbers: raise `-ub` to 2048 or more when experts sit in RAM, keep experts resident instead of memory-mapped (his `--no-mmap` maps to `-lm`/`--load-mode` at b12418), avoid q4 KV, and leave `--threads` at its default (he lost 20% or more with 16 threads).
Details are in [llamacpp/README.md](../llamacpp/README.md).

### Nick Lothian's agentic SQL benchmark

[LLM SQL Benchmark](https://sql-benchmark.nicklothian.com/) ([post](https://www.reddit.com/r/LocalLLaMA/comments/1s7r9wu/i_tested_as_many_of_the_small_local_and/), [repo](https://github.com/nlothian/llm-sql-benchmark), data 2026-03-17 to 2026-06-10) runs 25 text-to-SQL questions through a `run_sql_query` / `results_ok` tool loop with at most 20 tool calls.
Temperature is hard-coded to 0.1, scoring checks only row count, column names and the first row, each model ran once, and most models ran through OpenRouter with unpinned providers.
The default per-question timeout is 120 s; the author's local small-model runs used 300 s.
There is no RTX 5080 data in it: local runs were on a GTX 1070 8 GB.

| Model | Where it ran | Score /25 | Note |
|---|---|---|---|
| Qwen3.5-27B | OpenRouter | 23 | Two commenters reproduced 23/25 locally with bartowski Q4_K_L |
| Qwen3.5-35B-A3B | OpenRouter | 21 | 23,393 output tokens |
| Qwen3.5-35B-A3B UD-IQ3_S, experts on CPU | GTX 1070 8 GB | 20 (write-up says 19) | 16 tok/s effective wall-clock rate |
| Gemma 4 26B-A4B | OpenRouter | 21 | 7,110 output tokens |
| Qwen3-Coder-Next | OpenRouter | 16 | 4 tool errors |
| Nemotron 3 Nano 30B-A3B | OpenRouter | 15 | about 79k output tokens |
| gpt-oss-20b | OpenRouter paid, 120 s timeout | 10 | 14 errors |
| gpt-oss-20b | OpenRouter free, 600 s timeout | 20 | provider and timeout, not the model |

Two findings carry over: small Qwen3.5 models did slightly worse with thinking on (9B 16 vs 15, 4B 10 vs 7, at temperature 0.1), and 2-bit quants of small models collapsed (Qwen3.5-4B UD-IQ2_XXS 1/25 against 10/25 for Q4_0 and Q8_0) ([write-up](https://sql-benchmark.nicklothian.com/#quantization)).
The "effective" speeds in its result files are output tokens over summed question time, including prefill and SQL execution, not decode rates.

### LibreDB Studio agent harness (first-hand, M5 Max)

This is the user's own harness, not a community run, and it is the first first-hand tool-reliability evidence in this knowledge base; the full 40-row table and per-model caveats are in [agent-harness-evidence.md](agent-harness-evidence.md).
Every figure in this subsection is (measured on the M5 Max, libredb-studio).
LibreDB Studio's Agent mode drives a model through six surfaces (Investigate, Optimize, Assess, Operate, Analyze, Plan) against an embedded SQLite sample, on an Apple M5 Max with 64 GB of unified memory under Ollama 0.32.13 and 0.33.0 ([docs/llms/methodology.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/methodology.md)).
A model is listed only when all six surfaces passed five consecutive times, 30/30 in all ([docs/llms/README.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/README.md)).
The prompts are short SQLite tasks, each turn is capped at 90 s (150 s for nine models), and the published figures are whole-run durations (median and slowest passing run, in seconds), not tok/s.
SQL correctness is largely unchecked, and the loop adds report reminders and empty-turn retries that OpenCode does not have.

The transfer basis is bandwidth: the M5 Max has 614 GB/s of unified memory bandwidth ([Apple MacBook Pro specs](https://www.apple.com/macbook-pro/specs/)) against 896 GB/s here.
A model that fits fully in VRAM should run faster here, up to 896/614 = 1.46x on bandwidth-bound decode (estimate), and a model that spills to DDR5 (102.4 GB/s peak) should run slower.
Pass/fail transfers to this laptop as tool-reliability evidence; the durations do not.

Selected rows, for the KB's picks and the candidates worth measuring first:

| Ollama tag | Mac median / slowest run (s) | KB status | Note |
|---|---|---|---|
| qwen3.8:latest | 72 / 195 | Rank 1 | Ollama Q4_K_M, not the KB's llama.cpp UD-Q3_K_XL path |
| qwen3.6:35b | 13 / 172 | Rank 2 | Passed only with thinking suppressed on agent turns |
| gemma4:26b | 36 / 92 | Rank 5 | Needed an empty-turn retry; one loss was an empty completion after 11 tool calls |
| gpt-oss:20b | 37 / 169 | Rank 9 | Passed with reasoning suppressed on agent turns |
| qwen3-coder:30b | 11 / 21 | Honorable mention | Every cell on its first attempt |
| nemotron-3.5-lightning:30b | 29 / 145 | Honorable mention | Passed with thinking on and a 150 s limit |
| gemma4:12b | 11 / 63 | Honorable mention | Passed only with reasoning suppressed |
| qwen3.5:9b | 33 / 98 | Honorable mention | Conflicts with the 5/25 malformed-tool-call llama.cpp result in the SQL benchmark above |
| ministral-3:8b | 13 / 26 | Excluded | Every cell on its first attempt at the defaults |
| qwen2.5:7b | 6 / 21 | Excluded | Cheapest cross-check that the harness reproduces on this GPU |

### Other single rows

| Model | Hardware | Runtime | Result | Label | Source |
|---|---|---|---|---|---|
| Qwen3.5 4B Instruct Q4_K_M | "NVIDIA GeForce RTX 5080" (desktop presumed) | Ollama | 163.8 tok/s decode, context not stated | measured, 1 community run | [ModelFit](https://modelfit.io/gpu/rtx-5080/) |
| Qwen3.5-27B | RTX 4090 24 GB, Windows (1.1 GB used by desktop) | llama.cpp, bartowski Q4_K_L | fits 90112 ctx with f16 KV, 159744 with q8_0 KV | measured, commenter | [r/LocalLLaMA thread](https://www.reddit.com/r/LocalLLaMA/comments/1s7r9wu/i_tested_as_many_of_the_small_local_and/) |

### Numbers that do not transfer

- Artificial Analysis speed, latency and price data come from cloud API providers ([AA methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking)).
- BenchLM's tok/s and latency columns are hosted-API metrics, and its self-host VRAM lines are full-precision datacenter estimates ([BenchLM](https://benchlm.ai/best/open-source)).
- OpenRouter's "fastest" figures (for example Qwen3.8 27B on Cerebras, gpt-oss-20b on Groq) are rolling hosted throughput on datacenter hardware ([OpenRouter rankings](https://openrouter.ai/rankings)).
- Unsloth's NVFP4 speed tables are vLLM on 1x B200 at up to 128 concurrency, for example Qwen3.8-27B NVFP4 133.7 vs BF16 89.8 tok/s at batch 1 (vendor-reported, [Unsloth docs](https://unsloth.ai/docs)).
- The example outputs in the vLLM benchmarking docs come from unspecified hardware and show output format only ([vLLM docs](https://docs.vllm.ai/en/latest/benchmarking/)).
- LibreDB Studio harness durations are whole agent runs on an M5 Max at 614 GB/s, not tok/s; only their pass/fail column transfers ([agent-harness-evidence.md](agent-harness-evidence.md)).

## Calculator estimates

None of these four sites measures anything.
CanIRun, ModelFit and Periscale each mark at least one of our working configurations as not fitting, for reasons that are artifacts of the calculator; Twoody models CPU spill but not which tensors go to RAM.

| Calculator | Laptop entry | Speed model | Fit model | What it misses here | Credibility |
|---|---|---|---|---|---|
| [CanIRun.ai](https://canirun.ai/device/rtx-5080-laptop) | Yes: 16 GB, 896 GB/s, 7680 cores | round(896 / effective GB x 0.7); MoE effective GB = active/total x file size; spill uses a fixed 50 GB/s RAM term x 0.85 | can-run up to 0.85 x VRAM, tight up to 1.1 x VRAM, offload only if system RAM > VRAM | Its system RAM comes from the browser (navigator.deviceMemory caps at 8, so desktops show 16 GB), so every model over 17.6 GB is "Too heavy" until you enter 61 GB by hand; no KV, context, runtime, power or MXFP4/UD/QAT sizes | low |
| [ModelFit](https://modelfit.io/gpu/rtx-5080/) ([dataset](https://modelfit.io/api/dataset/performance/)) | No: desktop 960 GB/s only; no discrete laptop GPUs among 126 hardware keys | (weight bytes x MoE factor + alpha x fp16 KV bytes) / (eta x bandwidth) + overhead, fitted on 4,831 runs; 10.7% median out-of-sample error (8.1% CUDA, 14.2% CUDA MoE) | 90% of 16 GB = 14.4 GB usable (the FAQ says 15.5 GB); Q4_K_M and fp16 KV with a generic 8 heads x 128 dim class except Qwen hybrids | No expert offload ("rough estimate, offload not measured", no error band); sliding-window KV overstated; gpt-oss weights at safetensors size (13.8 GB) | low to medium |
| [Twoody](https://www.twoody.com/local-llm-calculator/rtx-5080) ([data.json](https://www.twoody.com/local-llm-calculator/data.json)) | No: desktop only | Device tg128 on Llama 2 7B Q4_0 (184.68 from #15013) scaled by bytes read per token, gpu_efficiency 0.85, cpu_efficiency 0.55; displays anything above 100 as "100+"; claims 4.5% median error on 20 held-out runs (not published) | file + mmproj + 0.3 GiB vision buffers + KV (q8_0 = 0.53125 x f16 for hybrids, f16 otherwise) + 0.3 GiB work buffers against 14.6 "GB" (actually GiB) | Offload strategy (layers or experts) unspecified; static page assumes 32 GB RAM, the 64 GB figures come from the [results endpoint](https://www.twoody.com/local-llm-calculator/results?device=rtx-5080&memory=64) | medium |
| [Periscale](https://periscale.ai/gpus/rtx-5080-laptop) | Name only: laptop and desktop pages are identical; bandwidth field empty | None | weights (fixed bits per weight per quant name) + FP16 KV at 8K on every layer + 1.56-1.96 overhead; "headroom" within 85% of VRAM | All MoE parameters in VRAM; KV 4x too high for Qwen3.6 hybrids; units look like GiB labelled GB; "largest context 128K" contradicts its own totals | low |

### What they say about our picks, next to the baseline arithmetic

Calculator figures are decode tok/s estimates; "desktop" means the calculator modelled the 960 GB/s card.
The last column is the baseline's own bandwidth arithmetic for this laptop (estimate, from the per-model verification records); six of these rows have since been measured, as listed after the table.

| Pick and our config | CanIRun (laptop) | ModelFit (desktop) | Twoody (desktop) | Periscale | Baseline estimate for this laptop |
|---|---|---|---|---|---|
| gpt-oss-20b, ggml-org MXFP4 11.28 GiB, all on GPU | about 324 (Q4_K_M) | 168 (158.9 on the 896 GB/s 5070 Ti proxy); "about 2k" fp16 context on the page vs 24,554 in its dataset | 100+ | fits at Q4_K_M, 83% at 8K | about 191 tg128 (desktop 204.85 x 0.933); bandwidth ceiling about 240 |
| Qwen3.6-35B-A3B, Q4_K_M 19.13 GiB, --n-cpu-moe 21 | Too heavy at the 16 GB RAM default; 94 with 64 GB entered | No fit; 83 rough | 27-54 (Q4_K_M, 64 GB RAM, 64k q8_0) | Will not run at Q4_K_M; Q2_K only | 40-90, ceiling about 160 |
| Qwen3.8-27B, UD-Q3_K_XL 12.24 GiB, all on GPU | 50 (Q3_K_M), 44 (Q4_K_M, tight) | Partial offload, 10 (Q4_K_M 16.5 GB) | 3.1-6.3 (Q4_K_M spilled) | Q2_K only, 87% | 40-45, ceiling about 72; 4080 SUPER measured about 38-39 at zero context |
| Gemma 4 26B-A4B, Google QAT Q4_0 13.45 GiB, all on GPU | 296 (Q4_K_M, tight) | Partial offload, 125 rough (Q4_K_M 16 GB) | 100+, "15.5 of 14.6" yet labelled Runs | 111% at Q4_K_M; Q2_K headroom | 150-220, ceiling about 375 |
| Gemma 4 12B, honorable mention (vLLM QAT W4A16 9.56 GiB) | about 95 (Q4_K_M) | 95.5 (Q4) | 81-117 (QAT Q4_0) | not checked | 65-90 in vLLM, ceiling about 110 |
| Nemotron 3.5 Lightning, honorable mention, Q4_0 17.60 GiB, --n-cpu-moe 16 | about 394 (Q4_K_M 15.9 GB, tight) | not listed | not listed | Q2_K, 83% | 60-110, ceiling about 170 |
| Qwen3-Coder-Next, UD-IQ4_XS 35.79 GiB, --n-cpu-moe 37 | cannot run at the default RAM; can-run-slow with 61 GB entered | not listed | not listed (Qwen3-Next-80B-A3B, the non-Coder sibling, is 13-25 at Q4_K_M with 64 GB RAM) | Will not run | 30-50, ceiling about 100-110 |
| GLM-4.7-Flash, honorable mention, Q4_K_M 17.05 GiB, --n-cpu-moe 20 | not in catalog | not listed | 45-78 (Q4_K, 64 GB RAM) | Q2_K, 98% | 40-80 (estimated at --n-cpu-moe 18; the recipe now uses 20, see [llamacpp/models.md](../llamacpp/models.md)) |

Measured on this laptop on 2026-11-16 with llama.cpp `server-cuda-b12407` ([llamacpp/models.md](../llamacpp/models.md)):

- gpt-oss-20b decoded at 166 tok/s: ModelFit's desktop 168 was the closest figure, the baseline's 191 was 15% high, and CanIRun's 324 was about double.
- Qwen3.6-35B-A3B at `--n-cpu-moe 21` decoded at 67.3 tok/s, inside the baseline's 40-90; with `-lm none` and a 3,200 MiB expert cache at `--n-cpu-moe 25` it reached 122.5 tok/s, above every figure in its row.
- Qwen3.8-27B decoded at 36.4 tok/s, below the baseline's 40-45 and CanIRun's 50 for Q3_K_M, and close to the 38-39 measured on the 4080 SUPER.
- Gemma 4 26B-A4B did not fit all on the GPU as the row assumes; with the experts of 12 blocks in pinned RAM and a 2,368 MiB expert cache it decoded at 148.2 tok/s, just under the baseline's 150-220 and above ModelFit's 125, while CanIRun's 296 was double.
- Nemotron 3.5 Lightning at `--n-cpu-moe 25` with a 3,200 MiB expert cache decoded at 143.6 tok/s, above the baseline's 60-110 and close to its 170 ceiling, while CanIRun's 394 was 2.7 times too high.
- Qwen3-Coder-Next at `--n-cpu-moe 38` decoded at 71.5 tok/s with f16 KV and a 1,504 MiB expert cache (80.9 with q8_0 KV and a 2,944 MiB cache), above the baseline's 30-50 and far above Twoody's 13-25 for its non-Coder sibling, while CanIRun's "can-run-slow" and Periscale's "will not run" did not anticipate expert offload with a GPU cache.

Why they mislead here:

- MoE offload: CanIRun, ModelFit and Periscale treat any MoE model that does not fit in VRAM as unusable or unmodelled, while llama.cpp `--n-cpu-moe` keeps only the needed experts in the about 40 GiB of free RAM (see [llamacpp/README.md](../llamacpp/README.md)).
- MoE speed: CanIRun reads only active/total of the file per token and ignores attention, shared weights, the output head and KV, so it gives gpt-oss-20b about 324 tok/s on the slower laptop card against a measured 204.85 on the desktop.
- Model-specific quants: none of them lists the files we use (ggml-org MXFP4, Google QAT Q4_0, unsloth UD quants, NVFP4), so they size a generic Q4_K_M that is often larger.
  Twoody is the exception and its file sizes match the Hugging Face byte counts.
- KV layout: ModelFit's generic class gives gpt-oss-20b 32 GB of fp16 KV at 128k; the real layout (24 layers, 8 KV heads, 64-dim heads, half the layers on a 128-token sliding window) gives about 6 GiB at most with full attention on every layer (estimate).
  Periscale counts all 40 Qwen3.6 layers as full attention when only 10 are.
- Budget: ModelFit's 14.4 GB (13.41 GiB) and Twoody's 14.6 GiB are rules of thumb; our vLLM budget is 0.92 x 16303 MiB = 14.65 GiB minus a 1.2 GiB reserve ([vllm/README.md](../vllm/README.md)).

Twoody's per-token KV sizes (f16) are the one calculator output worth reusing, because they match the configs: Qwen3.6-35B-A3B 20480 B, Qwen3.8-27B 65536 B, Gemma 4 26B-A4B 10240 B global plus 102400 B sliding (window 1024), gpt-oss-20b 24576 B plus 24576 B sliding (window 128), GLM-4.7-Flash 54144 B ([data.json](https://www.twoody.com/local-llm-calculator/data.json)).

Generic size tables from blogs fail the same way.
[Tech Centurion](https://www.techcenturion.com/local-llm-gpu-rankings) estimates Gemma 4 26B-A4B at about 18.2 GB and Qwen3.6 35B-A3B at about 23.4 GB for Q4 at 32K, and puts 16 GB cards in Tier C because its tiers count only full-VRAM fit.
It ranks the RTX 5080 Laptop #3 of 108 laptop GPUs with a composite score of 52.0, but says 94 of its 108 laptop scores are estimates scaled from desktop results and does not mark which.
[MindStudio](https://www.mindstudio.ai/blog/best-gpu-for-running-local-llms-24gb) says a 30B model at 4-bit is "around 15 to 18GB" and that 16 GB is a tight fit, assuming full VRAM residency; our 30-35B MoE files are 17.05-19.13 GiB (18.3-20.5 GB), at or above that range, and run with offload.

## Quality rankings for runnable models

Quality scores below come from full-precision cloud endpoints on harnesses that are not OpenCode.
The effect of our Q3/Q4 quants on them is unknown.
The ranked recommendation lives in [model-selection.md](model-selection.md).

### Artificial Analysis (independent, high credibility)

[Artificial Analysis](https://artificialanalysis.ai/) runs every eval itself, and its Terminal-Bench 2.1 scores agree with NVIDIA's separate same-harness table to within 1.8 points.
The headline is now Intelligence Index v5.0.1 (November 2026), built from hard agentic evals; small models score low on it and most of our picks have no v5 score ([methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking)).
Terminal-Bench 2.1 was replaced by Terminal-Bench 4.0 in v4.3 (September 2026) and tau2 was superseded in v4.1, so these legacy columns are still the only common ground for older models; tau3-Banking is still run as a standalone eval on new releases.
Harnesses: TB2.1 uses Terminus 2 with at most 250 episodes and 3 repeats; TB 4.0 uses mini-swe-agent with no context compaction and at most 500 steps; temperature is 0.6 for reasoning models and 0 otherwise.

| Model (AA variant) | TB2.1 | tau2 | tau3-Banking | TB-Hard | Other | Source |
|---|---|---|---|---|---|---|
| Qwen3.8 27B xhigh | 79.8 | n/a | 48.0 | n/a | TB 4.0 5.6, II v5 20.1 | [AA](https://artificialanalysis.ai/models/qwen3-8-27b) |
| Qwen3.8 27B medium | 65.2 | n/a | 47.4 | n/a | TB 4.0 5.1, II 17.5 | same |
| Qwen3.8 27B low | 67.4 | n/a | 32.2 | n/a | TB 4.0 2.5, II 15.7 | same |
| Qwen3.8 27B non-reasoning | 49.1 | n/a | 20.0 | n/a | TB 4.0 0, II 10.0 | same |
| Qwen3.6-35B-A3B reasoning | 44.9 | 95.3 | 9.3 | 34.8 | no v5 score | [AA](https://artificialanalysis.ai/models/qwen3-6-35b-a3b) |
| Qwen3.6-35B-A3B non-reasoning | 41.6 | 85.1 | 5.4 | 25.8 | | same |
| Gemma 4 26B A4B reasoning | 39.0 | 43.6 | n/a | 13.6 | | [AA](https://artificialanalysis.ai/models/gemma-4-26b-a4b) |
| Qwen3-Coder-Next | 38.2 | 79.5 | n/a | 18.2 | TB 4.0 0 | [AA](https://artificialanalysis.ai/) |
| Gemma 4 12B reasoning | 27.3 | 36.3 | n/a | n/a | | [AA](https://artificialanalysis.ai/) |
| Nemotron 3.5 Lightning | 24.3 | n/a | 8.9 | n/a | II 5.5 | [AA](https://artificialanalysis.ai/) |
| gpt-oss-20b high | 13.9 | 60.2 | n/a | 10.6 | | [AA](https://artificialanalysis.ai/models/gpt-oss-20b) |
| gpt-oss-20b low | n/a | 50.3 | n/a | 4.5 | AA has no medium variant | same |
| GLM-4.7-Flash reasoning | n/a | 98.8 | n/a | 22.0 | | [AA](https://artificialanalysis.ai/) |
| DeepSeek R1 0528 Qwen3 8B | n/a | 0 | n/a | 1.5 | | [AA](https://artificialanalysis.ai/) |

New candidates from this run that AA scores: K2 Horizon MoVA and K2 Horizon 7B are now ranks 3 and 4 and Nanbeige4.3-33B is an honorable mention; their fit and runtime verdicts live in [model-selection.md](model-selection.md), and none has been run here:

| Model | Scores | Notes | Source |
|---|---|---|---|
| Nanbeige4.3-33B | II v5 20.4, Progress Index 306, TB 4.0 29.8 | Dense LlamaForCausalLM, 33.1B, Apache-2.0, 2026-10-25; no TB2.1 or tau2; architecture supported natively by both runtimes, but no vLLM checkpoint fits 16 GB, llama.cpp needs a 2-3 bit quant, and its chat prompts mis-tokenize until open PR [#31986](https://github.com/ggml-org/llama.cpp/pull/31986) ships | [AA](https://artificialanalysis.ai/models/nanbeige4-3-33b) |
| K2 Horizon MoVA 36B A4B | TB2.1 56.6, tau3-Banking 32.0, Progress 281 | K2HorizonForCausalLM, supported in [vLLM v0.33.0](https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/vllm/model_executor/models/registry.py) (no checkpoint fits 16 GB) and in llama.cpp since [PR #29535](https://github.com/ggml-org/llama.cpp/pull/29535) (b11454), so expert offload is the path | [AA small open-weight view](https://artificialanalysis.ai/models/open-source/small) |
| K2 Horizon 7B | TB2.1 44.2, tau3-Banking 27.6 | Same architecture and runtime support; Q8_0 GGUF (8.92 GiB) fits all on GPU at 64k with q8_0 KV (estimate) | [AA](https://artificialanalysis.ai/models/k2-horizon-7b) |

Ornith-1.5-9B has no AA page (404 on 2026-11-15), so its quality figures are vendor-reported only.

Reasoning effort costs time locally: AA reports average output tokens per Intelligence Index task of 109.8k for Qwen3.8 27B xhigh, 80.2k at low and 44.9k with thinking off ([AA](https://artificialanalysis.ai/models/qwen3-8-27b)).
At the baseline estimate of 40-45 tok/s for Qwen3.8-27B, 109.8k tokens is about 41-46 minutes of pure decode per task (estimate).

### NVIDIA same-harness table (vendor-run)

The [Nemotron 3.5 Lightning model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) runs competing models through one harness: SWE-bench Verified / Terminal-Bench 2.1 of 70.12 / 44.38 for Qwen3.6-35B-A3B, 57.40 / 37.22 for Gemma 4 26B-A4B, 52.44 / 15.17 for gpt-oss-20b and 51.56 / 24.58 for Nemotron Lightning (vendor-run).
Prefer these over each lab's self-reported numbers, for example OpenAI's 60.7 SWE-bench Verified for gpt-oss-20b or Alibaba's 73.4 for Qwen3.6-35B-A3B ([BenchLM ledgers](https://benchlm.ai/best/open-source), vendor-reported).

### BenchLM (aggregator, medium credibility)

[BenchLM](https://benchlm.ai/best/open-source) (BenchAlign v5.8, data refreshed 2026-11-13) blends external leaderboards and benchmark rows into a modelled score.
For runnable models it rests mostly on vendor self-reports: Qwen3.6-35B-A3B, Ternary Bonsai 2 27B and Nemotron 3.5 Bolt have 0 independent runs, and the 90% ranges are 19-29 points wide.
Do not use rank position for close models: an "Estimated" row with fewer than two source families drops below rows within 2 points under it, which is why Gemma 4 26B-A4B (41.2) is listed above Qwen3.6-35B-A3B (42.0) ([methodology](https://benchlm.ai/methodology)).

| Model | BenchLM score | Status | Open-weight rank |
|---|---|---|---|
| Qwen3.8-27B | 54.0 (range 41.45-66.63) | Supported | 21 |
| Ternary Bonsai 2 27B | 51.6 | Estimated, all 23 results self-reported by Prism ML | 27 |
| Qwen3.6-27B | 48.0 | Estimated | 36 |
| Qwen3.5-35B-A3B | 45.2 | Supported | 40 |
| Qwen3.6-35B-A3B | 42.0 (range 27.65-56.35) | Estimated, 40 of 47 rows Alibaba-reported | 51 |
| Nemotron 3.5 Bolt 74B-A9B | 41.6 | Estimated, all 8 rows NVIDIA-reported | 46 |
| Gemma 4 26B-A4B | 41.2 | Estimated | 47 |
| GLM-4.7-Flash | 38.7 | Estimated | 54 |
| gpt-oss-20b | 36.9 | Estimated | 56 |
| Gemma 4 12B | 34.1 | Estimated | 59 |
| Ornith-1.5-9B | 32.5 | Estimated | 62 |

BenchLM also lists independent Vals AI runs for Qwen3.8-27B: Terminal-Bench 2.1 58.4% against Alibaba's self-reported 73.0% ([BenchLM](https://benchlm.ai/models/qwen3-8-27b)).
Ternary Bonsai 2 27B ships PTQ1_0 and PQ2_0 GGUFs of 5.95 GB and 7.21 GB per the Hugging Face API (BenchLM's text says 5.93 and 7.25).

### Small community evals and quant quality

| Eval | Hardware and runtime | Result | Label | Source |
|---|---|---|---|---|
| TechFuelHQ coding eval v1: 8 tasks including two-turn tool calling, 2 reps, executed checkers | Desktop RTX 5080, Ollama | gpt-oss:20b 8/8 stable, qwen3-coder:30b 6/8, devstral:24b (4k) 6/8, qwen2.5-coder:14b 5/8, qwen2.5-coder:7b 4/8 (both qwen2.5-coder failed Tools) | measured | [TechFuelHQ](https://techfuelhq.com/data/rtx-5080-llm-throughput/) |
| Gemma 4 26B-A4B QAT conversion quality vs BF16 QAT | llama.cpp KLD, hardware n/a | Unsloth UD-Q4_K_XL (13.27 GiB): mean KLD 0.098, top-1 85.6%; naive Q4_0 at the size of Google's QAT Q4_0 file: KLD 0.361, top-1 70.2% | vendor-reported | [Unsloth Gemma 4 QAT](https://unsloth.ai/docs/models/gemma-4/qat) |

The Unsloth table matters for pick 5 (Gemma 4 26B-A4B), which uses Google's QAT Q4_0 file: A/B it against the UD-Q4_K_XL file on this laptop before settling (see [llamacpp/models.md](../llamacpp/models.md)).

## Takeaways for this machine

1. Treat every number in this document as an upper bound or a hint until our own runs exist; the first job is to measure, following [llamacpp/benchmarking.md](../llamacpp/benchmarking.md) and [vllm/benchmarking.md](../vllm/benchmarking.md).
2. Run gpt-oss-20b on llama.cpp first and compare with the desktop anchor (tg128 204.85, pp2048 7477): that one ratio tells us how much the 140 W limit and the smaller core count cost on this laptop.
3. Run Llama 2 7B Q4_0 pp512/tg128 once as a cross-machine anchor against the #15013 rows (desktop 5080 184.68 / 9487.70, 5070 Ti 182.43 / 8419.56).
4. Expect prefill, not decode, to be the laptop's weak spot (about 0.62 of desktop compute at full boost, estimate), and OpenCode's long prompts make prefill the dominant cost per turn.
5. For MoE models with experts in RAM, the only RTX 5080 evidence is desktop: Gemma 4 26B-A4B UD-Q6_K_XL with `--n-cpu-moe 23` at about 41 tok/s, and Qwen3.6-35B-A3B Q6 with 29 GB in RAM at 35 tok/s.
   The laptop's DDR5 and the `--n-cpu-moe` count we use will decide our numbers.
6. Ignore calculator "does not fit" verdicts for configurations the baseline sized from real file bytes and config.json; do measure the tight all-on-GPU fits (Gemma 4 26B-A4B at 64k q8_0 KV, Qwen3.8-27B at 64k) with a real load, because calculators and blogs disagree on them.
7. Test q8_0 against f16 KV on Qwen3.8-27B, because the community reports conflict.
8. Quality evidence for picks is cloud full-precision and non-OpenCode; check quantized quality locally, especially Qwen3.8-27B at 3 bits against Qwen3.6-35B-A3B at 4 bits.

## Sources

- NVIDIA, [GeForce RTX 50 Series Laptops](https://www.nvidia.com/en-us/geforce/laptops/50-series/) and [GeForce RTX 5080](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/), read 2026-11-15.
- ggml-org, [gpt-oss guide, discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396) and [CUDA scoreboard, discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013).
- llama.cpp issues and PRs [#27623](https://github.com/ggml-org/llama.cpp/issues/27623), [#25859](https://github.com/ggml-org/llama.cpp/issues/25859), [#30489](https://github.com/ggml-org/llama.cpp/pull/30489), [#30196](https://github.com/ggml-org/llama.cpp/pull/30196), [#31802](https://github.com/ggml-org/llama.cpp/pull/31802).
- TechFuelHQ, [RTX 5080 LLM Throughput dataset v1.1.0](https://huggingface.co/datasets/iBlessi/rtx-5080-llm-throughput) ([canonical page](https://techfuelhq.com/data/rtx-5080-llm-throughput/)) and [Ollama vs llama.cpp on RTX 5080](https://techfuelhq.com/self-hosted/ollama-vs-llamacpp-rtx-5080-2026/).
- u/grumd, [18 models on an RTX 5080](https://www.reddit.com/r/LocalLLaMA/comments/1s9mkm1/benchmarked_18_models_that_i_can_run_on_my_rtx/) and [Gemma 4 llama-benchy comment](https://www.reddit.com/r/LocalLLaMA/comments/1sdu8oz/bartowski_vs_unsloth_for_gemma_4/oem5j5k/).
- Nick Lothian, [LLM SQL Benchmark](https://sql-benchmark.nicklothian.com/), [repo](https://github.com/nlothian/llm-sql-benchmark) and [post](https://www.reddit.com/r/LocalLLaMA/comments/1s7r9wu/i_tested_as_many_of_the_small_local_and/).
- [Tech Centurion GPU rankings](https://www.techcenturion.com/local-llm-gpu-rankings), 2026-11-14.
- Calculators: [CanIRun.ai](https://canirun.ai/device/rtx-5080-laptop), [ModelFit](https://modelfit.io/gpu/rtx-5080/), [Twoody](https://www.twoody.com/local-llm-calculator/rtx-5080), [Periscale](https://periscale.ai/gpus/rtx-5080-laptop).
- [MindStudio, Fitting 30B Models in 16GB or 24GB VRAM](https://www.mindstudio.ai/blog/best-gpu-for-running-local-llms-24gb).
- [Artificial Analysis](https://artificialanalysis.ai/) and its [methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking), Intelligence Index v5.0.1.
- [BenchLM open-weight ranking](https://benchlm.ai/best/open-source) and [methodology](https://benchlm.ai/methodology), BenchAlign v5.8.
- NVIDIA, [Nemotron 3.5 Lightning model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16).
- [Unsloth documentation](https://unsloth.ai/docs), including [Gemma 4 QAT](https://unsloth.ai/docs/models/gemma-4/qat).
- [Dettmers and Zettlemoyer, The case for 4-bit precision (arXiv 2212.09720)](https://arxiv.org/abs/2212.09720).
- The user's LibreDB Studio agent harness ([docs/llms/README.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/README.md)), analysed in [agent-harness-evidence.md](agent-harness-evidence.md), and [Apple MacBook Pro specs](https://www.apple.com/macbook-pro/specs/) for the M5 Max bandwidth.
- Credibility ratings and review notes for each source are in [sources.md](sources.md).

## Maintenance

Before relying on this document, re-check the following.

- Whether anyone has published RTX 5080 Laptop (or RTX 5090 Laptop, same GB203M chip) llama.cpp or vLLM numbers; once our own runs exist, they supersede every desktop row here.
- The Artificial Analysis index version and per-eval columns, because AA retires metrics every few months.
- BenchLM scores and ranges, which move with each data refresh.
- llama.cpp #27623 and PR #31802 for the q8_0 KV penalty, and PR #31986 for the Nanbeige4.3-33B tokenizer fix.
- Calculator formulas and dates (ModelFit and Tech Centurion 2026-11-14, Twoody 2026-10-16, Periscale 2026-10-30, CanIRun live).
- The TechFuelHQ change log, which mentions an 80-row CSV that was not reviewed here.
