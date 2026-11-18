# llama.cpp model recipes for the RTX 5080 Laptop

Last verified: 2026-11-18 (all ten ranked picks and Nemotron 3.5 Lightning measured on this laptop on 2026-11-16, the four GGUF builds of the vLLM and SGLang models on 2026-11-17, and every measured recipe's OpenCode entry in the validated config since 2026-11-17; the other recipes are 2026-11-15 estimates).

All ten ranked picks run on llama.cpp `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407`, each with the exact GGUF file, `--n-cpu-moe` value, context, KV type, memory cap and thinking control below.
The default is Qwen3.6-35B-A3B Q4_K_M (MoE, experts of blocks 0-24 in pinned RAM with a 3,200 MiB GPU cache for the most-used ones, 128k); use Qwen3.8-27B UD-Q3_K_XL (dense, all on the GPU, 64k) when a task needs the strongest model.
VRAM, RAM, context and speed figures are estimates from file sizes and model configs unless a recipe has a "Measured on this laptop" line.
All ten ranked picks, the Nemotron 3.5 Lightning fallback and the [GGUF builds of the vLLM and SGLang models](#gguf-builds-of-the-vllm-and-sglang-models) are measured; every other recipe needs the smoke tests in [README.md](README.md#smoke-tests) and an OpenCode tool-call test before use.
The ranking and its evidence live in [../docs/model-selection.md](../docs/model-selection.md); this document only says how to run each model.

Shared background (image choice, why `GGML_CUDA_DISABLE_GRAPHS=1` must come before the image name, flag meanings, offload mechanics, thinking controls, known issues) is in [README.md](README.md).
Measurement procedures, including the `--n-cpu-moe` sweep, are in [benchmarking.md](benchmarking.md).
Host prerequisites (NVIDIA Container Toolkit with CDI, ufw, memory caps) are in [../docs/host-setup.md](../docs/host-setup.md), and the OpenCode entries are in [../docs/opencode.md](../docs/opencode.md).

## At a glance

Sizes are HF API byte totals converted to GiB, read again on 2026-11-15 through the tree API of each repo (linked in each recipe).
The GPU estimates were made against 15.92 GiB (16303 MiB) with 1.4-1.9 GiB assumed for the CUDA context and compute buffers.
Measured on 2026-11-16, CUDA can use 15,841 MiB (15.47 GiB; the driver reserves 462 MiB), the desktop holds about 112 MiB, and llama.cpp's CUDA context, compute buffers and recurrent state took only 0.7-0.8 GiB (Qwen3.8-27B and Qwen3.6-35B-A3B).
For llama.cpp the two corrections roughly cancel: Qwen3.8-27B was estimated at 15.2 GiB and peaked at 14.4 GiB.
A GPU cell marked measured is the `nvidia-smi` peak, which includes the desktop.
About 40 GiB of the 61 GiB of RAM is free while the desktop runs.

| Rank | Model | GGUF | File (GiB) | `--n-cpu-moe` | Context | KV | GPU (GiB) | Experts in RAM (GiB) | `--memory` |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3.8-27B | unsloth UD-Q3_K_XL | 12.24 | No (dense) | 65536 | q8_0 | about 15.2 (measured 14.4) | 0 | 16g |
| 2 | Qwen3.6-35B-A3B | ggml-org Q4_K_M | 19.13 | 25, plus `--moe-cache-mib 3200` | 131072 | f16 | measured 14.8 | 10.5, pinned (container peak 14.9, measured) | 20g |
| 3 | K2 Horizon MoVA 36B A4B | bartowski Q4_K_M | 21.14 | 48 (all), plus `--moe-cache-mib 1536` | 65536 | q8_0 | measured 14.5 | 14.9, pinned (container peak 19.7, measured) | 26g |
| 4 | K2 Horizon 7B | bartowski Q8_0 | 8.92 | No (dense) | 81920 | q8_0 | measured 14.7 | 0 | 16g |
| 5 | Gemma 4 26B-A4B IT | Google QAT Q4_0 | 13.45 | 12, plus `--moe-cache-mib 2368` | 131072 | f16 | measured 14.7 | 4.8, pinned (container peak 10.6, measured) | 20g |
| 6 | Qwen3-Coder-Next | unsloth UD-IQ4_XS | 35.79 | 38, plus `--moe-cache-mib 1504` | 131072 | f16 | measured 14.7 | 26.4, pinned (container peak 31.9, measured) | 35g |
| 7 | Nemotron 3.5 Bolt 74B-A9B | NVIDIA 24GB GGUF | 20.51 | 60, plus `--moe-cache-mib 2816` | 131072 | f16 | measured 14.7 | 10.2, pinned (container peak 17.8, measured) | 26g |
| 8 | Groot-19B-A3B-261023 | GrootTeam Q4_K_M | 10.86 | 0 | 131072 | f16 | about 13.9-14.4 (measured 13.5) | 0 | 16g |
| 9 | gpt-oss-20b | ggml-org MXFP4 | 11.28 | 0 | 131072 | q8_0 | about 13.9 (measured 13.1) | 0 | 16g |
| 10 | DeepSeek-R1-0528-Qwen3-8B (chat only) | unsloth Q8_0 | 8.11 | No (dense) | 65536 | q8_0 | about 13.7 (measured 13.0) | 0 | 16g |

Thinking and tools in one line each: ranks 1, 2, 5, 7 and 8 toggle thinking; ranks 3, 4 and 9 always think; rank 6 never thinks; rank 10 always thinks and has no tool calling, so it is not an OpenCode agent model.

## How to read a recipe

Every command follows the run template in [README.md](README.md#run-template), and the same rules apply to each:

- Remove the previous container first with `docker rm -f llamacpp 2>/dev/null`; only one model server runs at a time, because the GPU has 16 GiB and the RAM budget is shared.
- `-e GGML_CUDA_DISABLE_GRAPHS=1` sits before the image name, as the workaround for the CUDA-graph GPU hang on sm_120 laptops ([#27330](https://github.com/ggml-org/llama.cpp/issues/27330), open, reproduced in a comment on driver 595.99.02 and Ubuntu 26.04).
  After the image name, `llama-server` reads `-e` as `--escape` and refuses to start ([arg.cpp](https://github.com/ggml-org/llama.cpp/blob/b12418/common/arg.cpp)).
- The port is published on 127.0.0.1 only, because Docker-published ports bypass ufw ([../docs/host-setup.md](../docs/host-setup.md)).
- No restart policy and no `--rm`: start on demand, stop when done, and keep the stopped container's logs.
- `--memory` and `--memory-swap` are equal, so the container cannot swap.
  The cap is the RAM-resident experts plus the host prompt cache (`--cache-ram`, default 8192 MiB, [arg.cpp](https://github.com/ggml-org/llama.cpp/blob/b12407/common/arg.cpp)) plus about 2-3 GiB for the process (estimate).
  Where the experts are large, the recipe lowers `--cache-ram` instead of raising the cap.
  Read `docker stats` on the first run and adjust.
- The default load mode is mmap, so RAM-resident experts are file-backed pages that the kernel can evict under pressure and re-read from NVMe (a stall, not an OOM kill).
  `-lm none` keeps them in anonymous memory that counts fully against the cap; [benchmarking.md](benchmarking.md#moe-offload-sweep---n-cpu-moe) uses it for the sweep.
  It also pins them, so they upload to the GPU at full PCIe speed: on Qwen3.6-35B-A3B it raised prefill at 8.6k from 934 to 2,112 tok/s without changing decode (measured on this laptop, 2026-11-16), and llama.cpp itself warns that tensor overrides with mmap are slower.
- With VRAM left after the load, try `--moe-cache-mib` at the default CPU share of 0 before lowering `--n-cpu-moe`: the cache keeps the most-used host experts on the GPU, and on Qwen3.6-35B-A3B 1,536 MiB of it raised decode from 68 to 112 tok/s, while a CPU share of 0.95 gave only 85 (measured on this laptop, 2026-11-16; [README.md](README.md#moe-expert-offload)).
- `--alias local` matches the `modelID` in [../docs/opencode.md](../docs/opencode.md); the OpenCode entry name carries the model identity.
- `-hf repo:QUANT` downloads into the Hugging Face hub cache and also fetches an mmproj when the repo has one, so text-only recipes add `--no-mmproj` ([arg.cpp](https://github.com/ggml-org/llama.cpp/blob/b12407/common/arg.cpp)).
  Where a quant tag could match more than one file, the recipe names the file with `--hf-file`.
- `-np 1` because auto gives 4 slots, and `--jinja` (the default) because the GGUF's chat template drives tool-call and reasoning parsing ([README.md](README.md#run-template)).
- Sampling flags set server defaults; a client that sends its own `temperature` or `top_p` overrides them per request.
- `--n-cpu-moe N` moves the routed experts of blocks 0 to N-1 to RAM and counts every block, not only MoE blocks ([server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)).
  On an out-of-memory error at load, raise N by 1 or 2.
  With `-ngl` and `--n-cpu-moe` both set, `--fit` does not adjust them, and a CUDA pool OOM during prefill aborts the server ([#31974](https://github.com/ggml-org/llama.cpp/issues/31974), open).
- Speed figures are bandwidth arithmetic, labelled "(inferred)".
  The laptop GPU has 896 GB/s ([NVIDIA spec](https://www.nvidia.com/en-us/geforce/laptops/50-series/)), and RAM is DDR5-6400 dual channel by Intel's spec, about 102.4 GB/s peak, with the configured DIMM speed unverified ([Intel spec](https://www.intel.com/content/www/us/en/products/sku/245960/intel-core-ultra-9-processor-290hx-plus-36m-cache-up-to-5-50-ghz/specifications.html)).

## 1. Qwen3.8-27B

Dense 27.8B (including vision and the MTP head), Apache-2.0, announced 2026-08-14 ([model card](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/README.md)).
It is the strongest agent that fits, and it fits entirely on the GPU at 64k with a 3-bit Unsloth Dynamic quant.
Start it with `./llmctl up qwen3.8` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF/tree/main), `Qwen3.8-27B-UD-Q3_K_XL.gguf`, 13,146,393,504 B (12.24 GiB).
- **File history:** Unsloth replaced this file with a Dynamic V3.1 build on 2026-11-12 at the same size (commit "New UD3.1 UD-Q3_K_XL", [commit list](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF/commits/main)), although the [Unsloth Qwen3.8 page](https://unsloth.ai/docs/models/qwen3.8) says the 3-bit files were not updated.
  A copy downloaded before 2026-11-12 is the older build; delete it from the cache to get the new one.
- **Quality variant:** `Qwen3.8-27B-UD-IQ4_XS.gguf` (13.37 GiB) reaches only about 48-50k with q8_0 KV (estimate).

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 16g --memory-swap 16g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL \
  --alias local --no-mmproj \
  -ngl all -fa on -c 65536 -ctk q8_0 -ctv q8_0 -np 1 --jinja \
  --reasoning-effort medium \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

- **Memory (estimate):** KV is 2 x 16 full-attention layers x 4 KV heads x 256 x 2 B = 64 KiB/token in f16 and about 34 KiB in q8_0, so 2.1 GiB at 64k ([config.json](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)).
  The GGUF header puts about 11.7 GiB of weights on the GPU, because `token_embd` (0.51 GiB) stays on the host.
  With about 0.4 GiB CUDA context and compute buffers the total is about 15.2 GiB of 15.92, a margin of about 0.6 GiB.
- **Context:** 65536 is the target.
  About 84-88k with q8_0 KV is possible on paper but untested, and f16 KV fits only about 45k (estimate).
- **Measured on this laptop (2026-11-16, [record](../research/2026-11-16/llamacpp-qwen3.8-27b/)):** the server started from the cached file in about 6 s with `-c 65536` and q8_0 KV and used 14,768 MiB of GPU memory (14,790 MiB peak, including about 112 MiB for the desktop), against 15.47 GiB that CUDA reports, so about 1 GiB of margin remained; container RAM was 3.3 GiB.
- **Measured run command:** the recipe above with `--user "$(id -u):$(id -g)"`, the cache mounted at `/hf` with `-e HF_HOME=/hf`, and `-m /hf/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4af51b3542f775b0a8e4b310bfb3f17e01c306c9/Qwen3.8-27B-UD-Q3_K_XL.gguf` instead of `-hf`, because the file was pre-downloaded with the `hf` CLI ([server arguments](../research/2026-11-16/llamacpp-qwen3.8-27b/server-args.json)).
- **Measured speed (2026-11-16):** decode 36.4 tok/s at a short context (36.2 to 36.5 over 3 runs), 34.2 tok/s after an 8.6k prompt and 29.6 tok/s after a 32k prompt; prompt processing 1,326 tok/s for 8.6k tokens (6.5 s to the first token) and 1,111 tok/s for 24k new tokens (21.7 s); repeating the same prompt hit the cache and gave the first token in 0.09 to 0.12 s.
  The GPU drew up to 146 W at its 140 W limit and reached 72 C.
- **Measured behaviour (2026-11-16):** the OK test passed with 121 characters of reasoning at effort medium, the tool test returned `get_weather` with `{"city":"Paris"}`, and an OpenCode v2.3.0 write-and-read session passed in 14 s; its first request was 4,180 prompt tokens (3.2 s), and the two later steps processed only 22 and 70 new tokens thanks to the prompt cache.
- **KV speed caveat:** a commenter in [#27623](https://github.com/ggml-org/llama.cpp/issues/27623) (open) measured q8_0 KV at 5.9 tok/s against 13.3 tok/s for f16 (RTX 2000 Ada, sm_89, UD-IQ4_XS; not this laptop); the decode cliff above 80k in the title was retracted by the reporter, and the fix [PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802) is open.
  If decode falls below about 15 tok/s, switch to f16 KV at `-c 45056`, or to `UD-IQ3_XXS` (10.18 GiB) with f16 KV at about 78k (inferred, lower quality).
- **Sampling:** thinking on, temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0; thinking off, temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5 ([Unsloth Qwen3.8 page](https://unsloth.ai/docs/models/qwen3.8), [model card](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/README.md)).
- **Thinking:** on by default with `reasoning_effort` `xhigh`; the template accepts only `xhigh`, `medium` and `low`, and `enable_thinking: false` turns it off ([chat_template.jinja](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja)).
  `--reasoning-effort medium` sets the server default ([arg.cpp](https://github.com/ggml-org/llama.cpp/blob/b12407/common/arg.cpp)), and OpenCode can send `none`, `low` or `medium` per request ([../docs/opencode.md](../docs/opencode.md)).
  Medium matters because `xhigh` is very verbose: [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-27b) counted about 128M output tokens across its index at `xhigh`.
- **Tool calling:** Qwen3-Coder XML; with `--jinja`, [chat.cpp](https://github.com/ggml-org/llama.cpp/blob/b12418/common/chat.cpp) routes the template to the Qwen3-Coder parser.
- **Open issues:** [#29947](https://github.com/ggml-org/llama.cpp/issues/29947) (open: a Jinja `CallExpression` error with OpenCode and the opencode-tool-search plugin on the ggml-org GGUF; the reporter says the Unsloth template works), [#30357](https://github.com/ggml-org/llama.cpp/issues/30357) (prefix-cache misses on hybrid Qwen templates), [#30118](https://github.com/ggml-org/llama.cpp/issues/30118) (tool call leaking into content, reported on Qwen3.6; same parser).
  UD-Q3_K_XL holds 3.29 GiB of IQ3_S tensors, which a self-built nvcc 13.2.0/13.2.1 binary miscompiles ([#28581](https://github.com/ggml-org/llama.cpp/issues/28581)); the CUDA 12.8.1 image is not affected, but check the first answers anyway.
- **Speed estimate:** the decode ceiling is about 896 GB/s divided by about 13 GB read per token, roughly 65-70 tok/s, so about 40-45 tok/s in practice with f16 KV (inferred); the measured 36.4 tok/s with q8_0 KV is about 85% of the low end, and no q8_0 penalty of the size reported in #27623 showed up.
- **Ollama alternative:** `qwen3.8:27b` is 18 GB and spills dense layers to the CPU, so it is slow here ([Ollama tags](https://ollama.com/library/qwen3.8/tags)).
  Not recommended.
  `qwen3.8:latest`, `:27b` and `:27b-q4_K_M` are the same build: as checked on 2026-11-16, their live registry manifests share model blob `sha256:f5f1dd8920d4` (16,810,714,464 B) plus a 931,146,016 B projector ([manifest](https://registry.ollama.ai/v2/library/qwen3.8/manifests/latest), [review record](../research/2026-11-16/libredb-review/check_qwen-current.json)).
- **Evidence:** Artificial Analysis Terminal-Bench 2.1 79.8 (xhigh), 65.2 (medium), 67.4 (low), 49.1 (non-reasoning) and tau3-Banking 48.0, measured on BF16 through an API ([AA](https://artificialanalysis.ai/models/qwen3-8-27b)); the 3-bit quality loss is unmeasured.

## 2. Qwen3.6-35B-A3B

35B total, 3B active MoE (35.95B with vision), Apache-2.0, released 2026-04-16 ([model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/README.md)).
It is the 128k pick with the most mature runtime path; 10.5 GiB of experts sit in pinned RAM, and a 3,200 MiB GPU cache serves the most-used of them.
Start it with `./llmctl up qwen3.6` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [ggml-org/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/api/models/ggml-org/Qwen3.6-35B-A3B-GGUF?blobs=true), `Qwen3.6-35B-A3B-Q4_K_M.gguf`, 20,542,732,576 B (19.13 GiB).
  Experts are 16.875 GiB, 0.422 GiB per block over 40 blocks, all of which are MoE.
  The GGUF header, read on 2026-11-16, confirms these figures, 1.98 GiB of other GPU weights and 10 full-attention layers; the downloaded file's SHA-256 matched its LFS id.
- **Alternative template:** Unsloth's Qwen3.6 GGUFs add developer-role support and better nested-object tool-call parsing (vendor-documented, [Unsloth Qwen3.6 page](https://unsloth.ai/docs/models/qwen3.6)); `UD-IQ4_XS` (16.50 GiB) keeps more experts on the GPU.
  Neither is measured against the ggml-org file here.

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 20g --memory-swap 20g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf ggml-org/Qwen3.6-35B-A3B-GGUF:Q4_K_M \
  --alias local --no-mmproj \
  -ngl all --n-cpu-moe 25 -fa on -c 131072 -np 1 --jinja \
  -lm none -t 8 --moe-cache-mib 3200 \
  --reasoning off \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0
```

- **Memory:** KV is 2 x 10 full-attention layers x 2 KV heads x 256 x 2 B = 20 KiB/token, 2.5 GiB at 128k in f16 ([config.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json)).
  `--n-cpu-moe 25` keeps the experts of blocks 0-24 (10.5 GiB) in RAM, pinned by `-lm none`.
  The GPU holds 1.98 GiB of other weights, the experts of blocks 25-39 (6.3 GiB), 2.5 GiB of KV, the 3,200 MiB expert cache and about 0.7 GiB of CUDA context and buffers.
- **Measured on this laptop (2026-11-16, [record](../research/2026-11-16/llamacpp-qwen3.6-35b-a3b/README.md)):** the server started from the cached file in about 6 s and peaked at 15,190 MiB of GPU memory including about 112 MiB for the desktop, about 650 MiB below the 15,841 MiB that CUDA can use; container RAM peaked at 14.9 GiB with a 107k prompt in the prompt cache.
- **Measured run command:** the recipe above with `--user "$(id -u):$(id -g)"`, the cache mounted at `/hf` with `-e HF_HOME=/hf`, and `-m` with the cached file's path instead of `-hf`; the validation run added `-lv 4` only to log the cache statistics, which did not change the speed ([launch](../research/2026-11-16/llamacpp-qwen3.6-35b-a3b/run-command.sh)).
- **Measured speed (2026-11-16):** decode 122.5 tok/s at a short context (119.7 to 122.7 over 3 runs), 117.7 tok/s after an 8.6k prompt, 109.9 after 32k and 92.6 after 107k, and 114.8 tok/s over 1,193 tokens of code; prompt processing 1,990 tok/s for 8.6k tokens (4.3 s to the first token), 1,855 tok/s for 24k new tokens (13.0 s) and 1,507 tok/s for 75.7k new tokens (50.3 s); repeating a prompt hit the cache in 0.06 to 0.13 s.
  The GPU drew up to 144 W and reached 71 C.
- **Measured behaviour (2026-11-16):** the OK test answered with no reasoning at the server default, one request with `enable_thinking: true` produced 374 characters of reasoning, the tool test returned `get_weather` with `{"city":"Paris"}`, and an OpenCode v2.3.0 write-and-read session passed in 9 s; its first request was 4,186 prompt tokens (2.6 s), and the two later steps processed only 24 and 136 new tokens thanks to the prompt cache.
- **Tuning (measured 2026-11-16, [variants](../research/2026-11-16/llamacpp-qwen3.6-35b-a3b/variants/variants.jsonl)):** the recipe as first written here (`--n-cpu-moe 21`, default mmap, automatic threads, which chose 4) decoded at 67.3 tok/s with 934 tok/s prefill at 8.6k and 93 s to the first token at 107k, and peaked at 13,696 MiB.
  `-lm none` raised prefill to 2,112 tok/s, `-t 8` added 2% to decode, a 1,536 MiB expert cache at CPU share 0 raised decode to 112 tok/s (85 at share 0.95), and `--n-cpu-moe 25` with 3,200 MiB of cache gave 120 tok/s for 8% less prefill.
  All 40 blocks in RAM with a 9,600 MiB cache gave 126 tok/s but only 1,679 tok/s prefill and an 18.7 GiB container, so it was not adopted.
  Over the validation run the cache served 83% of decode-time expert lookups from VRAM.
- **Memory cap:** 20g covers 10.5 GiB of pinned experts, the 8 GiB default prompt cache and the process; the measured peak was 14.9 GiB.
- **KV choice:** f16 avoids the q8_0 decode penalty ([#27623](https://github.com/ggml-org/llama.cpp/issues/27623)).
  q8_0 would halve KV to about 1.3 GiB and leave about 1.2 GiB more for the expert cache, and it is what reaches 262144 (estimate, not measured).
- **Sampling** ([model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/README.md)): precise coding with thinking, temperature 0.6, top_p 0.95, top_k 20, presence_penalty 0; general thinking, temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5; non-thinking, temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5.
  Unsloth lists presence_penalty 0 for general thinking ([Unsloth Qwen3.6 page](https://unsloth.ai/docs/models/qwen3.6)); the card is the primary source.
- **Thinking:** on by default in the template; `--reasoning off` makes the server default off, and the OpenCode `thinking` and `none` variants send `chat_template_kwargs.enable_thinking` per request ([../docs/opencode.md](../docs/opencode.md)).
  Why off by default is explained in [README.md](README.md#thinking-control).
- **Tool calling:** Qwen3-Coder XML through the Qwen3-Coder parser ([chat.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12418/common/chat.cpp)).
- **Open issues:** [#30118](https://github.com/ggml-org/llama.cpp/issues/30118) (open: when the model omits the `<tool_call>` opener, the call lands in `content`), [#30357](https://github.com/ggml-org/llama.cpp/issues/30357) (open: whitespace not round-trippable, full re-prefill), [#25859](https://github.com/ggml-org/llama.cpp/issues/25859) (open: serial expert copies slow prefill; with pinned experts on this PCIe 5.0 x16 link prefill still reached 1,990 tok/s).
  Do not set `--ctx-checkpoints 0` on this hybrid model ([README.md](README.md#prompt-cache)).
  MTP drafting (`--spec-type draft-mtp`) has open bugs and is left out.
  All three were still open when re-checked on 2026-11-16, and neither leak nor full re-prefill showed up in the measured session.
- **Speed:** measured above.
  The 40-90 tok/s range inferred on 2026-11-15 assumed that every RAM-resident expert is read from DDR5 for every token; the expert cache avoids most of those reads.
- **Ollama alternative:** `qwen3.6:35b-a3b` (23 GB, whole-layer offload) with `OLLAMA_CONTEXT_LENGTH=65536` ([Ollama tags](https://ollama.com/library/qwen3.6/tags)); slower than `--n-cpu-moe` for a MoE that spills ([README.md](README.md)).
- **Evidence:** NVIDIA's same-harness table gives SWE-bench Verified 70.12 and Terminal-Bench 2.1 44.38 ([Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/README.md)), and Artificial Analysis gives TB2.1 44.9 and tau3-Banking 9.3 ([AA](https://artificialanalysis.ai/models/qwen3-6-35b-a3b)).

## 3. K2 Horizon MoVA 36B A4B

37.44B total (BF16 safetensors), about 4B active by the vendor's count, Apache-2.0, from IFM, whose HF org is not verified ([model card](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/raw/main/README.md)).
It runs only on llama.cpp here: the smallest vLLM build is 20.25 GiB.
llama.cpp support (`k2-horizon` arch, dedicated chat parser) arrived in b11454 on 2026-10-06, about six weeks before this document ([llama-arch.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/src/llama-arch.cpp), [PR #29535](https://github.com/ggml-org/llama.cpp/pull/29535)).
Start it with `./llmctl up mova` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [bartowski/K2-Horizon-MoVA-36B-A4B-GGUF](https://huggingface.co/api/models/bartowski/K2-Horizon-MoVA-36B-A4B-GGUF?blobs=true), `K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf`, 22,701,885,664 B (21.14 GiB), built with mainline b11462.
  The GGUF header, read on 2026-11-16, shows 48 blocks (0-2 dense, 3-47 MoE with 100 routed experts, 8 used, and a shared expert), 64 value experts per MoE block (4 used), full attention with 8 KV heads of 128 in every layer and a 250,624-token vocabulary ([plan record](../research/2026-11-16/next-model-plans/plan-k2-horizon-mova.json)); the downloaded file's SHA-256 matched its LFS id.
- **Official alternative:** [IFM/K2-Horizon-MoVA-36B-A4B-GGUF](https://huggingface.co/api/models/IFM/K2-Horizon-MoVA-36B-A4B-GGUF?blobs=true) Q4_K_M (20.83 GiB) was uploaded on 2026-09-22, before mainline support, and has not been load-tested; its embedded template is the only one with an `enable_thinking` off switch (unverified in use).

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 26g --memory-swap 26g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf bartowski/K2-Horizon-MoVA-36B-A4B-GGUF:Q4_K_M \
  --alias local \
  -ngl all --n-cpu-moe 48 -fa on -c 65536 -ctk q8_0 -ctv q8_0 -np 1 --jinja \
  -lm none -t 8 --moe-cache-mib 1536 \
  --reasoning-effort high \
  --temp 1.0 --top-p 0.95
```

- **Memory:** MoE blocks 3-47 hold about 0.332 GiB of `ffn_*_exps` each.
  `--n-cpu-moe` and the expert cache touch only `ffn_*_exps` ([common.h](https://github.com/ggml-org/llama.cpp/blob/b12407/common/common.h)), so the 3.96 GiB of `attn_v_exps` (the "mixture of value attention" tensors) always stay on the GPU.
  KV is heavy: 192 KiB/token in f16, about 6.375 GiB at 64k in q8_0, and f16 at 64k does not fit.
  `--n-cpu-moe 48` keeps all routed experts (14.9 GiB) in pinned RAM, and the GPU holds about 1.9 GiB of other weights, the value experts, the KV, the 1,536 MiB expert cache and about 0.8 GiB of context and buffers.
- **Measured on this laptop (2026-11-16, [record](../research/2026-11-16/llamacpp-k2-horizon-mova/)):** the server started from the cached file in about 8 s and peaked at 14,848 MiB including about 112 MiB for the desktop, about 990 MiB below the 15,841 MiB that CUDA can use; container RAM peaked at 19.7 GiB, so the 26g cap keeps room for the default 8 GiB prompt cache.
  The run added `--user`, the cache at `/hf`, `-m` instead of `-hf` and `-lv 4` ([server arguments](../research/2026-11-16/llamacpp-k2-horizon-mova/server-args.json)).
- **Measured speed (2026-11-16):** decode 37.2 tok/s at a short context (36.9 to 37.5 over 3 runs), 31.8 tok/s after an 8k prompt and 26.2 after 30k; prompt processing 1,226 tok/s for 8k tokens (6.6 s to the first token) and 975 tok/s for 22k new tokens (22.8 s), because every prefill batch uploads experts from RAM; repeating a prompt hit the cache in 0.05 to 0.09 s.
  The GPU drew up to 144 W and reached 70 C, and the cache served 43% of decode-time expert lookups.
- **Measured behaviour (2026-11-16):** the OK test answered after 1,415 characters of reasoning at high effort, and `reasoning_effort: "low"` on one request cut that to 31; the tool test returned `get_weather` with `{"city":"Paris"}`, and an OpenCode v2.3.0 write-and-read session passed in 20 s with a first request of 3,489 prompt tokens.
  As on K2 Horizon 7B, the code-generation request spent its whole 3,000-token budget on reasoning (9,180 characters) without an answer.
- **Tuning (measured 2026-11-16, [variants](../research/2026-11-16/llamacpp-k2-horizon-mova/variants/variants.jsonl)):** at the same VRAM, keeping blocks 46-47 whole on the GPU with an 806 MiB cache decoded at 32.7 tok/s against 37.2, so the cache beats whole blocks even when it covers only 10% of the host experts.
  One more 256 MiB step (1,792 MiB) raised the hit rate to 48% but decode only about 2%, within run-to-run noise.
- **Context:** 65536 with q8_0 KV.
  128k would need q4_0 KV and is not recommended.
- **Sampling:** temperature 1.0, top_p 0.95, `reasoning_effort` high on every request; the card gives no top_k, min_p or penalties ([model card](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/raw/main/README.md)).
- **Thinking:** always on.
  Effort `high` (default), `medium` or `low` via `chat_template_kwargs.reasoning_effort`, and the card says to always use high.
  bartowski's GGUF embeds the newer template without `enable_thinking`, so `--reasoning off` does not turn reasoning off: the parser honours it only when the template mentions `enable_thinking` ([k2-horizon.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/common/parsers/k2-horizon.cpp)).
  The medium and low end-tag bug [#30246](https://github.com/ggml-org/llama.cpp/issues/30246) was closed on 2026-11-12 and its fix is in b12407.
- **Tool calling:** dedicated `k2-horizon` PEG parser (`<ifm|tool_calls>` blocks, xml by default, also json and xml_typed), selected automatically by [chat.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/common/chat.cpp).
  The strict template raises an error on an assistant turn without a thinking field; llama.cpp injects an empty `reasoning_content`, so OpenCode history works.
  Set OpenCode `limit.output` to about 16k at 64k context; a 32k output budget, which the card suggests, would leave only about 26k for input.
- **Speed:** measured above.
  It decodes at about the speed of Qwen3.8-27B and thinks at length at high effort, so turns are long; the 25-40 tok/s range inferred on 2026-11-15 held.
- **Ollama alternative:** `ifm/k2-horizon:36b` (22 GB) is in a vendor namespace, not the official library, needs Ollama 0.40.4 or newer, and does not pass earlier reasoning back in tool loops ([Ollama tags](https://ollama.com/ifm/k2-horizon/tags), fix pending in [ollama PR #18281](https://github.com/ollama/ollama/pull/18281)).
  Not recommended.
- **Evidence:** Artificial Analysis TB2.1 56.6, tau3-Banking 32.0, Terminal-Bench 4.0 0 ([AA](https://artificialanalysis.ai/models/k2-horizon-mova-36b-a4b)).
  The vendor card gives TB2.1 58.6 and tau3 26.8, which differ from AA for an unknown reason.

## 4. K2 Horizon 7B

9.0B dense (7B core plus untied embeddings and head), Apache-2.0, from the same unverified IFM org ([model card](https://huggingface.co/IFM/K2-Horizon-7B/raw/main/README.md)).
It is the simplest credible agent here: dense, all on the GPU, the same `k2-horizon` parser as MoVA.
On 2026-10-01 IFM promoted an anti-doom-loop (FTPO) checkpoint to main ([model card](https://huggingface.co/IFM/K2-Horizon-7B)).
Start it with `./llmctl up k2-7b` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [bartowski/K2-Horizon-7B-GGUF](https://huggingface.co/api/models/bartowski/K2-Horizon-7B-GGUF?blobs=true), `K2-Horizon-7B-Q8_0.gguf`, 9,573,973,280 B (8.92 GiB).
  It was made on 2026-10-07, after the FTPO promotion, so it most likely carries the released weights (inferred).
  The GGUF header, read on 2026-11-16, shows 36 dense blocks with 8 KV heads of 128 in every layer, a 250,624-token vocabulary, an untied 1.016 GiB `token_embd` and output head, and a native context of 524,288 ([plan record](../research/2026-11-16/next-model-plans/plan-k2-horizon-7b.json)); the downloaded file's SHA-256 matched its LFS id.
- **Longer context:** Q8_0 already reaches 81,920 with q8_0 KV (measured).
  `K2-Horizon-7B-Q5_K_M.gguf` (6.39 GiB) would leave room for about 110k (estimate), but decode is already below 28 tok/s at 61k (see the speed line).
- **Not chosen:** the official [IFM/K2-Horizon-7B-GGUF](https://huggingface.co/api/models/IFM/K2-Horizon-7B-GGUF?blobs=true) files were uploaded on 2026-09-22, before the promotion, so they are probably the pre-FTPO weights (inferred).

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 16g --memory-swap 16g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf bartowski/K2-Horizon-7B-GGUF:Q8_0 \
  --alias local \
  -ngl all -fa on -c 81920 -ctk q8_0 -ctv q8_0 -np 1 --jinja \
  --chat-template-kwargs '{"reasoning_effort":"high","tool_call_format":"xml"}' \
  --temp 1.0 --top-p 0.95
```

- **Memory:** KV is 144 KiB/token in f16 and about 76.5 KiB in q8_0, so 6.0 GiB at 81,920 with q8_0.
  `token_embd` (1.016 GiB) stays in host memory, so the GPU holds 7.89 GiB of weights, the KV and about 0.7 GiB of context and buffers; f16 KV fits only up to about 40k.
- **Measured on this laptop (2026-11-16, [record](../research/2026-11-16/llamacpp-k2-horizon-7b/)):** the server started from the cached file in about 2 s with `-c 81920` and q8_0 KV and peaked at 15,008 MiB including about 112 MiB for the desktop, about 830 MiB below the 15,841 MiB that CUDA can use; container RAM peaked at 5.5 GiB.
- **Measured run command:** the recipe above with `--user "$(id -u):$(id -g)"`, the cache mounted at `/hf` with `-e HF_HOME=/hf`, and `-m` with the cached file's path instead of `-hf` ([server arguments](../research/2026-11-16/llamacpp-k2-horizon-7b/server-args.json)).
- **Measured speed (2026-11-16):** decode 66.9 tok/s at a short context (66.8 to 67.0 over 3 runs of 1,500 tokens), 58.0 tok/s after an 8k prompt, 40.0 after 30k and 27.7 after 61k; prompt processing 4,333 tok/s for 8k tokens (1.9 s to the first token), 2,183 tok/s for 22k new tokens (10.2 s) and 1,092 tok/s for 30k new tokens between 30k and 61k (27.9 s); repeating a prompt hit the cache in 0.03 to 0.11 s.
  The GPU drew up to 145 W and reached 70 C.
- **KV type (measured 2026-11-16, [variant](../research/2026-11-16/llamacpp-k2-horizon-7b/variants/)):** with f16 KV at `-c 40960`, decode after the 30k prompt was 41.4 tok/s against 40.0 with q8_0, and prefill was about 3% faster, so the q8_0 penalty of [#27623](https://github.com/ggml-org/llama.cpp/issues/27623) is small here.
  The fall with depth comes from the attention reads themselves: this dense model keeps 8 KV heads in all 36 layers.
- **Measured behaviour (2026-11-16):** the OK test answered after 357 characters of reasoning at the server's high effort, and `reasoning_effort: "low"` on one request cut that to 49 characters; the tool test returned `get_weather` with `{"city":"Paris"}`, and an OpenCode v2.3.0 write-and-read session passed in 10 s with a first request of 3,482 prompt tokens.
  The code-generation request (an LRU cache module with tests) spent its whole 3,000-token budget on 12,454 characters of reasoning and returned no answer, which is the card's warning in practice: give it at least 32,768 output tokens.
- **Sampling:** temperature 1.0, top_p 0.95, `reasoning_effort` high, and at least 32,768 output tokens per the card ([model card](https://huggingface.co/IFM/K2-Horizon-7B/raw/main/README.md)).
  The card gives no top_k or min_p, so llama.cpp's defaults apply.
- **Thinking:** always on, and neither `--reasoning off` nor `reasoning_effort: "none"` disables it, because the template has no `enable_thinking` ([k2-horizon.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/common/parsers/k2-horizon.cpp)).
  Effort is set per request through `chat_template_kwargs.reasoning_effort` or the OpenAI `reasoning_effort` field ([server-common.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/tools/server/server-common.cpp)).
  Only `high`, `medium` and `low` are accepted; another value, such as `minimal` or `xhigh`, makes the template raise an error ([plan record](../research/2026-11-16/next-model-plans/plan-k2-horizon-7b.json)).
- **Tool calling:** as for MoVA.
  Set OpenCode `limit.output` to 32768, which leaves about 49k for input at 81,920, and watch for truncated answers.
- **Speed:** measured above.
  The inferred 65-80 tok/s near an empty context held at 67, and decode more than halved by 61k.
- **Ollama alternative:** `ifm/k2-horizon:7b` (5.6 GB, q4_K_M) in the vendor namespace ([Ollama tags](https://ollama.com/ifm/k2-horizon/tags)); effort control there is unverified.
- **vLLM:** a weaker AWQ path exists; see [../vllm/models.md](../vllm/models.md).
- **Evidence:** Artificial Analysis TB2.1 44.2 and tau3-Banking 27.6 ([AA](https://artificialanalysis.ai/models/k2-horizon-7b)); the vendor's SWE-bench Verified 72.4 is not independently confirmed.

## 5. Gemma 4 26B-A4B IT

25.2B total, 3.8B active MoE, Apache-2.0, released 2026-04-02 ([model card](https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/README.md)).
Google's official QAT Q4_0 GGUF does not fit entirely on the GPU at 64k as first estimated: the file has no separate output head, so llama.cpp puts a second copy of the 577.5 MiB `token_embd` on the GPU, and an all-GPU load at 64k with q8_0 KV would leave about 28 MiB.
With the experts of 12 of its 30 blocks in pinned RAM and a 2,368 MiB GPU cache for the most-used of them, it runs at 128k with f16 KV and decodes at 148 tok/s (measured).
Start it with `./llmctl up gemma4-26b` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [google/gemma-4-26B-A4B-it-qat-q4_0-gguf](https://huggingface.co/api/models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf?blobs=true), `gemma-4-26B_q4_0-it.gguf`, 14,439,363,584 B (13.45 GiB), not gated.
  Experts, attention and dense FFN are Q4_0; `token_embd` is Q6_K.
  It embeds the 2026-07-09 chat template, which fixed tool-calling loops; keep that template.
  The GGUF header, read on 2026-11-16, shows 30 blocks, all MoE with 128 experts and 8 used, 408.4 MiB of experts per block and 0.905 GiB of other block weights; attention alternates five sliding-window layers (window 1024, 8 KV heads of 256) with one global layer (2 KV heads of 512), five global layers in all ([plan record](../research/2026-11-16/next-model-plans/plan-gemma-4-26b-a4b.json)).
  The downloaded file's SHA-256 matched its LFS id.

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 20g --memory-swap 20g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf google/gemma-4-26B-A4B-it-qat-q4_0-gguf \
  --hf-file gemma-4-26B_q4_0-it.gguf \
  --alias local --no-mmproj \
  -ngl all --n-cpu-moe 12 -fa on -c 131072 -np 1 --jinja \
  -lm none -t 8 --moe-cache-mib 2368 -ctxcp 16 \
  --reasoning off \
  --temp 1.0 --top-p 0.95 --top-k 64
```

- **Memory:** global KV is 5 layers x 2 KV heads x (512 + 512) x 2 B = 20,480 B/token in f16, 2.5 GiB at 128k, and the sliding-window cache is fixed at 1,536 cells (the window plus one micro-batch), 300 MiB in f16.
  The GPU holds 0.905 GiB of other block weights, the 577.5 MiB output copy of `token_embd`, the experts of blocks 12-29 (7.2 GiB), the 2,368 MiB expert cache, the KV and about 0.8 GiB of context and buffers.
  f16 KV is deliberate: the global layers' 512-wide heads have no vector flash-attention kernel at b12407, so quantized KV would be converted to f16 on every decode step ([plan record](../research/2026-11-16/next-model-plans/plan-gemma-4-26b-a4b.json)).
- **Measured on this laptop (2026-11-16, [record](../research/2026-11-16/llamacpp-gemma-4-26b-a4b/)):** the server started from the cached file in about 5 s and peaked at 15,064 MiB including about 112 MiB for the desktop, about 780 MiB below the 15,841 MiB that CUDA can use; container RAM peaked at 10.6 GiB.
- **Measured run command:** the recipe above with `--user "$(id -u):$(id -g)"`, the cache mounted at `/hf` with `-e HF_HOME=/hf`, `-m` with the cached file's path instead of `-hf`, and `-lv 4` for the cache statistics ([server arguments](../research/2026-11-16/llamacpp-gemma-4-26b-a4b/server-args.json)).
- **Measured speed (2026-11-16):** decode 148.2 tok/s at a short context (147.9 to 150.1 over 3 runs), 132.4 tok/s after a 9.9k prompt, 124.9 after 37k and 94.9 after 124k, and 148.1 tok/s over 1,273 tokens of code; prompt processing 3,716 tok/s for 9.9k tokens (2.7 s to the first token), 2,935 tok/s for 27k new tokens (9.3 s) and 1,787 tok/s for 87k new tokens (48.8 s); repeating a prompt hit the cache in 0.07 to 0.17 s.
  The GPU drew up to 145 W and reached 71 C, and the cache served 92% of decode-time expert lookups.
  Gemma's tokenizer turns the probe's prompts into about 15% more tokens than Qwen's, so compare by tokens, not by the probe's targets.
- **Measured behaviour (2026-11-16):** the OK test answered with no reasoning at the server default, `enable_thinking: true` on one request produced 267 characters of reasoning, the tool test returned `get_weather` with `{"city":"Paris"}`, and an OpenCode v2.3.0 write-and-read session passed in 6 s with a first request of 3,824 prompt tokens.
- **Tuning (measured 2026-11-16):** the first launch, `--n-cpu-moe 9` with a 1,152 MiB cache, decoded at 140.6 tok/s with a 78% decode hit rate and peaked at 15,072 MiB ([record](../research/2026-11-16/llamacpp-gemma-4-26b-a4b/first-launch/)).
  Moving 3 more blocks to RAM and their VRAM into the cache raised the hit rate to 92% and decode by 5.4% at about 1% less prefill, the same trade as on Qwen3.6.
- **Memory cap:** 20g covers 4.8 GiB of pinned experts, the 8 GiB default prompt cache, up to 16 sliding-window checkpoints of 300 MiB each (`-ctxcp 16`; the default is 32) and the process.
- **Images:** the mmproj (1.11 GiB) would need about 1.1 GiB of VRAM, taken from the expert cache; leave it out for coding.
- **Sampling:** temperature 1.0, top_p 0.95, top_k 64 ([model card](https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/README.md), [Unsloth Gemma 4 page](https://unsloth.ai/docs/models/gemma-4)).
- **Thinking:** off by default in the template; turn it on per request with `chat_template_kwargs: {"enable_thinking": true}` ([chat_template.jinja](https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/chat_template.jinja)).
  The template drops old thoughts from history except on tool-call turns, so do not feed thought blocks back by hand.
  Switch it with `enable_thinking`, not with `reasoning_effort: "none"`, which [#30759](https://github.com/ggml-org/llama.cpp/issues/30759) (open) reports does not reliably turn it off on Gemma 4.
- **Tool calling:** native Gemma 4 tool tokens; [chat.cpp](https://github.com/ggml-org/llama.cpp/blob/b12407/common/chat.cpp) detects `<|tool_call>call:` and uses the Gemma 4 PEG parser.
- **Open issues:** [#29655](https://github.com/ggml-org/llama.cpp/issues/29655) (open) reports partial or malformed multi-line write and edit arguments while streaming.
  Test OpenCode's edit and write tools on a multi-line change before relying on this model.
  In the user's LibreDB Studio harness `gemma4:26b` passed only with an empty-turn retry, and the one loss whose last turn was read was an empty completion after 11 tool calls (measured on the M5 Max, libredb-studio; [agent-harness-evidence.md](../docs/agent-harness-evidence.md)); OpenCode has no such retry, so watch for empty turns deep in a loop.
  [#32362](https://github.com/ggml-org/llama.cpp/issues/32362) (open since 2026-11-15) reports raw thought and tool-call tokens leaking into streamed content during editor tool calls, on Unsloth's 26B QAT file at b12426; the short measured session did not show it.
- **Speed:** measured above; the 150-220 tok/s range inferred on 2026-11-15 was close, at 148 tok/s with 12 blocks of experts in RAM.
- **Ollama alternative:** `gemma4:26b-a4b-it-qat` (model 14.44 GB plus projector 1.19 GB; the same tensors as Google's file with different metadata, 2,144 B smaller) with `OLLAMA_CONTEXT_LENGTH=65536`, `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` ([manifest](https://registry.ollama.ai/v2/library/gemma4/manifests/26b-a4b-it-qat)).
  The default `gemma4:26b` is Q4_K_M at 17.07 GB and does not fit ([Ollama tags](https://ollama.com/library/gemma4/tags)).
  Given the duplicated output head above, the Ollama build likely spills too (inferred).
- **Evidence:** NVIDIA same-harness SWE-bench Verified 57.40 and TB2.1 37.22 ([Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/README.md)), Artificial Analysis TB2.1 39.0 ([AA](https://artificialanalysis.ai/models/gemma-4-26b-a4b)).

### A/B: Unsloth QAT UD-Q4_K_XL against Google Q4_0

Unsloth reports that a naive QAT-to-Q4_0 conversion of the 26B model, at 14.44 GB, has mean KL divergence 0.36094 and top-1 agreement 70.20% against the BF16 QAT model, while its own `UD-Q4_K_XL` scores 0.09788 and 85.63% (vendor-reported, [Unsloth Gemma 4 QAT page](https://unsloth.ai/docs/models/gemma-4/qat)).
The page does not name Google's repo, but 14.44 GB is exactly the size of `gemma-4-26B_q4_0-it.gguf`, and Unsloth's remark about Q6_K embeddings matches Google's file, so the comparison most likely targets it (inferred).
The Unsloth file, [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/api/models/unsloth/gemma-4-26B-A4B-it-qat-GGUF?blobs=true) `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`, is 14,249,047,104 B (13.27 GiB), 0.18 GiB smaller, so it drops into the same command with `-hf unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL` in place of the two Google lines.
Run both on this laptop before settling: `llama-perplexity --kl-divergence` from the full image, and the same short OpenCode edit task on each (see [benchmarking.md](benchmarking.md#quality-smoke-tests)).
Until that A/B is done, Google's file stays the default because it is the vendor's own QAT artifact and two independent harnesses scored the model.

## 6. Qwen3-Coder-Next

79.7B total, 3B active hybrid MoE, Apache-2.0, released 2026-02-03 ([model card](https://huggingface.co/Qwen/Qwen3-Coder-Next/raw/main/README.md)).
It is the best non-thinking tool caller, at the cost of about 26 GiB of experts in pinned RAM; it ran at 128k on 2026-11-16 (measured below).
Start it with `./llmctl up coder-next` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [unsloth/Qwen3-Coder-Next-GGUF](https://huggingface.co/api/models/unsloth/Qwen3-Coder-Next-GGUF?blobs=true), `Qwen3-Coder-Next-UD-IQ4_XS.gguf`, 38,429,272,064 B (35.79 GiB), a single file.
  Experts are 33.38 GiB, 0.695 GiB per block over 48 blocks.
- **Freshness:** Unsloth re-uploaded the GGUFs on 2026-02-04 after a llama.cpp `key_gdiff` fix that caused looping ([Unsloth Qwen3-Coder-Next page](https://unsloth.ai/docs/models/qwen3-coder-next)); the repo was last modified on 2026-03-06, so a fresh download is the fixed file.

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 35g --memory-swap 35g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf unsloth/Qwen3-Coder-Next-GGUF:UD-IQ4_XS \
  --alias local \
  -ngl all --n-cpu-moe 38 -fa on -c 131072 -np 1 --jinja \
  -lm none -t 8 --moe-cache-mib 1504 --cache-ram 4096 -ctxcp 8 \
  --temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01
```

- **Memory:** `--n-cpu-moe 38` keeps the experts of blocks 0-37 (38 x 712 MiB = 27,056 MiB) in pinned RAM, plus 0.31 GiB of host `token_embd`.
  The GPU holds 9,272 MiB of weights, 3,072 MiB of f16 KV (only 12 of the 48 layers carry attention, 24 KiB/token), 75 MiB of recurrent state, the 1,504 MiB expert cache and a 698 MiB compute buffer ([load log](../research/2026-11-16/llamacpp-qwen3-coder-next/server-load.log)).
- **Measured on this laptop (2026-11-16, [record](../research/2026-11-16/llamacpp-qwen3-coder-next/)):** the server started from the cached file in about 15 s with `-c 131072` and f16 KV and peaked at 15,076 MiB including about 112 MiB for the desktop, about 765 MiB below the 15,841 MiB that CUDA can use; container RAM was 27.3 GiB after load and peaked at 31.9 GiB.
  The run added `--user`, the cache at `/hf`, `-m` instead of `-hf` and `-lv 4` ([server arguments](../research/2026-11-16/llamacpp-qwen3-coder-next/server-args.json)).
  The first launch, with q8_0 KV and a 2,944 MiB cache, peaked at 15,082 MiB against a critic-corrected plan of about 15,100 MiB ([plan record](../research/2026-11-16/next-model-plans/critique.json)).
- **Measured speed (2026-11-16):** decode 71.5 tok/s at a short context (71.1 to 72.0 over 3 runs), 64.7 after an 8k prompt, 66.0 after 30k and 58.6 after 101k, and 68.8 tok/s over 1,445 tokens of code; prompt processing 1,114 tok/s for 8k tokens (7.3 s to the first token), 1,150 tok/s for 22.7k new tokens (19.7 s) and 1,026 tok/s for 71k new tokens (69.5 s); repeating a prompt hit the cache in 0.08 to 0.16 s.
  The GPU drew up to 141 W and reached 71 C, and the cache served 55% of decode-time expert lookups; every miss was uploaded to the GPU, none was computed on the CPU.
- **Measured behaviour (2026-11-16):** the OK test answered with no reasoning, the tool test returned `get_weather` with `{"city":"Paris"}`, and the OpenCode v2.3.0 write-and-read session passed in 13 s with a first request of 4,249 prompt tokens.
- **KV type:** f16, because q8_0 KV loses speed with depth on this model.
  At equal VRAM, the first launch with q8_0 KV and a 2,944 MiB cache decoded 80.9 tok/s at a short context and 77.8 over code, 13% faster than f16, but 68.1 after 30k (3% faster) and 48.7 after 101k (17% slower) ([first launch](../research/2026-11-16/llamacpp-qwen3-coder-next/first-launch/)).
  For sessions that stay under about 30k tokens, `-ctk q8_0 -ctv q8_0 --moe-cache-mib 2944` is the faster launch.
  The cause is most likely that llama.cpp's MMA flash-attention kernels convert a quantized K/V cache to f16 in a separate pass on models with 256-wide heads and a GQA ratio above 4, as this one has (16 query and 2 KV heads); [PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802), open on 2026-11-16, fuses that conversion, so re-measure q8_0 on an image that includes it (inferred from the PR description).
- **Offload variants (2026-11-16):** with q8_0 KV, a 3,200 MiB cache decoded 1% faster than 2,944 MiB but left only 505 MiB free, and `--n-cpu-moe 36` with a 1,664 MiB cache, two more whole blocks on the GPU, decoded 8% slower ([benchmarking.md](benchmarking.md#qwen3-coder-next-offload-variants)).
- **Memory cap:** 35g covers 26.4 GiB of pinned experts, the host `token_embd`, the 4 GiB prompt cache (`--cache-ram 4096`; the default is 8 GiB), up to 8 recurrent-state checkpoints of 75 MiB (`-ctxcp 8`; the default is 32) and the process.
  About 21 GiB of RAM stayed available while it ran, so close large desktop applications first and do not run a `heavy` gate at the same time ([../docs/host-setup.md](../docs/host-setup.md#container-memory-caps)).
- **262k (estimate):** f16 KV would need 6 GiB and does not fit; q8_0 KV needs 3,264 MiB, which leaves about 1,300 MiB for the cache, and decode would fall further than the 48.7 tok/s that q8_0 gave after 101k.
- **Sampling:** temperature 1.0, top_p 0.95, top_k 40, min_p 0.01 (llama.cpp's default is 0.05), repeat penalty off ([Unsloth Qwen3-Coder-Next page](https://unsloth.ai/docs/models/qwen3-coder-next)).
- **Thinking:** none; the card says it supports only non-thinking mode, so set `reasoning: {"supported": false}` in OpenCode ([model card](https://huggingface.co/Qwen/Qwen3-Coder-Next/raw/main/README.md), [../docs/opencode.md](../docs/opencode.md)).
- **Tool calling:** Qwen3-Coder XML through the Qwen3-Coder parser ([chat.cpp](https://github.com/ggml-org/llama.cpp/blob/b12418/common/chat.cpp)).
- **IQ3_S sanity check:** about 20.6 GiB of the experts in UD-IQ4_XS are IQ3_S tensors.
  [#28581](https://github.com/ggml-org/llama.cpp/issues/28581) affects only nvcc 13.2.0/13.2.1 builds, not the CUDA 12.8.1 image.
  On 2026-11-16 the answers on this image were coherent, including a 1,445-token code answer.
- **Speed:** measured above, well over the 30-50 tok/s inferred on 2026-11-15.
  Prefill, at about 1,000-1,150 tok/s, is close to Bolt's and MoVA's; `-lm none` pins the host experts instead of leaving them in the pageable memory of [#25859](https://github.com/ggml-org/llama.cpp/issues/25859), which doubled prefill on Qwen3.6.
- **Ollama alternative:** none that fits; `qwen3-coder-next` is 52 GB ([Ollama tags](https://ollama.com/library/qwen3-coder-next/tags)).
- **Evidence:** Artificial Analysis tau2 79.5 and TB2.1 38.2 ([AA](https://artificialanalysis.ai/models/qwen3-coder-next)); vendor SWE-bench Verified 70.6-71.3 ([tech report](https://raw.githubusercontent.com/QwenLM/Qwen3-Coder/main/qwen3_coder_next_tech_report.pdf)).

## 7. NVIDIA Nemotron 3.5 Bolt 74B-A9B

74B total, 9B active hybrid Mamba-2 plus MoE, OpenMDW-1.1 (permissive, commercial use allowed), released 2026-10-15 ([model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/raw/main/README.md), [license](https://raw.githubusercontent.com/OpenMDW/OpenMDW/main/1.1/LICENSE.OpenMDW-1.1)).
NVIDIA ships one GGUF built for 24 GB machines: IQ2_XXS QAT experts, NVFP4 Mamba, Q8_0 for the rest.
This laptop is below NVIDIA's stated 32 GB+ hardware floor, so this is not a vendor-supported setup, but it ran at 128k on 2026-11-16 (measured below).
Start it with `./llmctl up bolt` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF](https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/tree/main), `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`, 22,018,326,528 B (20.51 GiB), plus a 1.61 GiB mmproj that is not used.
  The repo has no Q4_K_M file, so `-hf` alone would fall back to the first file; name it with `--hf-file`.
- **Arch support:** `nemotron_h_moe` with per-layer expert widths, merged in [PR #30520](https://github.com/ggml-org/llama.cpp/pull/30520) on 2026-10-15 ([llama-arch.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/src/llama-arch.cpp), [nemotron-h.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/src/models/nemotron-h.cpp)).

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 26g --memory-swap 26g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF \
  --hf-file NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
  --alias local --no-mmproj \
  -ngl all --n-cpu-moe 60 -fa on -c 131072 -np 1 --jinja \
  -lm none -t 8 --moe-cache-mib 2816 -ctxcp 16 \
  --reasoning off \
  --temp 1.0 --top-p 0.95
```

- **Memory:** from the GGUF header, non-routed tensors are 4.617 GiB and routed experts 15.882 GiB, over 88 blocks: 40 Mamba-2, 40 MoE (496 routed experts, 9 used, per-layer widths) and 8 attention layers ([plan record](../research/2026-11-16/next-model-plans/plan-nemotron-3.5-bolt.json)).
  KV is only 8 KiB/token in f16 (1.00 GiB at 128k), and the Mamba state is about 165 MiB per sequence.
  `--n-cpu-moe 60` keeps the routed experts of the MoE blocks among blocks 0-59 (10.2 GiB) in pinned RAM, and a 2,816 MiB cache holds the most-used of them on the GPU.
- **Measured on this laptop (2026-11-16, [record](../research/2026-11-16/llamacpp-nemotron-3.5-bolt/)):** the server started from the cached file in about 7 s with `-c 131072` and f16 KV and peaked at 15,064 MiB including about 112 MiB for the desktop, about 780 MiB below the 15,841 MiB that CUDA can use; container RAM peaked at 17.8 GiB.
  The run added `--user`, the cache at `/hf`, `-m` instead of `-hf` and `-lv 4` ([server arguments](../research/2026-11-16/llamacpp-nemotron-3.5-bolt/server-args.json)).
- **Measured speed (2026-11-16):** decode 73.5 tok/s at a short context (73.1 to 73.7 over 3 runs), 69.1 after an 8.3k prompt, 71.1 after 31k and 65.8 after 104k, and 72.2 tok/s over 1,327 tokens of code; prompt processing 1,519 tok/s for 8.3k tokens (5.5 s to the first token), 1,616 tok/s for 23k new tokens (14.5 s) and 1,411 tok/s for 73k new tokens (52.2 s); repeating a prompt hit the cache in 0.10 to 0.19 s.
  Decode barely falls with depth, because only 8 of 88 layers carry attention; prefill is held back by expert uploads.
  The GPU drew up to 145 W and reached 71 C, and the cache served 69% of decode-time expert lookups.
- **Measured behaviour (2026-11-16):** the OK test answered with no reasoning at the server default, `enable_thinking: true` on one request produced 72 characters of reasoning, and the tool test returned `get_weather` with `{"city":"Paris"}`.
  The OpenCode v2.3.0 write-and-read session passed in 13 s with a first request of 4,372 prompt tokens, after one recovery: the model issued `write` and `read` as parallel calls in one step, the read ran before the file existed and failed, and the model then used `glob` and read the file again.
- **Context:** 131072 with f16 KV fits; 65536 is no longer needed as a first step.
- **Memory cap:** 26g covers 10.2 GiB of pinned experts, 0.53 GiB of host `token_embd`, the 8 GiB default prompt cache, up to 16 Mamba-state checkpoints of 165 MiB (`-ctxcp 16`; the default is 32) and the process.
- **Sampling:** temperature 1.0, top_p 0.95 for both thinking and non-thinking ([model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/raw/main/README.md)).
- **Thinking:** on by default in the template; `--reasoning off` sets the server default off.
  Per request, `chat_template_kwargs: {"enable_thinking": false}` turns it off and `{"enable_thinking": true, "medium_effort": true}` gives medium effort ([model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/raw/main/README.md)).
- **Tool calling:** the template is routed to llama.cpp's Qwen3-Coder handler ([chat.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/common/chat.cpp)), so [#30118](https://github.com/ggml-org/llama.cpp/issues/30118) is a risk on the same path; run the tool-call smoke test.
  The `pixtral` pre-tokenizer bug in NVIDIA's GGUF ([#30894](https://github.com/ggml-org/llama.cpp/issues/30894)) was fixed by [PR #30960](https://github.com/ggml-org/llama.cpp/pull/30960) before b12407.
- **Speed:** measured above, inside the 55-80 tok/s range inferred on 2026-11-15; Nemotron 3.5 Lightning decodes about twice as fast here.
  NVIDIA's own agent replay was 32% slower than Qwen3.6 ([model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/raw/main/README.md)).
- **Drafter:** NVIDIA's LiLiCorr drafter GGUF (0.93 GiB) would push about two more blocks to RAM, and its benefit with CPU-resident experts is unmeasured; leave it out.
- **Ollama alternative:** `nemotron-3.5-bolt:74b-a9b` (24.75 GB with projector and drafter) is a slower fallback, because Ollama offloads whole layers ([manifest](https://registry.ollama.ai/v2/library/nemotron-3.5-bolt/manifests/74b-a9b)).
- **Evidence:** NVIDIA's same-harness table gives TB2.1 50.98 against 44.38 for Qwen3.6, plus MCP-Atlas 52.2 and tau3 17.0 (vendor-run, [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/raw/main/README.md)); there is no independent run and no SWE-bench Verified score.

## 8. Groot-19B-A3B-261023

About 19B total, 3B active, Qwen3.6-style `qwen35moe` architecture, published 2026-10-23 by an anonymous org ([model card](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/README.md)).
The card metadata says Apache-2.0, but the repo has no LICENSE file, and the pruning method and data are undisclosed.
It is the fast all-in-VRAM option at 128k, with the same tool format and parser as Qwen3.6; treat its quality as unproven until an A/B against Qwen3.6.
Start it with `./llmctl up groot` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [GrootTeam/Groot-19B-A3B-261023-GGUF](https://huggingface.co/api/models/GrootTeam/Groot-19B-A3B-261023-GGUF?blobs=true), `Groot-19B-A3B-261023-Q4_K_M.gguf`, 11,657,618,560 B (10.86 GiB), plus a 0.84 GiB mmproj that is not used.
- **Template difference:** the official GGUF embeds `chat_template.jinja`, which re-emits every past assistant turn's reasoning and ignores `preserve_thinking`, while [bartowski's GGUF](https://huggingface.co/api/models/bartowski/GrootTeam_Groot-19B-A3B-261023-GGUF?blobs=true) (Q4_K_M 11.63 GiB) embeds the `tokenizer_config.json` template, which drops old reasoning unless `preserve_thinking` is true, as the card describes.
  With thinking on in long OpenCode sessions, the official template grows the context faster; compare both if that matters.

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 16g --memory-swap 16g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf GrootTeam/Groot-19B-A3B-261023-GGUF:Q4_K_M \
  --alias local --no-mmproj \
  -ngl all -fa on -c 131072 -np 1 --jinja \
  --reasoning off \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0
```

- **Memory (estimate):** KV is 2 x 10 full-attention layers x 2 KV heads x 256 x 2 B = 20 KiB/token, 2.50 GiB at 128k in f16, plus 0.06 GiB recurrent state and 0.5-1.0 GiB compute: about 13.9-14.4 GiB, plus about 0.4 GiB CUDA context.
  262144 with q8_0 KV (about 2.66 GiB) comes to about 14.6 GiB plus the CUDA context, about 15.0 of 15.92, tight; f16 at 262144 does not fit.
- **Measured on this laptop (2026-11-16, [record](../research/2026-11-16/llamacpp-groot-19b-a3b/)):** the server started from the cached file in about 2 s with `-c 131072` and f16 KV and peaked at 13,822 MiB including about 112 MiB for the desktop, 2 GiB below the 15,841 MiB that CUDA can use; container RAM peaked at 4.4 GiB.
  The run added `--user`, the cache at `/hf`, `-m` instead of `-hf` and `--n-cpu-moe 0` ([server arguments](../research/2026-11-16/llamacpp-groot-19b-a3b/server-args.json)).
- **Measured speed (2026-11-16):** decode 169.8 tok/s at a short context (169.5 to 170.0 over 3 runs), 164.5 after an 8.6k prompt, 150.1 after 32k and 112.6 after 107k, and 166.5 tok/s over 1,147 tokens of code; prompt processing 5,737 tok/s for 8.6k tokens (1.5 s to the first token), 4,605 tok/s for 24k new tokens (5.3 s) and 2,743 tok/s for 76k new tokens (27.7 s); repeating a prompt hit the cache in 0.04 to 0.12 s.
  It is the fastest model measured here, about 40% faster than Qwen3.6-35B-A3B at a short context, because all of it fits in VRAM.
  The GPU drew up to 146 W and reached 67 C.
- **Measured behaviour (2026-11-16):** the OK test answered with no reasoning at the server default, `enable_thinking: true` on one request produced 327 characters of reasoning, the tool test returned `get_weather` with `{"city":"Paris"}`, and an OpenCode v2.3.0 write-and-read session passed in 6 s with a first request of 4,186 prompt tokens.
  That short task does not test quality or path fidelity.
- **Sampling** ([model card](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/README.md), lines 86-88): precise coding, temperature 0.6, top_p 0.95, top_k 20, presence_penalty 0; general thinking, temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5; non-thinking, temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5.
- **Thinking:** toggle, on by default; `--reasoning off` for the server default and `chat_template_kwargs.enable_thinking` per request, as for Qwen3.6.
- **Tool calling:** Qwen3-Coder XML through the Qwen3-Coder parser ([chat.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/common/chat.cpp)); [#30118](https://github.com/ggml-org/llama.cpp/issues/30118) and [#30357](https://github.com/ggml-org/llama.cpp/issues/30357) probably apply, because the architecture and template family match Qwen3.6 (inferred, untested on Groot).
- **Risk for a coding agent:** a community report says the model changes Polish words in file paths even when asked to copy them verbatim ([discussion #5](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/discussions/5), open).
  In the A/B, check that file paths and identifiers come back unchanged.
- **Speed:** measured above.
- **Ollama alternative:** no official library entry; a community upload lists only the vision capability, with no tools or thinking badge ([Ollama tags](https://ollama.com/fauxpaslife/Groot-19B-A3B-261023/tags)).
  Not recommended.
- **vLLM:** an NVFP4 path exists; see [../vllm/models.md](../vllm/models.md).
- **Evidence:** every agentic score is self-reported, and on the vendor's own chart Groot loses to its base on SWE-bench Verified (31.66 vs 36.27) ([model card](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/README.md)).
  One third-party run exists without a Qwen3.6 comparator ([Blackwall](https://guitaripod.github.io/blackwall/#groot-19b-a3b-nvfp4~medium)).

## 9. gpt-oss-20b

21B total, 3.6B active MoE, Apache-2.0, released 2025-08-05 ([model card](https://huggingface.co/openai/gpt-oss-20b/raw/main/README.md)).
It is the oldest pick and weak on agentic benchmarks, but the most mature and fastest, so it is the pipeline baseline.
Start it with `./llmctl up gpt-oss` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/api/models/ggml-org/gpt-oss-20b-GGUF?blobs=true), `gpt-oss-20b-MXFP4.gguf`, 12,109,566,624 B (11.28 GiB).
  The ggml-org guide says to always use the original MXFP4 model; K-quants are not smaller ([guide](https://github.com/ggml-org/llama.cpp/discussions/15396)).

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 16g --memory-swap 16g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf ggml-org/gpt-oss-20b-GGUF:MXFP4 \
  --alias local \
  -ngl all -fa on -c 131072 -ctk q8_0 -ctv q8_0 -np 1 --jinja \
  --reasoning-effort medium \
  --temp 1.0 --top-p 1.0
```

- **Memory (estimate):** 12 full-attention layers x 2 x 8 KV heads x 64 x 2 B = 24 KiB/token in f16, 1.59 GiB at 128k in q8_0; the sliding-window layers (window 128) add under 0.1 GiB.
  Total about 13.9 GiB.
  128k with f16 KV is about 15.1 GiB, tight; 64k with f16 is about 13.6 GiB.
- **Measured on this laptop (2026-11-16, recipe above with `--user` and the `/hf` cache mount, [record](../research/2026-11-16/llamacpp-gpt-oss-20b/)):** GPU memory 13,360 MiB in use after loading at `-c 131072` with q8_0 KV (13,408 MiB peak, including about 112 MiB for the desktop); container RAM 2.1 to 2.5 GiB.
- **Measured speed (2026-11-16):** decode 166 tok/s for 1,500 generated tokens at a short context (165.7 to 166.2 over 3 runs), 145 tok/s after an 8k prompt and 105 to 108 tok/s after a 30k prompt; prompt processing 7,436 tok/s for 8k tokens and 5,546 tok/s for 22k new tokens; repeating the same prompt hit the prompt cache and gave a first token in 0.05 s (8k) and 0.10 s (30k).
  The GPU drew about 143 W at its 140 W limit, with the SM clock up to 2,242 MHz and at most 65 C.
- **Measured behaviour (2026-11-16):** "Reply with exactly: OK" returned OK with 151 characters of reasoning; the tool test returned `finish_reason` `tool_calls` with `get_weather` and `{"city":"Paris"}`; an OpenCode v2.3.0 session wrote a file and read it back correctly in 8 s.
- **Sampling:** temperature 1.0, top_p 1.0 (OpenAI default); Unsloth adds top_k 0 ([Unsloth gpt-oss page](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune)).
- **Thinking:** always on and cannot be turned off; effort `low`, `medium` (default) or `high` through `--reasoning-effort` or the request's `reasoning_effort` ([model card](https://huggingface.co/openai/gpt-oss-20b/raw/main/README.md)).
- **Tool calling:** Harmony format through the dedicated [gpt-oss parser](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12418/common/parsers/gpt-oss.cpp); `tool_choice: "required"` becomes a strict grammar, so it is enforced, unlike on vLLM.
- **Open issue:** [#30781](https://github.com/ggml-org/llama.cpp/issues/30781) (open): the gpt-oss template returns HTTP 500 on array tool parameters without `items`.
  Whether it hits the 20b GGUF with OpenCode's tool schemas is inferred, not tested.
- **Speed reference:** 204.85 tok/s tg128 on a desktop RTX 5080 (measured on a desktop RTX 5080, llama.cpp build 6210, [ggml-org guide](https://github.com/ggml-org/llama.cpp/discussions/15396)); the bandwidth-scaled estimate for this laptop was about 190 tok/s, and the measured 166 tok/s above is about 87% of it, with CUDA graphs disabled and the 140 W limit.
- **Ollama alternative:** `gpt-oss:20b` (14 GB) with `OLLAMA_CONTEXT_LENGTH=65536` ([Ollama tags](https://ollama.com/library/gpt-oss/tags)).
- **vLLM:** a 64k option with `--gpu-memory-utilization 0.965`, `--enforce-eager`, FP8 KV and `TRITON_ATTN` is in [../vllm/models.md](../vllm/models.md).
- **Evidence:** Artificial Analysis TB2.1 13.9 ([AA](https://artificialanalysis.ai/models/gpt-oss-20b)) and NVIDIA same-harness SWE-bench Verified 52.44, TB2.1 15.17 ([Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/README.md)).

OpenCode entry (llmctl checks it against the recipe; the validated config carries it under `llamacpp-docker` beside the vLLM gpt-oss-20b entry):

```json
"gpt-oss-20b": {
  "modelID": "local",
  "name": "gpt-oss-20b MXFP4 (llama.cpp)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true, "effort": ["low", "medium", "high"] } },
  "limit": { "context": 131072, "output": 16384 }
}
```

## 10. DeepSeek-R1-0528-Qwen3-8B (chat and reasoning only)

8.19B dense, MIT, released 2025-05-29 ([model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/raw/main/README.md)).
It is the newest DeepSeek checkpoint that runs on 16 GB; the January 2025 R1 distills are older and also lack tool calling.
Use it for reasoning or questions next to an agent model, not as an OpenCode agent.
Start it with `./llmctl up deepseek-r1` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF](https://huggingface.co/api/models/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/tree/main), `DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf`, 8,709,519,872 B (8.11 GiB).

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 16g --memory-swap 16g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q8_0 \
  --alias local \
  -ngl all -fa on -c 65536 -ctk q8_0 -ctv q8_0 -np 1 --jinja \
  --reasoning-format deepseek \
  --temp 0.6 --top-p 0.95 --min-p 0.01
```

- **Memory:** KV is 2 x 36 layers x 8 KV heads x 128 x 2 B = 144 KiB/token in f16, about 76.5 KiB in q8_0, so 4.78 GiB at 64k.
  f16 KV at 64k (17.9 GiB) does not fit.
  Q4_K_M (4.68 GiB) reaches 128k with q8_0 KV at about 15.0 GiB, tight.
- **Measured on this laptop (2026-11-16, [record](../research/2026-11-16/llamacpp-deepseek-r1-0528-qwen3-8b/)):** the server started from the cached file in about 2 s and peaked at 13,286 MiB including about 112 MiB for the desktop, 2.5 GiB below the 15,841 MiB that CUDA can use; container RAM peaked at 3.0 GiB.
  The run added `--user`, the cache at `/hf` and `-m` instead of `-hf` ([server arguments](../research/2026-11-16/llamacpp-deepseek-r1-0528-qwen3-8b/server-args.json)).
- **Measured speed (2026-11-16):** decode 70.6 tok/s at a short context (70.3 to 70.7 over 3 runs), 60.5 after an 8k prompt and 41.1 after 30k; prompt processing 4,291 tok/s for 8k tokens (2.0 s to the first token) and 2,166 tok/s for 22k new tokens (10.3 s); repeating a prompt hit the cache in 0.06 to 0.09 s.
  A code-generation request returned a complete answer (5,699 characters after 2,109 of reasoning) in 2,045 tokens at 69.3 tok/s.
  The GPU drew up to 145 W and reached 70 C.
- **Sampling:** temperature 0.6, top_p 0.95 ([model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/raw/main/README.md)); Unsloth's examples add min_p 0.01 ([Unsloth R1-0528 page](https://unsloth.ai/docs/models/tutorials/deepseek-r1-0528-how-to-run-locally)).
- **Thinking:** always on, with no official off switch in the template ([tokenizer_config.json](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/raw/main/tokenizer_config.json)).
  `--reasoning-format deepseek` puts the reasoning in `reasoning_content`.
- **Tool calling:** no.
  The official template never renders tools, and Artificial Analysis measured tau2 at 0% ([AA](https://artificialanalysis.ai/models/deepseek-r1-qwen3-8b)).
  Measured on 2026-11-16: the probe's one-tool request did come back as a parsed `get_weather` call, but an OpenCode v2.3.0 session with tools enabled for the test failed: after 72 s and 3,673 output tokens the model wrote the `write` tool's arguments into its answer as a JSON block and never called the tool, so no file was created ([events](../research/2026-11-16/llamacpp-deepseek-r1-0528-qwen3-8b/opencode-e2e-tools-test-events.jsonl)).
  Keep `tools: false`.
  Unsloth's GGUF template does inject a tool list, but llama.cpp has no dedicated R1 tool handler and its docs still call R1 tool calling work in progress ([function-calling.md](https://github.com/ggml-org/llama.cpp/blob/b12418/docs/function-calling.md)).
  Do not set `capabilities.tools` for it in OpenCode.
- **Ollama alternative:** `deepseek-r1:8b` is q4_K_M (5.2 GB); `deepseek-r1:8b-0528-qwen3-q8_0` (8.9 GB) matches this recipe.
  Ollama's `think` field works, and its 8b template has no tools block ([manifest](https://registry.ollama.ai/v2/library/deepseek-r1/manifests/8b)).

OpenCode entry (tools off; use it in plan or chat mode, never as the build agent):

```json
"deepseek-r1-0528-qwen3-8b": {
  "modelID": "local",
  "name": "DeepSeek-R1-0528-Qwen3-8B Q8_0, chat only (llama.cpp)",
  "capabilities": { "tools": false, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 65536, "output": 16384 }
}
```

## GGUF builds of the vLLM and SGLang models

These four models also have recipes on [vLLM](../vllm/models.md) and, except Mellum2.1, on [SGLang](../sglang/models.md); their GGUF builds were measured on 2026-11-17 so that every engine runs every model it can.
With these recipes, `./llmctl up` without `--runtime` starts the four names on llama.cpp, and `--runtime vllm`, or `--runtime sglang` for the three with an SGLang recipe, starts another engine's build.
Each one fits whole on the GPU at 131072 tokens with f16 KV.
None of them has run an OpenCode session yet; the clients e2e covers that.
The measurements started each recipe through `./llmctl up` from a copy of llmctl, with the GGUF file already in the page cache and the K2 Horizon MoVA download of the same lane running beside it ([records](../research/2026-11-17/README.md)).

### Qwen3.5-9B Q8_0

The GGUF build of the model that [vLLM](../vllm/models.md#qwen35-9b-nvfp4) and [SGLang](../sglang/models.md#qwen35-9b-nvfp4) serve as NVFP4.
Start it with `./llmctl up qwen3.5` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.
The NVFP4 build is faster: start it with `./llmctl up qwen3.5 --runtime vllm` or `--runtime sglang` when decode speed matters more than the 8-bit weights.

- **GGUF:** [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF/tree/main), `Qwen3.5-9B-Q8_0.gguf`, 9,527,502,048 B (8.87 GiB), at revision `3885219b6810b007914f3a7950a8d1b469d598a5`.
  The repository also holds mmproj files, so the launch adds `--no-mmproj`.

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 16g --memory-swap 16g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf unsloth/Qwen3.5-9B-GGUF \
  --hf-file Qwen3.5-9B-Q8_0.gguf \
  --alias local --no-mmproj \
  -ngl all -fa on -c 131072 -np 1 --jinja \
  --reasoning off \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0
```

- **Measured on this laptop (2026-11-17, [record](../research/2026-11-17/llamacpp-qwen3.5-9b/)):** `llmctl up` reached ready in 3.4 s and the GPU peaked at 12,782 MiB including about 112 MiB for the desktop; container RAM peaked at 14.95 GiB under the 16 GiB cap, of which 9.07 GiB was page cache of the mmapped GGUF and 5.87 GiB anonymous memory.
  The model has 4 KV heads of 256 for 16 query heads, and f16 KV holds 128k, so q8_0 KV was not tried.
- **Measured speed (2026-11-17):** decode 66.0 tok/s at a short context (65.4 to 66.2 over 3 runs), and 1,525 tokens of code at 66.2 tok/s; prompt processing 4,495 tok/s for 8.6k tokens (1.9 s to the first token), 3,834 tok/s for 24k new tokens (6.3 s) and 2,571 tok/s for 76k new tokens (29.5 s), where decode was 45.1 tok/s; repeating a prompt hit the cache in 0.05 to 0.13 s.
  The GPU drew up to 146.0 W against its 145 W limit and reached 71 C.
- **Against the other engines:** the NVFP4 build decodes 90.8 tok/s on vLLM v0.33.0 ([record](../research/2026-11-17/vllm-qwen3.5-9b/README.md)) and 93.7 tok/s on SGLang v0.5.23 ([record](../research/2026-11-17/sglang-qwen3.5-9b/README.md)) (measured on the MSI Raider 18 Max HX, 2026-11-17), so this Q8_0 build is about 28% slower; that Q8_0 reads more weight bytes per token than NVFP4 is the likely cause (inferred from the formats, not measured).
- **Measured behaviour (2026-11-17):** `llmctl test` passed, the tool test returned `get_weather` with `{"city":"Paris"}` and the second turn answered from the tool result, Hermes' title request with a strict `json_schema` returned a title that parses, and a forced `tool_choice` request returned the tool call.
- **Thinking:** toggle; `--reasoning off` makes the server default off, and `chat_template_kwargs.enable_thinking` true turns it on per request (735 characters of reasoning, measured); no `reasoning_effort` value turns it on.
- **Sampling:** temperature 0.6, top_p 0.95, top_k 20, min_p 0 and presence_penalty 0, the values of the vLLM recipe of the same model.

OpenCode entry (the validated config carries it under `llamacpp-docker` beside the vLLM and SGLang entries of the same model):

```json
"qwen3.5-9b": {
  "modelID": "local",
  "name": "Qwen3.5-9B Q8_0 (llama.cpp)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 131072, "output": 16384 },
  "variants": [
    { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
    { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
  ]
}
```

### Gemma 4 12B IT QAT Q4_0

Google's QAT Q4_0 GGUF of the model that [vLLM](../vllm/models.md#gemma-4-12b-it-qat-w4a16) and [SGLang](../sglang/models.md#gemma-4-12b-it-qat-w4a16) serve as QAT W4A16, and the fastest of the three builds here.
Start it with `./llmctl up gemma4-12b` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/api/models/google/gemma-4-12B-it-qat-q4_0-gguf/tree/main), `gemma-4-12b-it-qat-q4_0.gguf`, 6,975,879,296 B (6.50 GiB), at revision `29d097773436b69ff9feafd636ab4cf873786537`, plus a 0.16 GiB mmproj that is not used.
- **Args:** those of [Gemma 4 26B-A4B](#5-gemma-4-26b-a4b-it), same family, sampling and thinking switch, without its expert offload; f16 KV stays for the 512-wide global heads.

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 20g --memory-swap 20g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf google/gemma-4-12B-it-qat-q4_0-gguf \
  --hf-file gemma-4-12b-it-qat-q4_0.gguf \
  --alias local --no-mmproj \
  -ngl all -fa on -c 131072 -np 1 --jinja \
  -ctxcp 16 \
  --reasoning off \
  --temp 1.0 --top-p 0.95 --top-k 64
```

- **Measured on this laptop (2026-11-17, [record](../research/2026-11-17/llamacpp-gemma-4-12b/)):** `llmctl up` reached ready in 3.5 s and the GPU peaked at 9,772 MiB including about 112 MiB for the desktop; container RAM peaked at 12.94 GiB, of which 6.68 GiB was page cache of the mmapped GGUF and 6.25 GiB anonymous memory, mostly llama-server's host prompt cache (`--cache-ram`, 8 GiB by default) holding the long prompts.
- **Measured speed (2026-11-17):** decode 77.1 tok/s at a short context (76.5 to 77.3 over 3 runs), and 1,628 tokens of code at 72.9 tok/s; prompt processing 2,986 tok/s for 9.9k tokens (3.4 s to the first token), 2,355 tok/s for 27k new tokens (11.6 s) and 1,312 tok/s for 87k new tokens (66.4 s), where decode was 58.6 tok/s; repeating a prompt hit the cache in 0.09 to 0.19 s.
  The GPU drew up to 146.1 W against its 140 W limit and reached 71 C.
- **Against the other engines:** vLLM v0.33.0 decodes the QAT W4A16 build at 69.5 tok/s ([record](../research/2026-11-17/vllm-gemma-4-12b/README.md)) and SGLang v0.5.23 at 70.4 tok/s ([record](../research/2026-11-17/sglang-gemma-4-12b/README.md)) (measured on the MSI Raider 18 Max HX, 2026-11-17).
- **Measured behaviour (2026-11-17):** `llmctl test` passed, the tool test returned `get_weather` with `{"city":"Paris"}`, streamed and not, and the second turn answered from the tool result, Hermes' title request with a strict `json_schema` returned a title that parses, and a forced `tool_choice` request returned the tool call.
- **Thinking:** toggle; `--reasoning off` makes the server default off, and `chat_template_kwargs.enable_thinking` true turns it on per request (233 characters of reasoning, measured); no `reasoning_effort` value turns it on.
- **Sampling:** temperature 1.0, top_p 0.95 and top_k 64, as for Gemma 4 26B-A4B.

OpenCode entry (the validated config carries it under `llamacpp-docker` beside the vLLM and SGLang entries of the same model):

```json
"gemma-4-12b": {
  "modelID": "local",
  "name": "Gemma 4 12B QAT Q4_0 (llama.cpp)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 131072, "output": 16384 },
  "variants": [
    { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
    { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
  ]
}
```

### Gemma 4 E4B IT QAT Q4_0

Google's QAT Q4_0 GGUF of the smoke-test model that [vLLM](../vllm/models.md#smoke-test-model-gemma-4-e4b-it) and [SGLang](../sglang/models.md#gemma-4-e4b-it-qat-w4a16) serve as QAT W4A16 with 64k context; on llama.cpp it holds twice that context at the highest decode speed of the three.
Start it with `./llmctl up gemma4-e4b` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [google/gemma-4-E4B-it-qat-q4_0-gguf](https://huggingface.co/api/models/google/gemma-4-E4B-it-qat-q4_0-gguf/tree/main), `gemma-4-E4B_q4_0-it.gguf`, 5,154,941,280 B (4.80 GiB), at revision `4b4a2c1d584be7264f87aac328a1bc739ce81b6c`, plus a 0.92 GiB mmproj that is not used, so the vision input is not loaded.
- **Args:** those of [Gemma 4 26B-A4B](#5-gemma-4-26b-a4b-it) without its expert offload, as for Gemma 4 12B; 131072 is the model's own context length.

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 20g --memory-swap 20g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf google/gemma-4-E4B-it-qat-q4_0-gguf \
  --hf-file gemma-4-E4B_q4_0-it.gguf \
  --alias local --no-mmproj \
  -ngl all -fa on -c 131072 -np 1 --jinja \
  -ctxcp 16 \
  --reasoning off \
  --temp 1.0 --top-p 0.95 --top-k 64
```

- **Measured on this laptop (2026-11-17, [record](../research/2026-11-17/llamacpp-gemma-4-e4b/)):** `llmctl up` reached ready in 2.4 s, b12407 loaded E4B's per-layer embeddings without a warning, and the GPU peaked at 5,422 MiB including about 112 MiB for the desktop; container RAM peaked at 7.86 GiB, of which 5.00 GiB was page cache of the mmapped GGUF and 2.84 GiB anonymous memory.
- **Measured speed (2026-11-17):** decode 151.3 tok/s at a short context (150.5 to 151.6 over 3 runs), and 1,673 tokens of code at 148.6 tok/s; prompt processing 11,772 tok/s for 9.9k tokens (0.87 s to the first token), 8,834 tok/s for 27k new tokens (3.1 s) and 4,949 tok/s for 87k new tokens (17.7 s), where decode was 84.5 tok/s; repeating a prompt hit the cache in 0.02 to 0.13 s.
  The GPU drew up to 149.4 W against its 140 W limit and reached 70 C.
- **Against the other engines:** vLLM v0.33.0 runs the QAT W4A16 build only in eager mode at 64k, at 45.2 tok/s ([record](../research/2026-11-17/vllm-gemma-4-e4b/README.md)), and SGLang v0.5.23 at 64k at 141.3 tok/s ([record](../research/2026-11-17/sglang-gemma-4-e4b/README.md)) (measured on the MSI Raider 18 Max HX, 2026-11-17).
- **Measured behaviour (2026-11-17):** `llmctl test` passed, the tool test returned `get_weather` with `{"city":"Paris"}` and the second turn answered from the tool result, Hermes' title request with a strict `json_schema` returned a title that parses, and a forced `tool_choice` request returned the tool call.
- **Thinking:** toggle; `--reasoning off` makes the server default off, and `chat_template_kwargs.enable_thinking` true turns it on per request (259 characters of reasoning, measured); no `reasoning_effort` value turns it on.
- **Sampling:** temperature 1.0, top_p 0.95 and top_k 64, as for Gemma 4 26B-A4B.

OpenCode entry (the validated config carries it under `llamacpp-docker` beside the vLLM and SGLang entries of the same model):

```json
"gemma-4-e4b": {
  "modelID": "local",
  "name": "Gemma 4 E4B QAT Q4_0 (llama.cpp)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 131072, "output": 16384 },
  "variants": [
    { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
    { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
  ]
}
```

### Mellum2.1 12B-A2.5B Thinking Q6_K

JetBrains' GGUF of the thinking MoE that [vLLM](../vllm/models.md#mellum21-12b-a25b-thinking-awq-int4) serves as AWQ-INT4; it does not load on SGLang v0.5.23 ([SGLang](../sglang/models.md#models-that-do-not-run-on-sglang)), so llama.cpp is its second engine.
Start it with `./llmctl up mellum` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF](https://huggingface.co/api/models/JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF/tree/main), `Mellum2.1-12B-A2.5B-Thinking-Q6_K.gguf`, 10,879,834,400 B (10.13 GiB), at revision `4159d684ffdb09ba89dc2a58feb3e75c54eef036`.
  The repository also holds other quantizations and MTP files, so the launch names the file with `--hf-file`.
- **Args:** those of [Groot-19B-A3B](#8-groot-19b-a3b-261023), a small-active MoE held whole on the GPU with the same temperature, top_p and top_k as the vLLM Mellum2.1 recipe, without `--reasoning off`, because that recipe keeps thinking on by default.

```bash
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8080:8080 \
  --memory 16g --memory-swap 16g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF \
  --hf-file Mellum2.1-12B-A2.5B-Thinking-Q6_K.gguf \
  --alias local \
  -ngl all -fa on -c 131072 -np 1 --jinja \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0
```

- **Measured on this laptop (2026-11-17, [record](../research/2026-11-17/llamacpp-mellum2.1-12b/)):** `llmctl up` reached ready in 4.5 s and the GPU peaked at 12,674 MiB including about 112 MiB for the desktop; container RAM peaked at 13.16 GiB under the 16 GiB cap, of which 10.32 GiB was page cache of the mmapped GGUF and 2.83 GiB anonymous memory.
- **Measured speed (2026-11-17):** decode 188.2 tok/s at a short context with thinking on, reasoning included (187.6 to 189.6 over 3 runs); prompt processing 6,594 tok/s for 9.9k tokens (1.6 s to the first token), 5,422 tok/s for 27k new tokens (5.1 s) and 3,204 tok/s for 87k new tokens (27.2 s), where decode was 121.1 tok/s; repeating a prompt hit the cache in 0.04 to 0.12 s.
  A code request spent its whole 3,000-token budget on reasoning at 180.4 tok/s, so give thinking-on coding requests a large output budget.
  The GPU drew up to 145.1 W against its 140 W limit and reached 70 C.
- **Against the other engines:** vLLM v0.33.0 decodes the AWQ-INT4 build at 257.2 tok/s with thinking on ([record](../research/2026-11-17/vllm-mellum2.1-12b/README.md)), and SGLang v0.5.23 does not load it ([record](../research/2026-11-17/sglang-mellum2.1-12b/README.md)) (measured on the MSI Raider 18 Max HX, 2026-11-17).
- **Measured behaviour (2026-11-17):** b12407 parses Mellum's tool calls and reasoning: `llmctl test` passed, the tool test returned `get_weather` with `{"city": "Paris"}` and the second turn answered from the tool result, Hermes' title request with `reasoning_effort` `none` and a strict `json_schema` returned a title that parses with no reasoning, and a forced `tool_choice` request with a weather question returned the tool call after its reasoning.
  A forced `tool_choice` on "Reply with exactly: OK" with thinking off and a 256-token cap ended at the cap with its thinking as text in `content` and no tool call.
- **Thinking:** toggle, on by default; `chat_template_kwargs.enable_thinking` false or `reasoning_effort` `none` turns it off per request, while `low` and `high` keep it on (measured).
- **Sampling:** temperature 0.6, top_p 0.95, top_k 20, min_p 0 and presence_penalty 0.

OpenCode entry (the validated config carries it under `llamacpp-docker` beside the vLLM entry of the same model):

```json
"mellum2.1-12b": {
  "modelID": "local",
  "name": "Mellum2.1 12B-A2.5B Thinking Q6_K (llama.cpp)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 131072, "output": 16384 },
  "variants": [
    { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
    { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
  ]
}
```

## Fallbacks and trials

These are honorable mentions in [../docs/model-selection.md](../docs/model-selection.md), not top-10 picks.
The commands use the same docker flags as above; only the model lines change.

### Nemotron 3.5 Lightning 30B-A3B (fallback if Bolt does not fit)

Start it with `./llmctl up lightning` ([llmctl](../docs/llmctl.md)), which runs the launch below from the pinned file in the cache with the checks and rules applied.

- **GGUF:** [ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/api/models/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/tree/main), `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`, 18,898,091,584 B (17.60 GiB).
  The repo also holds three MTP files and BF16 and Q8_0 main files, so name the main file with `--hf-file`; its Q4_K_M file was deleted on 2026-08-14, and the repo was re-uploaded on 2026-11-12 ([plan record](../research/2026-11-16/next-model-plans/plan-nemotron-3.5-lightning.json)).
  The GGUF header, read on 2026-11-16, shows 52 hybrid blocks: 23 MoE (128 experts, 6 used, plus a shared expert that stays on the GPU), 23 Mamba-2 and 6 full-attention layers with 2 KV heads of 128; the downloaded file's SHA-256 matched its LFS id.
- **Model lines:** `-hf ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF --hf-file NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf --alias local -ngl all --n-cpu-moe 25 -fa on -c 131072 -np 1 --jinja -lm none -t 8 --moe-cache-mib 3200 --reasoning off --temp 1.0 --top-p 0.95`, with `--memory 18g --memory-swap 18g`.
- **Memory:** KV is only 6 layers x 2 KV heads x (128 + 128) x 2 B = 6 KiB/token, 768 MiB at 128k in f16, and the Mamba-2 state is about 48 MiB.
  `--n-cpu-moe 25` keeps the experts of the 11 MoE blocks among blocks 0-24 (7.4 GiB) in pinned RAM, because N counts all hybrid blocks.
  For 256k keep f16 KV: it adds only 768 MiB, which fits by shrinking the expert cache (estimate).
- **Measured on this laptop (2026-11-16, [record](../research/2026-11-16/llamacpp-nemotron-3.5-lightning/)):** the server started from the cached file in about 5 s and peaked at 15,106 MiB including about 112 MiB for the desktop, about 735 MiB below the 15,841 MiB that CUDA can use; container RAM peaked at 10.3 GiB.
  The run added `--user`, the cache at `/hf`, `-m` instead of `-hf` and `-lv 4` ([server arguments](../research/2026-11-16/llamacpp-nemotron-3.5-lightning/server-args.json)).
- **Measured speed (2026-11-16):** decode 143.6 tok/s at a short context (143.4 to 145.0 over 3 runs), 131.0 after an 8.3k prompt, 137.6 after 31k and 128.2 after 104k, and 141.0 tok/s over 1,254 tokens of code; prompt processing 2,912 tok/s for 8.3k tokens (2.9 s to the first token), 3,025 tok/s for 23k new tokens (7.8 s) and 2,463 tok/s for 73k new tokens (29.9 s); repeating a prompt hit the cache in 0.05 to 0.14 s.
  It keeps its speed at depth better than any other model measured here, because only 6 of its 52 layers carry attention.
  The GPU drew up to 146 W and reached 72 C.
- **Measured behaviour (2026-11-16):** the OK test answered with no reasoning at the server default, `enable_thinking: true` on one request produced 460 characters of reasoning, the tool test returned `get_weather` with `{"city":"Paris"}`, and an OpenCode v2.3.0 write-and-read session passed in 6 s with a first request of 4,370 prompt tokens.
- **Tuning (measured 2026-11-16):** the first launch, `--n-cpu-moe 21` with a 1,792 MiB cache, decoded at 136.5 tok/s with a 64% decode hit rate ([record](../research/2026-11-16/llamacpp-nemotron-3.5-lightning/first-launch/)).
  `--n-cpu-moe 25` with 3,200 MiB raised the hit rate to 78% and decode by 5%, at about 2% less prefill.
- **Thinking:** toggle, on by default in the template; `--reasoning off` makes the server default off, and `chat_template_kwargs.enable_thinking` switches it per request.
  The template has no effort or budget keys.
- **Sampling:** temperature 1.0 and top_p 0.95 in both modes, per NVIDIA's card ([Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/README.md), line 59); the 0.6 and 0.2 values that Unsloth relays do not match it.
- **Tools:** Qwen3-Coder handler.
  The `force_nonempty_content` template argument seen in some guides does nothing in llama.cpp; leave it out.
- **Evidence:** NVIDIA same-harness TB2.1 24.58 ([Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/README.md)), and Artificial Analysis TB2.1 24.3 (see [../docs/model-selection.md](../docs/model-selection.md)).

### GLM-4.7-Flash

- **GGUF:** [unsloth/GLM-4.7-Flash-GGUF](https://huggingface.co/api/models/unsloth/GLM-4.7-Flash-GGUF/tree/main), `GLM-4.7-Flash-Q4_K_M.gguf` (17.05 GiB), or Unsloth's recommended `GLM-4.7-Flash-UD-Q4_K_XL.gguf` (16.32 GiB), which needs its `--n-cpu-moe` value re-derived.
- **Model lines:** `-hf unsloth/GLM-4.7-Flash-GGUF:Q4_K_M --alias local -ngl all --n-cpu-moe 20 -fa on -c 65536 -np 1 --jinja --temp 0.7 --top-p 1.0 --min-p 0.01`, with `--memory 18g --memory-swap 18g`.
- **Offload (estimate):** block 0 is dense, so 17-18 is too tight; 20 moves 19 MoE layers (about 6.5 GiB) to RAM.
  KV uses MLA at 54,144 B/token in f16, 3.30 GiB at 64k.
- **Sampling:** tool calling temperature 0.7, top_p 1.0; general temperature 1.0, top_p 0.95; min_p 0.01 ([Unsloth GLM-4.7-Flash page](https://unsloth.ai/docs/models/tutorials/glm-4.7-flash)).
  Files from before the 2026-01-21 re-upload loop; download fresh.
- **Evidence:** Artificial Analysis tau2 98.8 and Terminal-Bench Hard 22.0, with no TB2.1 score ([AA](https://artificialanalysis.ai/models/glm-4-7-flash)); released 2026-01 ([model card](https://huggingface.co/zai-org/GLM-4.7-Flash/raw/main/README.md)).

### Nanbeige4.3-33B (trial only, blocked by a tokenizer bug)

- **Status:** at b12407 the SPM tokenizer produces wrong role-token ids and extra spaces in Nanbeige's chat prompts; the fix, [PR #31986](https://github.com/ggml-org/llama.cpp/pull/31986), was open and unmerged on 2026-11-15.
  Wait for an image that contains it.
- **GGUF when it ships:** [bartowski/Nanbeige_Nanbeige4.3-33B-GGUF](https://huggingface.co/api/models/bartowski/Nanbeige_Nanbeige4.3-33B-GGUF?blobs=true) IQ2_M (11.35 GiB), which carries the `tokenizer.ggml.metaspace.prepend_scheme` key the fix needs.
  On 16 GB the dense 33B only fits at 2-bit with q4_0 KV at 64k (about 15.5-15.7 of 15.92 GiB including the CUDA context, estimate), with unmeasured quality loss.
- **Evidence:** AA Intelligence Index 20.4 ([AA](https://artificialanalysis.ai/models/nanbeige4-3-33b)); it reasons for about 158.5k output tokens per AA task.

## Rejected GGUF options

| Option | Size | Why not |
|---|---|---|
| [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/api/models/ggml-org/Qwen3.8-27B-GGUF/tree/main) Q4_K_M | 19.4 GB | Does not fit on the GPU, and its template triggers [#29947](https://github.com/ggml-org/llama.cpp/issues/29947) with an OpenCode plugin |
| Ollama `qwen3.8:27b` | 18 GB | Dense layers spill to the CPU ([tags](https://ollama.com/library/qwen3.8/tags)) |
| unsloth Qwen3.8-27B UD-IQ4_XS | 13.37 GiB | Kept as the quality variant only: about 48-50k with q8_0 KV, below the 64k target (estimate) |
| [unsloth Qwen3.6-35B-A3B](https://huggingface.co/api/models/unsloth/Qwen3.6-35B-A3B-GGUF?blobs=true) UD-Q2_K_XL | 11.45 GiB | Fits fully on the GPU, but it sits at Unsloth's floor for agentic use ([Dynamic 3.0](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)); keep experts in RAM at Q4 instead |
| ggml-org Qwen3.6-35B-A3B-MTP GGUF | 37.8 GB | Q8_0 only, and MTP drafting has open bugs |
| Ollama `gemma4:26b` | 17.07 GB plus projector | Q4_K_M does not fit; use the `-it-qat` tag ([tags](https://ollama.com/library/gemma4/tags)) |
| Qwen3-Coder-Next official Q4_K_M or unsloth UD-Q4_K_XL | 45.09 / 46.20 GiB | 35-36 GiB of experts in RAM against about 40 GiB free; one desktop window evicts experts |
| Ollama `qwen3-coder-next` | 52 GB | Above VRAM plus free RAM with whole-layer offload ([tags](https://ollama.com/library/qwen3-coder-next/tags)) |
| IFM K2 Horizon official GGUFs (7B and MoVA) | 8.92 / 20.83 GiB | Uploaded 2026-09-22, before mainline support and before the 7B FTPO promotion; untested (inferred pre-FTPO for the 7B) |
| [unsloth Nemotron 3.5 Bolt](https://huggingface.co/api/models/unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF/tree/main?recursive=true) UD-IQ1_M / UD-IQ1_S | 18.91 / 17.24 GiB | 1-bit quants break tool calling per Unsloth ([Dynamic 3.0](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)) |
| Community REAP-pruned Bolt GGUF (anik-jha, 16GB-REAP50) | 12.46 GiB | Half the experts pruned by a third party; quality self-reported, long context not benchmarked |
| bartowski Groot Q4_K_M | 11.63 GiB | Not rejected outright: 0.77 GiB larger than the official file; use it only to compare the other chat template |
| gpt-oss-20b K-quants | 11.5-12.1 GB | No smaller than MXFP4; the guide says to use MXFP4 ([guide](https://github.com/ggml-org/llama.cpp/discussions/15396)) |
| gpt-oss-120b MXFP4 | 59.03 GiB | Needs about 49-52 GiB of RAM for experts against about 40 GiB free ([HF API](https://huggingface.co/api/models/ggml-org/gpt-oss-120b-GGUF?blobs=true)) |
| Nanbeige4.3-33B (any GGUF at b12407) | 10.87-19.36 GiB | Tokenizer bug until [PR #31986](https://github.com/ggml-org/llama.cpp/pull/31986) ships; mradermacher files also lack the `prepend_scheme` key the fix needs |
| [Xing4.0-29B-A4B](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B/raw/main/README.md) official GGUFs | 10.99-31.82 GiB | Built before the 2026-11-05 mainline merge ([PR #31589](https://github.com/ggml-org/llama.cpp/pull/31589)), no dedicated parser at b12407, and a public report of failed write and edit calls |
| [LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) QAD-Q4_0 | 12.65 GiB | Vendor-only quality and a revenue cap in the licence; [#30490](https://github.com/ggml-org/llama.cpp/issues/30490) (tool calls lost as content) was closed on 2026-10-26, but its fix in b12407 is not verified |
| [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/raw/main/README.md) PQ2_0 | not measured | The stock image runs only Q2_0, and the only independent test regresses against Qwen3.8-27B ([InsiderLLM](https://insiderllm.com/guides/bonsai-2-27b-vs-qwen3-8-27b-rtx-3090/)) |
| Devstral Small 2 24B (any 4-bit GGUF) | dense 24B | About 32k with q8_0 KV; 64k needs q4_0 KV or 3-bit weights |

Model-level exclusions and their evidence are in [../docs/model-selection.md](../docs/model-selection.md).

## First run checklist per model

1. Start the container with the recipe and follow `docker logs -f llamacpp` until the server listens; the first start downloads the file.
2. Run the smoke tests in [README.md](README.md#smoke-tests); for always-thinking models (ranks 3, 4, 9, 10) expect `reasoning_content` in test 3, and raise `max_tokens`.
3. Read device memory from the load log and `nvidia-smi`, and RAM from `docker stats`; replace the estimates in the table at the top.
4. For offloaded models (ranks 2, 3, 6, 7), run the sweep in [benchmarking.md](benchmarking.md#moe-offload-sweep---n-cpu-moe) and keep one or two blocks of margin.
5. Run one OpenCode session with a multi-line edit, a file write and a few parallel tool calls before relying on the model.
6. Record results in the template in [benchmarking.md](benchmarking.md#results-template).

## Sources

- Rankings and evidence: [../docs/model-selection.md](../docs/model-selection.md), [../docs/community-benchmarks.md](../docs/community-benchmarks.md), [../docs/sources.md](../docs/sources.md).
- llama.cpp at b12407 and b12418: [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/b12407/common/arg.cpp), [common/common.h](https://github.com/ggml-org/llama.cpp/blob/b12407/common/common.h), [common/chat.cpp](https://github.com/ggml-org/llama.cpp/blob/b12418/common/chat.cpp), [common/parsers/k2-horizon.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/common/parsers/k2-horizon.cpp), [common/parsers/gpt-oss.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12418/common/parsers/gpt-oss.cpp), [src/llama-arch.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/src/llama-arch.cpp), [tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md), [docs/function-calling.md](https://github.com/ggml-org/llama.cpp/blob/b12418/docs/function-calling.md).
- llama.cpp issues and PRs: [#27330](https://github.com/ggml-org/llama.cpp/issues/27330), [#27623](https://github.com/ggml-org/llama.cpp/issues/27623), [#28581](https://github.com/ggml-org/llama.cpp/issues/28581), [#29655](https://github.com/ggml-org/llama.cpp/issues/29655), [#29947](https://github.com/ggml-org/llama.cpp/issues/29947), [#30118](https://github.com/ggml-org/llama.cpp/issues/30118), [#30246](https://github.com/ggml-org/llama.cpp/issues/30246), [#30357](https://github.com/ggml-org/llama.cpp/issues/30357), [#30490](https://github.com/ggml-org/llama.cpp/issues/30490), [#30781](https://github.com/ggml-org/llama.cpp/issues/30781), [#25859](https://github.com/ggml-org/llama.cpp/issues/25859), [#31974](https://github.com/ggml-org/llama.cpp/issues/31974), [PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802), [PR #31986](https://github.com/ggml-org/llama.cpp/pull/31986), [PR #29535](https://github.com/ggml-org/llama.cpp/pull/29535), [PR #30520](https://github.com/ggml-org/llama.cpp/pull/30520).
- Model cards: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/README.md), [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/README.md), [K2 Horizon MoVA](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/raw/main/README.md), [K2 Horizon 7B](https://huggingface.co/IFM/K2-Horizon-7B/raw/main/README.md), [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/README.md), [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next/raw/main/README.md), [Nemotron 3.5 Bolt GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/raw/main/README.md), [Groot](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/README.md), [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/raw/main/README.md), [DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/raw/main/README.md), [Nemotron 3.5 Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/README.md).
- GGUF sizes: the Hugging Face tree and blob APIs linked in each recipe, read on 2026-11-15.
- Independent scores: [Artificial Analysis](https://artificialanalysis.ai/) model pages linked in each recipe.
- Vendor docs for their own quants: [Unsloth docs](https://unsloth.ai/docs) (Qwen3.8, Qwen3.6, Gemma 4 QAT, Qwen3-Coder-Next, gpt-oss, GLM-4.7-Flash, Dynamic 3.0).
- Ollama: [library tags](https://ollama.com/library) and registry manifests linked in each recipe.
- Speed reference: [ggml-org gpt-oss guide](https://github.com/ggml-org/llama.cpp/discussions/15396).

## Maintenance

Before relying on a recipe, re-check:

- The GGUF file size and last commit in the repo's tree API; Unsloth and others replace files under the same name (Qwen3.8 UD-Q3_K_XL changed on 2026-11-12).
- The image tag in [README.md](README.md#image); move to a newer `server-cuda-bNNNNN` only after re-running the smoke tests, and re-check the flags used here against its `--help`.
- Open issues per model: [#27330](https://github.com/ggml-org/llama.cpp/issues/27330), [#27623](https://github.com/ggml-org/llama.cpp/issues/27623) with [PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802), [#30118](https://github.com/ggml-org/llama.cpp/issues/30118), [#30357](https://github.com/ggml-org/llama.cpp/issues/30357), [#29655](https://github.com/ggml-org/llama.cpp/issues/29655), [#29947](https://github.com/ggml-org/llama.cpp/issues/29947), [#30781](https://github.com/ggml-org/llama.cpp/issues/30781), and [PR #31986](https://github.com/ggml-org/llama.cpp/pull/31986) for Nanbeige.
- The Gemma 4 A/B result, and whether Google or Unsloth has re-uploaded either QAT file.
- After the first run, replace every "(estimate)" and "(inferred)" in the table at the top with measured values from [benchmarking.md](benchmarking.md), including the configured DDR5 speed from `sudo dmidecode -t memory`, which every offload speed estimate depends on.
