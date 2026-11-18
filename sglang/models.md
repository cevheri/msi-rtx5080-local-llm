# SGLang model recipes for the RTX 5080 Laptop

Last verified: 2026-11-18 (Qwen3.5-9B, gpt-oss-20b, Gemma 4 12B and Gemma 4 E4B measured on 2026-11-17, K2 Horizon 7B at fraction 0.88 on 2026-11-18).

SGLang v0.5.23 (`lmsysorg/sglang:v0.5.23-runtime`) serves five of the active models as llmctl recipes on this 16 GB GPU: Qwen3.5-9B NVFP4, gpt-oss-20b and Gemma 4 12B QAT W4A16 at 128k context and K2 Horizon 7B AWQ-INT4 and Gemma 4 E4B QAT W4A16 at 64k.
gpt-oss-20b fits only through [Paged Experts](README.md#paged-experts), which keeps 24 of its 32 experts per layer on the GPU and pages the rest from pinned host memory.
Every SGLang recipe shares its name with a llama.cpp or vLLM recipe, which `./llmctl up` prefers, so each one starts only with `--runtime sglang` ([llmctl](../docs/llmctl.md)).
Only Gemma 4 E4B is a material speed gain over the fastest other recipe of the same model, 141.3 tok/s against 45.2 on vLLM in eager mode; Qwen3.5-9B decodes 93.7 tok/s against 90.8 on vLLM, K2 Horizon 7B 89.6 against 94.8 on vLLM, gpt-oss-20b 124.6 against 166.2 on llama.cpp, and Gemma 4 12B 70.4 against 69.5 on vLLM (measured on the MSI Raider 18 Max HX, each with its runtime, 2026-11-16 to 2026-11-18; records linked in each section and in [vllm/models.md](../vllm/models.md) and [llamacpp/models.md](../llamacpp/models.md)).
The two Gemma 4 recipes serve text only: they skip the audio tower that SGLang v0.5.23 cannot build from these checkpoints, and Gemma 4 E4B's vision tower stays uninitialized.

llmctl adds two things to the launches below.
It mounts a model overlay at `/llmctl/model` instead of the snapshot: links into the pinned snapshot plus a `generation_config.json` that carries the recipe's sampling, because SGLang v0.5.23 reads its default sampling only from that file ([Sampling and the model overlay](README.md#sampling-and-the-model-overlay)).
After the server is ready it posts `/flush_cache`, reads the KV pool from `/server_info` and the free GPU memory from `nvidia-smi`, and stops a start whose pool is shorter than the context or that leaves less than 768 MiB free ([Readiness and the memory check](README.md#readiness-and-the-memory-check)), the room that long prompts and grammar requests need ([Grammar requests and NCCL](README.md#grammar-requests-and-nccl), [Post-ready margins](benchmarking.md#post-ready-margins)).
Each recipe's `[evidence] record` names the folder with its measurements; the methods are the ones of [research/2026-11-17/sglang-runtime/](../research/2026-11-17/sglang-runtime/README.md).

## Which models run

| Model | Recipe | Checkpoint | Context and KV | Pool (`token_capacity`) | Decode | Record |
|---|---|---|---|---|---|---|
| [Qwen3.5-9B NVFP4](#qwen35-9b-nvfp4) | `qwen3.5-9b` | `RedHatAI/Qwen3.5-9B-NVFP4` | 131072, BF16 | 140,981 tokens | 93.7 tok/s | [2026-11-17](../research/2026-11-17/sglang-qwen3.5-9b/README.md) |
| [K2 Horizon 7B AWQ-INT4](#k2-horizon-7b-awq-int4) | `k2-horizon-7b` | `cyankiwi/K2-Horizon-7B-AWQ-INT4` | 65536, FP8 | 77,988 tokens | 89.6 tok/s | [2026-11-18](../research/2026-11-18/sglang-k2-horizon-7b/README.md) |
| [gpt-oss-20b](#gpt-oss-20b) | `gpt-oss-20b` | `openai/gpt-oss-20b` (MXFP4) | 131072, FP8 | 169,853 tokens | 124.6 tok/s | [2026-11-17](../research/2026-11-17/sglang-gpt-oss-20b/README.md) |
| [Gemma 4 12B IT QAT W4A16](#gemma-4-12b-it-qat-w4a16) | `gemma-4-12b` | `google/gemma-4-12B-it-qat-w4a16-ct` | 131072, FP8 | 204,181 tokens | 70.4 tok/s | [2026-11-17](../research/2026-11-17/sglang-gemma-4-12b/README.md) |
| [Gemma 4 E4B IT QAT W4A16](#gemma-4-e4b-it-qat-w4a16) | `gemma-4-e4b` | `google/gemma-4-E4B-it-qat-w4a16-ct` | 65536, BF16 | 101,791 tokens | 141.3 tok/s | [2026-11-17](../research/2026-11-17/sglang-gemma-4-e4b/README.md) |

Every figure in this table is measured on the MSI Raider 18 Max HX, SGLang v0.5.23, at the shipped settings, on the date its record names.
The models that do not run, or not yet as a recipe, are in [Models that do not run on SGLang](#models-that-do-not-run-on-sglang).

## Qwen3.5-9B NVFP4

Start it with `./llmctl up qwen3.5 --runtime sglang` ([llmctl](../docs/llmctl.md)); its recipe takes the server flags from the block below ([record](../research/2026-11-17/sglang-qwen3.5-9b/README.md)).

| Item | Value |
|---|---|
| Checkpoint | `RedHatAI/Qwen3.5-9B-NVFP4` at `a4df85a20fb7e54740f564f15af0c2f5d9c6616c`, the vLLM recipe's snapshot, so nothing downloads |
| Weights loaded | 8.39 GB, because SGLang v0.5.23 always loads the vision tower: `--language-model-only` was refused for this architecture (measured) |
| Memory | `--mem-fraction-static 0.88`, 8 Mamba state slots instead of the automatic 41, no prefill CUDA graph |
| Parsers | `qwen3_coder` and `qwen3`, thinking off by default through `--default-chat-template-kwargs` |
| llmctl needs | `gpu_mib` 15193, `ram_gib` 6, `memory_gib` 32 |

Measured at the shipped settings:

- Startup: the ready line came 28.7 s after `docker run`, the BF16 pool held 140,981 tokens, and 911 MiB were free right after ready (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [measure-b](../research/2026-11-17/sglang-qwen3.5-9b/measure-b/README.md)).
- Margins: a fresh 124,499-token prompt left 545 MiB, the first grammar request and a forced tool call then moved nothing, and the lowest reading over the whole protocol was 537 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- With all eight runtime variables, 937 MiB were free after the flush that llmctl's check makes, 41 MiB above the 896 MiB that the threshold's rule asks of a shipped recipe (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18; [measure-d](../research/2026-11-18/sglang-qwen3.5-9b/README.md)).
- Speed: 93.7 tok/s decode with thinking off, the median of three runs of 664 to 854 tokens; 0.572 s to the first token for 8,591 tokens and 2.398 s for 32,153 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Load: 66 requests in 300 s with prompts up to 13,448 tokens and no error, at a 15,305 MiB GPU peak and 5.73 GiB of non-reclaimable host memory (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Checks: plain and streamed replies, a `get_weather` tool call and its second turn, a `json_schema` reply and Hermes' title request passed; thinking stayed off by default and turned on with `enable_thinking: true` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- A cold start from an empty cache volume without a network reached `/ready` in 48.3 s with no download line (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [M15](../research/2026-11-17/sglang-runtime/measure-b/README.md#m15-cold-starts-without-a-network)).
- Any `reasoning_effort` except `none` turns its thinking on, although the server default is off (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17), because SGLang maps every effort but `none` to `enable_thinking` (SG-61 in [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md)).

```bash
docker run -d --name sglang \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -e FLASHINFER_NO_DOWNLOAD=1 \
  -e SYNC_TOKEN_IDS_ACROSS_TP=1 \
  -e NCCL_MAX_P2P_NCHANNELS=1 \
  -e TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data \
  -e PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v sglang-cache:/root/.cache/sglang \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
  --model-path /root/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-9B-NVFP4/snapshots/a4df85a20fb7e54740f564f15af0c2f5d9c6616c \
  --served-model-name qwen3.5-9b \
  --context-length 131072 \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.88 \
  --max-running-requests 1 \
  --max-mamba-cache-size 8 \
  --disable-prefill-cuda-graph \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

Sampling: llmctl pins `{"temperature": 0.6, "top_p": 0.95, "top_k": 20}` in the overlay's `generation_config.json`, the vLLM recipe's values; the checkpoint's own file has no sampling keys, so a request without sampling fields gets temperature 0.6, top_p 0.95 and top_k 20 from the overlay and min_p 0.0 and repetition_penalty 1.0 from SGLang's defaults (the start logged `{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}`, measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; the defaults are SG-59 and a request's own fields win by SG-58 in [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md)).

OpenCode entry:

```json
"qwen3.5-9b": {
  "modelID": "qwen3.5-9b",
  "name": "Qwen3.5-9B NVFP4 (SGLang)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 131072, "output": 16384 },
  "variants": [
    { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
    { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
  ]
}
```

## K2 Horizon 7B AWQ-INT4

Start it with `./llmctl up k2-7b --runtime sglang` ([llmctl](../docs/llmctl.md)); its recipe takes the server flags from the block below ([record](../research/2026-11-18/sglang-k2-horizon-7b/README.md)).

| Item | Value |
|---|---|
| Checkpoint | `cyankiwi/K2-Horizon-7B-AWQ-INT4` at `530d5ceccd8ea660ceaa2a9f3a4bcbb68ba44291`, the vLLM recipe's snapshot |
| Memory | `--mem-fraction-static 0.88` with FP8 KV, no prefill CUDA graph; at 0.90 the start left 865 MiB free after ready, below the 896 MiB a shipped recipe needs |
| Parsers | `k2_horizon` for tool calls and reasoning, with `reasoning_effort` high and XML tool calls through `--default-chat-template-kwargs`; `--trust-remote-code` |
| llmctl needs | `gpu_mib` 14908, `ram_gib` 5, `memory_gib` 32 |

Measured at the shipped settings:

- Startup: the ready line came 36.0 s after `docker run`, the FP8 pool held 77,988 tokens, 12,452 above the context, and 1,152 MiB were free right after ready and after the flush (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18; [measure-d](../research/2026-11-18/sglang-k2-horizon-7b/measure-d/README.md)).
- Margins: a fresh 62,254-token prompt, the first grammar request and a forced tool call left 828 MiB, and the lowest reading over the whole protocol was 822 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Speed: 89.6 tok/s decode, the median of runs of 1,286 to 1,500 tokens; 2.196 s to the first token for 8,085 tokens and 9.233 s for 30,261 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Load: 51 requests in 300 s with no error, at a 15,020 MiB GPU peak and 4.92 GiB of anonymous and shared host memory (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Checks: plain and streamed replies, a `get_weather` tool call and its second turn, a `json_schema` reply and Hermes' title request passed, and the model reasoned for every effort and switch, as its `always` type says (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- FP8 against BF16 KV at fraction 0.90: 20 of 20 correct answers each on 20 fixed prompts at temperature 0, with identical final answers, while BF16 KV held only 41,198 tokens, short of 64k (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [M13](../research/2026-11-17/sglang-runtime/measure-b/README.md#m13-fp8-against-bf16-kv)).
- A cold start at fraction 0.90 from an empty cache volume without a network reached `/ready` in 137.9 s with no download line (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [M15](../research/2026-11-17/sglang-runtime/measure-b/README.md#m15-cold-starts-without-a-network)).

```bash
docker run -d --name sglang \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -e FLASHINFER_NO_DOWNLOAD=1 \
  -e SYNC_TOKEN_IDS_ACROSS_TP=1 \
  -e NCCL_MAX_P2P_NCHANNELS=1 \
  -e TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data \
  -e PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v sglang-cache:/root/.cache/sglang \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
  --model-path /root/.cache/huggingface/hub/models--cyankiwi--K2-Horizon-7B-AWQ-INT4/snapshots/530d5ceccd8ea660ceaa2a9f3a4bcbb68ba44291 \
  --served-model-name k2-horizon-7b \
  --context-length 65536 \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.88 \
  --max-running-requests 1 \
  --disable-prefill-cuda-graph \
  --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --tool-call-parser k2_horizon \
  --reasoning-parser k2_horizon \
  --default-chat-template-kwargs '{"reasoning_effort":"high","tool_call_format":"xml"}'
```

Sampling: llmctl pins `{"temperature": 1.0, "top_p": 0.95}` in the overlay's `generation_config.json`, the vLLM recipe's values; the checkpoint's own file has no sampling keys, so a request without sampling fields gets temperature 1.0 and top_p 0.95 from the overlay and top_k -1, min_p 0.0 and repetition_penalty 1.0 from SGLang's defaults (the start logged `{'temperature': 1.0, 'top_p': 0.95}`, measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18; the defaults are SG-59 and a request's own fields win by SG-58 in [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md)).

OpenCode entry:

```json
"k2-horizon-7b": {
  "modelID": "k2-horizon-7b",
  "name": "K2 Horizon 7B AWQ-INT4 (SGLang)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 65536, "output": 16384 }
}
```

## gpt-oss-20b

Start it with `./llmctl up gpt-oss --runtime sglang` ([llmctl](../docs/llmctl.md)); its recipe takes the server flags from the block below ([record](../research/2026-11-17/sglang-gpt-oss-20b/README.md)).

| Item | Value |
|---|---|
| Checkpoint | `openai/gpt-oss-20b` at `6cee5e81ee83917806bbde320786a8fb61efebee` without `original/` and `metal/`, the vLLM recipe's snapshot |
| Memory | Paged Experts with 24 of 32 experts per layer on the GPU and the Marlin MoE runner, `--mem-fraction-static 0.90`, FP8 KV, `--swa-full-tokens-ratio 0.1`, 1,024-token prefill chunks, no prefill CUDA graph |
| Parsers | `gpt-oss` for tool calls and reasoning; the effort comes from each request |
| llmctl needs | `gpu_mib` 14724, `ram_gib` 16, `memory_gib` 20 |

Measured at the shipped settings:

- Startup: the ready line came 50.7 s after `docker run`, 11.44 GB of weights loaded, the FP8 pool held 169,853 tokens (16,985 sliding-window), and 1,347 MiB were free right after ready (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [measure-c](../research/2026-11-17/sglang-gpt-oss-20b/measure-c/README.md)).
- Margins: a fresh 124,459-token prompt left 1,006 MiB, and the first grammar request and a forced tool request then moved nothing (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Speed: 124.6 tok/s decode, the median of three 1,500-token runs; 1.472 s to the first token for 8,058 tokens and 4.935 s for 30,003 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Load: 73 requests in 300 s with no error under the 20 GiB memory cap, at a 14,836 MiB GPU peak and a 17.04 GiB cgroup peak with no OOM kill (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Host memory: the pinned expert store holds all 32 experts of every layer as 10.37 GiB of shared memory, and non-reclaimable memory peaked at 15.45 GiB; `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` in the runtime file brought the store down from 19.19 GiB at the same decode and margins (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17), because the image's PyTorch otherwise rounds each pinned tensor of the store up to a power of two (inferred from the image's PyTorch source, `CachingHostAllocator.h` lines 340 to 349).
- Experts on the GPU: K 16, 20, 24, 25 and 28 decoded 87.9, 112.3, 125.5, 130.0 and 137.6 tok/s; K 28 holds only 62,643 tokens, short of 128k; K 25 held 142,701 tokens, 11,629 above the context, at 130.0 tok/s but did not go through the margin protocol, so the recipe keeps K 24, which holds 128k with 37,384 tokens to spare (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17, without the allocator variable; [measure-b](../research/2026-11-17/sglang-gpt-oss-20b/measure-b/README.md)).
- Checks: plain and streamed replies, a `get_weather` tool call and its second turn, a `json_schema` reply and Hermes' title request passed, and every effort answered; a forced `tool_choice` comes back with HTTP 200, `finish_reason` `stop` and no `tool_calls` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17), the defect of issue #41747 (SG-89 in [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md)), which no client sends.
- `TIKTOKEN_ENCODINGS_BASE` in the runtime file loads the Harmony vocabulary from the image: with it a cold start without a network reached `/ready` in 253.6 s with no download line, and without it the server started with `/v1/responses` disabled (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [M15](../research/2026-11-17/sglang-runtime/measure-b/README.md#m15-cold-starts-without-a-network)).

```bash
docker run -d --name sglang \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 20g --memory-swap 20g \
  -p 127.0.0.1:30000:30000 \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -e FLASHINFER_NO_DOWNLOAD=1 \
  -e SYNC_TOKEN_IDS_ACROSS_TP=1 \
  -e NCCL_MAX_P2P_NCHANNELS=1 \
  -e TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data \
  -e PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v sglang-cache:/root/.cache/sglang \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
  --model-path /root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee \
  --served-model-name gpt-oss-20b \
  --context-length 131072 \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.90 \
  --max-running-requests 1 \
  --kv-cache-dtype fp8_e4m3 \
  --swa-full-tokens-ratio 0.1 \
  --chunked-prefill-size 1024 \
  --disable-prefill-cuda-graph \
  --moe-runner-backend marlin \
  --enable-paged-experts \
  --paged-experts-num-resident 24 \
  --tool-call-parser gpt-oss \
  --reasoning-parser gpt-oss
```

Sampling: llmctl pins `{"temperature": 1.0, "top_p": 1.0}` in the overlay's `generation_config.json`, the vLLM recipe's values; the checkpoint's own file sets only `do_sample`, so a request without sampling fields gets temperature 1.0 and top_p 1.0 from the overlay, which equal SGLang's defaults, and top_k -1, min_p 0.0 and repetition_penalty 1.0 from those defaults (the start logged `{'temperature': 1.0, 'top_p': 1.0}`, measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; the defaults are SG-59 and a request's own fields win by SG-58 in [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md)).

OpenCode entry:

```json
"gpt-oss-20b": {
  "modelID": "gpt-oss-20b",
  "name": "gpt-oss-20b MXFP4, Paged Experts (SGLang)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true, "effort": ["low", "medium", "high"] } },
  "limit": { "context": 131072, "output": 16384 }
}
```

## Gemma 4 12B IT QAT W4A16

Start it with `./llmctl up gemma4-12b --runtime sglang` ([llmctl](../docs/llmctl.md)); its recipe takes the server flags from the block below ([record](../research/2026-11-17/sglang-gemma-4-12b/README.md)).

| Item | Value |
|---|---|
| Checkpoint | `google/gemma-4-12B-it-qat-w4a16-ct` at `1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee`, the vLLM recipe's snapshot |
| Weights loaded | 8.77 GB, with Triton attention, which SGLang chose for this model (measured) |
| Memory | `--mem-fraction-static 0.90` with FP8 KV and `--swa-full-tokens-ratio 0.1`, no prefill CUDA graph |
| Audio | `--json-model-override-args '{"audio_config": null}'` builds the model without its audio projection, so the recipe serves text only |
| Parsers | `gemma4` for tool calls and reasoning; the chat template keeps thinking off by default |
| llmctl needs | `gpu_mib` 14891, `ram_gib` 6, `memory_gib` 32 |

Measured at the shipped settings:

- Startup: the ready line came 33.4 s after `docker run`, the FP8 pool held 204,181 tokens (20,418 sliding-window), and 1,099 MiB were free right after ready (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [measure-b](../research/2026-11-17/sglang-gemma-4-12b/measure-b/README.md)).
- Margins: a fresh 124,431-token prompt took 107.86 s and left 839 MiB, and the first grammar request and a forced tool call then moved nothing (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- With all eight runtime variables, 1,119 MiB were free after the flush that llmctl's check makes, 223 MiB above the 896 MiB that the threshold's rule asks of a shipped recipe, and the lowest reading after the prompt and both grammar requests was 841 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18; [measure-d](../research/2026-11-18/sglang-gemma-4-12b/README.md)).
- Speed: 70.4 tok/s decode with thinking off, the median of three runs of 842 to 929 tokens, against 69.5 on vLLM; 4.102 s to the first token for 9,936 tokens and 15.411 s for 37,194 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Long prompts are slow: the 124k-token prompt took 108 s, against 23 to 44 s for the other recipes that measurement B ran, because Triton attention runs over 40 sliding-window layers (the times and the backend are measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; the cause is inferred).
- Load: 41 requests in 300 s with prompts up to 13,112 tokens and no error, at a 15,003 MiB GPU peak and 5.92 GiB of non-reclaimable host memory (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Audio: without the override the start died one second into loading with `AssertionError: Attempted to load weight (torch.Size([3840, 640])) into parameter (torch.Size([3840, 1536]))` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17), because SGLang sizes the audio projection from `audio_config.output_proj_dims`, which this checkpoint's config lacks (inferred from the error and `model_gemma4_unified.py` at tag v0.5.23).
- FP8 against BF16 KV: 19 of 20 correct answers each on 20 fixed prompts at temperature 0, with the same 20 final answers; BF16 KV at `--swa-full-tokens-ratio 0.06` held 139,214 tokens, but its long prefills took 1.35 to 1.64 times as long, so the recipe keeps FP8 KV (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Checks: plain and streamed replies, a `get_weather` tool call and its second turn, a `json_schema` reply and Hermes' title request passed; thinking stayed off by default, and `enable_thinking: true` and the efforts `low` and `high` reasoned and answered in `content` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17), because SGLang maps every effort but `none` to `enable_thinking` (SG-61 in [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md)).
- Known risk: issue [#42791](https://github.com/sgl-project/sglang/issues/42791) reports that the `gemma4` tool-call parser loop "blocks the whole server" on a stray `]` in an array argument (third-party report, SG-92 in [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md)); the tool calls above did not hit it, so watch for a server that stops answering during a tool call ([Known issues](README.md#known-issues)).
- A cold start from an empty cache volume without a network reached `/ready` in 137.0 s with no download line (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [M15](../research/2026-11-17/sglang-runtime/measure-b/README.md#m15-cold-starts-without-a-network)).

```bash
docker run -d --name sglang \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -e FLASHINFER_NO_DOWNLOAD=1 \
  -e SYNC_TOKEN_IDS_ACROSS_TP=1 \
  -e NCCL_MAX_P2P_NCHANNELS=1 \
  -e TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data \
  -e PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v sglang-cache:/root/.cache/sglang \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
  --model-path /root/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-w4a16-ct/snapshots/1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee \
  --served-model-name gemma-4-12b \
  --context-length 131072 \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.90 \
  --max-running-requests 1 \
  --kv-cache-dtype fp8_e4m3 \
  --swa-full-tokens-ratio 0.1 \
  --disable-prefill-cuda-graph \
  --json-model-override-args '{"audio_config": null}' \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4
```

Sampling: llmctl pins `{"temperature": 1.0, "top_p": 0.95, "top_k": 64}` in the overlay's `generation_config.json`, the vLLM recipe's values, which the checkpoint's own file already holds, so a request without sampling fields gets temperature 1.0, top_p 0.95 and top_k 64 from the overlay and min_p 0.0 and repetition_penalty 1.0 from SGLang's defaults (the start logged `{'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}`, measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; the defaults are SG-59 and a request's own fields win by SG-58 in [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md)).

OpenCode entry:

```json
"gemma-4-12b": {
  "modelID": "gemma-4-12b",
  "name": "Gemma 4 12B QAT W4A16 (SGLang)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 131072, "output": 16384 },
  "variants": [
    { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
    { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
  ]
}
```

## Gemma 4 E4B IT QAT W4A16

Start it with `./llmctl up gemma4-e4b --runtime sglang` ([llmctl](../docs/llmctl.md)); its recipe takes the server flags from the block below ([record](../research/2026-11-17/sglang-gemma-4-e4b/README.md)).
It is the smoke-test model, as on vLLM: small and fast, but not a coding agent ([vLLM section](../vllm/models.md#smoke-test-model-gemma-4-e4b-it)).

| Item | Value |
|---|---|
| Checkpoint | `google/gemma-4-E4B-it-qat-w4a16-ct` at `6cd26aaa2357fb2bad8c51699a7558a4d1a965bb`, the vLLM recipe's snapshot |
| Weights loaded | 9.29 GB, with Triton attention, which SGLang chose for this model (measured) |
| Memory | `--mem-fraction-static 0.85` with BF16 KV and `--swa-full-tokens-ratio 0.1`, no prefill CUDA graph |
| Audio and vision | `--json-model-override-args '{"audio_config": null}'` builds the model without its audio tower; the vision tower's quantized layers stay uninitialized, so image input does not work and the recipe serves text only |
| Parsers | `gemma4` for tool calls and reasoning; the chat template keeps thinking off by default |
| llmctl needs | `gpu_mib` 14327, `ram_gib` 6, `memory_gib` 32 |

Measured at the shipped settings:

- Startup: the ready line came 28.0 s after `docker run`, the BF16 pool held 101,791 tokens (10,179 sliding-window), and 1,407 MiB were free right after ready (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [measure-b](../research/2026-11-17/sglang-gemma-4-e4b/measure-b/README.md)).
- Pool: with the default sliding-window ratio the same 3.40 GB held only 42,413 tokens, short of 64k, and ratio 0.1 raised it to 101,791 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17); 128k would need about 4.38 GiB of KV (estimate from `config.json`, checked against the start logs).
- Margins: a fresh 62,244-token prompt took 22.52 s and left 1,403 MiB, and the first grammar request and a forced tool call then moved nothing (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- With all eight runtime variables, 1,923 MiB were free after the flush that llmctl's check makes, and the lowest reading after the prompt and both grammar requests was 1,627 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18; [measure-d](../research/2026-11-18/sglang-gemma-4-e4b/README.md)).
- Speed: 141.3 tok/s decode with thinking off, the median of three runs of 981 to 1,099 tokens, three times the vLLM recipe's 45.2 tok/s in eager mode; 1.617 s to the first token for 9,932 tokens and 8.632 s for 37,190 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Load: 80 requests in 300 s with prompts up to 13,108 tokens and no error, at a 14,439 MiB GPU peak and 5.73 GiB of non-reclaimable host memory (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Audio: the hands-on start failed while building the audio tower with `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'weight'`, and a text-only architecture found no processor (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [record](../research/2026-11-17/sglang-gemma-4-e4b/README.md)), because the encoder is built with the checkpoint's quantization config (SG-73 in [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md)); with `audio_config` null the model builds without it (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Vision: the start log names 114 weights not initialized from the checkpoint, among them the quantized layers of all 16 vision-encoder layers (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17), so image input cannot work (inferred from the log; image input not run); llmctl's client entries send text only.
- Thinking: off by default, and `enable_thinking: false` and `reasoning_effort: "none"` gave plain answers; with thinking on, through `enable_thinking: true` or any other effort, the model sometimes answered without a thought block, and the `gemma4` reasoning parser then returned the whole reply as `reasoning_content`, leaving `content` empty, streamed and not (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
  A client that turns thinking on (OpenCode's `#thinking` variant, Pi at medium, dsh at Medium, Kimi with an effort) can therefore get an empty answer to a short question (inferred from the measured replies), so keep this recipe's thinking off.
- Checks: plain and streamed replies, a `get_weather` tool call, a `json_schema` reply and Hermes' title request passed; one non-streamed second turn after the tool call ended after one token with empty content, while its streamed copy answered, which at temperature 1.0 is intermittent (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Known risk: issue [#42791](https://github.com/sgl-project/sglang/issues/42791) reports that the `gemma4` tool-call parser loop "blocks the whole server" on a stray `]` in an array argument (third-party report, SG-92 in [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md)); the tool calls above did not hit it, so watch for a server that stops answering during a tool call ([Known issues](README.md#known-issues)).
- A cold start from an empty cache volume without a network reached `/ready` in 138.1 s with no download line (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [M15](../research/2026-11-17/sglang-runtime/measure-b/README.md#m15-cold-starts-without-a-network)).

```bash
docker run -d --name sglang \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -e HF_HUB_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e MAX_JOBS=2 \
  -e FLASHINFER_NO_DOWNLOAD=1 \
  -e SYNC_TOKEN_IDS_ACROSS_TP=1 \
  -e NCCL_MAX_P2P_NCHANNELS=1 \
  -e TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data \
  -e PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v sglang-cache:/root/.cache/sglang \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
  --model-path /root/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/6cd26aaa2357fb2bad8c51699a7558a4d1a965bb \
  --served-model-name gemma-4-e4b \
  --context-length 65536 \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.85 \
  --max-running-requests 1 \
  --swa-full-tokens-ratio 0.1 \
  --disable-prefill-cuda-graph \
  --json-model-override-args '{"audio_config": null}' \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4
```

Sampling: llmctl pins `{"temperature": 1.0, "top_p": 0.95, "top_k": 64}` in the overlay's `generation_config.json`, the vLLM recipe's values, which the checkpoint's own file already holds, so a request without sampling fields gets temperature 1.0, top_p 0.95 and top_k 64 from the overlay and min_p 0.0 and repetition_penalty 1.0 from SGLang's defaults (the start logged `{'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}`, measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; the defaults are SG-59 and a request's own fields win by SG-58 in [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md)).

OpenCode entry:

```json
"gemma-4-e4b": {
  "modelID": "gemma-4-e4b",
  "name": "Gemma 4 E4B QAT W4A16 smoke test (SGLang)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 65536, "output": 16384 },
  "variants": [
    { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
    { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
  ]
}
```

## Models that do not run on SGLang

The first three rows are estimates, the last three measured failures of measurement B, which ran each vLLM recipe's checkpoint and sampling on SGLang with SGLang's own flags.
The arithmetic below takes the static budget of SGLang v0.5.23 on this GPU as 15.06 GiB times `--mem-fraction-static`, 13.55 GiB at 0.90, for the weights, the KV pool, the Mamba state and a 0.10 GiB multimodal reserve together (estimate, from the hands-on start logs).
"Free RAM" means the about 40 GiB that the desktop leaves free, and the budget itself is explained in [Memory budget](README.md#memory-budget).

| Model | Why it does not run |
|---|---|
| `qwen3.8-27b` | It is dense, so Paged Experts does not apply (inferred), and the smallest W4A16 build is 15.59 GiB (SG-105), which with the vision tower exceeds the 13.55 GiB budget (estimate). `--cpu-offload-gb` streams every offloaded weight to the GPU on every forward pass (SG-80); in PR #42557 it gave 16.6 tok/s with 3 GB offloaded, serving the MoE Qwen3-30B-A3B in int4 on a 16 GB desktop RTX 5070 Ti (SG-82, vendor-reported), against 36.4 tok/s for Qwen3.8-27B on llama.cpp here (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-16; [llamacpp/models.md](../llamacpp/models.md)). |
| `nemotron-3.5-bolt` | No public NVFP4 build answers (HTTP 401), and the BF16 checkpoint is 139.5 GiB of safetensors (SG-102), far above 16 GB of VRAM plus the free RAM (estimate). |
| `qwen3-coder-next` | No public NVFP4 build answers (HTTP 401), and the official FP8 build is 74.86 GiB (SG-103); an int4 build would pin about 38 GiB of experts in host memory against the free RAM (estimate), and Intel's AutoRound format is not among the formats Paged Experts accepts (SG-28, inferred). |
| `mellum2.1-12b` | The vLLM recipe's checkpoint `cyankiwi/Mellum2.1-12B-A2.5B-Thinking-AWQ-INT4` died two seconds into loading with `RuntimeError: The size of tensor a (288) must match the size of tensor b (28) at non-singleton dimension 1` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [record](../research/2026-11-17/sglang-mellum2.1-12b/README.md)), because SGLang's compressed-tensors WNA16 MoE loader expects the asymmetric zero points in the transposed layout, and its Triton WNA16 MoE backend accepts only symmetric INT4 (inferred from the source at tag v0.5.23). vLLM v0.33.0 serves the same files at 257.2 tok/s (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17). |
| `groot-19b-a3b` | The vLLM recipe's NVFP4 checkpoint loaded its weights, then preparing the weight-only NVFP4 linears for the FP4 Marlin kernel raised `AttributeError: 'CompressedTensorsConfig' object has no attribute 'group_size'` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [record](../research/2026-11-17/sglang-groot-19b-a3b/README.md)), because `prepare_nvfp4_layer_for_marlin` reads a field that the compressed-tensors config lacks, so no `--fp4-gemm-backend` avoids it on this GPU (inferred from the source and the log). vLLM v0.33.0 serves it at 128k at 193.4 tok/s (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17). |
| `deepseek-r1-0528-qwen3-8b` | With `--quantization fp8` the start ran out of GPU memory before loading, `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB.` in the `lm_head` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [record](../research/2026-11-17/sglang-deepseek-r1-0528-qwen3-8b/README.md)), because SGLang v0.5.23 creates every linear weight of the BF16 checkpoint in BF16 before it quantizes, and the 15.26 GiB BF16 model does not fit in the 15.06 GiB that CUDA leaves free (inferred from the source; the sizes are arithmetic from `config.json` checked against the log), so no fraction, context or KV setting helps. vLLM v0.33.0 quantizes while it loads and holds the model in 9.12 GiB (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [record](../research/2026-11-17/vllm-deepseek-r1-0528-qwen3-8b/README.md)). |

The sources rows (SG-28 to SG-105) are in [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md), each with its verdict.
