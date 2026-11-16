# vLLM model recipes for the RTX 5080 Laptop

Last verified: 2026-11-16 (gpt-oss-20b measured; Ornith note updated from the agent-harness review; other rows are 2026-11-15 estimates).

Only small checkpoints run fully under vLLM v0.33.0 on this 16 GB GPU: Groot-19B-A3B NVFP4, K2 Horizon 7B AWQ-INT4 and DeepSeek-R1-0528-Qwen3-8B (chat only) from the ranked picks, plus Gemma 4 12B QAT W4A16, Qwen3.5-9B NVFP4 and Mellum2.1 as honorable mentions and Gemma 4 E4B as a stack smoke test.
gpt-oss-20b starts only with `--enforce-eager` and 256 batched tokens, and then runs at half the llama.cpp speed (measured on 2026-11-16, [gpt-oss-20b](#gpt-oss-20b)).
The 2026-11-16 measurement also showed that the budget method behind the estimates below is optimistic by about 1.5 to 2.5 GiB ([README.md](README.md#memory-budget)), so every context figure on this page except gpt-oss-20b must be confirmed against the first startup log, and Groot NVFP4 (10.94 GiB on disk) may not fit at all.
None of the four highest-ranked models (Qwen3.8-27B, Qwen3.6-35B-A3B, K2 Horizon MoVA, Gemma 4 26B-A4B) fits the 14.65 GiB vLLM budget, and vLLM cannot offload MoE experts to RAM on this version ([#66264](https://github.com/vllm-project/vllm/issues/66264)), so they run on llama.cpp ([llamacpp/models.md](../llamacpp/models.md)).
For every model below, llama.cpp is the primary runtime in the ranking ([model-selection.md](../docs/model-selection.md)); vLLM is the second path, useful mainly for gpt-oss-20b, Groot and the vLLM-native honorable mentions.
Apart from the gpt-oss-20b measurement, nothing here has been run on this laptop: the other context, memory and speed figures are estimates from config files and file sizes.

The image, run template, budget method, flag reference, parser table and smoke tests live in [README.md](README.md); this page only adds the per-model values.
Load testing is in [benchmarking.md](benchmarking.md), host prerequisites (NVIDIA Container Toolkit with CDI, installed and verified on 2026-11-15) in [host-setup.md](../docs/host-setup.md), and the OpenCode provider in [opencode.md](../docs/opencode.md).

## Which models fit

Budget method used for this table (2026-11-15): 16303 MiB x 0.92 = 14.65 GiB, minus the weights vLLM loads, minus a 1.2 GiB reserve, gives the KV pool (estimate).
The measured basis is lower: CUDA reports 15.47 GiB in total, weights can load larger than the file, and the reserve is about 2 GiB ([README.md](README.md#memory-budget)).
KV per token is 2 x full-attention layers x KV heads x head dim x bytes; sliding-window, Gated DeltaNet and Mamba layers add a fixed amount per sequence instead.

| Model | Status in the ranking | vLLM checkpoint | Weights loaded (GiB) | KV per token, BF16 | Context here (estimate) | Use it for |
|---|---|---|---|---|---|---|
| [gpt-oss-20b](#gpt-oss-20b) | Rank 9 | `openai/gpt-oss-20b` (MXFP4) | 13.8 in GPU memory (12.82 on disk), measured | 24 KiB | Measured: 65536 only at util 0.965 with `--enforce-eager`, 81,504 FP8 KV tokens, 83.5 tok/s | Runtime comparison only; use llama.cpp (131072, 166 tok/s) |
| [Groot-19B-A3B-261023](#groot-19b-a3b-261023-nvfp4) | Rank 8 | `GrootTeam/Groot-19B-A3B-261023-NVFP4`, `--language-model-only` | 10.94 | 20 KiB | 131072 with FP8 KV | Fast all-in-VRAM agent, vendor claims only |
| [K2 Horizon 7B](#k2-horizon-7b-awq-int4) | Rank 4 | `cyankiwi/K2-Horizon-7B-AWQ-INT4` | 7.56 | 144 KiB | 65536 with FP8 KV | Weak alternative; llama.cpp Q8_0 is primary |
| [DeepSeek-R1-0528-Qwen3-8B](#deepseek-r1-0528-qwen3-8b-chat-only) | Rank 10, chat only | BF16 repo with online `fp8_per_tensor` | 8.79 | 144 KiB | 65536 with FP8 KV (3% margin) | Reasoning and Q&A, no tools |
| [Gemma 4 12B IT](#gemma-4-12b-it-qat-w4a16) | Honorable | `google/gemma-4-12B-it-qat-w4a16-ct` | 9.56 | 16 KiB global + sliding | 131072 with FP8 KV | The vLLM-native Gemma |
| [Qwen3.5-9B](#qwen35-9b-nvfp4) | Honorable | `RedHatAI/Qwen3.5-9B-NVFP4`, `--language-model-only` | 7.42 | 32 KiB | 131072 with BF16 KV | Conservative vLLM-native 9B |
| [Mellum2.1 12B-A2.5B Thinking](#mellum21-12b-a25b-thinking-awq-int4) | Honorable | `cyankiwi/Mellum2.1-12B-A2.5B-Thinking-AWQ-INT4` | 7.15 | 14 KiB | 131072 with BF16 KV | Fast sub-agent |
| [Gemma 4 E4B IT](#smoke-test-model-gemma-4-e4b-it) | Excluded as an agent | `google/gemma-4-E4B-it-qat-w4a16-ct`, `--language-model-only` | 9.83 | 16 KiB | 65536 (131072 possible) | First run of the whole stack |

Ranks 1, 2, 3, 5, 6 and 7 do not fit; see [Models that do not fit in vLLM](#models-that-do-not-fit-in-vllm).
Ornith-1.5-9B, which earlier drafts used as the vLLM primary, is now excluded; see [Ornith-1.5-9B](#ornith-15-9b-excluded).

## Conventions for every recipe

- Every command is the [README run template](README.md#run-template) with the model's flags filled in: CDI GPU access, `-p 127.0.0.1:8000:8000`, `--memory 32g --memory-swap 32g` (the template's cap, an estimate that leaves room for first-start JIT), no restart policy, and persistent `vllm-cache` and `flashinfer-cache` volumes.
- Start one model at a time with `--name vllm`, and stop it with `docker stop vllm` when you are done, so the dGPU can power down ([host-setup.md](../docs/host-setup.md)).
- `--served-model-name` is the OpenCode `modelID`, and `--max-model-len` is `limit.context` ([opencode.md](../docs/opencode.md#matching-rules)).
- A quantized build is a separate repository, not a flag: `--quantization` cannot turn a BF16 repo into the NVFP4 one (lesson from the user's DGX Spark runbook).
  The one exception here is DeepSeek, which uses vLLM's online FP8.
- FP8 KV on sm_120 leaves FlashAttention 2 for FlashInfer, which has a race under async scheduling whose fix is not in v0.33.0 ([#61239](https://github.com/vllm-project/vllm/issues/61239)), so every recipe with `--kv-cache-dtype fp8` also passes `--attention-backend TRITON_ATTN` (inferred mitigation, see [README.md](README.md#quantization-paths-on-sm_120)).
- Toggle-thinking models start with thinking off (`--default-chat-template-kwargs '{"enable_thinking":false}'`), and OpenCode opts in with a `#thinking` variant ([opencode.md](../docs/opencode.md#thinking-variants-per-runtime)).
- Server-side sampling defaults use `--override-generation-config`, which is merged with the model's `generation_config.json` under the default `--generation-config auto` ([model.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/model.py)); request values still win.
- `--max-num-seqs 1`: one sequence at a time, so the per-sequence sliding-window and recurrent state is reserved once; other OpenCode requests (titles, sub-agents) queue.
  Raise it only where the pool leaves room.
- After the first start, read the "KV cache size" log line and replace the estimate in the card ([README.md](README.md#memory-budget)).
- After every start, run the three smoke tests in [README.md](README.md#smoke-tests), then one OpenCode tool-call and edit test.

The OpenCode entries in the cards go under the `vllm-docker` provider of the validated config in [opencode.md](../docs/opencode.md#the-validated-config), which already sets `compatibility.reasoningField: "reasoning"` and turns off the built-in vLLM discovery.
All eight entries, placed together in that provider, validated against a copy of [https://opencode.ai/config.json](https://opencode.ai/config.json) with Python `jsonschema` 4.19.2 on 2026-11-15 with 0 errors; a negative test with a V1 `npm` key and a string `tools` value was rejected.

## gpt-oss-20b

Rank 9, the required gpt-oss family member and the pipeline baseline ([model-selection.md](../docs/model-selection.md)).
Run it on llama.cpp for the full 131072 context ([llamacpp/models.md](../llamacpp/models.md)); use vLLM when you want the official checkpoint at 64k.

| Item | Value |
|---|---|
| Checkpoint | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b), native MXFP4 MoE, Apache-2.0, released 2025-08-05 |
| Size | Three top-level shards, 13,761,316,904 B = 12.82 GiB ([HF API](https://huggingface.co/api/models/openai/gpt-oss-20b?blobs=true)); the repo also carries a 12.82 GiB `original/` copy that vLLM does not need |
| Architecture | 21B total, 3.6B active; 12 full-attention and 12 sliding-window layers (window 128), 8 KV heads, head dim 64, 131072 max positions ([config.json](https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json)) |
| Kernel path on sm_120 | MXFP4 MoE on MARLIN with BF16 activations by default ([mxfp4.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py)); attention needs sinks, which FA2 lacks, and Triton supports sinks with FP8 KV ([triton_attn.py](https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/vllm/v1/attention/backends/triton_attn.py)) |
| Parsers | `--tool-call-parser openai`, `--reasoning-parser openai_gptoss` ([tool parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/tool_parsers/__init__.py), [reasoning parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/reasoning/__init__.py)) |

Measured on 2026-11-16 ([record](../research/2026-11-16/vllm-gpt-oss-20b/), [attempt log](../research/2026-11-16/vllm-gpt-oss-20b/attempts.md)): the estimate below did not hold.
The weights load as 13.8 GiB, CUDA reports 15.47 GiB in total, and the start at 0.95 failed with `Available KV cache memory: -0.74 GiB`.
The only configuration that started is the one in the serve command below: utilization 0.965, `--enforce-eager` and `--max-num-batched-tokens 256`, which gave 0.94 GiB of FP8 KV, 81,504 tokens, and 15,698 MiB of GPU memory in use.
In that configuration vLLM decoded at 83.5 tok/s (3 runs), took 1.81 s to the first token for an 8k prompt and 8.87 s for a 30k prompt, and returned to the first token in under 0.1 s when the same prompt was repeated; the OK test, the tool-call test and an OpenCode write-and-read session all passed.
llama.cpp runs the same model at 166 tok/s with 128k context and 1.12 s for an 8k prompt ([../llamacpp/models.md](../llamacpp/models.md#9-gpt-oss-20b)), so use llama.cpp for gpt-oss-20b on this laptop; keep this vLLM recipe only for runtime comparisons.

Original budget (estimate, superseded by the measurement above):

| Step | At util 0.92 | At util 0.95 |
|---|---|---|
| Budget | 14.65 GiB | 15.13 GiB |
| Weights | 12.82 GiB | 12.82 GiB |
| Reserve | 1.2 GiB | 1.2 GiB |
| KV pool | 0.63 GiB | 1.11 GiB |
| KV per token | 2 x 12 x 8 x 64 x 2 B = 24,576 B BF16, 12,288 B FP8 | same |
| Context | about 27.6k BF16, 55.1k FP8 | about 48.5k BF16, 97.0k FP8 |

The estimate concluded that 65536 needs util 0.95 plus FP8 KV; the FP8 KV needs `TRITON_ATTN` because of [#61239](https://github.com/vllm-project/vllm/issues/61239), which was reproduced with gpt-oss-20b on one GPU, and that part still applies.
The sliding layers add only about 3 MiB per sequence.
The iGPU drives the desktop and the dGPU idles at about 112 MiB (observed host state in [host-setup.md](../docs/host-setup.md#current-host-state-observed)), which is why 0.95 is plausible here; it is still the highest value [README.md](README.md#memory-budget) recommends trying.

Pre-download without the duplicate folders: the repo holds 12.84 GiB at the top level plus a 12.82 GiB `original/` copy and a 12.81 GiB `metal/` copy, and only the top level is needed ([HF tree](https://huggingface.co/api/models/openai/gpt-oss-20b/tree/main?recursive=true), checked 2026-11-16).
The `hf` CLI runs from the image (observed on 2026-11-16), and `--exclude` takes glob patterns ([download.py](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/cli/download.py)).
Running it as the user with the cache at `/hf` keeps the files owned by the user:

```bash
docker run --rm --memory 8g --memory-swap 8g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf -e HF_TOKEN \
  -v "$HOME/.cache/huggingface:/hf" \
  --entrypoint hf vllm/vllm-openai:v0.33.0 \
  download openai/gpt-oss-20b --exclude 'original/*' --exclude 'metal/*'
```

The `hf` download opens many parallel connections and took most of the bandwidth from a concurrent llama.cpp `-hf` download on this machine (observed), so run large downloads one at a time.

Serve:

```bash
docker run -d --rm --name vllm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN \
  -e VLLM_API_KEY \
  -e VLLM_NO_USAGE_STATS=1 \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v vllm-cache:/root/.cache/vllm \
  -v flashinfer-cache:/root/.cache/flashinfer \
  vllm/vllm-openai:v0.33.0 \
  openai/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --gpu-memory-utilization 0.95 \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --reasoning-effort-map '{"none":"low","minimal":"low","xhigh":"high","max":"high"}' \
  --override-generation-config '{"temperature":1.0,"top_p":1.0}'
```

The log line "Using 'MARLIN' Mxfp4 MoE backend." appeared on startup (observed on 2026-11-16).
The serve command above is the original recipe and does not start on this laptop; the configuration that started (observed) changes these flags, and the HF cache is mounted read-only with `HF_HUB_OFFLINE=1` so the root-run server cannot write root-owned files into the user's cache:

```bash
docker run -d --name vllm \
  --device nvidia.com/gpu=all --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:8000:8000 \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v vllm-cache:/root/.cache/vllm -v flashinfer-cache:/root/.cache/flashinfer \
  vllm/vllm-openai:v0.33.0 \
  openai/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --gpu-memory-utilization 0.965 \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 256 \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss \
  --reasoning-effort-map '{"none":"low","minimal":"low","xhigh":"high","max":"high"}' \
  --override-generation-config '{"temperature":1.0,"top_p":1.0}'
```

Thinking and sampling:

- Always on; only the effort changes: `reasoning_effort` low, medium (default) or high per request, and the Harmony path rejects any other value with HTTP 400 unless `--reasoning-effort-map` rewrites it ([cli_args.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/common/cli_args.py)).
- Sampling temperature 1.0, top_p 1.0, effort medium, as in the ranking ([model-selection.md](../docs/model-selection.md)).
- Tool calling supports `tool_choice` auto only on vLLM, which is what OpenCode sends; llama.cpp enforces calls with a grammar instead ([gpt-oss.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12418/common/parsers/gpt-oss.cpp)).

OpenCode entry (`limit.context` equals `--max-model-len`):

```json
"gpt-oss-20b": {
  "modelID": "gpt-oss-20b",
  "name": "gpt-oss-20b (vLLM)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true, "effort": ["low", "medium", "high"] } },
  "limit": { "context": 65536, "output": 16384 }
}
```

Caveats and open issues:

- Keep the default Python frontend: the Rust frontend refuses `--tool-call-parser openai` at startup ([#66989](https://github.com/vllm-project/vllm/issues/66989)).
- [#65770](https://github.com/vllm-project/vllm/issues/65770): after a tool name containing special-token text, every later turn returns HTTP 500; start a new session.
  [#58825](https://github.com/vllm-project/vllm/issues/58825) and [#46464](https://github.com/vllm-project/vllm/issues/46464) are further open Harmony parsing failures.
- The first start downloads the Harmony tiktoken vocabulary; if that fails, pre-download the files and set `TIKTOKEN_ENCODINGS_BASE` ([vLLM recipe](https://github.com/vllm-project/vllm/blob/v0.33.0/recipes/models/openai/gpt-oss-20b.yaml)).
- Do not copy the recipe's Blackwell flags (`--quantization-config.moe.activation mxfp8`, FlashInfer TRTLLM MoE): they target the B200 family, and the TRTLLM MXFP4 path needs capability family 100 ([mxfp4.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py)).
- Agentic strength is weak on independent measures: SWE-bench Verified 52.44 and Terminal-Bench 2.1 15.17 in NVIDIA's same-harness table ([Nemotron 3.5 Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/README.md)), and Terminal-Bench 2.1 13.9 on [Artificial Analysis](https://artificialanalysis.ai/models/gpt-oss-20b).
- Speed: about 205 tok/s tg128 with llama.cpp (measured on a desktop RTX 5080, 960 GB/s, llama.cpp build 6210, [ggml-org guide](https://github.com/ggml-org/llama.cpp/discussions/15396)); no vLLM number on an RTX 50 card exists, and the laptop part is slower.

## Groot-19B-A3B-261023 NVFP4

Rank 8, the fast all-in-VRAM option, on vendor claims only ([model-selection.md](../docs/model-selection.md)).
The ranking's primary runtime is llama.cpp with the official Q4_K_M ([llamacpp/models.md](../llamacpp/models.md)); this NVFP4 build is the second path and uses the same tool format and parsers as Qwen3.6.

| Item | Value |
|---|---|
| Checkpoint | [`GrootTeam/Groot-19B-A3B-261023-NVFP4`](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023-NVFP4), created 2026-10-23 |
| Size | Shards 3.73 + 3.73 + 3.73 + 0.60 = 11.78 GiB, plus a separate 0.82 GiB `mtp.safetensors` ([HF API](https://huggingface.co/api/models/GrootTeam/Groot-19B-A3B-261023-NVFP4?blobs=true)); about 10.94 GiB with `--language-model-only` (estimate: the BF16 vision tower is taken as the 0.84 GiB mmproj size) |
| Quantization | NVFP4 W4A16 for routed and shared experts and lm_head, FP8 W8A8 for attention and GDN projections, vision BF16, FP8 `kv_cache_scheme` ([config.json](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023-NVFP4/raw/main/config.json)) |
| Architecture | `Qwen3_5MoeForConditionalGeneration`, identical to Qwen3.6-35B-A3B except 128 instead of 256 routed experts; 40 layers, 10 full-attention (2 KV heads, head dim 256) and 30 Gated DeltaNet ([config.json](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/config.json)) |
| Kernel path on sm_120 | Weight-only NVFP4 MoE on Marlin, which avoids the W4A4 FlashInfer JIT RAM spike (inferred from the selection order in [nvfp4.py](https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py)) |
| Parsers | `--tool-call-parser qwen3_coder` (`qwen3_xml` is the same class), `--reasoning-parser qwen3` ([tool parsers](https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/vllm/tool_parsers/__init__.py), [reasoning parsers](https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/vllm/reasoning/__init__.py)) |
| Licence | Apache-2.0 in the card metadata, but the repo has no LICENSE file ([README](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/README.md)) |

Budget (estimate):

| Step | Value |
|---|---|
| Budget at 0.92 | 14.65 GiB |
| Weights, language model only | 10.94 GiB |
| Reserve | 1.2 GiB |
| GDN state per sequence | 30 x (32 x 128 x 128 x 4 B + 8192 x 3 x 2 B) = about 0.06 GiB |
| KV pool | 14.65 - 10.94 - 1.2 - 0.06 = 2.45 GiB |
| KV per token | 2 x 10 x 2 x 256 x 2 B = 20,480 B BF16, 10,240 B FP8 |
| Context | about 128.5k BF16, 257k FP8 (cap 262144) |
| 131072 with FP8 KV | 1.25 GiB of KV, about 1.2 GiB spare |

131072 is just above the BF16 estimate, so the recipe uses FP8 KV; to stay on FlashAttention 2 instead, drop the two FP8 flags and use `--max-model-len 98304` (estimate).
MTP speculative decoding would load the extra 0.82 GiB and cut the pool to about 85k BF16 or 171k FP8 tokens; it is untested here, so it is left out.

```bash
docker run -d --rm --name vllm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN \
  -e VLLM_API_KEY \
  -e VLLM_NO_USAGE_STATS=1 \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v vllm-cache:/root/.cache/vllm \
  -v flashinfer-cache:/root/.cache/flashinfer \
  vllm/vllm-openai:v0.33.0 \
  GrootTeam/Groot-19B-A3B-261023-NVFP4 \
  --served-model-name groot-19b-a3b \
  --language-model-only \
  --max-model-len 131072 \
  --max-num-seqs 1 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'
```

The card's own serve commands use the repo id `Groot/Groot-19B-A3B-261023`, which does not exist; use `GrootTeam/...` as above ([README](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/README.md)).

Thinking and sampling:

- Toggle, on by default in the template; `chat_template_kwargs {"enable_thinking": false}` appends an empty think block, and there are no `/think` or `/nothink` switches ([chat_template.jinja](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/chat_template.jinja)).
- Card presets: precise coding temperature 0.6, top_p 0.95, top_k 20, presence 0 (used above); thinking general temperature 1.0, top_p 0.95, top_k 20, presence 1.5; non-thinking temperature 0.7, top_p 0.8, top_k 20, presence 1.5 ([README](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/README.md)).
- The card recommends keeping at least 128K of context to preserve thinking quality ([README](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/README.md)).

OpenCode entry:

```json
"groot-19b-a3b": {
  "modelID": "groot-19b-a3b",
  "name": "Groot-19B-A3B NVFP4 (vLLM)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 131072, "output": 16384 },
  "variants": [
    { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
    { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
  ]
}
```

Caveats and open issues:

- [#54331](https://github.com/vllm-project/vllm/issues/54331): hybrid-GDN NVFP4 models on the Marlin NvFp4 MoE path on sm_120 died under sustained high-concurrency load with CUDA graphs on; this is the same architecture and kernel path.
  Keep `--enforce-eager` in reserve, and fall back to llama.cpp if it crashes.
- Hybrid prefix caching runs in `align` mode ([README.md](README.md#prefix-caching-and-chunked-prefill)); on Qwen3.5-family GDN models the hit rate can silently drop to 0% ([#45238](https://github.com/vllm-project/vllm/issues/45238)), so long OpenCode turns may re-prefill (likely applies to Groot, inferred).
- Qwen tool-parser issues apply: [#61206](https://github.com/vllm-project/vllm/issues/61206), [#65774](https://github.com/vllm-project/vllm/issues/65774), [#48753](https://github.com/vllm-project/vllm/issues/48753).
- The repo's `chat_template.jinja` re-emits reasoning for every past assistant turn, while the template in `tokenizer_config.json` drops it unless `preserve_thinking` is set; which one vLLM applies was not confirmed ([chat_template.jinja](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/chat_template.jinja)).
- Evidence is thin.
  Every agentic score is self-reported, some Qwen3.6 baselines were copied rather than run, and on the vendor's own chart Groot loses to its base on SWE-bench Verified (31.66 vs 36.27) ([README](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/README.md)).
- One third-party run exists: quality index 78.5 and tools 100.0 (measured on an RTX PRO 6000 Blackwell with vLLM NVFP4 plus MTP, [Blackwall](https://guitaripod.github.io/blackwall/#groot-19b-a3b-nvfp4~medium)), with no Qwen3.6 comparator.
- The publisher is an anonymous Hugging Face user account, and the pruning method and data are undisclosed ([HF user API](https://huggingface.co/api/users/GrootTeam/overview)); community reports describe altered verbatim path text, which is a direct risk for a coding agent.
  Run an A/B against Qwen3.6 on the same OpenCode tasks before relying on it.

## K2 Horizon 7B AWQ-INT4

Rank 4, but the vLLM path is weak: run the ranking's llama.cpp recipe (bartowski Q8_0, 65536) first ([llamacpp/models.md](../llamacpp/models.md)).
Use this card only when you need the vLLM server, and expect pre-FTPO weights.

| Item | Value |
|---|---|
| Base model | [`IFM/K2-Horizon-7B`](https://huggingface.co/IFM/K2-Horizon-7B), Apache-2.0; 9.0B dense BF16 (7B core plus untied embeddings), 36 full-attention layers, 8 KV heads, head dim 128, 524,288 native context ([config.json](https://huggingface.co/IFM/K2-Horizon-7B/raw/main/config.json)) |
| Checkpoint | [`cyankiwi/K2-Horizon-7B-AWQ-INT4`](https://huggingface.co/cyankiwi/K2-Horizon-7B-AWQ-INT4), compressed-tensors W4A16 int4 group 32, lm_head BF16 |
| Size | 8,122,473,440 B = 7.56 GiB ([HF API](https://huggingface.co/api/models/cyankiwi/K2-Horizon-7B-AWQ-INT4?blobs=true)) |
| Why not the BF16 or official FP8 build | BF16 is 16.76 GiB and does not fit; the official FP8 (10.29 GiB) reaches only about 46k tokens with FP8 KV at 0.92 (estimate) |
| Weights version | Uploaded 2026-09-24, before IFM promoted the anti-doom-loop (FTPO) checkpoint to main on 2026-10-01, so it most likely carries pre-FTPO weights (inferred from dates, [commits](https://huggingface.co/api/models/IFM/K2-Horizon-7B/commits/main)) |
| vLLM support | `K2HorizonForCausalLM` first appears in v0.30.0 ([k2_horizon.py at v0.30.0](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/k2_horizon.py); absent at v0.29.0) |
| Parsers | `k2_horizon` for both tools and reasoning ([tool parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/tool_parsers/__init__.py), [reasoning parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/reasoning/__init__.py)) |

Budget (estimate): 14.65 - 7.56 - 1.2 = 5.88 GiB of pool.
KV per token is 2 x 36 x 8 x 128 x 2 B = 147,456 B (144 KiB) BF16 or 73,728 B FP8, so the pool holds about 42.8k tokens with BF16 KV and 85.7k with FP8.
65536 therefore needs FP8 KV (4.5 GiB at 64k).

```bash
docker run -d --rm --name vllm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN \
  -e VLLM_API_KEY \
  -e VLLM_NO_USAGE_STATS=1 \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v vllm-cache:/root/.cache/vllm \
  -v flashinfer-cache:/root/.cache/flashinfer \
  vllm/vllm-openai:v0.33.0 \
  cyankiwi/K2-Horizon-7B-AWQ-INT4 \
  --served-model-name k2-horizon-7b \
  --trust-remote-code \
  --chat-template-content-format string \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --enable-auto-tool-choice \
  --tool-call-parser k2_horizon \
  --reasoning-parser k2_horizon \
  --default-chat-template-kwargs '{"reasoning_effort":"high","tool_call_format":"xml"}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}'
```

`--trust-remote-code` and `--chat-template-content-format string` come from the vLLM recipe ([K2-Horizon-7B.yaml](https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/recipes/models/IFM/K2-Horizon-7B.yaml)); there is no in-tree config class, so vLLM runs the repo's `configuration_k2_horizon.py`.
That code comes from a third-party repo here, so read the repo's Python files before the first start.

Thinking and sampling:

- Always on; there is no `enable_thinking`.
  Effort is `reasoning_effort` high (default), medium or low through `chat_template_kwargs` ([chat_template.jinja](https://huggingface.co/IFM/K2-Horizon-7B/raw/main/chat_template.jinja)).
- Stay on high on vLLM: the `k2_horizon` reasoning parser accepts only the requested effort's end tag, and the model often closes medium and low with the generic tag, so reasoning leaks into content ([#62743](https://github.com/vllm-project/vllm/issues/62743)).
- Card: temperature 1.0, top_p 0.95, effort high, at least 32,768 output tokens ([README](https://huggingface.co/IFM/K2-Horizon-7B/raw/main/README.md)).
  At 64k context, cap OpenCode output at about 16k so input keeps room.
- Whether the parser sees `reasoning_effort` from `--default-chat-template-kwargs` rather than from the request is untested; high is also the template default, so the two agree.

OpenCode entry (the provider's `reasoningField: "reasoning"` is required: the strict template raises on assistant history without a reasoning field, and OpenCode sends one only when the field is set):

```json
"k2-horizon-7b": {
  "modelID": "k2-horizon-7b",
  "name": "K2 Horizon 7B AWQ-INT4 (vLLM)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 65536, "output": 16384 }
}
```

Caveats and open issues:

- The template also rejects a tool schema whose `required` list names a property that is missing from `properties`; check MCP tool schemas ([chat_template.jinja](https://huggingface.co/IFM/K2-Horizon-7B/raw/main/chat_template.jinja)).
- The IFM organisation is not HF-verified ([org API](https://huggingface.co/api/organizations/IFM/overview)).
- Evidence: Terminal-Bench 2.1 44.2 and tau3-Banking 27.6 (measured by [Artificial Analysis](https://artificialanalysis.ai/models/k2-horizon-7b) at full precision, not this AWQ build); vendor SWE-bench Verified 72.4 is not confirmed independently ([README](https://huggingface.co/IFM/K2-Horizon-7B/raw/main/README.md)).
- Speed: about 145 tok/s decode ceiling for AWQ-INT4 at 896 GB/s (estimate); always-on high-effort thinking makes turns long regardless.

## DeepSeek-R1-0528-Qwen3-8B (chat only)

Rank 10 through the required-family swap: chat and reasoning only, with no tool calling, so it is not an OpenCode agent model ([model-selection.md](../docs/model-selection.md)).
llama.cpp Q8_0 is the ranking's primary path ([llamacpp/models.md](../llamacpp/models.md)); vLLM works with online FP8.

| Item | Value |
|---|---|
| Checkpoint | [`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B), BF16, MIT, released 2025-05-29 |
| Download | 16,381,516,796 B = 15.26 GiB ([HF API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B?blobs=true)); too big for vLLM as BF16 |
| Loaded weights | About 8.79 GiB after online FP8 (estimate: embeddings and lm_head stay BF16, linears go to FP8; it matches the 9.43 GB of a third-party FP8-Dynamic build) |
| Architecture | `Qwen3ForCausalLM`, 36 full-attention layers, 8 KV heads, head dim 128, 131072 max positions via YaRN ([config.json](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/raw/main/config.json)) |
| Quantization flag | `--quantization fp8_per_tensor`; plain `fp8` is deprecated for online use in v0.33.0 and is rewritten with a warning ([weight_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/model_loader/weight_utils.py#L355-L361), [quantization/\_\_init\_\_.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/layers/quantization/__init__.py)) |
| Parsers | `--reasoning-parser deepseek_r1`; no tool parser works, because the `deepseek_v3` parser raises at init when the tool tokens are missing from this tokenizer ([deepseekv3_tool_parser.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/tool_parsers/deepseekv3_tool_parser.py)) |

Budget (estimate): 14.65 - 8.79 - 1.2 = 4.66 GiB of pool.
KV per token is 2 x 36 x 8 x 128 x 2 B = 147,456 B BF16 or 73,728 B FP8, so the pool holds about 33.9k tokens BF16 and 67.8k FP8.
65536 with FP8 KV leaves only about 3% margin; if startup refuses it, use `--max-model-len 32768` without the two FP8 flags.
For more room, [`Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound`](https://huggingface.co/api/models/Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound?blobs=true) is 5.68 GiB and would leave about 56.5k BF16 or 113k FP8 tokens (estimate; its Marlin path on sm_120 is not checked).

```bash
docker run -d --rm --name vllm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN \
  -e VLLM_API_KEY \
  -e VLLM_NO_USAGE_STATS=1 \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v vllm-cache:/root/.cache/vllm \
  -v flashinfer-cache:/root/.cache/flashinfer \
  vllm/vllm-openai:v0.33.0 \
  deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --served-model-name deepseek-r1-8b \
  --quantization fp8_per_tensor \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --reasoning-parser deepseek_r1 \
  --override-generation-config '{"temperature":0.6,"top_p":0.95}'
```

There is deliberately no `--enable-auto-tool-choice`.

Thinking and sampling:

- Always on, with no official off switch; vLLM silently drops an `enable_thinking` kwarg for R1 distills ([reasoning_outputs.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/features/reasoning_outputs.md)).
- Card sampling temperature 0.6, top_p 0.95 ([README](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/raw/main/README.md)).

OpenCode entry (tools off; use it in plan or chat mode, never as the build agent):

```json
"deepseek-r1-8b": {
  "modelID": "deepseek-r1-8b",
  "name": "DeepSeek-R1-0528-Qwen3-8B, chat only (vLLM)",
  "capabilities": { "tools": false, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 65536, "output": 16384 }
}
```

Caveats and open issues:

- [#59738](https://github.com/vllm-project/vllm/issues/59738): streamed `deepseek_r1` reasoning deltas are not detokenized, so streamed reasoning can show byte-level glyphs.
- The official chat template never renders tools ([tokenizer_config.json](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/raw/main/tokenizer_config.json)), and Artificial Analysis measured tau2 at 0% ([Artificial Analysis](https://artificialanalysis.ai/models/deepseek-r1-qwen3-8b)).
- It is the newest DeepSeek checkpoint that fits 16 GB, not the only one: the January 2025 R1 distills are older and also lack tool calling ([model-selection.md](../docs/model-selection.md)).
- Reasoning is long (the card says the upgraded R1-0528 averages about 23K tokens per AIME question, against 12K for the previous version, [README](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/raw/main/README.md)), so a 64k window fills quickly.

## Gemma 4 12B IT QAT W4A16

Honorable mention: the vLLM-native Gemma, fully in VRAM at 128k ([model-selection.md](../docs/model-selection.md)).
The ranked Gemma (rank 5, 26B-A4B) does not fit vLLM; this is the fallback or sub-agent.

| Item | Value |
|---|---|
| Checkpoint | [`google/gemma-4-12B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct), official Google QAT, compressed-tensors int4 symmetric group 32, Apache-2.0 |
| Size | 10,264,229,896 B = 9.56 GiB ([HF API](https://huggingface.co/api/models/google/gemma-4-12B-it-qat-w4a16-ct?blobs=true)); it stores a 2.01 GB BF16 lm_head copy that vLLM ties to the embedding, so resident weights may be nearer 7.7 GiB (inferred from [gemma4.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/models/gemma4.py#L1447-L1454)) |
| Architecture | `Gemma4UnifiedForConditionalGeneration`, 11.96B dense, encoder-free; 48 layers: 40 sliding (window 1024, 8 KV heads x 256) and 8 global (1 KV head x 512, K=V); 262144 max positions ([config.json](https://huggingface.co/google/gemma-4-12B-it/raw/main/config.json)) |
| Minimum vLLM | v0.23.0 ([recipe](https://github.com/vllm-project/recipes/blob/main/models/Google/gemma-4-12B-it.yaml)); the recipe's 131072 context and "nightly required" note are stale |
| Kernel path on sm_120 | Marlin for W4A16 (Machete needs capability 9.0); attention is forced to `TRITON_ATTN` because FA4 does not cover capability 12 ([kernels/linear/\_\_init\_\_.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/kernels/linear/__init__.py), [models/config.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/models/config.py)) |
| Parsers | `gemma4` for tools and reasoning, always together ([tool parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/tool_parsers/__init__.py), [reasoning parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/reasoning/__init__.py)) |

Budget (estimate, file-size method):

| Step | Value |
|---|---|
| KV pool | 14.65 - 9.56 - 1.2 = 3.89 GiB |
| Global KV per token | 2 x 8 x 1 x 512 x 2 B = 16,384 B BF16, 8,192 B FP8 (vLLM caches K and V separately despite K=V) |
| Sliding reservation per sequence | Window plus 2 x 2048 in-flight tokens under async scheduling: 5136 tokens x 327,680 B = 1.57 GiB BF16, 0.78 GiB FP8 ([kv_cache_interface.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/kv_cache_interface.py#L918-L955), [#49878](https://github.com/vllm-project/vllm/issues/49878)) |
| 131072, BF16 KV | 2.0 + 1.57 = 3.57 GiB, about 0.32 GiB spare |
| 131072, FP8 KV | 1.0 + 0.78 = 1.78 GiB, about 2.1 GiB spare (recommended) |
| 262144, FP8 KV | 2.0 + 0.78 = 2.78 GiB, fits |

Each extra concurrent sequence costs another 0.78 GiB of sliding cache with FP8 KV, which is why `--max-num-seqs 1`.

```bash
docker run -d --rm --name vllm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN \
  -e VLLM_API_KEY \
  -e VLLM_NO_USAGE_STATS=1 \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v vllm-cache:/root/.cache/vllm \
  -v flashinfer-cache:/root/.cache/flashinfer \
  vllm/vllm-openai:v0.33.0 \
  google/gemma-4-12B-it-qat-w4a16-ct \
  --served-model-name gemma-4-12b \
  --max-model-len 131072 \
  --max-num-seqs 1 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":64}'
```

`TRITON_ATTN` is what Gemma 4 gets on sm_120 anyway; passing it keeps FP8 KV away from FlashInfer explicitly.
`--limit-mm-per-prompt` with zero images and audio skips multimodal profiling for text-only use ([README.md](README.md#flags-reference)).

Thinking and sampling:

- Toggle, off by default in the template (`enable_thinking` defaults to false), so no server default is needed; opt in with `chat_template_kwargs {"enable_thinking": true}` ([chat_template.jinja](https://huggingface.co/google/gemma-4-12B-it/raw/main/chat_template.jinja)).
- Card sampling temperature 1.0, top_p 0.95, top_k 64 ([README](https://huggingface.co/google/gemma-4-12B-it/raw/main/README.md)).

OpenCode entry (as in the validated config):

```json
"gemma-4-12b": {
  "modelID": "gemma-4-12b",
  "name": "Gemma 4 12B QAT W4A16 (vLLM)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 131072, "output": 16384 },
  "variants": [
    { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
    { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
  ]
}
```

Caveats and open issues:

- Open gemma4 parser bugs hit agent loops: dropped tool calls after the reasoning channel ([#54256](https://github.com/vllm-project/vllm/issues/54256)), a missing empty thought channel after a tool result with thinking off ([#66430](https://github.com/vllm-project/vllm/issues/66430)), empty `tool_calls` under load ([#50889](https://github.com/vllm-project/vllm/issues/50889)) and `tool_choice="required"` not enforced ([#53363](https://github.com/vllm-project/vllm/issues/53363)).
  Test with thinking on and off.
- [#53658](https://github.com/vllm-project/vllm/issues/53658): `TRITON_ATTN` illegal memory access when a CUDA-graph capture size exceeds `--max-model-len`, first hit with this model on an RTX 5090; it does not trigger at 131072 with default capture sizes.
- KV-cache CPU offloading is broken for Gemma 4 ([#42348](https://github.com/vllm-project/vllm/issues/42348)).
- Evidence: Terminal-Bench 2.1 27.3 and tau2-Telecom 36.3 (measured by [Artificial Analysis](https://artificialanalysis.ai/models/gemma-4-12b)), against a vendor tau2 average of 69.0 in a different setting ([README](https://huggingface.co/google/gemma-4-12B-it/raw/main/README.md), vendor-reported).
- Speed: about 110 tok/s decode ceiling at short context (estimate from about 8.1 GB read per token at 896 GB/s); nothing measured.

## Qwen3.5-9B NVFP4

Honorable mention: the conservative vLLM-native 9B if every vendor-only small model is rejected ([model-selection.md](../docs/model-selection.md)).

| Item | Value |
|---|---|
| Base model | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B), Apache-2.0, released February 2026 |
| Checkpoint | [`RedHatAI/Qwen3.5-9B-NVFP4`](https://huggingface.co/RedHatAI/Qwen3.5-9B-NVFP4), 8.72 GiB file ([HF API](https://huggingface.co/api/models/RedHatAI/Qwen3.5-9B-NVFP4?blobs=true)); 7.42 GiB with `--language-model-only` (minus the 0.85 GiB vision tower and 0.45 GiB MTP head, read from the safetensors header) |
| Architecture | 32 layers, 8 full-attention (4 KV heads, head dim 256) and 24 Gated DeltaNet, 262144 max positions ([config.json](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json)) |
| Kernel path on sm_120 | FlashInfer CUTLASS NVFP4 GEMM, which JIT-compiles on first start and can take about 52 GiB of host RAM without `MAX_JOBS=2` ([#46268](https://github.com/vllm-project/vllm/issues/46268), [PR #66864](https://github.com/vllm-project/vllm/pull/66864)) |
| Parsers | `qwen3_coder` and `qwen3` ([tool parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/tool_parsers/__init__.py), [reasoning parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/reasoning/__init__.py)) |

Budget (estimate): 14.65 - 7.42 - 1.2 = 6.03 GiB, minus about 49 MiB of GDN state per sequence.
KV per token is 2 x 8 x 4 x 256 x 2 B = 32,768 B BF16, so the pool holds about 197k tokens BF16 and the 262144 cap with FP8.
131072 with BF16 KV uses 4.0 GiB and keeps attention on FlashAttention 2.

```bash
docker run -d --rm --name vllm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN \
  -e VLLM_API_KEY \
  -e VLLM_NO_USAGE_STATS=1 \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v vllm-cache:/root/.cache/vllm \
  -v flashinfer-cache:/root/.cache/flashinfer \
  vllm/vllm-openai:v0.33.0 \
  RedHatAI/Qwen3.5-9B-NVFP4 \
  --served-model-name qwen3.5-9b \
  --language-model-only \
  --max-model-len 131072 \
  --max-num-seqs 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'
```

`MAX_JOBS=2` is load-bearing for this model: the first start compiles for about 8.5 minutes at about 14-16 GiB of host RAM ([#46268](https://github.com/vllm-project/vllm/issues/46268)), and the `flashinfer-cache` volume keeps the result.
`--linear-backend cutlass` might avoid the JIT, but that is inferred and untested.

Thinking and sampling:

- Toggle, on by default; `enable_thinking: false` per request or server-wide, and the `/think` and `/nothink` switches are not supported ([README](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/README.md)).
- Coding preset temperature 0.6, top_p 0.95, top_k 20, presence 0 (used above).

OpenCode entry:

```json
"qwen3.5-9b": {
  "modelID": "qwen3.5-9b",
  "name": "Qwen3.5-9B NVFP4 (vLLM)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 131072, "output": 16384 },
  "variants": [
    { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
    { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
  ]
}
```

Caveats and open issues:

- RedHatAI validated the NVFP4 build only on a B200 with a vLLM nightly, not on sm_120 ([model card](https://huggingface.co/RedHatAI/Qwen3.5-9B-NVFP4)).
- Hybrid GDN prefix caching can silently fall to 0% hits ([#45238](https://github.com/vllm-project/vllm/issues/45238)), and agent traces recompute 1.6-1.7x more prompt than on a full-attention model ([#67674](https://github.com/vllm-project/vllm/issues/67674)).
- Qwen parser issues: [#65774](https://github.com/vllm-project/vllm/issues/65774), [#61206](https://github.com/vllm-project/vllm/issues/61206), [#48753](https://github.com/vllm-project/vllm/issues/48753), and [#38696](https://github.com/vllm-project/vllm/issues/38696) for `json_schema` output.
- Evidence is modest and second-hand: SWE-bench Verified 53.2 and Terminal-Bench 2.1 21.3 come from the Ornith card's comparison table ([Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B)), another vendor's harness.

## Mellum2.1 12B-A2.5B Thinking AWQ-INT4

Honorable mention: a fast, fully in-VRAM JetBrains sub-agent with 128k on either runtime, vendor-only and weaker than Qwen3.5-9B in its own table ([model-selection.md](../docs/model-selection.md)).
The ranking's preferred file is the official GGUF on llama.cpp ([llamacpp/models.md](../llamacpp/models.md)).

| Item | Value |
|---|---|
| Base model | [`JetBrains/Mellum2.1-12B-A2.5B-Thinking`](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking), Apache-2.0, announced 2026-10-08; 28 layers, 7 full-attention (4 KV heads x 128) and 21 sliding (window 1024) |
| Checkpoint | [`cyankiwi/Mellum2.1-12B-A2.5B-Thinking-AWQ-INT4`](https://huggingface.co/cyankiwi/Mellum2.1-12B-A2.5B-Thinking-AWQ-INT4), 7,677,168,648 B = 7.15 GiB ([HF API](https://huggingface.co/api/models/cyankiwi/Mellum2.1-12B-A2.5B-Thinking-AWQ-INT4?blobs=true)) |
| vLLM support | Through the Transformers modeling backend (`MellumForCausalLM` maps to `TransformersMoEForCausalLM`, [registry.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/models/registry.py)); the transformers version inside the image was not read |
| Parsers | `--tool-call-parser hermes`, `--reasoning-parser qwen3`, as on the card ([README](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking/raw/main/README.md)) |

Budget (estimate): 14.65 - 7.15 - 1.2 = 6.30 GiB of pool.
KV per token is 2 x 7 x 4 x 128 x 2 B = 14,336 B BF16, plus about 42 MiB of sliding cache per sequence, so 131072 needs about 1.79 GiB and the model's cap, not memory, sets the context.

```bash
docker run -d --rm --name vllm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN \
  -e VLLM_API_KEY \
  -e VLLM_NO_USAGE_STATS=1 \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v vllm-cache:/root/.cache/vllm \
  -v flashinfer-cache:/root/.cache/flashinfer \
  vllm/vllm-openai:v0.33.0 \
  cyankiwi/Mellum2.1-12B-A2.5B-Thinking-AWQ-INT4 \
  --served-model-name mellum2.1 \
  --max-model-len 131072 \
  --max-num-seqs 1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'
```

Thinking is left on by default here, unlike the other toggle models: every vendor evaluation ran in thinking mode, and non-thinking quality is undocumented ([README](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking/raw/main/README.md)).
`enable_thinking: false` appends an empty think block, so the `#none` variant works ([chat_template.jinja](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking/raw/main/chat_template.jinja)).
Sampling follows the card quickstart (temperature 0.6, top_p 0.95, top_k 20); `generation_config.json` carries no sampling values, so the override sets them.

OpenCode entry:

```json
"mellum2.1": {
  "modelID": "mellum2.1",
  "name": "Mellum2.1 12B-A2.5B Thinking AWQ-INT4 (vLLM)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 131072, "output": 16384 },
  "variants": [
    { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
    { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
  ]
}
```

Caveats: no test of a quantized community build on the Transformers MoE backend was found, and the vendor's own table puts it below Qwen3.5-9B (SWE-bench Verified 47 vs 50, Terminal-Bench 2.1 17.4 vs 21.7, vendor-reported, [README](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking/raw/main/README.md)).

## Smoke-test model: Gemma 4 E4B IT

Use this first, before any ranked model, to prove the whole chain: Container Toolkit with CDI, image pull, HF cache, vLLM startup on sm_120, tool and reasoning parsers, and the OpenCode provider.
It is excluded as a daily agent: vendor tau2 is 42.2% and there are no SWE-bench or Terminal-Bench numbers from Google ([README](https://huggingface.co/google/gemma-4-E4B-it/raw/main/README.md), vendor-reported).

| Item | Value |
|---|---|
| Checkpoint | [`google/gemma-4-E4B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-E4B-it-qat-w4a16-ct), official QAT W4A16, Apache-2.0 |
| Size | 11,513,861,828 B = 10.72 GiB ([HF API](https://huggingface.co/api/models/google/gemma-4-E4B-it-qat-w4a16-ct?blobs=true)); 9.83 GiB with `--language-model-only` (minus 0.32 GiB vision and 0.58 GiB audio, read from the safetensors header); the 5.25 GiB per-layer-embedding table stays on the GPU |
| Architecture | 42 layers; only layers 0-23 own KV (4 full-attention, 2 KV heads x 512), the rest share it; 131072 max positions ([config.json](https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json)) |
| Budget (estimate) | 14.65 - 9.83 - 1.2 = 3.62 GiB pool; 16,384 B per token BF16, so 65536 uses 1.0 GiB and 131072 uses 2.0 GiB, plus about 100 MiB of sliding cache per sequence ([#49878](https://github.com/vllm-project/vllm/issues/49878)) |
| BF16 checkpoint | 14.0 GiB text-only, does not fit |

```bash
docker run -d --rm --name vllm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN \
  -e VLLM_API_KEY \
  -e VLLM_NO_USAGE_STATS=1 \
  -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v vllm-cache:/root/.cache/vllm \
  -v flashinfer-cache:/root/.cache/flashinfer \
  vllm/vllm-openai:v0.33.0 \
  google/gemma-4-E4B-it-qat-w4a16-ct \
  --served-model-name gemma-4-e4b \
  --language-model-only \
  --max-model-len 65536 \
  --max-num-seqs 2 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":64}'
```

BF16 KV keeps the run simple (attention is on Triton for Gemma 4 on sm_120 regardless).
Then run the [README smoke tests](README.md#smoke-tests) with `"model":"gemma-4-e4b"`: `/v1/models`, "Reply with exactly: OK", and the `get_weather` tool call, streamed and non-streamed.
A pass proves the plumbing, not the quality of any other model.

OpenCode entry for the first OpenCode run:

```json
"gemma-4-e4b": {
  "modelID": "gemma-4-e4b",
  "name": "Gemma 4 E4B QAT W4A16 smoke test (vLLM)",
  "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
  "limit": { "context": 65536, "output": 16384 }
}
```

Known issues: the same gemma4 parser bugs as the 12B, plus [#39468](https://github.com/vllm-project/vllm/issues/39468) and [#39392](https://github.com/vllm-project/vllm/issues/39392); [#44788](https://github.com/vllm-project/vllm/issues/44788) (crash on KV-shared checkpoints without `k_norm`) looks handled in the v0.33.0 code but is untested.

## Ornith-1.5-9B (excluded)

Earlier drafts of this knowledge base, including the worked example in [README.md](README.md#memory-budget), used `ornith-ai/Ornith-1.5-9B-NVFP4` (8.21 GiB file, 6.91 GiB with `--language-model-only`, [HF API](https://huggingface.co/api/models/ornith-ai/Ornith-1.5-9B-NVFP4?blobs=true)) as the vLLM primary.
The 2026-11-15 re-ranking gave it no votes: every score comes from an anonymous, unverified organisation, and the hybrid-GDN NVFP4 crash [#54331](https://github.com/vllm-project/vllm/issues/54331) applies ([model-selection.md](../docs/model-selection.md)).
Groot and K2 Horizon 7B now hold the fast in-VRAM slot, and Qwen3.5-9B above is the conservative swap with the same architecture and parsers.
This exclusion covers Ornith-1.5-9B only: the user's LibreDB Studio harness passed Ornith-1.0-9B (Ollama repo `ornith`, tag `ornith:9b`) 30/30 (measured on the M5 Max, libredb-studio; [agent-harness-evidence.md](../docs/agent-harness-evidence.md)), a different model and runtime that has not been judged here.
If you still want to try it, use the Qwen3.5-9B command with that repo id, `--trust-remote-code` and `--tool-call-parser qwen3_xml` as on its card ([README](https://huggingface.co/ornith-ai/Ornith-1.5-9B/raw/main/README.md)), and FP8 KV with `TRITON_ATTN` for contexts above about 212k (estimate).

## Models that do not fit in vLLM

None of these leaves a 64k KV pool in 14.65 GiB, and vLLM v0.33.0 cannot rescue MoE models by offloading experts: `--cpu-offload-gb` with `--cpu-offload-params experts` is silently undone when Marlin repacks the weights, and loading runs out of memory ([#66264](https://github.com/vllm-project/vllm/issues/66264)).
Run them on llama.cpp, where `--n-cpu-moe` works ([llamacpp/models.md](../llamacpp/models.md)).

| Model | Status | Smallest vLLM checkpoint | Size (GiB) | Why it does not fit (estimate) | Instead |
|---|---|---|---|---|---|
| Qwen3.8-27B | Rank 1 | [grevinden/qwen3.8-27b-w4a16-autoround-rtx3090](https://huggingface.co/api/models/grevinden/qwen3.8-27b-w4a16-autoround-rtx3090?blobs=true) | 15.59 | Weights alone exceed the 14.65 GiB budget; the next smallest, [RedHatAI INT4](https://huggingface.co/api/models/RedHatAI/Qwen3.8-27B-INT4?blobs=true), is 18.12 | llama.cpp UD-Q3_K_XL at 65536 |
| Qwen3.6-35B-A3B | Rank 2 | [Intel int4-mixed AutoRound](https://huggingface.co/api/models/Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound?blobs=true) | 18.75 language model only (20.02 file) | Over budget, and expert offload is undone ([#66264](https://github.com/vllm-project/vllm/issues/66264)) | llama.cpp Q4_K_M with `--n-cpu-moe 21` |
| K2 Horizon MoVA 36B A4B | Rank 3 | [Siladrim GPTQ-Int4](https://huggingface.co/api/models/Siladrim/K2-Horizon-MoVA-36B-A4B-GPTQ-Int4?blobs=true) | 20.25 | Over budget; vLLM's MoVA value-expert path also takes no quant flags, so a build that quantizes `v_experts` is likely to fail (inferred from [k2_horizon.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/models/k2_horizon.py)) | llama.cpp Q4_K_M with `--n-cpu-moe 46` |
| Gemma 4 26B-A4B IT | Rank 5 | [Intel int4 AutoRound](https://huggingface.co/api/models/Intel/gemma-4-26B-A4B-it-int4-AutoRound?blobs=true) | 13.24 language model only (14.31 file) | Loads, but the pool is about 0.21 GiB: about 11.8k tokens with FP8 KV | llama.cpp official QAT Q4_0 at 65536 |
| Qwen3-Coder-Next | Rank 6 | [Intel int4 AutoRound](https://huggingface.co/api/models/Intel/Qwen3-Coder-Next-int4-AutoRound?blobs=true) | 40.53 | Far over budget | llama.cpp UD-IQ4_XS with `--n-cpu-moe 38` and a 1,504 MiB expert cache (measured on 2026-11-16, [llama.cpp models](../llamacpp/models.md#6-qwen3-coder-next)) |
| Nemotron 3.5 Bolt 74B-A9B | Rank 7 | [nvidia 24GB build](https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB?blobs=true) | 22.43 | Over budget, and its IQ2_XXS/Q8_0 blocks need the b12x kernels, which the image does not install ([b12x.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/features/quantization/b12x.md)) | llama.cpp GGUF with `--n-cpu-moe 50` |
| Nanbeige4.3-33B | Honorable (trial) | [Tech2wild W4A16 AutoRound](https://huggingface.co/api/models/Tech2wild/Nanbeige4.3-33B-W4A16-AutoRound?blobs=true) | 19.34 | Dense 33B over budget | llama.cpp IQ2_M trial |
| Nemotron 3.5 Lightning 30B-A3B | Honorable | [nvidia NVFP4](https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4?blobs=true) | 20.08 (community W4A16 16.60) | Over budget | llama.cpp Q4_0 with `--n-cpu-moe 16` |
| GLM-4.7-Flash | Honorable | [Intel int4 AutoRound](https://huggingface.co/api/models/Intel/GLM-4.7-Flash-int4-AutoRound?blobs=true) | 16.06 | Over budget | llama.cpp Q4_K_M with `--n-cpu-moe 20` |
| LFM2.5-24B-A2B | Excluded | [myroslavtryhubets NVFP4A16](https://huggingface.co/api/models/myroslavtryhubets/LFM2.5-24B-A2B-NVFP4A16?blobs=true) | 12.92 | Loads, but the pool is about 0.53 GiB: about 28k BF16 or 56k FP8 tokens at 0.92 (20 KiB per token BF16); 64k only at util 0.95 with FP8 | Excluded for quality and licence |
| Devstral Small 2 24B | Excluded | [androiddrew AWQ-4bit](https://huggingface.co/api/models/androiddrew/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit?blobs=true) | 13.99 (13.17 language model only) | About 3.6k tokens with FP8 KV | Excluded |
| Xing4.0-29B-A4B | Excluded | [official FP8](https://huggingface.co/api/models/XingChen-AGI/Xing4.0-29B-A4B-FP8?blobs=true) | 30.89 | Far over budget, and the architecture is not in-tree in v0.33.0 (out-of-tree plugin only) | Excluded |
| Ternary Bonsai 2 27B | Excluded | None | n/a | No vLLM-loadable checkpoint or architecture | Excluded |
| gpt-oss-120b | Excluded | [openai/gpt-oss-120b](https://huggingface.co/api/models/openai/gpt-oss-120b?blobs=true) | 60.77 (top-level shards) | Far over budget, and even llama.cpp needs about 49-52 GiB of RAM | Excluded |
| DeepSeek V4-Flash | Excluded | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash?blobs=true) | 159.6 GB | Far above VRAM plus free RAM | Excluded |

The reasons for each exclusion, and the full ranking, are in [model-selection.md](../docs/model-selection.md).

## What to measure first

1. Gemma 4 E4B: the smoke tests pass and the OpenCode `get_weather` and edit tests work end to end.
2. gpt-oss-20b at util 0.95: the startup "KV cache size" line confirms 65536 fits, and the MARLIN and Triton backends appear in the log.
3. Groot NVFP4: real KV capacity, decode tok/s, a 30-minute OpenCode session without the [#54331](https://github.com/vllm-project/vllm/issues/54331) crash, and an A/B against Qwen3.6 on llama.cpp with verbatim path checks.
4. K2 Horizon 7B: whether the AWQ build behaves like the llama.cpp Q8_0 on the same OpenCode tasks.
5. For every model, record the numbers with the template in [benchmarking.md](benchmarking.md).

## Sources

- vLLM v0.33.0 source: [registry.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/models/registry.py), [tool parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/tool_parsers/__init__.py), [reasoning parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/reasoning/__init__.py), [cli_args.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/common/cli_args.py), [model.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/model.py), [mxfp4.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py), [nvfp4.py](https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py), [kv_cache_interface.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/kv_cache_interface.py)
- vLLM recipes at v0.33.0: [gpt-oss-20b](https://github.com/vllm-project/vllm/blob/v0.33.0/recipes/models/openai/gpt-oss-20b.yaml), [K2-Horizon-7B](https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/recipes/models/IFM/K2-Horizon-7B.yaml); [Gemma 4 12B recipe](https://github.com/vllm-project/recipes/blob/main/models/Google/gemma-4-12B-it.yaml)
- vLLM issues and PRs cited above, state read on 2026-11-15: [#66264](https://github.com/vllm-project/vllm/issues/66264), [#61239](https://github.com/vllm-project/vllm/issues/61239), [#54331](https://github.com/vllm-project/vllm/issues/54331), [#46268](https://github.com/vllm-project/vllm/issues/46268), [#62743](https://github.com/vllm-project/vllm/issues/62743), [#65770](https://github.com/vllm-project/vllm/issues/65770), [#66989](https://github.com/vllm-project/vllm/issues/66989), [#49878](https://github.com/vllm-project/vllm/issues/49878), [#45238](https://github.com/vllm-project/vllm/issues/45238), [#59738](https://github.com/vllm-project/vllm/issues/59738)
- Model cards and HF API listings: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), [GrootTeam/Groot-19B-A3B-261023](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023), [IFM/K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B), [deepseek-ai/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B), [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), [RedHatAI/Qwen3.5-9B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.5-9B-NVFP4), [JetBrains/Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking); sizes rechecked through the HF API on 2026-11-15
- Independent scores: [Artificial Analysis](https://artificialanalysis.ai/models/gpt-oss-20b) model pages, NVIDIA's same-harness table in the [Nemotron 3.5 Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/README.md), [Blackwall](https://guitaripod.github.io/blackwall/#groot-19b-a3b-nvfp4~medium)
- llama.cpp desktop RTX 5080 gpt-oss numbers: [ggml-org discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396)
- OpenCode config schema: [https://opencode.ai/config.json](https://opencode.ai/config.json)
- The user's DGX Spark runbook (private), for the repository-not-flag and thinking-off lessons
- Full catalog with evidence classes: [sources.md](../docs/sources.md)

## Maintenance

Re-verify before relying on this page:

- The vLLM release and image digest; the v0.34.0rc4 tag drops the fixed 0.92 `gpu_memory_utilization` default ([cache.py at v0.34.0rc4](https://github.com/vllm-project/vllm/blob/v0.34.0rc4/vllm/config/cache.py)), which invalidates every budget table here ([README.md](README.md#maintenance)).
- [#66264](https://github.com/vllm-project/vllm/issues/66264): a fix would reopen vLLM for Qwen3.6-35B-A3B and other MoE models with expert offload.
- [#61239](https://github.com/vllm-project/vllm/issues/61239) (drop `TRITON_ATTN` once the fix ships), [#54331](https://github.com/vllm-project/vllm/issues/54331), [#46268](https://github.com/vllm-project/vllm/issues/46268), [#62743](https://github.com/vllm-project/vllm/issues/62743) and [#65770](https://github.com/vllm-project/vllm/issues/65770).
- Checkpoint sizes and upload dates, especially whether an FTPO-era AWQ build of K2 Horizon 7B or an official Groot LICENSE file appears.
- The ranking in [model-selection.md](../docs/model-selection.md), which decides which cards stay here.
- After the first real runs, replace every "(estimate)" context and speed figure with the measured value and its date.
