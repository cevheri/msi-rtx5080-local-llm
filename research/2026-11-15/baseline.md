# Verified baseline (as of 2026-11-15)

This file is the verified baseline from the research phase.
Every item was checked against primary sources or observed on the machine.
"Estimate" means arithmetic from config.json and real file sizes; nothing has been run on this machine yet.
Raw research records: scratchpad/disc/*.json (discovery lanes) and scratchpad/ver/*.json (per-model verification, adversarial checks, platform resolvers, judges, synthesis.json, audit.json).

## Machine (observed with nvidia-smi, lscpu, free, lsblk, docker info)

- GPU: NVIDIA GeForce RTX 5080 Laptop GPU, Blackwell GB203M, compute capability 12.0 (sm_120), 16303 MiB GDDR7, 256-bit, memory clock 14001 MHz, power limit 140 W (default 80 W, max 175 W), PCIe 5.0 x16.
  NVIDIA spec page: 896 GB/s, 1334 AI TOPS, 80-150 W.
- Desktop RTX 5080 is a different card (960 GB/s, 360 W): its benchmark numbers are not laptop numbers.
- Hybrid graphics, PRIME on-demand; the Intel iGPU drives the desktop; about 112 MiB VRAM used at idle.
- Driver 595.99.02 (nvidia-driver-595-open), CUDA driver API 13.2.
  No CUDA toolkit on the host.
- CPU Intel Core Ultra 9 290HX Plus, 24 cores / 24 threads; Intel spec DDR5-6400 dual channel, 102.4 GB/s peak (configured DIMM speed unverified).
- RAM 61 GiB usable; desktop uses about 20 GiB, so about 40 GiB is free for models.
- Disk 2 TB Samsung NVMe (405 GB free on /, 562 GB free on /home); Docker root on /.
- Ubuntu 26.04.1 LTS, kernel 7.0.0-43.
  Docker 29.9.3, Compose v5.7.1, cgroup v2, systemd driver.
  ufw is active and enabled.
- NVIDIA Container Toolkit not installed yet.
  OpenCode v2.3.0 installed.
  No Ollama, no llama.cpp on the host.
- The user's `heavy` wrapper (systemd-run --user --scope, heavy.slice MemoryMax=32G) does not cap Docker containers; containers need --memory.

## vLLM

- Stable v0.33.0 (GitHub release named v0.33.0, tag v0.33.0 = commit 4ba787e, 2026-11-02).
  Image vllm/vllm-openai:v0.33.0 = :latest, digest sha256:d825baf9ce4b, 9.74 GB compressed amd64; CUDA 13.0.3, PyTorch 2.14.0, TORCH_CUDA_ARCH_LIST includes 12.0.
  v0.34.0rc4 (2026-11-14) is a pre-release.
- Default --gpu-memory-utilization 0.92 (14.65 GiB of 16303 MiB).
  Budget method: 14.65 GiB minus weights minus 1.2 GiB reserve = KV pool.
- Default frontend is Python; the Rust frontend breaks gpt-oss parser flags (#66989).
- With tools in a request and no tool_choice, v0.33.0 sets tool_choice "auto" (protocol.py check_tool_usage).
  Empty tools [] returns HTTP 400.
- Prefix caching is on by default (cache.py enable_prefix_caching=True); hybrid Mamba/GDN models switch to mamba_cache_mode "align".
- FP8 KV moves attention from FlashAttention 2 to FlashInfer or Triton on sm_120; FlashInfer race #61239 (fix merged after v0.33.0), so use --attention-backend TRITON_ATTN when FP8 KV is needed.
- MXFP4 MoE (gpt-oss) uses MARLIN on sm_120.
  NVFP4 W4A4 uses FlashInfer CUTLASS; first start JIT can use about 52 GiB host RAM (#46268; fix PR #66864 open); workaround MAX_JOBS=2 plus persistent caches.
  NVFP4 W4A16 builds use Marlin and avoid it.
  Hybrid GDN NVFP4 crash under load with CUDA graphs on sm_120 (#54331).
- MoE expert CPU offload (--cpu-offload-gb with --cpu-offload-params experts) is silently undone for Marlin-repacked weights (#66264): no vLLM offload plans.
- GGUF and bitsandbytes are out-of-tree plugins, not in the image.
  b12x kernels not installed in the image.
- Parsers: Qwen3.5/3.6/3.8 and Qwen3-Coder(-Next): tool qwen3_coder (Qwen3.6 model card) or qwen3_xml (vLLM docs list it for Qwen3-Coder; worked on the user's DGX with Qwen3.6), reasoning qwen3.
  Gemma 4: gemma4 / gemma4.
  gpt-oss: openai / openai_gptoss.
  Nemotron 3.5: qwen3_coder / nemotron_v3.
  GLM-4.7: glm47 / glm45.
  Mistral: mistral.
  DeepSeek R1: reasoning deepseek_r1, no tool calling.
  Reasoning text is returned in the "reasoning" field.
- Thinking default off for all callers: --default-chat-template-kwargs '{"enable_thinking":false}'; per request chat_template_kwargs.
- NGC images nvcr.io/nvidia/vllm: 26.05 CUDA 13.2.1 vLLM 0.20.1; 26.08 CUDA 13.4.1 vLLM 0.27.1; 26.09 CUDA 13.4.1 vLLM 0.29.0; 26.10 CUDA 13.4.3 vLLM 0.31.0 (registry image config, read by the lead).

## llama.cpp

- Latest build b12418 (2026-11-15).
  Newest image tags: ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 (CUDA 12.8.1, NVIDIA_REQUIRE_CUDA cuda>=12.8) and server-cuda13-b12407 (CUDA 13.5.0, cuda>=13.5, no GeForce brand clause: likely refused on driver 595 / CUDA 13.2).
  No b12418 image tag.
- Laptop CUDA-graph GPU hang on RTX 50 Laptop (#27330, reproduced on 595.99.02 / Ubuntu 26.04): pass docker -e GGML_CUDA_DISABLE_GRAPHS=1 before the image name (after it, llama-server parses -e as --escape and fails).
- Flags at b12418: -np auto gives 4 slots (use -np 1), -fa auto, --jinja default, --fit on (places experts automatically when -ngl/-ncmoe unset), --n-cpu-moe N counts block indices from 0 (not MoE layers), --cpu-moe, --moe-cache-mib (VRAM cache for host experts), -ctk/-ctv q8_0, --no-mmap and --mlock removed (use -lm/--load-mode), --reasoning on|off|auto, --reasoning-budget N, --chat-template-kwargs JSON, --cache-prompt default on, --ctx-checkpoints default 32.
- -hf downloads into HF hub layout (LLAMA_CACHE, HF_HUB_CACHE, HF_HOME/hub, ~/.cache/huggingface/hub), so vLLM and llama.cpp can share one cache.
- Open issues: #30357 (Qwen3.5+ hybrid models: reasoning/whitespace not round-trippable, prefix-cache invalidation, full re-prefill), #30118 (Qwen3.6 tool call leaks into content), #29655 (Gemma 4 multi-line streaming tool args), #28581 (IQ3_S garbage with nvcc 13.2 builds), #27623 (q8_0 KV decode penalty, cliff in title retracted), #25859 (serial expert copies slow prefill).
  OpenCode '^ses' schema issue fixed by llama.cpp PR #26939.

## Ollama

- v0.40.16 (2026-11-09); docs list RTX 5080 under compute capability 12.0.
  Default context 4k on GPUs under 24 GiB; OpenCode needs 64k or more, so OLLAMA_CONTEXT_LENGTH=65536, OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE=q8_0.
  /v1 ignores tool_choice.
  Whole-layer offload is slower than llama.cpp --n-cpu-moe for MoE models that spill.

## Host

- NVIDIA Container Toolkit 1.20.1-1 (docs pin; 1.20.2-1 in repo); Ubuntu 26.04 is in the supported-platforms table.
  nvidia-ctk runtime configure --runtime=docker; CDI (--device nvidia.com/gpu=all) preferred because systemctl daemon-reload can drop GPUs from --gpus containers on systemd-cgroup hosts.
- Compose: gpus: all (Compose 2.30+), devices: ["nvidia.com/gpu=all"] (CDI), or deploy.resources.reservations.devices.
- Docker-published ports bypass ufw: publish on 127.0.0.1 only.
  Docker 29 is not affected by the pre-28.0 L2 issue.
- A running CUDA app keeps the dGPU awake (RTD3 cannot power it down): no restart policy on the laptop; start on demand.

## OpenCode v2.3.0

- Config keys: providers, package "@opencode/ai/providers/openai-compatible", settings.baseURL/apiKey/headerTimeout, models.<id>.modelID, capabilities.tools, limit.context/output, variants (body.chat_template_kwargs), compatibility.reasoningField ("reasoning" for vLLM, "reasoning_content" for llama.cpp).
- Built-in vLLM discovery sets tools false (local.ts: tools: spec.tools ?? false): use a custom provider and plugins ["-opencode.provider.vllm"].
- A two-provider config (llama.cpp :8080 Qwen3.6, vLLM :8000 Gemma 4 12B and gpt-oss-20b) validated against https://opencode.ai/config.json with 0 errors.
- Practical minimum context 64k (Ollama doc); fixed prompt overhead estimated 5.5k-7k tokens from source character counts.
- Open issues: #55378 (reasoning-only turn ends silently), #50814 (tool calls as text after compaction).
  #60202 (tool_choice omitted) does not bite vLLM v0.33.0.

## Model picks (Borda of three judges: quality, practicality, coverage)

| # | Model | Runtime and file | Size | Context (estimate) | Thinking | Tools |
|---|---|---|---|---|---|---|
| 1 | Qwen3.6-35B-A3B (HF 2026-04-15, Apache-2.0) | llama.cpp ggml-org Q4_K_M, --n-cpu-moe 21 | 19.13 GiB, about 9 GiB experts in RAM | 128k | toggle | Yes |
| 2 | Qwen3.8-27B (HF 2026-08-05, Apache-2.0) | llama.cpp unsloth UD-Q3_K_XL, all on GPU | 12.24 GiB | 64k q8_0 KV | toggle + reasoning_effort | Yes |
| 3 | Gemma 4 26B-A4B (HF 2026-03-11, Apache-2.0) | llama.cpp Google QAT Q4_0, all on GPU | 13.45 GiB | 64k q8_0 KV | toggle, default off | Yes |
| 4 | gpt-oss-20b (HF 2025-08-04, Apache-2.0) | llama.cpp ggml-org MXFP4 (vLLM second) | 11.28 GiB GGUF, 12.82 GiB safetensors | 128k llama.cpp, 64k vLLM (util 0.95 + FP8 KV + TRITON_ATTN) | always, low/medium/high | Yes |
| 5 | Qwen3-Coder-Next (HF 2026-01-30, Apache-2.0) | llama.cpp unsloth UD-IQ4_XS, --n-cpu-moe 37 | 35.79 GiB, about 26 GiB in RAM | 128k | no | Yes |
| 6 | Nemotron 3.5 Lightning 30B-A3B (HF 2026-08-01, OpenMDW-1.1) | llama.cpp ggml-org Q4_0, --n-cpu-moe 16 | 17.60 GiB, about 5 GiB in RAM | 128k-256k | toggle | Yes |
| 7 | Ornith-1.5-9B (HF 2026-08-18, MIT, unverified org) | vLLM NVFP4 W4A16, --language-model-only | 6.91 GiB | 128k | toggle | Yes |
| 8 | Gemma 4 12B (HF 2026-05-23, Apache-2.0) | vLLM Google QAT W4A16 | 9.56 GiB | 128k FP8 KV | toggle, default off | Yes |
| 9 | GLM-4.7-Flash (HF 2026-01-19, MIT) | llama.cpp unsloth Q4_K_M, --n-cpu-moe 20 | 17.05 GiB | 64k | toggle | Yes |
| 10 | DeepSeek-R1-0528-Qwen3-8B (HF 2025-05-29, MIT) | llama.cpp Q8_0 or Ollama | 8.11 GiB | 64k q8_0 KV | always | No (chat only) |

- Honorable: KAT-Coder-V2.5-Dev, Laguna XS 2.1, Qwen3.5-9B (RedHatAI NVFP4 7.42 GiB, vLLM).
- Excluded: gpt-oss-120b (RAM), Gemma 4 31B, Muse Glimmer 30B, Nemotron 3 Super, DeepSeek V4-Flash, Mistral Small 4, Ripple-65B, Qwen3-Coder-30B-A3B (older, weaker), Devstral Small 2 (about 32k context only), Granite 4.2, Ministral 3, Nemotron Nano 9B v2, Qwen3.5-27B / Qwen3.6-27B (superseded by Qwen3.8-27B), Flint / Ornith-35B / North Mini Code (unverified).
- Key evidence: NVIDIA same-harness table in the Nemotron 3.5 Lightning card (SWE-bench Verified / Terminal-Bench 2.1): Qwen3.6-35B-A3B 70.12 / 44.38, Gemma 4 26B-A4B 57.40 / 37.22, gpt-oss-20b 52.44 / 15.17, Nemotron Lightning 51.56 / 24.58.
  Artificial Analysis: Qwen3.8-27B TB2.1 79.8 (xhigh) / 65.2 (medium) / 67.4 (low) / 49.1 (non-reasoning), TB 4.0 5.6; Qwen3-Coder-Next tau2 79.5, TB2.1 38.2; Gemma 4 12B TB2.1 27.3; DeepSeek R1 8B tau2 0.
- Measured anchor: the user's DGX Spark (GB10, 273 GB/s unified) served RedHatAI/Qwen3.6-35B-A3B-NVFP4 at 42.7 tok/s single stream (2026-09-07).
  Desktop RTX 5080 llama.cpp gpt-oss-20b about 205 tok/s tg128 (ggml-org guide).

## Lessons from the user's DGX Spark runbook

- Thinking default off measured: Qwen3.6 with thinking on used all 1024 tokens (23.7 s, no answer); off answered in 1.5 s.
  Thinking still improves agent quality (Qwen3.8 TB2.1 49.1 off vs 67.4 low), so compare both in measurement.
- Wrong tool parser does not error; tool calls arrive as plain text.
- Model support ships inside the vLLM version; check the card's minimum version first.
- Download through the container into the shared cache; check for *.incomplete files.
- Health checks: /v1/models, "Reply with exactly: OK", plus a tool-call smoke test.
