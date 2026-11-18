# vLLM on the RTX 5080 Laptop

Last verified: 2026-11-18.

Use vLLM here only for models whose whole checkpoint fits in about 14.65 GiB of VRAM with room left for the KV cache: from the ranked picks, gpt-oss-20b as a 64k-context option, Groot-19B-A3B-261023 NVFP4 (W4A16), and the weaker K2 Horizon 7B AWQ-INT4 path, plus Gemma 4 12B QAT W4A16 as the vLLM-native honorable mention ([model-selection.md](../docs/model-selection.md)).
DeepSeek-R1-0528-Qwen3-8B (chat only) runs on vLLM v0.33.0 only through llmctl's tokenizer overlay, because the image loads the model's own tokenizer config with the wrong class and garbles every reply without it ([models.md](models.md#deepseek-r1-0528-qwen3-8b-chat-only)); llama.cpp stays its primary path.
Every larger model (Qwen3.8-27B, Qwen3.6-35B-A3B, K2 Horizon MoVA, Gemma 4 26B-A4B, Qwen3-Coder-Next, Nemotron 3.5 Bolt) runs on llama.cpp, because none fits the budget and vLLM v0.33.0 silently undoes MoE expert CPU offload for Marlin-repacked weights ([#66264](https://github.com/vllm-project/vllm/issues/66264)) and llama.cpp's `--n-cpu-moe` works ([llama.cpp README](../llamacpp/README.md)).
The pinned image is `vllm/vllm-openai:v0.33.0`, started on demand with CDI GPU access, a localhost-only port and a hard RAM cap.
gpt-oss-20b ran on this laptop on 2026-11-16 ([attempts](../research/2026-11-16/vllm-gpt-oss-20b/attempts.md)) and the seven other recipes on 2026-11-17 ([records](../research/2026-11-17/README.md)), with the results in [benchmarking.md](benchmarking.md#results) and [models.md](models.md).
Every figure without a measured label is read from source code or an estimate.

Per-model commands live in [models.md](models.md), benchmarking in [benchmarking.md](benchmarking.md), the ranked picks in [model-selection.md](../docs/model-selection.md), host prerequisites in [host-setup.md](../docs/host-setup.md) and the OpenCode side in [opencode.md](../docs/opencode.md).

## When vLLM, when llama.cpp

| Situation | Runtime | Why |
|---|---|---|
| Dense or small MoE checkpoint at or below about 13 GiB, official or vendor 4-bit build exists | vLLM | Full model in VRAM, OpenAI-compatible server with tool and reasoning parsers, prefix caching on by default ([cache.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py)). |
| MoE checkpoint larger than the 14.65 GiB budget | llama.cpp | vLLM expert offload is undone for Marlin-repacked weights ([#66264](https://github.com/vllm-project/vllm/issues/66264)); the smallest Qwen3.6-35B-A3B vLLM build (Intel AutoRound int4, 18.75 GiB language-model-only) does not fit ([HF API](https://huggingface.co/api/models/Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound?blobs=true)). |
| gpt-oss-20b at 128k context | llama.cpp | vLLM reaches 65536 only with `--gpu-memory-utilization 0.965`, `--enforce-eager`, 256 batched tokens and FP8 KV, at half the llama.cpp decode speed (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-16; see [Memory budget](#memory-budget)). |
| GGUF file, or a bitsandbytes checkpoint | llama.cpp | GGUF and bitsandbytes are out-of-tree vLLM plugins and are not in the image ([gguf.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/features/quantization/gguf.md), [bnb.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/features/quantization/bnb.md)). |
| Brand-new architecture | Check the model card's minimum vLLM version first | Model support ships inside the vLLM version; an older image fails at startup with "Model architectures [...] are not supported for now" ([registry.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/models/registry.py)). |

Run one model server at a time: vLLM refuses to start when free VRAM is below the requested fraction ([worker utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/utils.py)), so a running llama.cpp container blocks it.

## Image and versions

| Item | Value | Source |
|---|---|---|
| vLLM release | v0.33.0, published 2026-11-02 (the GitHub release is named v0.33.0 but carries tag_name v0.33.0rc3; the git tag v0.33.0 is commit 4ba787e) | [release](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3) |
| Image | `vllm/vllm-openai:v0.33.0`, same index digest as `:latest` on 2026-11-15 | [Docker Hub](https://hub.docker.com/r/vllm/vllm-openai/tags) |
| Index digest | `sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e` | [Docker Hub](https://hub.docker.com/r/vllm/vllm-openai/tags) |
| amd64 size | 9.74 GB compressed; `v0.33.0-x86_64-zstd` is the same release at 7.70 GB | [Docker Hub](https://hub.docker.com/r/vllm/vllm-openai/tags) |
| CUDA | 13.0.3 (`ARG CUDA_VERSION=13.0.3`) | [Dockerfile](https://github.com/vllm-project/vllm/blob/v0.33.0/docker/Dockerfile) |
| PyTorch | 2.14.0 | [requirements/cuda.txt](https://github.com/vllm-project/vllm/blob/v0.33.0/requirements/cuda.txt) |
| FlashInfer | 0.7.2.post1 | [release](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3) |
| Arch list | `7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0`, so sm_120 is built in | [Dockerfile](https://github.com/vllm-project/vllm/blob/v0.33.0/docker/Dockerfile) |
| ENTRYPOINT | `["vllm", "serve"]`: the model id and flags go straight after the image name | [Dockerfile](https://github.com/vllm-project/vllm/blob/v0.33.0/docker/Dockerfile) |
| Default frontend | Python; the Rust frontend runs only with `VLLM_USE_RUST_FRONTEND=1` | [envs.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/envs.py) |
| Next version | v0.34.0rc4 (tagged 2026-11-14) is a release-candidate tag with no GitHub release yet; it changes the `gpu_memory_utilization` default (see [Maintenance](#maintenance)) | [cache.py at v0.34.0rc4](https://github.com/vllm-project/vllm/blob/v0.34.0rc4/vllm/config/cache.py) |

Driver compatibility: CUDA 13 images need an R580 or newer driver ([vLLM Docker docs](https://docs.vllm.ai/en/latest/deployment/docker/)), and driver 595.99.02 is the CUDA 13.2 branch ([CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)), so the stock image runs without compatibility mode.
Do not set `VLLM_ENABLE_CUDA_COMPATIBILITY`: it "only supports select professional and datacenter NVIDIA GPUs" ([vLLM Docker docs](https://docs.vllm.ai/en/latest/deployment/docker/)).
Avoid the `cu134`/`cu135` nightly tags on this driver: CUDA 13.4 and 13.5 map to R615 and R620 ([CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)), and the vLLM docs require R620 or newer for their CUDA 13.5 staging image ([vLLM Docker docs](https://docs.vllm.ai/en/latest/deployment/docker/)).
Keep the Rust frontend off: gpt-oss cannot be served with tool or reasoning parser flags on it ([#66989](https://github.com/vllm-project/vllm/issues/66989)).

### Why not the NGC images

NVIDIA's `nvcr.io/nvidia/vllm` images trail upstream and their newer builds target a CUDA newer than this driver.

| NGC tag | CUDA | vLLM | Source |
|---|---|---|---|
| 26.05-py3 | 13.2.1 | 0.20.1 | [release notes 26.05](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-05.html) |
| 26.08-py3 | 13.4.1 | 0.27.1 | [release notes 26.08](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-08.html) |
| 26.09-py3 | 13.4.1 | 0.29.0 | [release notes 26.09](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-09.html) |
| 26.10-py3 | 13.4.3 | 0.31.0 | [release notes 26.10](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-10.html) |

26.05 matches the driver's CUDA 13.2 but ships vLLM 0.20.1, which predates several models in [models.md](models.md); the user's DGX Spark runbook hit exactly this (Nemotron 3.5 Lightning needs vLLM 0.27.1 or newer, [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)).
26.08 and later are CUDA 13.4, newer than R595, and would rely on minor-version compatibility ([CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).
Even 26.10 is two releases behind v0.33.0, which holds the sm_120 MXFP4 Marlin preference and the tool-parser fixes this document relies on ([release](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3)).
NGC images also have no `vllm serve` entrypoint, so commands copied from them carry an extra `vllm serve` that breaks on `vllm/vllm-openai` (see [Troubleshooting](#troubleshooting)).

## Run template

Prerequisites: NVIDIA Container Toolkit installed and the CDI spec generated, as described in [host-setup.md](../docs/host-setup.md).
CDI (`--device nvidia.com/gpu=all`) is preferred over `--gpus all` because on a systemd-cgroup host `systemctl daemon-reload` can strip the GPU from containers that got it through the legacy hook ([NVIDIA troubleshooting](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html)).
`--gpus all` (the upstream form in the [vLLM Docker docs](https://docs.vllm.ai/en/latest/deployment/docker/)) also works if CDI is not set up.

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
  <hf-repo-id> \
  --served-model-name <short-name> \
  --max-model-len 65536 \
  --max-num-seqs 2 \
  --enable-auto-tool-choice \
  --tool-call-parser <tool-parser> \
  --reasoning-parser <reasoning-parser> \
  --default-chat-template-kwargs '{"enable_thinking":false}'

docker logs -f vllm
```

What each part does:

- `-d --rm` with no `--restart`: the server runs only while you need it.
  A running CUDA process keeps the dGPU out of RTD3 power-down ([driver README](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/dynamicpowermanagement.html)), so do not let it come back after a reboot.
  Drop `--rm` while debugging a crash so `docker logs` survives.
- `--ipc=host`: PyTorch shares tensors between vLLM processes through shared memory; the docs accept `--ipc=host` or `--shm-size` and give no size ([vLLM Docker docs](https://docs.vllm.ai/en/latest/deployment/docker/)).
- `--memory 32g --memory-swap 32g`: equal values mean no swap ([Docker resource constraints](https://github.com/docker/docs/blob/main/content/manuals/engine/containers/resource_constraints.md)).
  Containers run under `system.slice` through dockerd, so the user's `heavy` wrapper does not cap them ([host-setup.md](../docs/host-setup.md)).
  32 GiB leaves room for the first-start JIT at `MAX_JOBS=2` (about 14-16 GiB peak, [#46268](https://github.com/vllm-project/vllm/issues/46268)) inside the about 40 GiB the desktop leaves free; the value itself is an estimate.
- `-p 127.0.0.1:8000:8000`: Docker-published ports bypass ufw, so publish on localhost only ([host-setup.md](../docs/host-setup.md)).
- `-e HF_TOKEN` and `-e VLLM_API_KEY` without a value pass the host variables through only if they are set, so no secret appears in the command line.
  vLLM reads `VLLM_API_KEY` as the API key when `--api-key` is not given ([envs.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/envs.py)).
  The key protects only `/v1`, `/v2` and `/inference` paths ([cli_args.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/common/cli_args.py)), and other endpoints such as `/invocations` stay open, so it is a convenience on top of the localhost binding, not a security boundary.
- `VLLM_NO_USAGE_STATS=1`: the image sets `VLLM_USAGE_SOURCE=production-docker-image` and reports usage stats unless opted out ([Dockerfile](https://github.com/vllm-project/vllm/blob/v0.33.0/docker/Dockerfile), [envs.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/envs.py)).
- `MAX_JOBS=2`: caps parallel JIT compile jobs; without it the first NVFP4 W4A4 start can use about 52 GiB of host RAM ([#46268](https://github.com/vllm-project/vllm/issues/46268)).
  Harmless for other models.
- HF cache bind mount: the same HF hub layout that llama.cpp `-hf` uses, so both runtimes share one download ([llamacpp/README.md](../llamacpp/README.md)).
- `vllm-cache` volume at `/root/.cache/vllm`: keeps torch.compile, Triton and AOT artifacts, so only the first start of a model and config pays the compile time ([docker.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/deployment/docker.md)).
  Changing the model, config, relevant `VLLM_*` variables, torch build or GPU invalidates it ([Faster Startup](https://docs.vllm.ai/en/latest/configuration/optimization/#faster-startup)).
- `flashinfer-cache` volume: FlashInfer keeps its JIT and autotune cache in `/root/.cache/flashinfer` ([vllm/snapshot/runtime.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/snapshot/runtime.py)).
- `--served-model-name`: must equal the OpenCode `modelID`, and `--max-model-len` must equal `limit.context` ([opencode.md](../docs/opencode.md)).
- `--max-num-seqs 2`: a single-user laptop does not need the API-server default of 256 sequences ([arg_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py)); the per-model value is in [models.md](models.md).

The image runs as root by default; to run as the built-in `vllm` user (UID 2000, GID 0), add `--user 2000:0` and mount the caches under `/home/vllm/.cache/` instead, with host directories writable by group 0 ([vLLM Docker docs](https://docs.vllm.ai/en/latest/deployment/docker/)).

To pull a model before serving, run the `hf` CLI from the same image (the image installs `huggingface_hub` 1.31 or newer, [requirements/common.txt](https://github.com/vllm-project/vllm/blob/v0.33.0/requirements/common.txt); it ran here on 2026-11-16, see [models.md](models.md#gpt-oss-20b) for the user-owned variant), then check for interrupted transfers:

```bash
docker run --rm --memory 8g --memory-swap 8g \
  -e HF_TOKEN \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint hf vllm/vllm-openai:v0.33.0 download <hf-repo-id>

find ~/.cache/huggingface/hub -name '*.incomplete' -size +0   # must print nothing
du -sh ~/.cache/huggingface/hub/models--*
```

A truncated shard does not fail the download; it fails later as a safetensors header or tensor-shape error at load (lesson from the user's DGX Spark runbook).

### Compose equivalent

Compose accepts the CDI device in `devices`, or `gpus: all` on Compose 2.30 and later ([Compose services reference](https://github.com/docker/docs/blob/main/content/reference/compose-file/services.md)); the host has Compose v5.7.1.
Leaving out `restart` keeps the Compose default of no restart.

```yaml
services:
  vllm:
    image: vllm/vllm-openai:v0.33.0
    container_name: vllm
    devices:
      - nvidia.com/gpu=all
    ipc: host
    mem_limit: 32g
    memswap_limit: 32g
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      - HF_TOKEN
      - VLLM_API_KEY
      - VLLM_NO_USAGE_STATS=1
      - MAX_JOBS=2
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - vllm-cache:/root/.cache/vllm
      - flashinfer-cache:/root/.cache/flashinfer
    command:
      - <hf-repo-id>
      - --served-model-name=<short-name>
      - --max-model-len=65536
      - --max-num-seqs=2
      - --enable-auto-tool-choice
      - --tool-call-parser=<tool-parser>
      - --reasoning-parser=<reasoning-parser>
      - --default-chat-template-kwargs={"enable_thinking":false}

volumes:
  vllm-cache:
  flashinfer-cache:
```

Start with `docker compose up -d` and stop with `docker compose down` when you are done, so the GPU can power down.

## Memory budget

vLLM claims a fixed fraction of total VRAM at startup and splits it into weights, a working reserve and the KV pool.

Measured on this laptop on 2026-11-16, the method below was too optimistic in three ways ([attempt log](../research/2026-11-16/vllm-gpt-oss-20b/attempts.md)):

- CUDA reports a total of 15.47 GiB, not the 16,303 MiB (15.92 GiB) that `nvidia-smi` shows, and about 15.08 GiB was free at startup; vLLM multiplies `--gpu-memory-utilization` by the 15.47 GiB figure and refuses a value above the free share (observed).
- Weights in GPU memory can exceed the file size: gpt-oss-20b loads as 13.8 GiB against 12.82 GiB of safetensors on the MXFP4 Marlin path (observed).
- The reserve is larger than 1.2 GiB: activations, CUDA graphs and non-torch memory took about 2.1 GiB with the defaults (2,048 batched tokens, 0.40 GiB of CUDA graphs) and about 1.7 GiB with 1,024 batched tokens and four small graph sizes (derived from the logged budget and KV figures).

On 2026-11-17 the first starts of the seven other recipes, all at the default 0.92, gave KV pools that missed the old-method estimates of [models.md](models.md) in both directions ([records](../research/2026-11-17/README.md)) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17):

| Recipe | Estimated weights | Loaded weights | Estimated pool | Measured pool | Miss |
|---|---|---|---|---|---|
| K2 Horizon 7B (cold compile cache) | 7.56 GiB | 7.72 GiB | 5.88 GiB | 4.44 GiB | -1.44 GiB |
| Qwen3.5-9B | 7.42 GiB | 7.48 GiB | 6.03 GiB | 4.71 GiB | -1.32 GiB |
| DeepSeek-R1-0528-Qwen3-8B (cold compile cache) | 8.79 GiB | 9.12 GiB | 4.66 GiB | 3.83 GiB | -0.83 GiB |
| Groot-19B-A3B | 10.94 GiB | 10.99 GiB | 2.45 GiB | 1.87 GiB | -0.58 GiB |
| Gemma 4 E4B (`--enforce-eager`) | 9.83 GiB | 10.13 GiB | 3.62 GiB | 3.67 GiB | +0.05 GiB |
| Mellum2.1 | 7.15 GiB | 7.20 GiB | 6.30 GiB | 6.37 GiB | +0.07 GiB |
| Gemma 4 12B | 9.56 GiB | 8.28 GiB | 3.89 GiB | 5.37 GiB | +1.48 GiB |

The estimates used a 14.65 GiB budget and a 1.2 GiB reserve; with the measured 14.23 GiB budget, the rest of each start (activations, CUDA graphs and non-torch memory) took 0.4 to 2.1 GiB (derived from the logged budget, weights and pool of each start).
So use 15.47 GiB as the total, take the weights from the `Model loading took` line once a model has started, because they came out 1.28 GiB below to 0.33 GiB above the file-size estimates (and about 1 GiB above for gpt-oss-20b's MXFP4 Marlin path), and plan a reserve of about 2 GiB unless a startup log shows less.
An unmeasured context estimate in [models.md](models.md) can miss its pool by about 1.5 GiB either way, so check it against the first startup log (estimate).

1. Budget: `--gpu-memory-utilization` defaults to 0.92 in v0.33.0 ([cache.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py)).
   The original estimate used 16303 MiB x 0.92 = 14.65 GiB; the measured basis is 15.47 GiB x 0.92 = 14.23 GiB, and 0.965 (14.93 GiB) is the highest value that started here.
2. Weights: the real on-disk size of the checkpoint files vLLM loads (language model only when you pass `--language-model-only`).
3. Reserve: originally 1.2 GiB for activations, CUDA graphs and non-torch allocations; measured about 1.7 to 2.1 GiB for gpt-oss-20b and 0.4 to 2.1 GiB for the 2026-11-17 recipes (compare against the "Available KV cache memory" line in the startup log, [gpu_worker.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu_worker.py)).
4. KV pool = budget - weights - reserve.
5. KV per token = 2 (K and V) x full-attention layers x KV heads x head dim x bytes per element (2 for BF16, 1 for FP8).
   Linear-attention (GDN, Mamba) and sliding-window layers add a bounded amount instead of a per-token cost.
6. Context = KV pool / KV per token, capped by the model's maximum.

Ornith-1.5-9B is now excluded from the ranking ([models.md](models.md#ornith-15-9b-excluded)), but its numbers still show the method.
Worked example, Ornith-1.5-9B NVFP4 with `--language-model-only` (a Qwen3.5-9B fine-tune with 8 full-attention layers, 4 KV heads, head dim 256), all figures estimates:

| Step | Value |
|---|---|
| Budget at 0.92 | 14.65 GiB |
| Weights (language model only) | 6.91 GiB ([HF API](https://huggingface.co/api/models/ornith-ai/Ornith-1.5-9B-NVFP4?blobs=true)) |
| Reserve | 1.2 GiB |
| KV pool | 14.65 - 6.91 - 1.2 = 6.54 GiB |
| KV per token, BF16 | 2 x 8 x 4 x 256 x 2 B = 32 KiB |
| Context, BF16 KV | (6.54 GiB - about 0.05 GiB fixed GDN state) / 32 KiB = about 212k tokens, so 131072 fits with margin |
| Context, FP8 KV | about 425k, capped at the model's 262144 |

The old method predicted that gpt-oss-20b fits 65536 at 0.95 with FP8 KV; on this laptop that start failed with `Available KV cache memory: -0.74 GiB`, and only `--gpu-memory-utilization 0.965 --enforce-eager --max-num-batched-tokens 256` started, with 81,504 tokens of FP8 KV and half the llama.cpp decode speed (measured, see [models.md](models.md#gpt-oss-20b)).
Per-model budgets are in [models.md](models.md).
After the first start, record the "KV cache size: N tokens, Maximum concurrency for M tokens per request" log line ([kv_cache_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/kv_cache_utils.py)) and replace the estimate.
If the pool is too small, startup fails with a message that gives "the estimated maximum model length" ([kv_cache_planning.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/kv_cache_planning.py)); use that number or a smaller one.

Third-party calculators use other rules, for example ModelFit's 90% of 16 GB with a 1.6 GB reserve (estimates for the desktop RTX 5080 at 960 GB/s, no laptop page, [ModelFit](https://modelfit.io/gpu/rtx-5080/)); see [community-benchmarks.md](../docs/community-benchmarks.md) and treat them only as a sanity check.

## Flags reference

All flags exist in v0.33.0 ([arg_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py), [cli_args.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/common/cli_args.py)).

| Flag | Purpose | Recommended here | Source |
|---|---|---|---|
| `--served-model-name` | API-visible model id | Short name equal to the OpenCode `modelID` | [arg_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py) |
| `--max-model-len` | Context limit | From the memory budget; 65536 is the practical floor for OpenCode | [opencode.md](../docs/opencode.md) |
| `--gpu-memory-utilization` | Fraction of total VRAM vLLM claims | Default 0.92; 0.965 with `--enforce-eager` for gpt-oss-20b at 64k, 0.94 for K2 Horizon 7B | [cache.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py) |
| `--kv-cache-dtype` | KV element type | `auto` (BF16) where context fits; `fp8` to double context; never `fp8_e5m2` ([#41343](https://github.com/vllm-project/vllm/issues/41343)) | [cache.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py) |
| `--attention-backend` | Force an attention backend | `TRITON_ATTN` with FP8 KV, for gpt-oss, and for Mistral3 | [cuda.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/platforms/cuda.py) |
| `--max-num-seqs` | Concurrent sequences | 1-4 (default 256 for the API server on a GPU under 70 GiB) | [arg_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py) |
| `--max-num-batched-tokens` | Tokens per scheduler step, so prefill chunk size | Default 2048 on this GPU; raise to 8192 for faster long prefills if memory allows (untested) | [arg_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py) |
| `--enable-prefix-caching` | Reuse KV for shared prefixes | On by default; leave it on | [cache.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py) |
| `--enable-auto-tool-choice` | Turn on tool-call parsing | Always, with `--tool-call-parser` (startup fails without one) | [cli_args.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/common/cli_args.py) |
| `--tool-call-parser` | Model-family tool format | See [Tool calling and reasoning](#tool-calling-and-reasoning) | [tool_parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/tool_parsers/__init__.py) |
| `--reasoning-parser` | Split thinking into the `reasoning` field | Always for thinking models | [reasoning](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/reasoning/__init__.py) |
| `--default-chat-template-kwargs` | Server-wide template defaults, request values win | `'{"enable_thinking":false}'` for toggle models | [cli_args.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/common/cli_args.py) |
| `--reasoning-effort-map` | Map client effort values to ones the model accepts | gpt-oss: `'{"none":"low","minimal":"low","xhigh":"high","max":"high"}'` | [cli_args.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/common/cli_args.py) |
| `--language-model-only` | Skip the vision or audio tower | Always for text-only work on multimodal checkpoints | [arg_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py) |
| `--limit-mm-per-prompt` | Per-modality item limit | Gemma 4: `'{"image":0,"audio":0}'` | [arg_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py) |
| `--enforce-eager` | Skip torch.compile and CUDA graphs | Off; keep in reserve for hybrid-GDN NVFP4 crashes ([#54331](https://github.com/vllm-project/vllm/issues/54331)) | [arg_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py) |
| `--no-async-scheduling` | Disable async scheduling | Alternative to `TRITON_ATTN` for the FlashInfer race ([#61239](https://github.com/vllm-project/vllm/issues/61239)), inferred mitigation | [arg_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py) |
| `--api-key` | Require a bearer key | Prefer the `VLLM_API_KEY` environment variable | [cli_args.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/common/cli_args.py) |
| `--trust-remote-code` | Run model repo Python code | Only for repos that need it and that you trust | [arg_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py) |
| `--cpu-offload-gb`, `--cpu-offload-params` | UVA weight offload to RAM | Do not use for MoE experts on v0.33.0 ([#66264](https://github.com/vllm-project/vllm/issues/66264)) | [offload.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/offload.py) |
| `--kv-offloading-size`, `--kv-offloading-backend` | Copy KV blocks to host RAM | Off; optional experiment, see [CPU offload](#cpu-offload) | [cache.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py) |

The docs at docs.vllm.ai/en/latest are built from main and can describe newer behaviour than the pinned image (for example a 1.0 utilization default); read the docs at the tag ([v0.33.0 docs](https://github.com/vllm-project/vllm/tree/v0.33.0/docs)) when they disagree.

## Tool calling and reasoning

Parser flags are per model family and are not interchangeable.
The parser names below are registered in v0.33.0 ([tool parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/tool_parsers/__init__.py), [reasoning parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/reasoning/__init__.py)).

| Family | `--tool-call-parser` | `--reasoning-parser` | Thinking control | Source |
|---|---|---|---|---|
| Qwen3.5, Qwen3.6, Qwen3.8, Groot-19B-A3B (Qwen3.6 architecture), Ornith-1.5 (Qwen3.5 fine-tune) | `qwen3_coder` (`qwen3_xml` also worked on the user's DGX Spark with Qwen3.6) | `qwen3` | `enable_thinking`, on by default in the template | [Qwen3.6 card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Qwen3-Coder, Qwen3-Coder-Next | `qwen3_coder` | none (no thinking mode) | Not applicable | [Qwen3-Coder-Next card](https://huggingface.co/Qwen/Qwen3-Coder-Next) |
| Gemma 4 | `gemma4` | `gemma4`, always paired with the tool parser | `enable_thinking`, off by default | [Gemma 4 12B card](https://huggingface.co/google/gemma-4-12B-it) |
| gpt-oss | `openai` | `openai_gptoss` | Always thinks; `reasoning_effort` low, medium or high | [gpt-oss-20b card](https://huggingface.co/openai/gpt-oss-20b) |
| Nemotron 3.5 | `qwen3_coder` | `nemotron_v3` | `enable_thinking`, on by default | [Nemotron 3.5 Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) |
| GLM-4.7-Flash | `glm47` (`glm45` maps to the same parser) | `glm45` or `glm47` (same adapter) | `enable_thinking` | [GLM-4.7-Flash card](https://huggingface.co/zai-org/GLM-4.7-Flash) |
| Mistral / Ministral 3 | `mistral` | `mistral` is registered; check the card for the reasoning variants | Per model variant; also force `TRITON_ATTN` ([#53481](https://github.com/vllm-project/vllm/issues/53481)) | [tool parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/tool_parsers/__init__.py) |
| K2 Horizon 7B | `k2_horizon` | `k2_horizon` | Always thinks; keep `reasoning_effort` high, because medium and low leak reasoning ([#62743](https://github.com/vllm-project/vllm/issues/62743)) | [models.md](models.md#k2-horizon-7b-awq-int4) |
| DeepSeek-R1-0528-Qwen3-8B | none: tool calling is not supported | `deepseek_r1` | Always thinks; no switch | [reasoning parsers](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/reasoning/__init__.py) |

How the server behaves in v0.33.0:

- Wrong or missing tool parser: no error; the call arrives as plain text in `content` and `finish_reason` is `stop`, so an agent loop stalls (lesson from the user's DGX Spark runbook).
  `--enable-auto-tool-choice` without `--tool-call-parser`, or an unknown parser name, fails at startup ([cli_args.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/common/cli_args.py)).
- Missing reasoning parser: thinking text lands in `content`; for Gemma 4 the channel tokens leak.
- `tool_choice` default: when a request has `tools` and no `tool_choice`, the server sets `"auto"` ([protocol.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/openai/chat_completion/protocol.py)).
  So OpenCode, which omits `tool_choice`, works on stock v0.33.0 ([opencode.md](../docs/opencode.md)).
- `"tools": []` returns HTTP 400 ("`tools` must not be an empty array"), and `tool_choice` without `tools` also returns 400 ([protocol.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/openai/chat_completion/protocol.py)).
  Do not force `tool_choice` in client config.
- Reasoning text comes back in the `reasoning` field of the message; an incoming `reasoning_content` field is renamed to `reasoning` ([protocol.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/openai/chat_completion/protocol.py)).
  OpenCode needs `compatibility.reasoningField: "reasoning"` for vLLM ([opencode.md](../docs/opencode.md)).
- `finish_reason` reflects what happened, so a parsed tool call ends with `tool_calls` ([release notes, #59324](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3)).

Thinking default off: `--default-chat-template-kwargs '{"enable_thinking":false}'` sets the default for every caller, and request-level `chat_template_kwargs` take precedence ([cli_args.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/common/cli_args.py)).
A single request opts back in with:

```json
{ "chat_template_kwargs": { "enable_thinking": true } }
```

Why off by default: on the user's DGX Spark, Qwen3.6 with thinking on spent all 1024 `max_tokens` reasoning and returned no answer in 23.7 s, while thinking off answered correctly in 1.5 s (measured on a DGX Spark GB10 with vLLM 0.20.1, 2026-09-05).
Thinking still helps agent quality: Artificial Analysis gives Qwen3.8-27B 49.1 on Terminal-Bench 2.1 without reasoning and 67.4 at low effort ([Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-27b), measured by Artificial Analysis through the Alibaba API at BF16, not on this hardware), so benchmark both through the OpenCode variants in [opencode.md](../docs/opencode.md).
For gpt-oss the switch is `reasoning_effort` in the request, and `--reasoning-effort-map` keeps unsupported values from returning 400 ([cli_args.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/common/cli_args.py)).
For DeepSeek R1 distills the `enable_thinking` kwarg is "harmlessly filtered out" and has no effect ([reasoning_outputs.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/features/reasoning_outputs.md)).

## Quantization paths on sm_120

| Checkpoint format | Kernel path on sm_120 in v0.33.0 | Notes | Source |
|---|---|---|---|
| MXFP4 MoE (gpt-oss) | MARLIN with BF16 activations by default, then BATCHED_MARLIN, TRITON, HUMMING | The source keeps OAI Triton opt-in on consumer Blackwell because its 99 KB shared-memory budget caps tile size and Marlin prefills about 10% faster | [mxfp4.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py) |
| NVFP4 W4A16 (weights FP4, activations BF16), and int4 W4A16 compressed-tensors such as Google's Gemma 4 QAT | Marlin | No FlashInfer FP4 JIT, so no first-start RAM spike (inferred from source; RAM was not traced); the Groot NVFP4 and Gemma 4 12B builds in [models.md](models.md) use this path, and their startup logs on this laptop name `MarlinNvFp4LinearKernel` and the `MARLIN` NvFp4 MoE backend for Groot and `MarlinLinearKernel` for Gemma 4 12B (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [Groot record](../research/2026-11-17/vllm-groot-19b-a3b/README.md), [Gemma 4 12B record](../research/2026-11-17/vllm-gemma-4-12b/README.md)) | [kernels/linear/\_\_init\_\_.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/kernels/linear/__init__.py) |
| NVFP4 W4A4 (ModelOpt-style, FP4 activations) | FlashInfer CUTLASS for MoE and `FlashInferCutlassNvFp4LinearKernel` for dense GEMM | First start JIT-compiles `fp4_gemm_cutlass_sm120` at about 6 GiB per compile job, about 52 GiB by default; set `MAX_JOBS=2` (about 14-16 GiB, about 8.5 min); fix PR [#66864](https://github.com/vllm-project/vllm/pull/66864) open | [#46268](https://github.com/vllm-project/vllm/issues/46268) |
| FP8 weights, MoE | Triton (no CUTLASS FP8 MoE on SM120) | Slower MoE path | [#43507](https://github.com/vllm-project/vllm/issues/43507) |
| GGUF, bitsandbytes | Not in the image | Out-of-tree `vllm-gguf-plugin` and `vllm-bnb-plugin` need a derived image; use llama.cpp for GGUF | [gguf.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/features/quantization/gguf.md), [bnb.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/features/quantization/bnb.md) |
| b12x SM120 kernels | Not installed | Optional extra `vllm[b12x]`, selected explicitly with `--linear-backend b12x` / `--moe-backend b12x`; not auto-selected | [b12x.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/features/quantization/b12x.md) |

Do not force W4A4 checkpoints onto Marlin: Unsloth reports Marlin about 2.5x slower for W4A4 (vendor-reported, measured on 1x B200 at 128 concurrency, [Unsloth Qwen3.6 docs](https://unsloth.ai/docs/models/qwen3.6)), and an explicit `--moe-backend cutlass` for NVFP4 MoE returned empty completions on an RTX 5090 Laptop ([#51525](https://github.com/vllm-project/vllm/issues/51525)).
Leave `--moe-backend` unset unless a model note in [models.md](models.md) says otherwise.
To tell W4A16 from W4A4 before downloading, read the `quantization_config` in the checkpoint's `config.json` (or `hf_quant_config.json`) and check whether activations are quantized.
MoE decode on GeForce Blackwell runs without tuned fused-MoE configs, so it is slower than its size suggests ([#48732](https://github.com/vllm-project/vllm/issues/48732)).

FP8 KV cache and the attention backend:

- sm_120 runs FlashAttention 2 (FA3 is SM90 only, FA4 SM100 only), and FA2 rejects quantized KV ([flash_attn.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/attention/backends/flash_attn.py), [fa_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/attention/backends/fa_utils.py)).
- With `--kv-cache-dtype fp8`, selection falls through to FlashInfer, then Triton ([cuda.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/platforms/cuda.py)).
- FlashInfer has a metadata-builder race under async scheduling, reproduced with gpt-oss-20b on one GPU ([#61239](https://github.com/vllm-project/vllm/issues/61239)); the fix [#64165](https://github.com/vllm-project/vllm/pull/64165) was merged on 2026-11-11, after v0.33.0.
  So when you need FP8 KV, add `--attention-backend TRITON_ATTN` (inferred mitigation, not upstream guidance).
- gpt-oss needs attention sinks, which FA2 lacks; Triton supports both FP8 and sinks ([triton_attn.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/attention/backends/triton_attn.py)).
- Gemma 4 already ends up on Triton on sm_120 ([models/config.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/models/config.py)).
- Mistral3 gives degenerate output on FlashInfer, so force `TRITON_ATTN` for it ([#53481](https://github.com/vllm-project/vllm/issues/53481)).
- If the context fits with BF16 KV, prefer it: attention stays on FA2 and the race does not apply.

## Prefix caching and chunked prefill

- Prefix caching is on by default (`enable_prefix_caching: bool = True`, [cache.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py)).
  OpenCode resends a near-identical system prompt and tool schema every step, so cache hits cut prefill time on every turn after the first.
- Chunked prefill is on by default (`enable_chunked_prefill: bool = True`, [scheduler.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/scheduler.py)).
  On a GPU under 70 GiB the API server defaults to `max_num_batched_tokens` 2048 per step ([arg_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py)), which is the prefill chunk size.
- Hybrid models (Qwen3.5/3.6/3.8 Gated DeltaNet, Nemotron Mamba) switch `mamba_cache_mode` from `none` to `align` when prefix caching is on, and assert that chunked prefill is enabled ([models/config.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/models/config.py)).
  In `align` mode the recurrent state is cached only for the last token of each scheduler step and at block-size positions ([cache.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py)), so a prefix hit on these models resumes from the nearest saved checkpoint rather than any token.
- Prefix caching skews benchmarks: repeated identical prompts measure cache hits, not prefill.
  See [benchmarking.md](benchmarking.md).

## CPU offload

MoE expert offload to system RAM does not work for the models this laptop needs it for.
`--cpu-offload-params experts` does nothing unless `--cpu-offload-gb` is above 0, and the UVA path reads offloaded weights over PCIe on every forward pass ([offload.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/offload.py)).
On v0.33.0 the offload is silently undone when Marlin repacking rebinds the parameter, so the experts land back on the GPU and loading runs out of memory ([#66264](https://github.com/vllm-project/vllm/issues/66264), open on 2026-11-15, filed against v0.33.0 on sm_120).
Use llama.cpp `--n-cpu-moe` for models that spill ([llamacpp/README.md](../llamacpp/README.md)).

KV offloading is a different feature: `--kv-offloading-size <GiB>` with `--kv-offloading-backend native` (default) or `lmcache` copies finished KV blocks to pinned host memory and promotes hits back ([cache.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py), [KV offloading docs](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)).
It extends the prefix cache across turns; it does not raise the maximum live context beyond what fits in VRAM (inferred).
The host buffer counts against the container `--memory` cap, and it is broken for Gemma 4 ([#42348](https://github.com/vllm-project/vllm/issues/42348)).
Treat it as an untested experiment here.

## Known issues

State as read on GitHub on 2026-11-15.

| Issue | Effect here | State | Workaround |
|---|---|---|---|
| [#46268](https://github.com/vllm-project/vllm/issues/46268) NVFP4 W4A4 FlashInfer JIT uses about 52 GiB host RAM | First start of a W4A4 checkpoint can OOM the laptop | Open; fix PR [#66864](https://github.com/vllm-project/vllm/pull/66864) open | `-e MAX_JOBS=2`, persistent cache volumes, prefer W4A16 builds |
| [#61239](https://github.com/vllm-project/vllm/issues/61239) FlashInfer race under async scheduling | Illegal memory access with FP8 KV, reproduced with gpt-oss-20b | Open; fix [#64165](https://github.com/vllm-project/vllm/pull/64165) merged 2026-11-11, not in v0.33.0 | `--attention-backend TRITON_ATTN` or `--no-async-scheduling` |
| [#66264](https://github.com/vllm-project/vllm/issues/66264) UVA offload undone for repacked weights | No vLLM MoE expert offload | Open | Use llama.cpp for MoE models above the budget |
| [#54331](https://github.com/vllm-project/vllm/issues/54331) sm_120 hybrid-GDN NVFP4 dies under sustained load with CUDA graphs | Groot and other Qwen3.5-family NVFP4 builds may crash | Open | `--enforce-eager`; llama.cpp Q8_0 as fallback |
| [#66989](https://github.com/vllm-project/vllm/issues/66989) Rust frontend rejects gpt-oss parser flags | gpt-oss fails to start with `VLLM_USE_RUST_FRONTEND=1` | Open | Keep the default Python frontend |
| [#65770](https://github.com/vllm-project/vllm/issues/65770) gpt-oss HTTP 500 when a prior tool name contains special-token text | Agent session breaks after a malformed call | Open | Start a new session |
| [#53363](https://github.com/vllm-project/vllm/issues/53363) Gemma 4 `tool_choice="required"` not enforced | Forced tool calls can come back as prose | Open | Use `auto` (OpenCode's default) |
| [#54256](https://github.com/vllm-project/vllm/issues/54256) Gemma 4 parser drops tool calls after the reasoning channel | Missing calls with thinking on | Open | Always pass both `gemma4` parsers; test with thinking on and off |
| [#48753](https://github.com/vllm-project/vllm/issues/48753) Qwen3 tool parser strips whitespace in string arguments | Exact-match edit tools can fail | Open | Run the scripted edit test in [opencode.md](../docs/opencode.md) before daily use |
| [#65774](https://github.com/vllm-project/vllm/issues/65774) `qwen3_coder` streamed arguments break on a repeated parameter | Malformed streamed tool calls | Open | None known; watch OpenCode edits |
| [#61206](https://github.com/vllm-project/vllm/issues/61206) three Qwen tool-calling parser problems | Intermittent bad tool calls on Qwen models | Open; fix PR [#61217](https://github.com/vllm-project/vllm/pull/61217) open | None known |
| [#64326](https://github.com/vllm-project/vllm/issues/64326) non-streaming parse treats markers written as text as real markers | Prose that quotes a tag becomes a tool call or ends reasoning | Open | Prefer streaming clients (OpenCode streams) |
| [#38696](https://github.com/vllm-project/vllm/issues/38696) Qwen3.5 `json_schema` output has garbled spaces | Structured output unreliable | Open | Avoid `response_format` with Qwen parsers |
| [#53481](https://github.com/vllm-project/vllm/issues/53481) FlashInfer degenerate output for Mistral3 on sm_120 | Garbage output | Open | `--attention-backend TRITON_ATTN` |
| [#43507](https://github.com/vllm-project/vllm/issues/43507) no CUTLASS FP8 MoE on SM120 | FP8 MoE runs on Triton, slower | Open | Prefer 4-bit builds |
| [#48732](https://github.com/vllm-project/vllm/issues/48732) no tuned fused-MoE configs for GeForce Blackwell | Slower MoE decode | Open | None; measure before comparing to desktop numbers |
| [#51525](https://github.com/vllm-project/vllm/issues/51525) explicit cutlass NVFP4 MoE backend returns empty output | Empty completions | Open | Leave `--moe-backend` unset |
| [#42348](https://github.com/vllm-project/vllm/issues/42348) Gemma 4 KV-cache CPU offloading broken | KV offloading unusable for Gemma 4 | Open | Do not enable it for Gemma 4 |
| [#41343](https://github.com/vllm-project/vllm/issues/41343) `fp8_e5m2` KV silently corrupts output | Wrong answers | Open | Use `fp8` (e4m3) only |
| [#66423](https://github.com/vllm-project/vllm/issues/66423) segfault with a reasoning parser on SM120 NVFP4 W4A4 | Not confirmed: two others could not reproduce it | Open | Note only |

One more issue was measured on this laptop: a model's first vLLM start, with an empty torch.compile cache in the `vllm-cache` volume, left less KV cache memory than later starts with the cache warm.
K2 Horizon 7B had 4.44 GiB of KV cache at util 0.92 cold against 6.54 GiB at 0.94 warm, a gain of 2.10 GiB of which the 0.02 utilization step explains only 0.31 GiB ([attempts](../research/2026-11-17/vllm-k2-horizon-7b/attempts.md)).
DeepSeek-R1-0528-Qwen3-8B had 3.83 GiB at 0.92 cold, short of the 4.5 GiB its 65536 context needs ([attempts](../research/2026-11-17/vllm-deepseek-r1-0528-qwen3-8b/attempts.md)) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17), and both e2e runs later started it at 0.92 in 25 s and 24 s ([first run's record 277](../research/2026-11-17/llmctl-dsh-e2e-run1/records/277-18-deepseek-r1-0528-qwen3-8b-vllm-up), [second run's record 276](../research/2026-11-17/llmctl-dsh-e2e/records/276-18-deepseek-r1-0528-qwen3-8b-vllm-up)) with the compile cache warm (inferred from those short starts; the records hold no server log, and the compile cache as the cause is not proven).
On 2026-11-18 its recipe with `--kv-cache-memory-bytes 5153960755`, the 4.8 GiB that warm starts got, started both cold, with a new empty compile-cache volume, in 44.2 s and warm in 20.3 s, each with 69,904 KV tokens (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18) ([record](../research/2026-11-18/vllm-deepseek-r1-0528-qwen3-8b-cold-start/README.md)).
A search of the vLLM issues on 2026-11-17 (`gh search issues --repo vllm-project/vllm` for "cold-cache KV cache", "cold start kv cache", "warm cache KV cache", "compile cache KV cache", "first startup KV cache smaller" and "torch.compile cache memory profiling") found one related report, [#62277](https://github.com/vllm-project/vllm/issues/62277) (open): a cold-cache start that ran out of memory while allocating the KV cache with `--kv-cache-memory-bytes` and `--enable-sleep-mode`, which it blames on allocator reservations the profile run leaves behind; the DeepSeek-R1 recipe sets `--kv-cache-memory-bytes` without `--enable-sleep-mode`, and its cold start passed (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18), so it did not meet that issue, and the other recipes use neither flag (inferred from its description).
So a recipe that starts with a warm cache may fail on a cold one, after a new image or a cleared `vllm-cache` volume; when a first start lacks KV cache by a small margin, start it once more at the same utilization before raising `--gpu-memory-utilization` (estimate).

## Smoke tests

Run these after every new model or flag change, in order.
The `Authorization` header is ignored when no key is set, so the commands work either way.

1. The model is listed under the expected name:

```bash
curl -s http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer ${VLLM_API_KEY:-none}" | jq -r '.data[].id'
```

2. The chat template works and thinking is off by default:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${VLLM_API_KEY:-none}" \
  -d '{"model":"<short-name>",
       "messages":[{"role":"user","content":"Reply with exactly: OK"}],
       "max_tokens":16}' \
  | jq '{content: .choices[0].message.content, reasoning: .choices[0].message.reasoning, finish: .choices[0].finish_reason}'
```

Expected: `content` is `OK`, `reasoning` is null and `finish_reason` is `stop`.
An empty `content` with `finish_reason` `length` means thinking is on and ate the budget.
gpt-oss always reasons, so for it expect a short `reasoning` as well.

3. A tool call is parsed:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${VLLM_API_KEY:-none}" \
  -d '{"model":"<short-name>",
       "messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
       "tools":[{"type":"function","function":{"name":"get_weather",
         "description":"Get the current weather for a city",
         "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
       "max_tokens":512}' \
  | jq '{finish: .choices[0].finish_reason, calls: .choices[0].message.tool_calls, content: .choices[0].message.content}'
```

Expected: `finish_reason` is `tool_calls`, and `tool_calls[0].function.name` is `get_weather` with `arguments` a JSON string containing `"city": "Paris"`.
If `finish_reason` is `stop` and the call appears as text in `content`, the tool parser is missing or wrong.
Repeat it with `"stream": true` and check that `delta.tool_calls` chunks arrive, because OpenCode streams and several parser bugs above are streaming-only.

4. Optional: send `"chat_template_kwargs":{"enable_thinking":true}` with test 2 and check that `reasoning` fills while `content` stays clean.

Then run the first OpenCode session described in [opencode.md](../docs/opencode.md), and record load numbers with [benchmarking.md](benchmarking.md).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Docker rejects `--device nvidia.com/gpu=all` or `--gpus all` | NVIDIA Container Toolkit missing or CDI spec not generated | Follow [host-setup.md](../docs/host-setup.md) |
| A long-running container loses the GPU ("Failed to initialize NVML: Unknown Error") | `systemctl daemon-reload` on a systemd-cgroup host with legacy `--gpus` injection | Use CDI ([NVIDIA troubleshooting](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html)) |
| "Free memory on device ... is less than desired GPU memory utilization" | Another CUDA process, usually a llama.cpp container, holds VRAM | `nvidia-smi`, stop the other server ([worker utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/utils.py)) |
| "No available memory for the cache blocks" or "... larger than the available KV cache", with "the estimated maximum model length" | Weights plus reserve leave too little KV pool | Use the suggested `--max-model-len`, FP8 KV with `TRITON_ATTN`, or a smaller checkpoint ([kv_cache_planning.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/kv_cache_planning.py)) |
| Container exits with code 137 during the first start | Host RAM cap hit, usually the FlashInfer JIT ([#46268](https://github.com/vllm-project/vllm/issues/46268)) | Keep `MAX_JOBS=2`; check `docker stats`; that is the cap working, so do not raise it blindly |
| First start takes many minutes, later starts are faster | torch.compile, Triton and FlashInfer JIT on a cold cache | Keep the `vllm-cache` and `flashinfer-cache` volumes |
| Argument parsing fails at startup and the command has `vllm serve` after the image name | NGC habit: `vllm/vllm-openai` already has `vllm serve` as ENTRYPOINT, so the words are passed twice | Put the model id directly after the image name ([Dockerfile](https://github.com/vllm-project/vllm/blob/v0.33.0/docker/Dockerfile)) |
| "invalid tool call parser" at startup | Typo or a parser name from a newer vLLM | Use a name from the [registry](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/tool_parsers/__init__.py) |
| "tool parser `openai` is not registered" | Rust frontend enabled | Unset `VLLM_USE_RUST_FRONTEND` ([#66989](https://github.com/vllm-project/vllm/issues/66989)) |
| Tool calls arrive as plain text, agent stalls | `--enable-auto-tool-choice` or `--tool-call-parser` missing or wrong | Fix the flags per the [parser table](#tool-calling-and-reasoning) |
| Empty `content`, `finish_reason: length` | Thinking on, budget spent on reasoning | Check `--default-chat-template-kwargs` and that no client sends `enable_thinking: true` |
| Reasoning text or channel tokens inside `content` | `--reasoning-parser` missing | Add the family's reasoning parser |
| HTTP 400 "`tools` must not be an empty array" | Client sends `"tools": []` | Omit the field when there are no tools ([protocol.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/openai/chat_completion/protocol.py)) |
| HTTP 400 "When using `tool_choice`, `tools` must be set" | Client forces `tool_choice` on requests without tools, for example OpenCode compaction | Remove the forced `tool_choice` from client config |
| HTTP 401 | `VLLM_API_KEY` was set in the container | Send the same key from the client (`{env:VLLM_API_KEY}` in OpenCode) |
| "Model architectures [...] are not supported for now" | Image vLLM is older than the model | Newer image, not a newer download ([registry.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/models/registry.py)) |
| Safetensors header or tensor-shape error at load | Truncated shard from an interrupted download | `find ~/.cache/huggingface/hub -name '*.incomplete'`, delete and re-download |
| Disk filling with no new models | Stale `*.incomplete` blobs | Same `find`, then delete; removing `models--<org>--<repo>` is the only way to reclaim a model's space |
| Illegal memory access under load with FP8 KV | FlashInfer race ([#61239](https://github.com/vllm-project/vllm/issues/61239)) | `--attention-backend TRITON_ATTN` |
| NVFP4 hybrid model crashes after minutes of load | [#54331](https://github.com/vllm-project/vllm/issues/54331) | `--enforce-eager`, or llama.cpp |
| Port 8000 reachable from the LAN | Published without `127.0.0.1:` (Docker bypasses ufw) | Re-run with `-p 127.0.0.1:8000:8000` ([host-setup.md](../docs/host-setup.md)) |
| Fans and battery drain while idle | A running container keeps the dGPU awake | `docker stop vllm` when done |

## Sources

- vLLM v0.33.0 release: https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3
- vLLM source at tag v0.33.0: https://github.com/vllm-project/vllm/tree/v0.33.0 (config/cache.py, config/scheduler.py, config/offload.py, engine/arg_utils.py, entrypoints/common/cli_args.py, entrypoints/openai/chat_completion/protocol.py, envs.py, platforms/cuda.py, fused_moe/oracle/mxfp4.py, tool_parsers, reasoning, docker/Dockerfile)
- vLLM Docker docs: https://docs.vllm.ai/en/latest/deployment/docker/ and https://github.com/vllm-project/vllm/blob/v0.33.0/docs/deployment/docker.md
- vLLM KV offloading docs: https://docs.vllm.ai/en/latest/features/kv_offloading_usage/
- Docker Hub tags for vllm/vllm-openai: https://hub.docker.com/r/vllm/vllm-openai/tags
- NVIDIA vLLM container release notes: https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html
- CUDA toolkit release notes (driver mapping): https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- NVIDIA Container Toolkit troubleshooting: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html
- Docker resource constraints: https://github.com/docker/docs/blob/main/content/manuals/engine/containers/resource_constraints.md
- Compose services reference: https://github.com/docker/docs/blob/main/content/reference/compose-file/services.md
- NVIDIA driver 595.99.02 README, dynamic power management: https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/dynamicpowermanagement.html
- vLLM GitHub issues and PRs cited in the tables above, state read on 2026-11-15
- Model cards: Qwen3.6-35B-A3B, Qwen3-Coder-Next, Gemma 4 12B, gpt-oss-20b, Nemotron 3.5 Lightning, GLM-4.7-Flash (linked in the parser table)
- Unsloth Qwen3.6 docs (W4A4 on Marlin): https://unsloth.ai/docs/models/qwen3.6
- Artificial Analysis, Qwen3.8-27B: https://artificialanalysis.ai/models/qwen3-8-27b
- ModelFit RTX 5080 (desktop) calculator page, estimates only: https://modelfit.io/gpu/rtx-5080/
- The user's DGX Spark runbook (private; measured 2026-09-05 and 2026-09-07 on a GB10 with NGC vLLM 0.20.1)
- Full catalog with evidence classes: [sources.md](../docs/sources.md)

## Maintenance

Re-verify before relying on this document:

- The newest stable vLLM release and the `vllm/vllm-openai` digest; v0.34 changes `gpu_memory_utilization` to an extensible-KV default, which invalidates the [Memory budget](#memory-budget) arithmetic.
- The state of [#46268](https://github.com/vllm-project/vllm/issues/46268) and [#66864](https://github.com/vllm-project/vllm/pull/66864), [#61239](https://github.com/vllm-project/vllm/issues/61239), [#66264](https://github.com/vllm-project/vllm/issues/66264), [#54331](https://github.com/vllm-project/vllm/issues/54331) and [#66989](https://github.com/vllm-project/vllm/issues/66989); a fix to #66264 would reopen vLLM for MoE models above the budget.
- The installed NVIDIA driver branch against the image's CUDA version.
- After the first real run: replace every estimate here with the logged KV cache size, the peak host RAM from `docker stats`, and the smoke-test results, and say which numbers are measured on this laptop.
