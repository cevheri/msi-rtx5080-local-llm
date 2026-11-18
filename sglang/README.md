# SGLang on the RTX 5080 Laptop

Last verified: 2026-11-18.

SGLang v0.5.23 is the third engine here, after llama.cpp and vLLM: it serves Qwen3.5-9B NVFP4, gpt-oss-20b, Gemma 4 12B QAT W4A16 and Qwen3.6-35B-A3B NVFP4 at 128k context and K2 Horizon 7B AWQ-INT4 and Gemma 4 E4B QAT W4A16 at 64k as llmctl recipes ([models.md](models.md)).
One recipe is a material speed gain over the fastest other recipe of the same model.
Qwen3.6-35B-A3B decodes 151.3 tok/s through Paged Experts (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18; [record](../research/2026-11-18/sglang-qwen3.6-35b-a3b/README.md)) against 122.5 on llama.cpp, 24% faster (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-16; [llamacpp/models.md](../llamacpp/models.md#2-qwen36-35b-a3b)), but `./llmctl up qwen3.6` still starts llama.cpp, the default, and the SGLang recipe needs `--runtime sglang`.
The smoke-test model Gemma 4 E4B decodes 141.3 tok/s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [record](../research/2026-11-17/sglang-gemma-4-e4b/README.md)) against 45.2 on vLLM in eager mode (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [record](../research/2026-11-17/vllm-gemma-4-e4b/README.md)), but its llama.cpp GGUF build decodes 151.3 tok/s with 128k context (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17; [record](../research/2026-11-17/llamacpp-gemma-4-e4b/README.md)), so it is no gain.
None of the other five is, so SGLang's value here is that speed for one model, a third engine that cross-checks the other two, and Paged Experts, which keeps part of a MoE model's experts on the GPU and pages the rest from pinned host memory.
Every SGLang recipe shares its name with a llama.cpp or vLLM recipe, which `./llmctl up` prefers, so an SGLang recipe starts only with `--runtime sglang` ([llmctl](../docs/llmctl.md)).
The pinned image is `lmsysorg/sglang:v0.5.23-runtime`, started on demand with CDI GPU access, a localhost-only port, a hard RAM cap, the model cache mounted read-only and a model overlay that llmctl builds before each start.
The runs on this laptop are archived under [research/2026-11-17/sglang-runtime/](../research/2026-11-17/sglang-runtime/README.md): the hands-on run and measurements A and B on 2026-11-17, and measurements C, D and E, which ran past midnight, on 2026-11-17 and 2026-11-18.
Every figure without a measured label is read from a source or an estimate.
An `SG-<n>` beside a source names its row in the independent check of the SGLang sources, [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md); every row cited here has a verdict that starts with `Holds`.

Per-model commands live in [models.md](models.md), the measurement method and results in [benchmarking.md](benchmarking.md), the ranked picks in [model-selection.md](../docs/model-selection.md), host prerequisites in [host-setup.md](../docs/host-setup.md) and the OpenCode side in [opencode.md](../docs/opencode.md).

## When SGLang, when the other engines

| Situation | Engine | Why |
|---|---|---|
| A model that has a llama.cpp or vLLM recipe and fits in VRAM | llama.cpp or vLLM, the default | Qwen3.5-9B decodes 93.7 tok/s on SGLang against 90.8 on vLLM, K2 Horizon 7B 89.6 against 94.8 on vLLM, and gpt-oss-20b 124.6 against 166.2 on llama.cpp (measured on the MSI Raider 18 Max HX, each with its runtime, 2026-11-16 to 2026-11-18; [models.md](models.md)) |
| A second opinion on a model's behaviour, parsers or tokenizer | SGLang with `--runtime sglang` | A third, independent server stack with its own parsers, grammar backend and loaders |
| A MoE safetensors checkpoint larger than VRAM | SGLang with Paged Experts, when its format pages | gpt-oss-20b runs at 128k only with Paged Experts, and Qwen3.6-35B-A3B NVFP4 decoded 151.3 tok/s with 88 of 256 experts resident against 122.5 on llama.cpp (measured on the MSI Raider 18 Max HX, SGLang v0.5.23 on 2026-11-18 and llama.cpp b12407 on 2026-11-16; [record](../research/2026-11-18/sglang-qwen3.6-35b-a3b/README.md)); see [Paged Experts](#paged-experts) |
| A dense model larger than VRAM, or a GGUF file | llama.cpp | Paged Experts applies only to MoE layers, and `--cpu-offload-gb` "streams *every* offloaded weight to the GPU on every forward pass, whatever the routing" ([#42557](https://github.com/sgl-project/sglang/pull/42557), SG-80) (vendor-reported); SGLang does not serve GGUF here (no SGLang start used one) |
| A checkpoint format that SGLang v0.5.23 does not load on sm_120 | llama.cpp or vLLM | See [Quantization paths on sm_120](#quantization-paths-on-sm_120) for the formats that failed here |

Run one model server at a time: SGLang sizes its static memory from whatever GPU memory is free when it starts ([Memory budget](#memory-budget)), so a server that still holds memory shrinks the next one's KV pool instead of stopping it.

## Image and versions

| Item | Value | Source |
|---|---|---|
| SGLang release | v0.5.23, the newest stable release on 2026-11-18: tag commit 297e54fd of 2026-10-29, GitHub release published on 2026-11-02 (vendor-reported) | [release](https://github.com/sgl-project/sglang/releases/tag/v0.5.23), SG-01 |
| Image | `lmsysorg/sglang:v0.5.23-runtime`, a smaller image of the same stable release (vendor-reported) | [install.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/get-started/install.mdx), SG-14 |
| Index digest | `sha256:3ce6da3e5ea0fee2dac6a23b528cb294fedd31ca74a55a4511f709c5e01e8652` (vendor-reported), the digest pulled here (measured on the MSI Raider 18 Max HX, Docker 29.9.3, 2026-11-17) | [Docker Hub tag record](https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/v0.5.23-runtime), SG-04; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md) |
| Size | 14.47 GB compressed for amd64, against 16.11 GB for `v0.5.23` without `-runtime` (vendor-reported); 49.7 GB on disk under the containerd snapshotter (measured on the MSI Raider 18 Max HX, Docker 29.9.3, 2026-11-18; [host-setup.md](../docs/host-setup.md#disk-budget)) | [Docker Hub tag record](https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/v0.5.23-runtime), SG-04, SG-05 |
| CUDA | 13.0.3: "ARG CUDA_VERSION=13.0.3" (vendor-reported); the pulled image carries `CUDA_VERSION=13.0.3` and `NVIDIA_REQUIRE_CUDA=cuda>=13.0` (measured on the MSI Raider 18 Max HX, docker image inspect, 2026-11-17) | [Dockerfile](https://github.com/sgl-project/sglang/blob/v0.5.23/docker/Dockerfile), SG-07; [image-inspect.txt](../research/2026-11-17/sglang-runtime/handson/logs/image-inspect.txt) |
| Packages | sglang 0.5.23, torch 2.14.1+cu130, flashinfer 0.7.2, sglang-kernel 0.4.10, transformers 5.19.0, triton 3.8.0, xgrammar 0.2.7, and huggingface_hub 1.33.0 (measured on the MSI Raider 18 Max HX, in-container checks, 2026-11-17) | [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md), [design probes](../research/2026-11-17/sglang-runtime/spec-probes/offline-image-probe.txt) |
| FlashInfer kernels | The release builds pass "--build-arg INSTALL_FLASHINFER_JIT_CACHE=1" (vendor-reported) | [release workflow](https://github.com/sgl-project/sglang/blob/v0.5.23/.github/workflows/_docker-build-and-publish.yml), SG-10 |
| Entrypoint and user | No SGLang entrypoint: `/opt/nvidia/nvidia_entrypoint.sh` with the command `/bin/bash`, and no user, so the server runs as root (measured on the MSI Raider 18 Max HX, docker image inspect, 2026-11-17) | [image-inspect.txt](../research/2026-11-17/sglang-runtime/handson/logs/image-inspect.txt) |
| Server command | `python3 -m sglang.launch_server ...`, which works and warns that `sglang serve` is the recommended entrypoint (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17); the quick start documents `sglang serve` (vendor-reported) | [quickstart.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/get-started/quickstart.mdx), SG-15; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md) |
| Next version | Releases came on 2026-09-05, 09-18, 10-02, 10-19 and 11-02, so v0.5.24 is due; none was out on 2026-11-18 (inferred from the release dates) | [releases](https://github.com/sgl-project/sglang/releases), SG-02 |

Driver compatibility: NVIDIA lists CUDA 13.2 with driver branch R595 and CUDA 13.0 with R580, so driver 595.99 runs the CUDA 13.0.3 image without minor version compatibility (vendor-reported; [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html), SG-97), and `nvidia-smi` inside the container saw the RTX 5080 Laptop GPU on 595.99.02 through CDI (measured on the MSI Raider 18 Max HX, SGLang v0.5.23 image, 2026-11-17; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).
Do not use the moving tags `latest`, `dev` or `dev-cu13`, nor `nightly-cu134*`, which is built for CUDA 13.4 (driver branch R615), so 595.99 would rely on minor version compatibility (vendor-reported; [tag list](https://hub.docker.com/r/lmsysorg/sglang/tags), SG-06, SG-97).
CUDA 12 images ended with v0.5.19: "CUDA 12 images (`-cu12` / `-cu129`) are no longer published" (vendor-reported; [install.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/get-started/install.mdx), SG-13).
Prefer the image to a pip install: [#49209](https://github.com/sgl-project/sglang/issues/49209) reports that the CUDA 13 wheel install on RTX 5080 and 5090 "cannot serve any model without manual workarounds" (third-party report, SG-87), while the image ran here without any (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).

## Run template

Prerequisites: NVIDIA Container Toolkit installed and the CDI spec generated, as described in [host-setup.md](../docs/host-setup.md).
This is the hand launch of a recipe whose snapshot is already in the Hugging Face cache, with the eight variables of the runtime file `recipes/sglang/_runtime.toml`; the per-model flags are in [models.md](models.md):

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
  --model-path /root/.cache/huggingface/hub/models--<org>--<name>/snapshots/<revision> \
  --served-model-name <short-name> \
  --context-length <context> \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static <at most 0.90> \
  --max-running-requests 1 \
  --disable-prefill-cuda-graph \
  <the recipe's other flags>

docker logs -f sglang
```

What each part does:

- No `--restart` and no `--rm`: the server runs only while you need it, and a stopped container keeps its log; llmctl stops a container whose post-ready check fails and keeps it for `./llmctl logs`.
- `--ipc=host`: every measured start used it (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18); it shares the host's `/dev/shm`, which does not hold the Paged Experts store ([Paged Experts](#paged-experts)), and each start leaves one small file there ([Known issues](#known-issues)).
- `--memory` equal to `--memory-swap`: no swap; the cap must hold a Paged Experts recipe's pinned store, because that store is host memory ([Paged Experts](#paged-experts)).
- `-p 127.0.0.1:30000:30000` with `--host 0.0.0.0 --port 30000`: the server binds 127.0.0.1:30000 by default (vendor-reported; [serving.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/fields/serving.py), SG-37), which inside a container is unreachable, while Docker publishes the port on localhost only, because Docker-published ports bypass ufw ([host-setup.md](../docs/host-setup.md)).
- The Hugging Face cache read-only with `HF_HUB_OFFLINE=1`: with a snapshot path and the variable set, the loader skips downloads (vendor-reported; [weight_utils.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/model_loader/weight_utils.py), SG-65), and no log of the hands-on starts shows a write error on the read-only mount (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).
- `HF_MODULES_CACHE=/tmp/hf_modules`: a `--trust-remote-code` model copies its code there instead of into the read-only cache; K2 Horizon 7B loaded its remote code this way (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [measurement A](../research/2026-11-17/sglang-runtime/measure-a/README.md)).
- `MAX_JOBS=2`: caps parallel JIT compile jobs, as in the vLLM runtime file.
- `FLASHINFER_NO_DOWNLOAD=1`: FlashInfer 0.7.2 downloads a missing kernel artifact from NVIDIA's artifact repository unless the variable is set, and with it raises instead (inferred from [cubin_loader.py](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2/flashinfer/jit/cubin_loader.py), SG-94, SG-95); no cold start here raised (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18; [benchmarking.md](benchmarking.md#cold-starts-without-the-network)).
- `SYNC_TOKEN_IDS_ACROSS_TP=1` and `NCCL_MAX_P2P_NCHANNELS=1`: they move the NCCL allocation of the first grammar request into the server's own warm-up and shrink it; see [Grammar requests and NCCL](#grammar-requests-and-nccl).
- `TIKTOKEN_ENCODINGS_BASE`: gpt-oss's Harmony vocabulary from a copy in the image instead of a download from `openaipublic.blob.core.windows.net` (inferred from [openai-harmony 0.0.4](https://github.com/openai/harmony/blob/v0.0.4/src/tiktoken_ext/public_encodings.rs), SG-96); without it a start without a network serves chat but logs `OpenAI Responses API (/v1/responses) disabled` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [measurement B](../research/2026-11-17/sglang-runtime/measure-b/README.md)).
- `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`: stops PyTorch from rounding the Paged Experts store up to powers of two; see [Paged Experts](#paged-experts).
- `sglang-cache` volume at `/root/.cache/sglang`: keeps the FlashInfer autotune results and the Triton and JIT caches, so only the first start of a model pays the compile and autotune time (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [benchmarking.md](benchmarking.md#cold-starts-without-the-network)).
- `--mem-fraction-static`, `--max-running-requests 1` and `--disable-prefill-cuda-graph`: llmctl requires the first two, and `--disable-prefill-cuda-graph` while the recipe's `[evidence] kind` is not `measured`, so only a recipe measured with the prefill graph on may drop it; every shipped recipe keeps it; see [Memory budget](#memory-budget).

What llmctl adds to this launch: it mounts the model overlay `~/.cache/llmctl/sglang/overlays/<id>/` read-only at `/llmctl/model` and passes `--model-path /llmctl/model` instead of the snapshot path ([Sampling and the model overlay](#sampling-and-the-model-overlay)), adds its labels, and after `/ready` runs the [memory check](#readiness-and-the-memory-check) before it writes any client file.
`./llmctl show <id> --runtime sglang` prints the whole command for a recipe.

Downloads: the image has no `hf` executable, so `./llmctl pull <id> --runtime sglang` runs `python3 -m huggingface_hub.cli.hf download <repo> --revision <revision>` in a throwaway container of the image as your user (measured on the MSI Raider 18 Max HX, offline probe, 2026-11-17; [design probes](../research/2026-11-17/sglang-runtime/spec-probes/offline-image-probe.txt)); every shipped recipe uses the snapshot of its vLLM twin, so nothing new downloads for them.

Privacy: no usage telemetry was found in v0.5.23, only streaming usage statistics, OpenTelemetry tracing behind `--enable-trace`, which is off by default, and FastAPI telemetry that SGLang switches off (inferred from absence in every text file of the source archive; [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md), SG-77, SG-110).
gpt-oss's built-in web search needs `EXA_API_KEY`, which llmctl never sets (vendor-reported; [environ.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/environ.py), SG-53).
Socket tables read inside running containers listed only the container's own address, the bridge gateway and loopback, a snapshot rather than proof of no traffic (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).

## Memory budget

SGLang sizes its static memory, the weights plus the KV pool, the Mamba state and a small multimodal reserve, as a share of the GPU memory that is free before the weights load, not of the device total: the KV pool is "currently-free GPU memory minus the non-static runtime slack (pre_model_load_memory * (1 - mem_fraction_static))" (vendor-reported; [kv_cache_configurator.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/kv_cache_configurator.py), SG-44).

Every start here saw 15.04 to 15.06 GiB free before loading, of the 16,611,278,848-byte device total that vLLM also sees, every start of the other five shipped recipes saw 15.06 GiB, and every start of Qwen3.6-35B-A3B 15.04 GiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md), [a 15.04 GiB start of Qwen3.6-35B-A3B](../research/2026-11-18/sglang-qwen3.6-35b-a3b/measure-e/logs/e1-k88-f088-mem27.log)).
So the pool, the Mamba state and a 0.10 GiB multimodal reserve together come to 15.06 GiB x fraction minus the loaded weights: Qwen3.5-9B at 0.88 gives 13.25 - 8.39 = 4.86 GiB against 4.83 logged, K2 Horizon 7B at 0.90 gives the 5.66 GiB logged, and paged gpt-oss-20b at 0.90 gives 2.09 against 2.12 logged (estimate, checked against the hands-on start logs of 2026-11-17; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).
The rest, 15.06 GiB x (1 - fraction), is what the CUDA graphs, FlashInfer's workspace and the kernels loaded while serving take (inferred from the start logs; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)), and what is left of it is the free memory after ready that the [memory check](#readiness-and-the-memory-check) reads.
SGLang's "GB" in its log lines are GiB (inferred from the 15.06 GB free before loading against the device total).

The rules every SGLang recipe follows, with the measurements behind them (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17, unless a rule says otherwise):

- `--mem-fraction-static` is required and at most 0.90.
  Left unset it would be about 0.75 here: below 20 GiB SGLang assumes 2,048-token prefill chunks and a decode batch of 8 (vendor-reported; [memory_hook.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/memory_hook.py), SG-42) and reserves 512 + 2,048 x 1.5 + 128 MiB, plus 16 MiB for the decode graph (`max_bs` x 2) and 8 MiB for each of the 42 prefill graph shapes logged here, 4,064 of 16,303 MiB, so (16,303 - 4,064) / 16,303 = 0.751, and 0.771 without the prefill graphs (estimate from the vendor formula; SG-43).
  Fractions of 0.95 and 0.97, and 0.90 with the prefill graph on, left 0.40 to 0.67 GiB free and crashed on the first grammar request, while 0.88 and 0.90 with the prefill graph off left 0.98 to 1.45 GiB and passed ([hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).
- The prefill CUDA graph stays off (`--disable-prefill-cuda-graph`) until a recipe measures it: the default breakable prefill graph took 0.58 to 0.71 GiB here, and [#40094](https://github.com/sgl-project/sglang/issues/40094) reports that it "reserves ~1.8 GB and starves quantized-KV long-context prefill on small cards" (third-party report, SG-86).
  Prefill speed did not suffer: Qwen3.5-9B took 0.566 and 2.394 s to the first token for 8,591 and 32,153 tokens without it against 0.580 and 2.426 s with it ([hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).
- `--max-running-requests 1`: one user, and every measured start ran with it ([lane records](../research/2026-11-17/sglang-runtime/README.md)).
- Mamba slots for hybrid Gated DeltaNet models: Qwen3.5-9B got 41 Mamba state slots (2.0 GiB) by default even with one running request, while `--max-mamba-cache-size 8` took 0.43 GiB and kept prefix caching working, with a repeated 124,667-token prompt answered in 0.057 s ([hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).
- A low `--swa-full-tokens-ratio` for sliding-window models: the sliding-window layers get a pool of 0.8 of the full pool by default (vendor-reported; [schedule.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/fields/schedule.py), SG-38), and gpt-oss-20b holds 128k only at 0.1, where its FP8 pool held 169,853 full and 16,985 sliding-window tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [record](../research/2026-11-17/sglang-gpt-oss-20b/measure-c/README.md)).
- FP8 KV is `--kv-cache-dtype fp8_e4m3`, with no plain `fp8` choice (vendor-reported; [server_arguments.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/server_arguments.mdx), SG-20); it runs with scales of 1.0 ("Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0."), and on K2 Horizon 7B FP8 and BF16 KV both answered 20 of 20 fixed prompts with identical answers ([M13](../research/2026-11-17/sglang-runtime/measure-b/README.md#m13-fp8-against-bf16-kv)).
- `--language-model-only` is refused for the architectures used here (`--language-model-only does not support ['Qwen3_5ForConditionalGeneration']`), so the vision tower always loads: Qwen3.5-9B's weights take 8.39 GiB against 7.48 on vLLM ([hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).
- The pool must hold the context: SGLang starts with a shorter pool and only warns ("KV pool holds 74233 tokens, fewer than --context-length 131072"), and `/v1/models` still reports the full context as `max_model_len`, so llmctl reads `token_capacity` itself ([hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).
- At least `min_free_at_ready_mib` free after ready, 768 MiB in the runtime file: the threshold covers the largest drop in free memory that a long prompt, Hermes' title request, a forced tool request and 300 s of load caused after ready, plus 256 MiB, rounded up to a multiple of 64; the inputs and the arithmetic are in [benchmarking.md](benchmarking.md#post-ready-margins) (estimate from margins measured on 2026-11-17 and 2026-11-18).
  A recipe ships only when its own free memory after ready is at least 768 + 128 = 896 MiB, the 128 MiB being room for a desktop that holds a little more than the 112 MiB it held during the runs (estimate).
- When a start falls short, lower `--mem-fraction-static` by 0.02, which frees about 300 MiB (15.06 GiB x 0.02, estimate): K2 Horizon 7B went from 865 MiB free at 0.90 to 1,152 MiB at 0.88, 287 MiB more, while its pool shrank from 82,397 to 77,988 FP8 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18; [record](../research/2026-11-18/sglang-k2-horizon-7b/measure-d/README.md)).

The threshold holds only with both NCCL variables of the runtime file, `SYNC_TOKEN_IDS_ACROSS_TP=1` and `NCCL_MAX_P2P_NCHANNELS=1`: without them the first grammar request takes about 605 MiB after ready instead of about 100 MiB before it ([Grammar requests and NCCL](#grammar-requests-and-nccl)), which the 768 MiB do not cover.
llmctl does not refuse a runtime override that drops them, so such an override belongs to an image that contains the fix [#48487](https://github.com/sgl-project/sglang/pull/48487), where it may set `min_free_at_ready_mib = 0`, or must measure its own margins as [benchmarking.md](benchmarking.md#method) describes.

Weights load at least as large as on vLLM, and larger when SGLang cannot drop the vision tower: Qwen3.5-9B 8.39 GiB against 7.48, K2 Horizon 7B 7.89 against 7.72, and gpt-oss-20b 14.45 GiB on the default FlashInfer CUTLASS MXFP4 path or 13.86 on Marlin against 13.8 on vLLM's Marlin path (measured on the MSI Raider 18 Max HX, SGLang v0.5.23 and vLLM v0.33.0, 2026-11-16 and 2026-11-17; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md), vLLM's [Qwen3.5-9B](../research/2026-11-17/vllm-qwen3.5-9b/README.md), [K2 Horizon 7B](../research/2026-11-17/vllm-k2-horizon-7b/README.md) and [gpt-oss-20b](../research/2026-11-16/vllm-gpt-oss-20b/attempts.md) records).
Unlike vLLM's cold compile cache, cold and warm starts gave the same pools: 74,233 tokens for Qwen3.5-9B and 82,394 for K2 Horizon 7B in both (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).

## Sampling and the model overlay

SGLang v0.5.23 has no server flag that sets the default sampling of OpenAI chat requests: the chat layer fills temperature, top_p, top_k, min_p and repetition_penalty on every request by "Priority: user value > model generation_config > OpenAI defaults" (vendor-reported; [protocol.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/entrypoints/openai/protocol.py), SG-58), with the defaults temperature 1.0, top_p 1.0, top_k -1, min_p 0.0 and repetition_penalty 1.0 (SG-59).
`--preferred-sampling-params` is merged under the request's fields (vendor-reported; [tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/tokenizer_manager.py), SG-57), and chat requests arrive with temperature already set (SG-109), so it never reached a chat request: with `{"temperature":0.6,"top_p":0.95,"top_k":20}` field-less requests still ran at 1.0, 1.0 and -1 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [measurement A](../research/2026-11-17/sglang-runtime/measure-a/README.md)).

What works is the model's own `generation_config.json`, from which SGLang reads only `repetition_penalty`, `temperature`, `top_k`, `top_p` and `min_p`, and only while `--sampling-defaults` stays `model`, its default (vendor-reported; [model_config.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/configs/model_config.py), SG-67, SG-19).
So llmctl builds a model overlay for every start that runs `docker run`:

1. The folder is `<cache_dir>/sglang/overlays/<id>/`, by default `~/.cache/llmctl/sglang/overlays/<id>/`, mounted read-only at `/llmctl/model` with `--model-path /llmctl/model`.
2. Every file of the pinned snapshot becomes a symbolic link to its container path, `/root/.cache/huggingface/hub/models--<org>--<name>/snapshots/<revision>/<file>`, with subfolders recreated as folders of links.
3. When the recipe's `[server] sampling` table is not empty, the top-level `generation_config.json` is a real file instead: the snapshot's JSON object, or `{}` when the snapshot has none, with each sampling key replaced or appended.
4. The folder is built aside and renamed into place, from the snapshot as it is at that moment, so a file added by a wider pull or a new revision appears in the next start.

`up` prints the overlay line before `docker run`, for example `wrote ~/.cache/llmctl/sglang/overlays/qwen3.5-9b: model overlay with 11 links into snapshot a4df85a20fb7 of RedHatAI/Qwen3.5-9B-NVFP4, and generation_config.json with temperature 0.6, top_p 0.95, top_k 20`.
The overlay started Qwen3.5-9B with the snapshot's weights, pool and parsers, logged `Using default chat sampling params from model generation config: {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}`, and ran field-less requests at those values, while `top_k` 1 in the overlay gave five identical replies to five identical requests; K2 Horizon 7B with `--trust-remote-code` loaded its remote code from the overlay's links (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [measurement A](../research/2026-11-17/sglang-runtime/measure-a/README.md)).
A snapshot without `generation_config.json` loads with no defaults, because the loader catches the missing file (vendor-reported; [common.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/utils/hf_transformers/common.py), SG-66).

What follows for recipes and overrides:

- A recipe pins the twin recipe's sampling in `[server] sampling`; [models.md](models.md) gives each recipe's `Sampling:` line with the effective values.
- A request's own sampling fields still win, so a client that sends a temperature overrides the pin.
- llmctl refuses `--sampling-defaults openai`, which would ignore the file, and `--preferred-sampling-params`, which does not reach chat requests.
- A key SGLang does not read from the file, such as `presence_penalty`, cannot be pinned.
- The links resolve only inside the container, so llmctl checks the snapshot on the host, never the overlay; a snapshot `generation_config.json` that is not a JSON object refuses the start with exit 3 before anything stops, and the refusal says to delete that file and its blob from the Hugging Face cache, then pull again.
- `down` leaves the folder, a few KB of links that the next `up` of the recipe replaces.

## Readiness and the memory check

SGLang's endpoints during a start, from 0.5 s polls of 15 starts (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [measurement A](../research/2026-11-17/sglang-runtime/measure-a/README.md)):

| Endpoint | During the start | Once ready | Use |
|---|---|---|---|
| `/ready` | Connection refused, then reset by Docker's port proxy, then 503 from the moment the HTTP server listens | 200 in about 1 ms | llmctl's readiness path |
| `/health` | As `/ready`, and 200 at the same poll in all 15 starts | 200 after 1.0 s, because it sleeps a second and generates one token (vendor-reported; [http_server.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/entrypoints/http_server.py), SG-47) | Not used: slow, and it runs a generation |
| `/v1/models` | 200 1.0 to 2.0 s before `/ready` in warm starts and 13 s before it in a cold one | 200, the served name and `max_model_len` | The served name, after `/ready` |
| `/server_info` | 200 as early as `/v1/models` | `internal_states[0].memory_usage` with `token_capacity` | The memory check |
| `/get_server_info` | As `/server_info` | Deprecated, a warning per call (vendor-reported; SG-49) | Not used |

`/ready` and `/health` both wait for `ServerStatus.Up`, which the server sets after its own warm-up request (vendor-reported; [http_server.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/entrypoints/http_server.py), SG-46), while `/v1/models` answers without looking at the server status (SG-48), so it alone is no readiness signal.
llmctl's `ready` therefore asks `GET /ready` for 200 and then `GET /v1/models` for the served name, so a foreign server on port 30000 is not taken for the recipe.
Starts on a warm cache volume reached the ready line in 27 to 64 s, apart from measurement A's two gpt-oss-20b starts with a network and without `TIKTOKEN_ENCODINGS_BASE`, which took 106 and 116 s, 52 and 63 s of it in the Harmony vocabulary step ([measurement A](../research/2026-11-17/sglang-runtime/measure-a/README.md)); the first start of a model on that volume took up to 233 s, and cold starts from an empty volume took 46 to 351 s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18; [benchmarking.md](benchmarking.md#cold-starts-without-the-network)); the runtime file's `start_timeout_s` is 900.

After ready, and before it writes any client file, llmctl checks the started server once:

1. `GET /server_info`: `internal_states[0].memory_usage.token_capacity` must be an integer and at least the recipe's context.
2. `POST /flush_cache`: the idle server drops PyTorch's cached blocks and an empty prefix cache, which makes the reading repeatable: Qwen3.5-9B had 709 MiB free after a cold start and 1,013 after a warm one, and 1,039 after the flush in both (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [measurement A](../research/2026-11-17/sglang-runtime/measure-a/README.md)).
3. `nvidia-smi` free memory of the whole GPU must be at least `min_free_at_ready_mib`, 768 MiB ([Memory budget](#memory-budget)).

On success `up` prints `memory: KV pool <token_capacity> tokens for context <context>; <free> MiB of GPU memory free after the start, at least 768 MiB needed`.
A short pool, low free memory, an unreadable `/server_info`, a failed flush or a failed `nvidia-smi` reading ends `up` with exit 4: llmctl prints the refusal, stops the container, keeps it for `./llmctl logs`, and writes no client file; a short pool and low free memory are named in one refusal, so one override can fix both ([docs/llmctl.md](../docs/llmctl.md)).
The free memory counts every process on the GPU, so a browser or another CUDA program that took memory during the load can fail the check although preflight passed; close it, and the same `up` removes the stopped container, rebuilds the overlay and starts again.
An `up` of the recipe that is already running skips the check and says so.

`startup_available`, which the start log and `/server_info` report, is not an input: it is logged before the server's own warm-up and overstated Qwen3.5-9B's headroom by 0.34 GiB, against 40 and 56 MiB on K2 Horizon 7B and gpt-oss-20b (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [measurement A](../research/2026-11-17/sglang-runtime/measure-a/README.md)).
Without an API key the native endpoints `/server_info` and `/flush_cache` are open on 127.0.0.1, and "Health/readiness/metrics endpoints are always allowed" even with one (vendor-reported; [auth.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/utils/auth.py), SG-52).

## Grammar requests and NCCL

SGLang v0.5.23 crashes a server whose first grammar-constrained request, a `json_schema` response format or a forced `tool_choice`, finds too little GPU memory free.
Hermes sends a `json_schema` title request in every session, so every Hermes session reaches this path (inferred from [clients.md](../docs/clients.md)).

The defect, as measured here:

- The failing requests died in `layers/sampler.py:726 in _sync_token_ids_across_tp` with "NCCL error ... unhandled cuda error" and "Failed to CUDA calloc 536870912 bytes"; the scheduler exited, the client got HTTP 500, and the container was gone 6 s after the crash (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md), [measurement A](../research/2026-11-17/sglang-runtime/measure-a/README.md)).
- The sampler runs a token-id `all_reduce` whenever `SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars`, so no flag skips it while a grammar is active (vendor-reported; [sampler.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/layers/sampler.py), SG-55), and it "does not check the size of the sync group" ([#43302](https://github.com/sgl-project/sglang/pull/43302), SG-79) (third-party report).
- On one GPU that first `all_reduce` creates an NCCL communicator with 64 peer-to-peer channels and a 512 MiB shared buffer; a one-rank probe in the image took 672 MiB on it, and the first grammar request took 602 to 608 MiB in the server on all three models (measured on the MSI Raider 18 Max HX, NCCL 2.30.7 in the image and SGLang v0.5.23, 2026-11-17, the split of the buffer inferred; [measurement A](../research/2026-11-17/sglang-runtime/measure-a/README.md)).
- The fix, [#48487](https://github.com/sgl-project/sglang/pull/48487) ("[Fix] sampler: skip the token-id all-reduce when the TP sync group has one rank"), was merged on 2026-11-06, after the v0.5.23 tag, and no release contained it on 2026-11-18 (vendor-reported, SG-78).

The two variables of the runtime file work around it:

- `SYNC_TOKEN_IDS_ACROSS_TP=1` forces the `all_reduce` on every batch, so the server's own warm-up creates the communicator before ready and its memory is gone when the [memory check](#readiness-and-the-memory-check) reads (vendor-reported for the gating, SG-55; measured: Qwen3.5-9B had 407 instead of 1,013 MiB free after ready with this variable alone).
- `NCCL_MAX_P2P_NCHANNELS=1` cuts the buffer to 8 MiB and the whole allocation from about 605 MiB to about 100 MiB: with both variables Qwen3.5-9B had 911 MiB free after ready (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- With both, the first grammar request moved no memory on any model measured, and decode stayed the same: 89.5 against 89.9 tok/s on K2 Horizon 7B and 125.5 against 125.5 on gpt-oss-20b (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [measurement B](../research/2026-11-17/sglang-runtime/measure-b/README.md)).
- What remains after ready is the growth of PyTorch's cache during a long prefill plus kernels loaded while serving, which the post-ready threshold covers ([benchmarking.md](benchmarking.md#post-ready-margins)).

A grammar request after a long prompt is still the riskiest moment, and the check does not watch the server after ready: if a server dies on a grammar request later, `./llmctl status` shows the container exited, and `./llmctl logs` holds the NCCL lines above.
On the first release that contains #48487, drop `SYNC_TOKEN_IDS_ACROSS_TP` and measure the margins again ([Maintenance](#maintenance)).

## Flags reference

The flag names exist in v0.5.23 ([server_arguments.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/server_arguments.mdx)) (vendor-reported); "Measured start" means a start on this laptop used the flag (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18).

| vLLM v0.33.0 | SGLang v0.5.23 | Recommended here | Source |
|---|---|---|---|
| `--gpu-memory-utilization` | `--mem-fraction-static` | Required, at most 0.90: a share of the memory free before the weights load | SG-24, SG-44; [Memory budget](#memory-budget) |
| `--max-model-len` | `--context-length` | Owned by llmctl, from `[server] context` | Measured start |
| `--served-model-name` | `--served-model-name` | Owned by llmctl, equal to the OpenCode `modelID` | Measured start |
| `--max-num-seqs` | `--max-running-requests` | 1, required | Measured start |
| `--max-num-batched-tokens` | `--chunked-prefill-size` | 2,048 by default below 20 GiB; 1,024 for gpt-oss-20b | SG-42 |
| `--enforce-eager` | `--disable-prefill-cuda-graph` and `--disable-decode-cuda-graph` | Prefill graph off in every recipe; `--disable-cuda-graph` is deprecated in favour of the per-phase flags and is refused | SG-16 |
| `--enable-auto-tool-choice` | No such flag | `--tool-call-parser` alone turns tool calling on | SG-33 |
| `--tool-call-parser` | `--tool-call-parser` | An explicit name; `auto` is refused | SG-17, SG-88; [Tool calling and reasoning](#tool-calling-and-reasoning) |
| `--reasoning-parser` | `--reasoning-parser` | An explicit name for every thinking model | SG-17 |
| `--default-chat-template-kwargs` | `--default-chat-template-kwargs` | `{"enable_thinking":false}` for toggle models; a request's own keys win | SG-25 |
| `--kv-cache-dtype fp8` | `--kv-cache-dtype fp8_e4m3` | BF16 where the pool holds the context, else FP8 | SG-20 |
| `--language-model-only` | `--language-model-only` | Refused for Qwen3.5, whose error lists four other supported architectures; leave it out | Measured start |
| `--override-generation-config` | No equivalent | The model overlay ([Sampling and the model overlay](#sampling-and-the-model-overlay)); `--sampling-defaults` stays `model` | SG-19, SG-67 |
| `--reasoning-effort-map` | Not needed | gpt-oss-20b answered every effort from `none` to `xhigh` | Measured start |
| No vLLM flag | `--max-mamba-cache-size` | 8 for Gated DeltaNet models | SG-26 |
| No vLLM flag | `--swa-full-tokens-ratio` | 0.1 for gpt-oss-20b (default 0.8) | SG-26, SG-38 |
| `--cpu-offload-gb` | `--enable-paged-experts --paged-experts-num-resident K` with `--moe-runner-backend marlin` | For MoE checkpoints larger than VRAM | SG-21, SG-28; [Paged Experts](#paged-experts) |
| `--trust-remote-code` | `--trust-remote-code` | Only for repositories that need it, as K2 Horizon 7B does | Measured start |
| `--api-key` | `--api-key`, `--admin-api-key` | Refused by llmctl, as is `--config`, whose YAML file can carry any option | SG-18 |
| No vLLM flag | `--watchdog-timeout` | Default 300 s: "If a forward batch takes longer than this, the server will crash to prevent hanging." | SG-22 |

The parser is Python's argparse with its default abbreviation matching, so `--context-len` reaches `--context-length` and a flag given twice keeps the last value (inferred from [server_args.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/server_args.py), SG-41); `--model` is an alias of `--model-path` (SG-39).
That is why llmctl refuses a recipe that names a checked flag twice or in a shortened form ([docs/llmctl.md](../docs/llmctl.md)).

## Tool calling and reasoning

The parser names below exist in v0.5.23 (vendor-reported; [parser_names.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/function_call/parser_names.py), [reasoning_parser_names.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/parser/reasoning_parser_names.py), SG-69).
Three differ from vLLM's: `gpt-oss` for `openai` and `openai_gptoss`, `deepseek-r1` for `deepseek_r1`, and `nemotron_3` for `nemotron_v3` (inferred from both registries, SG-70).

| Recipe and thinking type | `--tool-call-parser` | `--reasoning-parser` | `--default-chat-template-kwargs` | Behaviour on this laptop |
|---|---|---|---|---|
| `qwen3.5-9b`, toggle, off by default | `qwen3_coder` | `qwen3` | `{"enable_thinking":false}` | `chat_template_kwargs.enable_thinking` true and false honoured; `reasoning_effort` `none` keeps thinking off, `low` and `high` turn it on |
| `k2-horizon-7b`, always | `k2_horizon` | `k2_horizon` | `{"reasoning_effort":"high","tool_call_format":"xml"}` | Reasoned for every effort and switch |
| `gpt-oss-20b`, effort | `gpt-oss` | `gpt-oss` | None | Every effort from `none` to `xhigh` answered HTTP 200, and `none` still reasons |

The behaviour column is measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18 ([models.md](models.md) links each record).

How the server behaves in v0.5.23:

- Replies carry the chain of thought as `reasoning_content`, never `reasoning` (vendor-reported; [separate_reasoning.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/separate_reasoning.mdx), [protocol.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/entrypoints/openai/protocol.py), SG-35, SG-60), and the key is present even with thinking off; streams carry `reasoning_content` deltas (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).
- Any `reasoning_effort` other than `none` turns `enable_thinking` on at request validation, while a request's explicit `chat_template_kwargs` keys still win (vendor-reported; SG-61), so a client's effort setting turns a toggle model's thinking on although the server default is off (measured on Qwen3.5-9B, MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).
- Tool calls come back with an `id` of the form `call_` plus 24 hex digits, `arguments` as a JSON string, `content` `""` and `finish_reason` `tool_calls`, and the second turn with the tool result worked on Qwen3.5-9B, K2 Horizon 7B and gpt-oss-20b (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).
- `"tools": []`, `tool_choice` `auto` without tools, `tool_choice` `none` with tools, `prompt_cache_key`, `max_completion_tokens` and a wrong model name all get HTTP 200, where vLLM refuses the first two with HTTP 400; a `max_tokens` above the context gets HTTP 400 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md), [K2 Horizon 7B in measurement D](../research/2026-11-18/sglang-k2-horizon-7b/measure-d/README.md)).
- A forced `tool_choice` on gpt-oss-20b comes back with HTTP 200, `finish_reason` `stop`, no `tool_calls` and raw Harmony text in `content` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17), as [#41747](https://github.com/sgl-project/sglang/issues/41747) reports (third-party report, SG-89); no client forces `tool_choice`, and every forced choice takes the [grammar path](#grammar-requests-and-nccl).
- K2 Horizon 7B's template refuses a replayed assistant turn that carries only `reasoning` with HTTP 400 ("Assistant message is missing a thinking field. ...") and accepts it with `reasoning_content` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18), which fits [#44931](https://github.com/sgl-project/sglang/issues/44931): "the template reads `thinking`, SGLang only forwards `reasoning_content`" (third-party report, SG-90).
- A request with tools to a recipe without `--tool-call-parser` gets HTTP 200 and any call as text in `content`, instead of vLLM's HTTP 400 (inferred from serving_chat.py; [serving_chat.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/entrypoints/openai/serving_chat.py), SG-63); no SGLang chat-only recipe ships, so this was not run.
- Tool choice runs on xgrammar, "the default grammar backend" (vendor-reported; [tool_parser.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/tool_parser.mdx), SG-34), and `json_schema` replies were valid JSON on all six recipes (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18; measurement C's [Qwen3.6-35B-A3B](../research/2026-11-18/sglang-qwen3.6-35b-a3b/measure-c/README.md) and measurement B's [Qwen3.5-9B](../research/2026-11-17/sglang-qwen3.5-9b/measure-b/README.md), [K2 Horizon 7B](../research/2026-11-17/sglang-k2-horizon-7b/measure-b/README.md), [gpt-oss-20b](../research/2026-11-17/sglang-gpt-oss-20b/measure-b/README.md), [Gemma 4 12B](../research/2026-11-17/sglang-gemma-4-12b/measure-b/README.md) and [Gemma 4 E4B](../research/2026-11-17/sglang-gemma-4-e4b/measure-b/README.md) records).

What each client sends and how its entry is written is in [clients.md](../docs/clients.md) and [opencode.md](../docs/opencode.md).

## Quantization paths on sm_120

The README lists "RTX 30/40/50 series" among the supported GPUs (vendor-reported; [README](https://github.com/sgl-project/sglang/blob/v0.5.23/README.md), SG-12), and the ahead-of-time kernels are built for `sm_120a` (vendor-reported; [CMakeLists.txt](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/kernels/aot/CMakeLists.txt), SG-75).
The tree holds tuned fused-MoE configs for the RTX 4090 and the RTX PRO 6000 Blackwell but none for a GeForce RTX 50 GPU, so MoE decode may run below what the card allows (inferred from the tree listing, SG-76).

| Checkpoint format | Model here | Outcome on v0.5.23 | Record |
|---|---|---|---|
| NVFP4 W4A16 compressed-tensors, dense | Qwen3.5-9B | Runs; FlashInfer attention, Triton kernels for the Gated DeltaNet layers, and an FP4 GEMM runner on `auto` that the log does not name | [record](../research/2026-11-17/sglang-qwen3.5-9b/README.md) |
| AWQ int4 | K2 Horizon 7B | Runs natively as `K2HorizonForCausalLM` with FlashInfer attention | [record](../research/2026-11-18/sglang-k2-horizon-7b/README.md) |
| MXFP4 MoE | gpt-oss-20b | Runs; the default path logged "Detected SM120 and MXFP4 quantization format for GPT-OSS model, enabling FlashInfer CUTLASS MXFP4 MOE kernel." and loaded 14.45 GiB, the Marlin runner 13.86 GiB, and Paged Experts needs Marlin | [record](../research/2026-11-17/sglang-gpt-oss-20b/README.md) |
| ModelOpt NVFP4 MoE | Qwen3.6-35B-A3B | Runs with Paged Experts on Marlin as W4A16 | [record](../research/2026-11-18/sglang-qwen3.6-35b-a3b/README.md) |
| compressed-tensors WNA16 MoE with asymmetric zero points | Mellum2.1 12B-A2.5B AWQ | Fails while loading: `RuntimeError: The size of tensor a (288) must match the size of tensor b (28)` | [record](../research/2026-11-17/sglang-mellum2.1-12b/README.md) |
| Mixed NVFP4 compressed-tensors MoE | Groot-19B-A3B | Fails after loading: `AttributeError: 'CompressedTensorsConfig' object has no attribute 'group_size'` | [record](../research/2026-11-17/sglang-groot-19b-a3b/README.md) |
| BF16 with online `--quantization fp8` | DeepSeek-R1-0528-Qwen3-8B | Runs out of memory before loading: the model is created in BF16 first, 15.26 GiB | [record](../research/2026-11-17/sglang-deepseek-r1-0528-qwen3-8b/README.md) |
| int4 MoE with gated GELU | Gemma 4 26B-A4B | Fails: `AssertionError: Only gated SiLU/SiTU is supported, got gelu.` in the Marlin MoE runner | [record](../research/2026-11-18/sglang-gemma-4-26b-a4b/README.md) |
| Quantized MoVA | K2 Horizon MoVA | Fails: `ValueError: K2 Horizon MoVA supports unquantized bf16/fp16 weights only` | [record](../research/2026-11-18/sglang-k2-horizon-mova-36b/README.md) |
| QAT W4A16 Gemma 4 with an audio tower | Gemma 4 E4B, Gemma 4 12B | Fail while building the audio tower or its projection; load with `--json-model-override-args '{"audio_config": null}'` | [E4B](../research/2026-11-17/sglang-gemma-4-e4b/README.md), [12B](../research/2026-11-17/sglang-gemma-4-12b/README.md) |

Every outcome in this table is measured on the MSI Raider 18 Max HX, SGLang v0.5.23, on the date of its record.
The Gemma 4 E4B audio encoder is built with the checkpoint's quantization config, and only when the config has an `audio_config` (vendor-reported; [gemma4_mm.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/gemma4_mm.py), SG-73), which is why the override helps.
For ModelOpt NVFP4 MoE checkpoints, [#45520](https://github.com/sgl-project/sglang/pull/45520) reports that they "crash at warmup on SM120 with the default `--moe-runner-backend auto`" (third-party report, SG-85); Paged Experts sets `--moe-runner-backend marlin` anyway.
FP8 KV scales from a checkpoint are ignored for GPT-OSS, Qwen3-MoE and others, and "v0.5.23 has the same code" ([#48749](https://github.com/sgl-project/sglang/issues/48749), SG-93) (third-party report), so FP8 KV runs with unit scales; prefer BF16 KV where the pool holds the context, and compare FP8 against BF16 on fixed prompts before shipping FP8 ([benchmarking.md](benchmarking.md#method)).

## Paged Experts

Paged Experts keeps K experts per MoE layer on the GPU and pages the rest from pinned host memory, and "Computation stays on the GPU." (vendor-reported; [paged_experts.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/paged_experts.mdx), SG-27).
It arrived in v0.5.23 with [#42557](https://github.com/sgl-project/sglang/pull/42557), merged on 2026-10-19 (SG-80), and the MXFP4 and NVFP4 formats with [#45787](https://github.com/sgl-project/sglang/pull/45787), merged on 2026-10-25 (SG-84) (vendor-reported).

What it accepts and costs (vendor-reported; [paged_experts.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/paged_experts.mdx)):

- Formats: MXFP4 and ModelOpt NVFP4 experts page only with `--moe-runner-backend marlin`, ModelOpt NVFP4 running as W4A16; GPTQ, AWQ and compressed-tensors W4A16 go through Marlin, and unquantized, FP8 and INT8 experts through Triton, and other quantization methods and MoE runners are rejected at startup (SG-28).
- Rejected at startup: tensor, expert, pipeline and data parallelism, fused shared experts (use `--disable-shared-experts-fusion`) and speculative decoding (SG-29).
- `--paged-experts-num-resident` is required with `--enable-paged-experts`, an integer of at least the router's top-k, with no `auto` (SG-21), and v0.5.23 has no disk tier and no host-memory cap (SG-108).
- Each resident expert costs its bytes times the number of MoE layers of the static budget (SG-31), so K trades KV pool for decode speed.
- The host store holds every expert and "is page-locked, so the host must be able to pin that much memory" (SG-30).
- Upstream reports GSM8K 0.96 to 0.97 for gpt-oss-20b with 16 of 32 experts resident against 0.975 unpaged, and "Accuracy is unchanged" (SG-32); accuracy was not measured here.
- On a desktop RTX 5070 Ti, #42557 reports "205.7 vs 194.4 tok/s single-stream decode" against tuned llama.cpp on Qwen3-30B-A3B int4, and that llama.cpp has the lower time to first token on short prompts (SG-81).

Choosing K on gpt-oss-20b at fraction 0.90 with FP8 KV and 128k (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [M10](../research/2026-11-17/sglang-gpt-oss-20b/measure-b/README.md)):

| K of 32 | Weights on the GPU | `token_capacity` | Decode |
|---|---|---|---|
| 16 | 8.842 GiB | 376,049 | 87.9 tok/s |
| 20 | 10.209 GiB | 267,443 | 112.3 tok/s |
| 24 | 11.455 GiB | 168,456 | 125.5 tok/s |
| 25 | 11.779 GiB | 142,701 | 130.0 tok/s |
| 28 | 12.787 GiB | 62,643 | 137.6 tok/s |

Each resident expert slot cost 0.33 GiB of GPU weights, about 26,000 FP8 tokens of pool, and decode rose with K because fewer experts are paged in per token (arithmetic on the measured figures); the recipe keeps K 24, the one that went through the margin protocol, with 37,384 tokens to spare.
Choosing K on Qwen3.6-35B-A3B NVFP4 at 128k (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18; [M11](../research/2026-11-18/sglang-qwen3.6-35b-a3b/measure-c/README.md), and the 977 MiB after the flush from [measurement E](../research/2026-11-18/sglang-qwen3.6-35b-a3b/measure-e/README.md)):

| K of 256 | Fraction and KV | Weights on the GPU | `token_capacity` | Free after ready | Decode |
|---|---|---|---|---|---|
| 64 | 0.90, FP8 (the checkpoint's choice) | 7.760 GiB | 535,616 | 585 MiB | 132.3 tok/s |
| 88 | 0.90, BF16 | 9.480 GiB | 177,593 | 631 MiB | 149.5 tok/s |
| 96 | 0.90, BF16 | 9.980 GiB | 152,403 | 659 MiB | 158.6 tok/s |
| 104 | 0.90, BF16 | 10.723 GiB | 113,491 | 677 MiB | 164.1 tok/s |
| 88 | 0.88, BF16 (the recipe) | 9.48 GiB | 161,736 | 951 MiB, 977 after the flush | 151.3 tok/s |

Each resident slot cost about 0.078 GiB of GPU weights, about 4,000 BF16 tokens of pool (arithmetic on the measured figures); K 104 falls short of 128k, and at fraction 0.90 every K left less than the 896 MiB gate free, so the recipe runs K 88 at 0.88, which holds 128k with 30,664 tokens to spare.
Pick K the same way for a new model: the largest K whose pool still holds the context, then measure the margins and lower K or the fraction if the post-ready gate fails.

The pinned store against the container cap:

- The store is shared memory: the scheduler holds it in shared mappings of `/dev/zero (deleted)`, which is how the CUDA driver's pinned allocations appear, counted as cgroup `shmem` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; that `cudaHostAlloc` makes these mappings is inferred; [record](../research/2026-11-17/sglang-gpt-oss-20b/measure-c/README.md)).
- So it counts against `--memory`, and a recipe's `ram_gib` is the measured peak of cgroup anon plus shmem, rounded up, which must fit in `memory_gib`: gpt-oss-20b has `ram_gib` 16 and `memory_gib` 20, and its run under a 20 GiB cap passed the margin protocol and 300 s of load without an OOM kill at a 17.04 GiB cgroup peak (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- It does not sit in `/dev/shm` under `--ipc=host`: the host's `/dev/shm` grew by 29 to 37 MiB at ready in the SGLang starts and fell back after the container was gone, while the store was 10 to 26 GiB, so the size of `/dev/shm` does not bound it (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18; [measurement C](../research/2026-11-17/sglang-runtime/measure-c/README.md)).
- `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` in the runtime file brought gpt-oss-20b's store from 19.19 to 10.37 GiB and its non-reclaimable peak from 24.35 to 15.45 GiB at the same decode and margins (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17), because the image's PyTorch 2.14.1 otherwise rounds each pinned tensor of the store up to a power of two (inferred from the image's PyTorch source, `CachingHostAllocator.h` lines 340 to 349).
  With the setting the stores measured 10.37 GiB for gpt-oss-20b and 16.97 GiB for Qwen3.6-35B-A3B; the setting leaves pinned requests of at most 1 MiB rounded as before (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18; the second half inferred from the same source).
- Qwen3.6-35B-A3B's store holds all 256 experts of all 40 layers as 16.97 GiB of shared memory at every K, which the scheduler held as 17.02 GiB of `RssShmem` in `/dev/zero (deleted)` mappings while the host's `/dev/shm` grew by only 29 to 37 MiB; with about 5.7 GiB of process memory beside it, anon plus shmem peaked at 22.70 GiB in measurement C's warm protocol runs and 22.76 GiB in measurement E, and at 22.96 GiB in C's two cold starts, 23.19 GiB with the kernel's share, under a 32 GiB cap, so the recipe has `ram_gib` 23, which covers all of them, and `memory_gib` 27, and runs of the whole protocol and 300 s of load under a 27 GiB cap had no OOM kill, at non-reclaimable peaks of 22.92 and 22.98 GiB with the kernel's share, the cap squeezing only the page cache (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18; [record](../research/2026-11-18/sglang-qwen3.6-35b-a3b/README.md)).
  No cold start ran under the 27 GiB cap; that one fits there is arithmetic on the 23.19 GiB peak, with 3.8 GiB to spare (estimate).
- Docker's default memlock limit did not block the pinning (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md)).

With CUDA graphs off, the first Marlin compile happens on the first request and "can exceed the scheduler watchdog" ([#42557](https://github.com/sgl-project/sglang/pull/42557), SG-83) (vendor-reported); every recipe keeps the decode graph on.

## Known issues

State as read on GitHub on 2026-11-18: #48487 was merged, every other issue and pull request below was open and was opened by an outside contributor, so its quote is a third-party report ([sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md)).

| Issue | Effect here | Workaround |
|---|---|---|
| The first grammar request allocates NCCL memory; fix [#48487](https://github.com/sgl-project/sglang/pull/48487) merged after v0.5.23 (SG-78) | The server crashes when about 0.6 GiB is not free at that moment | The runtime file's two NCCL variables and the post-ready check ([Grammar requests and NCCL](#grammar-requests-and-nccl)) |
| [#41747](https://github.com/sgl-project/sglang/issues/41747) gpt-oss `tool_choice: "required"` returns raw Harmony control tokens as content (SG-89) | A forced tool call on gpt-oss-20b comes back as text with `finish_reason` `stop` (measured) | Do not force `tool_choice`, which no client sends |
| [#44931](https://github.com/sgl-project/sglang/issues/44931) gpt-oss drops replayed reasoning: "the template reads `thinking`, SGLang only forwards `reasoning_content`" (SG-90) | gpt-oss-20b loses its earlier chain of thought between tool turns, which costs quality, not a failure | None known |
| [#40739](https://github.com/sgl-project/sglang/issues/40739) `qwen3_coder` non-streaming parser drops visible text after a tool call: "Non-streaming returns `content="before"`" (SG-91) | Text after a tool call can vanish in non-streamed replies on Qwen recipes | Prefer streaming clients (OpenCode streams) |
| [#42791](https://github.com/sgl-project/sglang/issues/42791) the `gemma4` tool-call parser loop "blocks the whole server" on a stray `]` (SG-92) | A Gemma 4 recipe could hang the server | Both Gemma 4 recipes pass `--tool-call-parser gemma4`, and their measured tool calls did not hit the hang (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [models.md](models.md#gemma-4-12b-it-qat-w4a16)); watch for a server that stops answering during a tool call in e2e sessions |
| [#48749](https://github.com/sgl-project/sglang/issues/48749) checkpoint FP8 KV scales are ignored for GPT-OSS, Qwen3-MoE and others, and "v0.5.23 has the same code" (SG-93) | FP8 KV runs with unit scales | Prefer BF16 KV where the pool allows; compare FP8 and BF16 on fixed prompts (M13) |
| [#45520](https://github.com/sgl-project/sglang/pull/45520) ModelOpt NVFP4 MoE checkpoints "crash at warmup on SM120 with the default `--moe-runner-backend auto`" (SG-85) | NVFP4 MoE checkpoints such as Qwen3.6-35B-A3B | `--moe-runner-backend marlin`, which Paged Experts needs anyway |
| [#40094](https://github.com/sgl-project/sglang/issues/40094) the prefill CUDA graph "reserves ~1.8 GB and starves quantized-KV long-context prefill on small cards" (SG-86) | The prefill graph took 0.58 to 0.71 GiB here and left too little for the first grammar request (measured) | `--disable-prefill-cuda-graph`, in every shipped recipe; llmctl requires it until a recipe measures the prefill graph |
| [#49209](https://github.com/sgl-project/sglang/issues/49209) the CUDA 13 wheel install on RTX 5080 and 5090 "cannot serve any model without manual workarounds" (SG-87) | None with the image | Use the pinned image, not pip |
| The Harmony vocabulary download: openai-harmony fetches `o200k_base.tiktoken` at every gpt-oss start unless `TIKTOKEN_ENCODINGS_BASE` is set (SG-96) | Without the variable a start without a network disables `/v1/responses`, and a start with a network spent 52 to 63 s in that step (measured; that the time is the download is inferred) | `TIKTOKEN_ENCODINGS_BASE` in the runtime file |
| Every start under `--ipc=host` leaves one root-owned `sglang_loads_<hex>_<hex>.shm` file of 16,396 bytes in the host's `/dev/shm` | The files pile up until a reboot, which clears `/dev/shm` (measured; that a reboot clears `/dev/shm`, a tmpfs, is inferred) | None needed; llmctl does not remove root's files, and `sudo rm /dev/shm/sglang_loads_*.shm` removes them while no SGLang server runs |

"Measured" in this table means measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18 ([hands-on record](../research/2026-11-17/sglang-runtime/handson/README.md), [measurement A](../research/2026-11-17/sglang-runtime/measure-a/README.md), [measurement B](../research/2026-11-17/sglang-runtime/measure-b/README.md), [measurement C](../research/2026-11-17/sglang-runtime/measure-c/README.md), [K2 Horizon 7B in measurement D](../research/2026-11-18/sglang-k2-horizon-7b/measure-d/README.md)).

## Smoke tests

Run these after every new recipe or flag change, in order.

1. Readiness and the served name:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:30000/ready   # 200 once ready, 503 while starting
curl -s http://127.0.0.1:30000/v1/models | jq -r '.data[].id'
```

2. The chat template works and thinking follows the recipe's default:

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<short-name>",
       "messages":[{"role":"user","content":"Reply with exactly: OK"}],
       "max_tokens":64}' \
  | jq '{content: .choices[0].message.content, reasoning: .choices[0].message.reasoning_content, finish: .choices[0].finish_reason}'
```

Expected for a toggle recipe: `content` `OK`, empty `reasoning_content` and `finish_reason` `stop`; K2 Horizon 7B and gpt-oss-20b always add some reasoning.

3. A tool call is parsed:

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<short-name>",
       "messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
       "tools":[{"type":"function","function":{"name":"get_weather",
         "description":"Get the current weather for a city",
         "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
       "max_tokens":1024}' \
  | jq '{finish: .choices[0].finish_reason, calls: .choices[0].message.tool_calls}'
```

Expected: `finish_reason` `tool_calls` and `get_weather` with `{"city": "Paris"}`; if the call arrives as text in `content`, the tool parser is missing or wrong.

4. The pool and the free memory, as the memory check reads them:

```bash
curl -s http://127.0.0.1:30000/server_info | jq '.internal_states[0].memory_usage.token_capacity'
curl -s -X POST http://127.0.0.1:30000/flush_cache
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits   # at least 768
```

Then send a `json_schema` request, as Hermes' title request does, after a long prompt, and check that the server is still up; [benchmarking.md](benchmarking.md#method) has the full protocol.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `up` exits 4 with `the KV pool holds <n> tokens, fewer than the context` | Weights and graphs leave too little pool at this fraction | An override with a smaller context, FP8 KV or fewer Mamba slots ([Memory budget](#memory-budget)) |
| `up` exits 4 with `only <n> MiB of GPU memory is free after the start` | Another program holds GPU memory, or the fraction is too high for this model | Close other GPU programs (`nvidia-smi`), or an override with a fraction 0.02 lower ([Memory budget](#memory-budget)) |
| `up` exits 4 with `could not read internal_states[0].memory_usage.token_capacity` | The server answered `/server_info` without the field, or not at all | `./llmctl logs`; a newer image may have moved the field ([Readiness and the memory check](#readiness-and-the-memory-check)) |
| The server exits on the first `json_schema` or forced tool request, with "Failed to CUDA calloc 536870912 bytes" in the log | The NCCL allocation of the [grammar path](#grammar-requests-and-nccl) found too little memory | Keep both NCCL variables in the runtime file and the post-ready check on |
| "Raise --mem-fraction-static above ..." at start | The weights leave no KV memory at that fraction | A smaller model build, Paged Experts, or a different engine; above 0.90 the grammar path crashes |
| "--language-model-only does not support [...]" | The flag is supported for four other architectures only | Drop it; the vision tower loads |
| "KV pool holds <n> tokens, fewer than --context-length" in the log | The pool is short; SGLang starts anyway | llmctl refuses it after ready; lower the context or use FP8 KV |
| A start sits for minutes on "Capture ... CUDA graph" | First start on an empty `sglang-cache` volume compiles and autotunes kernels | Keep the volume; cold starts take up to about 6 minutes ([benchmarking.md](benchmarking.md#cold-starts-without-the-network)) |
| A request hangs and the server crashes after 300 s | A forward batch passed `--watchdog-timeout` (SG-22), for example a first Marlin compile with CUDA graphs off (SG-83) | Keep the decode graph on |
| Tool calls arrive as plain text | `--tool-call-parser` missing or wrong | Use the parser of the [table above](#tool-calling-and-reasoning) |
| `/v1/responses` answers 404 or logs "disabled" for gpt-oss | The Harmony vocabulary could not be loaded | Keep `TIKTOKEN_ENCODINGS_BASE` in the runtime file |
| Exit 137 during load | The container's `--memory` cap is below the pinned store plus the process | Raise `memory_gib` to the measured peak of anon plus shmem with margin ([Paged Experts](#paged-experts)) |
| Port 30000 reachable from the LAN | Published without `127.0.0.1:` | Publish on `127.0.0.1:30000:30000` ([host-setup.md](../docs/host-setup.md)) |

## Sources

- SGLang v0.5.23 release and tag: https://github.com/sgl-project/sglang/releases/tag/v0.5.23
- Docker Hub tag record: https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/v0.5.23-runtime and the tag list https://hub.docker.com/r/lmsysorg/sglang/tags
- SGLang Dockerfile and release workflow at v0.5.23: https://github.com/sgl-project/sglang/blob/v0.5.23/docker/Dockerfile and https://github.com/sgl-project/sglang/blob/v0.5.23/.github/workflows/_docker-build-and-publish.yml
- SGLang docs at v0.5.23 (README, install, quickstart, server_arguments, paged_experts, tool_parser, separate_reasoning): https://github.com/sgl-project/sglang/tree/v0.5.23/docs/docs
- SGLang source at v0.5.23, the files linked above: https://github.com/sgl-project/sglang/tree/v0.5.23/python/sglang
- Pull requests #48487, #43302, #42557, #45787 and #45520, and issues #40094, #49209, #48172, #41747, #44931, #40739, #42791 and #48749, state read on 2026-11-18
- FlashInfer v0.7.2 `cubin_loader.py`: https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2/flashinfer/jit/cubin_loader.py
- openai-harmony v0.0.4: https://github.com/openai/harmony/blob/v0.0.4/src/tiktoken_ext/public_encodings.rs
- CUDA toolkit release notes (driver mapping): https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- The independent check of every source row cited here: [sources-check.md](../research/2026-11-17/sglang-runtime/sources-check.md)
- The records of this laptop: [research/2026-11-17/sglang-runtime/](../research/2026-11-17/sglang-runtime/README.md), [research/2026-11-17/](../research/2026-11-17/README.md) and [research/2026-11-18/](../research/2026-11-18/README.md)
- Full catalog with evidence classes: [sources.md](../docs/sources.md)

## Maintenance

Re-verify before relying on this document:

- On every pin bump, re-run the probe set of [benchmarking.md](benchmarking.md#method) for every recipe: readiness, the pool, the margins, the grammar requests, decode, M12 and a cold start without the network.
- On the first release that contains [#48487](https://github.com/sgl-project/sglang/pull/48487), drop `SYNC_TOKEN_IDS_ACROSS_TP` from the runtime file and measure `min_free_at_ready_mib` again; the variable then only costs an `all_reduce` per batch.
- The state of the issues in [Known issues](#known-issues); a fix to #41747 or #44931 changes how gpt-oss-20b behaves in tool sessions.
- A newer PyTorch in the image: check whether the pinned allocator still rounds up to powers of two, and whether `pinned_max_round_threshold_mb` still exists.
- The installed NVIDIA driver branch against the image's CUDA version.
- A new tuned fused-MoE config for a GeForce RTX 50 GPU in the tree would change MoE decode speed.
