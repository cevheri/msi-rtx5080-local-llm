# llama.cpp server on the RTX 5080 Laptop

Last verified: 2026-11-18.

Run llama.cpp as `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407` (CUDA 12.8.1), with `-e GGML_CUDA_DISABLE_GRAPHS=1` placed before the image name, the port published on 127.0.0.1 only, and a container memory cap.
llama.cpp is the main runtime on this laptop because it can keep MoE expert weights in system RAM (`--n-cpu-moe`), and that is what lets 30B-class MoE models run on 16 GiB of VRAM, which vLLM cannot do here (see [../vllm/README.md](../vllm/README.md)).
It reads GGUF quants, including Unsloth Dynamic builds, and its `-hf` cache uses the Hugging Face hub layout, so it can share one cache with vLLM.
Eleven models ran on this laptop with this image on 2026-11-16 (gpt-oss-20b, Qwen3.8-27B, Qwen3.6-35B-A3B, Gemma 4 26B-A4B, K2 Horizon 7B and MoVA, Qwen3-Coder-Next, Nemotron 3.5 Bolt and Lightning, Groot-19B-A3B and DeepSeek-R1-0528-Qwen3-8B), and the GGUF builds of Qwen3.5-9B, Gemma 4 12B, Gemma 4 E4B and Mellum2.1 on 2026-11-17, all measured in [models.md](models.md).
The commands were checked against llama.cpp source at tag b12418, and a memory figure not marked as measured is an estimate.

Per-model commands are in [models.md](models.md), and measurement procedures are in [benchmarking.md](benchmarking.md).
Host prerequisites (NVIDIA Container Toolkit, CDI, ufw, memory caps) are in [../docs/host-setup.md](../docs/host-setup.md), and the OpenCode side is in [../docs/opencode.md](../docs/opencode.md).

## Image

Use `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407`.
It is built on CUDA 12.8.1 ([docker.yml lines 93-94](https://github.com/ggml-org/llama.cpp/blob/b12418/.github/workflows/docker.yml), [cuda.Dockerfile](https://github.com/ggml-org/llama.cpp/blob/b12418/.devops/cuda.Dockerfile)), and its image config sets `NVIDIA_REQUIRE_CUDA=cuda>=12.8 ...` (GHCR image config, read 2026-11-15), which the 595.99.02 driver (CUDA 13.2) satisfies.
The default CUDA arch list adds `120a-real` whenever the toolkit is 12.8 or newer, so the image ships native sm_120 code and needs no PTX JIT ([ggml-cuda/CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/b12418/ggml/src/ggml-cuda/CMakeLists.txt)).
On 2026-11-15 the floating `server-cuda` tag pointed at the same index digest as `server-cuda-b12407`: `sha256:22bd49f4cf68236a63c5a187d7d1867ffb08472611ddfbf0392b6842a1b30d44` (GHCR registry API, read 2026-11-15).
Pin the build tag (or the digest), never `server-cuda`, so that a rebuild cannot change the server under you.

Do not use `server-cuda13-b12407`.
It is built on CUDA 13.5.0 ([docker.yml lines 95-96](https://github.com/ggml-org/llama.cpp/blob/b12418/.github/workflows/docker.yml)), and its image config sets `NVIDIA_REQUIRE_CUDA` to `cuda>=13.5` followed by brand clauses such as `brand=nvidiartx,driver>=595,driver<596`, none of which names GeForce (GHCR image config, read 2026-11-15).
Space-separated constraints are alternatives, so on this GeForce card with a CUDA 13.2 driver none of them matches ([NVIDIA docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)).
The NVIDIA Container Toolkit refuses to start a container whose constraint fails ([NVIDIA docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)), so on this driver the cuda13 image is likely refused (inferred from the image config, not tested).
`NVIDIA_DISABLE_REQUIRE=true` would bypass the check, but there is no reason to take that risk when the CUDA 12.8.1 image runs through ordinary backward compatibility.

There is no image for the newest build yet.
Build b12418 was published on 2026-11-15 as a GitHub pre-release ([release b12418](https://github.com/ggml-org/llama.cpp/releases/tag/b12418)), and the newest stable release is [v1.0.0](https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0) (2026-11-13).
`b12407` is 47 commits after the `v1.0.0` tag, so the pinned image already contains everything in v1.0.0, including the MoE expert cache ([GitHub compare v1.0.0...b12407](https://github.com/ggml-org/llama.cpp/compare/v1.0.0...b12407), read 2026-11-16).
On 2026-11-15 the registry returned 404 for `server-cuda-b12418`, `server-cuda-b12419` and `server-cuda-b12420`, and 200 for `server-cuda-b12407`, `full-cuda-b12407`, `light-cuda-b12407` and `server-cuda13-b12407` (GHCR registry API).
The flag behaviour in this document was read at b12418; b12407 is 11 builds older, so re-check a flag against the server's `--help` if it is rejected.

The image comes in three variants, all from the same [cuda.Dockerfile](https://github.com/ggml-org/llama.cpp/blob/b12418/.devops/cuda.Dockerfile):

| Variant | Tag | Contents | Entrypoint |
|---|---|---|---|
| server | `server-cuda-b12407` | `llama-server` and `llama` only | `/app/llama-server` |
| light | `light-cuda-b12407` | `llama`, `llama-cli` and `llama-completion` | `/app/llama-cli` |
| full | `full-cuda-b12407` | every binary in `build/bin`, including `llama-bench`, `llama-perplexity` and `llama-fit-params`, plus the Python conversion scripts | `/app/tools.sh` |

`llama-bench` is only in the full image; run it with `--bench` ([tools.sh](https://github.com/ggml-org/llama.cpp/blob/b12418/.devops/tools.sh)), as shown in [benchmarking.md](benchmarking.md).
The server image sets `LLAMA_ARG_HOST=0.0.0.0` and `LLAMA_ARG_PORT=8080`, has a `HEALTHCHECK` on `/health`, and has no `USER` directive, so it runs as root ([cuda.Dockerfile](https://github.com/ggml-org/llama.cpp/blob/b12418/.devops/cuda.Dockerfile)).
Files it downloads into a bind-mounted cache are therefore owned by root on the host, the same as with the vLLM image.

## Run template

The NVIDIA Container Toolkit 1.20.1 is installed on this laptop, and GPU access from containers was verified on 2026-11-15 ([install record](../research/2026-11-15/probes/toolkit-install.md)).
The template uses the CDI device syntax (`--device nvidia.com/gpu=all`), which host-setup recommends on this systemd-cgroup host; `--gpus all` also works here (verified).

The example is rank 2 and the 128k pick, Qwen3.6-35B-A3B Q4_K_M (19.13 GiB), with the experts of the first 25 layers in pinned RAM and a 3,200 MiB GPU cache for the most-used of them, because it shows expert offload.
It is the launch measured on 2026-11-16 (122 tok/s decode, 128k context); [models.md](models.md#2-qwen36-35b-a3b) has the measurements, why those values and the rank 1 recipe.

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
  --alias local \
  --no-mmproj \
  -c 131072 -np 1 \
  -fa on --jinja \
  -ngl all --n-cpu-moe 25 \
  -lm none -t 8 --moe-cache-mib 3200 \
  --reasoning off
docker logs -f llamacpp
```

Why each docker flag is there:

- `-e GGML_CUDA_DISABLE_GRAPHS=1` works around a CUDA-graph GPU hang (RC watchdog, Xid 8) on sm_120 laptop GPUs ([#27330](https://github.com/ggml-org/llama.cpp/issues/27330), open, reported with the same GB203M die and reproduced in a 2026-10-12 comment on Ubuntu 26.04 with driver 595.99.02).
  It costs some speed; how much is not measured here.
  Ollama 0.40.16's bundled `llama-server` (CUDA 13 backend) runs with CUDA graphs on (`USE_GRAPHS = 1`, no override), and no hang occurred in five runs of 1 to 2 minutes each on 2026-11-16 (measured, [records](../research/2026-11-16/README.md)); that is too short to clear #27330, so keep the variable for this image.
- It must come before the image name.
  Everything after the image name goes to `llama-server`, where `-e` means `--escape` ([arg.cpp line 1832](https://github.com/ggml-org/llama.cpp/blob/b12418/common/arg.cpp)); `GGML_CUDA_DISABLE_GRAPHS=1` then becomes an unknown argument and the server exits with `error: invalid argument`.
- `-e HF_TOKEN` passes the host variable through without writing it into the command; llama.cpp reads `HF_TOKEN` as the default for `--hf-token` ([server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)).
  It is only needed for gated repos and to avoid anonymous rate limits.
- Tested variant on 2026-11-16: add `--user "$(id -u):$(id -g)"` and mount the cache at `/hf` with `-e HF_HOME=/hf`, so downloaded files belong to the user instead of root; mounting under `/root/.cache` fails for a non-root user with `Permission denied [/root/.cache/huggingface/hub]` because `/root` is not traversable (observed, [measurement record](../research/2026-11-16/llamacpp-gpt-oss-20b/run-command.sh)).
- `HF_HOME` plus the bind mount put `-hf` downloads in `~/.cache/huggingface/hub` on the host.
  The cache lookup order is `LLAMA_CACHE`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME/hub`, `XDG_CACHE_HOME/huggingface/hub`, then `$HOME/.cache/huggingface/hub`, all in the hub layout ([hf-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/b12418/common/hf-cache.cpp)), so vLLM with the same mount sees the same files.
  The cache lives on `/home`, which has 562 GB free.
- `-p 127.0.0.1:8080:8080` because Docker-published ports bypass ufw ([../docs/host-setup.md](../docs/host-setup.md)).
- `--memory 20g --memory-swap 20g` because the user's `heavy` wrapper does not cap Docker containers.
  The value covers 10.5 GiB of pinned experts, up to 8 GiB of host prompt cache (`--cache-ram` default 8192 MiB) and the process; the measured peak was 14.9 GiB with a 107k prompt cached.
  Size the cap for each model from its RAM-resident experts ([models.md](models.md)).
- No `--restart` policy and no `--rm`.
  A running CUDA process keeps the dGPU awake, so start the server on demand and stop it when done; keeping the stopped container keeps its logs for troubleshooting a hang.

Why each llama-server flag is there (defaults from the [server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md) at b12418):

| Flag | Default | Why |
|---|---|---|
| `-hf repo:quant` | unused | Downloads the GGUF into the hub cache; the quant defaults to Q4_K_M. |
| `--alias local` | model path | The model id the API reports; it matches the `modelID` in [../docs/opencode.md](../docs/opencode.md). |
| `--no-mmproj` | mmproj on | `-hf` also downloads and loads the vision projector; text-only coding does not need its VRAM. |
| `-c 131072` | 0 (model value, `--fit` may lower it) | An explicit context stops `--fit` from shrinking it; OpenCode needs 64k or more. |
| `-np 1` | -1 (auto, 4 slots with a unified KV cache) | One OpenCode user; auto gives 4 slots. |
| `-fa on` | auto | Flash attention, required for a quantized V cache. |
| `--jinja` | enabled | Uses the GGUF's Jinja chat template, which drives tool-call and reasoning parsing. Explicit for readability. |
| `-ngl all` | auto | All layers on the GPU, with the experts moved by `--n-cpu-moe`. |
| `--n-cpu-moe 25` | unset | Keeps the experts of blocks 0-24 in RAM; see the next section. |
| `-lm none` | auto (mmap) | Loads the RAM-resident experts into pinned memory, which doubled prefill here (measured). |
| `-t 8` | -1 (auto, which chose 4 here) | One thread per P-core; 2% faster decode than the automatic 4 (measured). |
| `--moe-cache-mib 3200` | 0 (off) | A VRAM cache for the most-used RAM-resident experts; it nearly doubled decode (measured). |
| `-ctk`, `-ctv` | f16 | Not set: f16 KV is 2.5 GiB at 128k on this model and avoids the q8_0 caveat in the KV section. |
| `--reasoning off` | auto | Thinking off by default; a request can turn it on (see the thinking section). |

The flags `--no-mmap` and `--mlock` no longer exist and are rejected with `error: invalid argument`; use `-lm none` or `-lm mlock` (`--load-mode`, default `auto` = mmap) instead ([arg.cpp](https://github.com/ggml-org/llama.cpp/blob/b12418/common/arg.cpp)).
Older guides, including the 16 GB example in the [ggml-org gpt-oss guide](https://github.com/ggml-org/llama.cpp/discussions/15396), still show `--no-mmap`.

### Compose equivalent

```yaml
services:
  llamacpp:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda-b12407
    container_name: llamacpp
    devices: ["nvidia.com/gpu=all"]     # CDI; or: gpus: all (Compose 2.30+)
    environment:
      GGML_CUDA_DISABLE_GRAPHS: "1"
      HF_HOME: /root/.cache/huggingface
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "127.0.0.1:8080:8080"
    mem_limit: 20g
    memswap_limit: 20g                 # equal to mem_limit: no swap
    command:
      - -hf
      - ggml-org/Qwen3.6-35B-A3B-GGUF:Q4_K_M
      - --alias
      - local
      - --no-mmproj
      - -c
      - "131072"
      - -np
      - "1"
      - -fa
      - "on"
      - --jinja
      - -ngl
      - all
      - --n-cpu-moe
      - "25"
      - -lm
      - "none"
      - -t
      - "8"
      - --moe-cache-mib
      - "3200"
      - --reasoning
      - "off"
```

Leave out the `restart` key; the Compose default is no restart.
For gated repos, add `HF_TOKEN` to the environment block.
Start it with `docker compose up -d`, follow it with `docker compose logs -f`, and stop it with `docker compose stop` (keeps the logs) or `docker compose down`.
In Compose the environment block is where `GGML_CUDA_DISABLE_GRAPHS` belongs; the `command` list is passed to `llama-server` exactly like arguments after the image name.
The GPU syntax and memory-cap keys follow [../docs/host-setup.md](../docs/host-setup.md).

## MoE expert offload

A MoE model reads only a few experts per token, so experts can sit in system RAM while attention, shared weights and the KV cache stay on the GPU.
Decode speed then depends on DDR5 bandwidth (102.4 GB/s theoretical for dual-channel DDR5-6400, configured DIMM speed unverified) rather than on the GPU's 896 GB/s, so any tokens/s figure for an offloaded model is an estimate until measured ([benchmarking.md](benchmarking.md)).
The expert cache below changes that picture: it serves the most-used RAM-resident experts from VRAM, and on Qwen3.6-35B-A3B it nearly doubled decode (measured on this laptop, 2026-11-16).

The flags at b12418 ([server README lines 79-91](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)):

- `-ncmoe, --n-cpu-moe N` keeps the MoE weights of the first N layers in RAM.
  N counts block indices from 0, not MoE layers.
  On a model where every block is MoE (Qwen3.6-35B-A3B has 40) the two are the same.
  On a hybrid such as Nemotron 3.5 Lightning, `--n-cpu-moe 16` moves blocks 0-15, which hold only 7 of its 23 MoE layers, so tune N by blocks ([Lightning config.json](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/config.json)).
- `-cmoe, --cpu-moe` keeps all expert weights in RAM: the lowest-VRAM setting, and without the expert cache the slowest.
  With all 40 blocks of Qwen3.6-35B-A3B in RAM and a 9,600 MiB cache it decoded fastest (126 tok/s) but had the slowest prefill and an 18.7 GiB container (measured, [models.md](models.md#2-qwen36-35b-a3b)).
- `-fit, --fit on|off` is on by default and adjusts only the arguments you did not set ([server README line 89](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)).
  When `-ngl` and `--n-cpu-moe` are unset it moves `blk.N.ffn_*_exps` tensors to the CPU until the model fits with a free margin of `-fitt` (default 1024 MiB), and it shrinks the context only when `-c` is 0 or unset ([fit-params README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/fit-params/README.md)).
  `--fit` assumes host memory is unlimited ([server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)), so it does not protect you from running out of RAM, and Unsloth notes that it ignores CPU-repacked weights ([Unsloth Gemma 4 QAT page](https://unsloth.ai/docs/models/gemma-4/qat)).
- `-ot, --override-tensor <regex>=<buffer>` places tensors by name.
  Unsloth's guides use it instead of `--n-cpu-moe`, in this order from least to most VRAM ([Unsloth docs](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune)): `-ot ".ffn_.*_exps.=CPU"` (all experts), `-ot ".ffn_(up|down)_exps.=CPU"`, `-ot ".ffn_(up)_exps.=CPU"`, and per-layer patterns such as `-ot "\.(6|7|8|9|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(gate|up|down)_exps.=CPU"` (experts from layer 6 on).
- `--moe-cache-mib N` is a VRAM cache for the most-used host-resident experts during decode and speculative steps (default 0, off).
  It is GPU memory, not host memory.
  `--moe-cache-cpu-share F` lets the CPU compute a share of the misses.
  Host experts are pinned only with `-lm none`, `-lm mlock` or `-lm dio`, not with the default mmap ([moe-cache.md](https://github.com/ggml-org/llama.cpp/blob/b12418/docs/moe-cache.md)).
  The tuning script `scripts/moe-cache-tune.py` is in the repository, not in any image.
  Measured on Qwen3.6-35B-A3B on 2026-11-16 with pinned experts: a 1,536 MiB cache at CPU share 0 raised decode from 68 to 112 tok/s, while share 0.95 gave only 85, as the doc predicts for pinned experts.
  At the same VRAM, moving 4 more blocks to RAM and giving their 1,728 MiB to the cache (`--n-cpu-moe 25`, 3,200 MiB) raised decode to 120 tok/s, so the cache gains more per MiB than whole blocks.
  Gemma 4 26B-A4B and Nemotron 3.5 Lightning showed the same on 2026-11-16: moving 3 or 4 whole blocks' VRAM into the cache raised the decode hit rate from 78% to 92% and from 64% to 78%, and decode by about 5% each ([models.md](models.md#5-gemma-4-26b-a4b-it)).
  K2 Horizon MoVA and Qwen3-Coder-Next confirmed it: at about equal VRAM, two whole blocks on the GPU decoded 12% and 8% slower than giving that VRAM to the cache ([benchmarking.md](benchmarking.md#qwen3-coder-next-offload-variants)).
  With `-lv 4` the cache logs its hit rate at exit; it served 83% of decode-time expert lookups ([models.md](models.md#2-qwen36-35b-a3b)).

To see what `--fit` would choose without starting the server, run `llama-fit-params` from the full image, for example `docker run --rm --device nvidia.com/gpu=all --memory 20g --memory-swap 20g -v "$HOME/.cache/huggingface:/root/.cache/huggingface" --entrypoint /app/llama-fit-params ghcr.io/ggml-org/llama.cpp:full-cuda-b12407 -m <path to the .gguf in the cache> -c 131072`; it prints the fitted `-c`, `-ngl` and `-ot` arguments ([fit-params README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/fit-params/README.md)).

RAM budget: about 40 GiB of the 61 GiB is free while the desktop runs.
Experts in RAM, the host prompt cache (`--cache-ram`, default 8192 MiB) and the context checkpoints all come out of that, and the container cap must sit above their sum.
Qwen3-Coder-Next with 26.4 GiB of pinned experts is the tight case: with `--cache-ram 4096` and `-ctxcp 8` its container peaked at 31.9 GiB under a 35g cap, so close desktop applications first (measured on 2026-11-16, [models.md](models.md#6-qwen3-coder-next)).

First-run procedure for an offloaded model:

1. Start with the `--n-cpu-moe` value from [models.md](models.md), or leave both `-ngl` and `--n-cpu-moe` unset and let `--fit` choose.
2. Read the load log for the device memory used and the context it got, and run `nvidia-smi`.
3. On a CUDA out-of-memory error, raise `--n-cpu-moe` by 1 or 2.
   With VRAM to spare, give it to `--moe-cache-mib` at CPU share 0 before lowering `--n-cpu-moe`, and keep about 0.5 GiB free at the peak; on Qwen3.6 the cache gained more per MiB than whole blocks (measured).
4. Expect slower prefill when experts are in RAM: experts are copied to the GPU serially and the GPU waits ([#25859](https://github.com/ggml-org/llama.cpp/issues/25859), open).
   Load with `-lm none` so the copies come from pinned memory: on this laptop that doubled prefill (measured), and the experts then count fully against the container cap.

## KV cache quantization and context

`-ctk` and `-ctv` accept f32, f16, bf16, f8_e4m3, q8_0, q4_0, q4_1, iq4_nl, q5_0 and q5_1; the default is f16 ([server README line 71](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)).
q8_0 halves the KV cache compared with f16 (8.5 bits per element), and that is what makes 64k to 128k context fit next to the weights for most picks.
A quantized V cache requires flash attention, so keep `-fa on` ([llama-context.cpp line 617](https://github.com/ggml-org/llama.cpp/blob/b12418/src/llama-context.cpp)).

The cost is decode speed.
In [#27623](https://github.com/ggml-org/llama.cpp/issues/27623) (open) a comment measured q8_0 KV at 5.9 tok/s against 13.3 tok/s for f16 (measured by a commenter on an RTX 2000 Ada 16 GB, sm_89, CUDA 13.3, Qwen3.8-27B UD-IQ4_XS, short prompt; not on this laptop).
A later commenter attributed the penalty to the q8_0-to-f16 conversion the f16-only MMA flash-attention kernel needs on every decode step, and AMD Vulkan and HIP data points in the same thread showed no q8_0 penalty, so it looks CUDA-specific.
The issue title claims a decode collapse above about 80k context; the reporter retracted that on 2026-08-26 as a measurement artifact.
The fused q8_0/q4_0/f8_e4m3 dequant fix, [PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802), was still open on 2026-11-16, so assume the penalty applies to b12407; the PR says it applies to models with large heads and a GQA ratio above 4.
Measured on 2026-11-16 on Qwen3-Coder-Next (256-wide heads, GQA 8): q8_0 decode fell 40% from a short context to a 101k prompt, against 18% with f16, and at equal VRAM f16 with a smaller expert cache overtook q8_0 between the 30k and 101k measurements ([benchmarking.md](benchmarking.md#kv-cache-type-at-depth)).
K2 Horizon 7B (128-wide heads) showed only a small q8_0 penalty after 30k.
Measure q8_0 against f16 on the first run for each model, at depth and not only at a short context ([benchmarking.md](benchmarking.md)); if q8_0 is much slower, use f16 at a shorter context, a smaller expert cache or a smaller quant.
If a model produces gibberish, Unsloth suggests trying `-ctk bf16 -ctv bf16` or a larger context ([Unsloth Qwen3.6 page](https://unsloth.ai/docs/models/qwen3.6)).

When a request exceeds the context, llama-server returns HTTP 400 `exceed_context_size_error` with `n_prompt_tokens` and `n_ctx` ([Unsloth Qwen3-Coder-Next page](https://unsloth.ai/docs/models/qwen3-coder-next)); raise `-c` or lower `limit.context` in OpenCode.

## Thinking control

llama.cpp returns reasoning in `message.reasoning_content` (DeepSeek style), not in `reasoning` as vLLM does ([server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)); OpenCode must be told so with `compatibility.reasoningField: "reasoning_content"` ([../docs/opencode.md](../docs/opencode.md)).

| Control | Scope | Notes |
|---|---|---|
| `--reasoning on\|off\|auto` | server default | Default `auto` (detected from the template). `off` sets the template default `enable_thinking=false`. |
| `chat_template_kwargs: {"enable_thinking": true}` | one request | Overrides the server default; must be a JSON boolean, a string is rejected. |
| `--reasoning-effort LEVEL` / `"reasoning_effort"` | server / request | Passed to the template (Qwen3.8 levels, gpt-oss low/medium/high); `"reasoning_effort": "none"` turns thinking off. |
| `--reasoning-budget N` | server | -1 unlimited (default), 0 ends thinking at once, N caps the thinking tokens. |
| `--reasoning-budget-message MSG` | server | Text injected before the end-of-thinking tag when the budget runs out. |
| `--reasoning-format none\|deepseek\|deepseek-legacy` | server or request | Default `auto`; `deepseek` fills `reasoning_content`, `none` leaves thoughts in `content`. |
| `--reasoning-preserve` | server | Default enabled; keeps past reasoning in history for templates that support it. |
| `--chat-template-kwargs JSON` | server | Generic template arguments, for example `{"preserve_thinking":true}`. |

Sources: the flag table in the [server README lines 221, 239-244](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md), and the request merge in [server-common.cpp lines 1625-1650](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/server-common.cpp), which applies per-request `chat_template_kwargs` over the server defaults.
Setting `enable_thinking` through `--chat-template-kwargs` still works but logs a deprecation warning that points to `--reasoning on|off`, and the same holds for `preserve_reasoning` and `--reasoning-preserve` ([arg.cpp lines 3576-3582](https://github.com/ggml-org/llama.cpp/blob/b12418/common/arg.cpp)).

Why the default is off: on the user's DGX Spark, Qwen3.6-35B-A3B with thinking on spent all 1024 tokens of a `max_tokens` budget reasoning and returned no answer (23.7 s), while thinking off answered correctly in 1.5 s (measured on a DGX Spark, vLLM, 2026-09-05).
An agent loop with a bounded `max_tokens` fails that step outright.
Thinking still helps agent quality on hard tasks ([Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-27b) reports Qwen3.8-27B Terminal-Bench 2.1 at 49.1 without reasoning and 67.4 at low effort, measured through an API on BF16 weights, see [../docs/model-selection.md](../docs/model-selection.md)), so the OpenCode config exposes thinking and non-thinking variants per model instead of fixing one.
gpt-oss always reasons; control it with the effort level, not with `enable_thinking`.

## Prompt cache

Prompt caching is on by default (`--cache-prompt`), and the `timings.cache_n` field of each response shows how many prompt tokens were reused ([server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)).
For an agent like OpenCode, which resends a long and mostly unchanged conversation on every turn, a cache hit is the difference between a few hundred prompt tokens and a full re-prefill of tens of thousands.

| Flag | Default | Meaning |
|---|---|---|
| `--cache-prompt` | enabled | Reuse the matching prefix of the previous prompt in the slot. |
| `-cram, --cache-ram N` | 8192 MiB | Host RAM for cached prompts; counts against the container memory cap. 0 disables, -1 means no limit. |
| `-ctxcp, --ctx-checkpoints N` | 32 per slot | Context checkpoints. Hybrid models (Gated DeltaNet or Mamba layers) cannot roll their recurrent state back, so prefix reuse depends on these. |
| `--cache-reuse N` | 0 (off) | Minimum chunk size for reuse through KV shifting; not supported on some architectures. |

Do not set `--ctx-checkpoints 0` on hybrid models.
In [#32316](https://github.com/ggml-org/llama.cpp/issues/32316) (open) a Qwen3.5 user saw a full re-prefill on every follow-up turn; a 2026-11-15 comment reproduced it and traced it to `--ctx-checkpoints 0`, and with the default checkpoints the second turn reused all 5906 cached tokens (measured on Apple Metal by the commenter).
The same issue shows that `--cache-reuse` is disabled at load on Qwen3.5-family models (`cache_reuse is not supported by this context`), so do not count on it for the Qwen picks.

Known hole: [#30357](https://github.com/ggml-org/llama.cpp/issues/30357) (open).
On Qwen3.5 and newer chat templates, reasoning and content whitespace is not round-trippable, so when the model emits non-canonical whitespace the re-rendered history no longer matches the cached prefix and a hybrid model re-prefills the whole context.
Watch `timings.cache_n` (or `llamacpp:prompt_tokens_total` with `--metrics`) during a long OpenCode session to see how often this happens.

## Tool calling

With `--jinja` (the default) llama.cpp renders the GGUF's chat template and picks a tool-call parser from it; for example, a template containing `<tool_call>`, `<function=` and `<parameter=` gets the Qwen3-Coder XML parser ([common/chat.cpp lines 1420-1427](https://github.com/ggml-org/llama.cpp/blob/b12418/common/chat.cpp)).
Tool calls then come back as OpenAI `tool_calls`.
There is no per-model parser flag as in vLLM, so a tool-call problem is usually a template problem: the fix is a newer GGUF, a different uploader's GGUF, or `--chat-template-file` with a corrected template.

Template sources differ.
Unsloth's Qwen3.6 and Qwen3.8 GGUFs add developer-role support for Codex and OpenCode and better parsing of nested objects in tool calls (vendor-documented, [Unsloth Qwen3.6 page](https://unsloth.ai/docs/models/qwen3.6)), and these templates are not the ones in the ggml-org GGUFs.
Unsloth also documents tool-call template fixes for gpt-oss, Qwen3-Coder-Next and GLM-4.7-Flash, some of which needed both a re-download and a newer llama.cpp ([Unsloth docs](https://unsloth.ai/docs)).
Re-download a GGUF when its repo announces a template fix; the `-hf` cache does not update itself.

OpenCode's subagent tool schema contains the pattern `^ses`, which older llama.cpp builds rejected with HTTP 400 (`Pattern must start with ^ and end with $`, [OpenCode #58019](https://github.com/anomalyco/opencode/issues/58019), still open).
llama.cpp [PR #26939](https://github.com/ggml-org/llama.cpp/pull/26939), merged 2026-08-20, turns an unsupported pattern into a warning and accepts any string, and b12407 includes it.
Bundles that embed an older llama.cpp can still hit the error.

Open tool-calling issues that matter for the picks:

- [#30118](https://github.com/ggml-org/llama.cpp/issues/30118) (open): when Qwen3.6 omits the `<tool_call>` opener, the call leaks into `content` instead of `tool_calls`.
- [#29655](https://github.com/ggml-org/llama.cpp/issues/29655) (open): Gemma 4 tool calls are unstable with multi-line arguments during streaming and partial parsing, which hits OpenCode edit and write calls.
- [#29947](https://github.com/ggml-org/llama.cpp/issues/29947) (open): Qwen3.8-27B with OpenCode and the opencode-tool-search plugin fails with a Jinja `CallExpression` error on the ggml-org GGUF; the reporter says the Unsloth GGUF template works.

Test every model with a scripted tool call before using it in OpenCode (see Smoke tests).

## Unsloth Dynamic quants

Unsloth Dynamic (UD) GGUFs, named `UD-Q3_K_XL`, `UD-IQ4_XS` and so on, are post-training quants in which the quant type is chosen per layer and sensitive tensors such as attention and `ssm_out` on hybrid models are kept at higher precision, with an importance matrix calibrated on agentic coding, chat and multilingual text ([Unsloth Dynamic 3.0](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)).
They are not QAT.
They load in stock llama.cpp like any GGUF.
Unsloth's quality numbers (KL divergence and top-1 agreement against BF16) are vendor-reported self-comparisons.
Unsloth says 1-bit quants break tool calling and puts the floor for agentic use at UD-Q2_K_XL (vendor-reported, [Unsloth Dynamic 3.0](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)).
Unsloth repos can replace a file under the same name: `unsloth/Qwen3.8-27B-GGUF` replaced `UD-Q3_K_XL` with a Dynamic V3.1 build on 2026-11-12 at the same size, although the doc page says the 3-bit files were not updated ([HF repo](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)), so a copy downloaded before that date is the older build.

Unsloth's Qwen3.6 page warns: "Do NOT use CUDA 13.2 as you may get gibberish outputs" ([Unsloth Qwen3.6 page](https://unsloth.ai/docs/models/qwen3.6)).
That concerns the CUDA toolkit llama.cpp is built with, not the driver.
[#28581](https://github.com/ggml-org/llama.cpp/issues/28581) (open) shows IQ1_S, IQ2_S and IQ3_S producing garbage from a local nvcc 13.2 build on an RTX 5060 Ti, with a fix reported in 13.2.2 and later.
The CUDA 12.8.1 image is not affected, so this is not a reason to avoid any image, but do not build llama.cpp yourself with CUDA 13.2.0 or 13.2.1, and sanity-check the first answers of a UD file that contains IQ3_S tensors (UD-Q3_K_XL of Qwen3.8-27B holds 3.29 GiB of them).

## Known issues

States read from the GitHub API on 2026-11-15.

| Issue | State | Effect here | Workaround |
|---|---|---|---|
| [#27330](https://github.com/ggml-org/llama.cpp/issues/27330) CUDA graphs hang the GPU on sm_120 laptops | Open | GPU channel hang, Xid 8, server abort under sustained agent load; reproduced on driver 595.99.02 and Ubuntu 26.04 | `-e GGML_CUDA_DISABLE_GRAPHS=1` before the image name |
| [#30357](https://github.com/ggml-org/llama.cpp/issues/30357) Qwen3.5+ whitespace not round-trippable | Open | Prefix-cache miss and full re-prefill on hybrid Qwen models | None; watch `timings.cache_n` |
| [#32316](https://github.com/ggml-org/llama.cpp/issues/32316) cache reuse disabled for IMROPE models | Open | `--cache-reuse` has no effect on Qwen3.5-family models; re-prefill seen with `--ctx-checkpoints 0` | Keep the default `--ctx-checkpoints` |
| [#30118](https://github.com/ggml-org/llama.cpp/issues/30118) Qwen3.6 tool call leaks into content | Open | An occasional tool call arrives as text | Retry; try the Unsloth template |
| [#29655](https://github.com/ggml-org/llama.cpp/issues/29655) Gemma 4 multi-line streaming tool args | Open | Broken edit and write calls with Gemma 4 | Test edits before use; see [models.md](models.md) |
| [#29947](https://github.com/ggml-org/llama.cpp/issues/29947) Qwen3.8 Jinja error with an OpenCode plugin | Open | Request fails with a template error | Unsloth GGUF template |
| [#27623](https://github.com/ggml-org/llama.cpp/issues/27623) q8_0 KV decode penalty | Open | Slower decode with `-ctk/-ctv q8_0`, growing with depth: 40% at 101k on Qwen3-Coder-Next (measured) | Measure; f16 at shorter context or with a smaller expert cache; fix [PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802) open |
| [#25859](https://github.com/ggml-org/llama.cpp/issues/25859) serial expert copies in offloaded prefill | Open | Slow prefill with `--n-cpu-moe` | Fewer blocks on CPU; measure |
| [#28581](https://github.com/ggml-org/llama.cpp/issues/28581) IQ3_S garbage with nvcc 13.2 | Open | Only for self-built CUDA 13.2.0/13.2.1 binaries | Use the CUDA 12.8.1 image |
| [PR #26939](https://github.com/ggml-org/llama.cpp/pull/26939) unsupported JSON-schema regex fallback | Merged 2026-08-20 | Fixes the OpenCode `^ses` HTTP 400 | Included in b12407 |

## Smoke tests

Run these after every start, in order.
The first start downloads the model, so wait until the log shows the server listening.

```bash
# 1. Liveness (public even with --api-key)
curl -s http://127.0.0.1:8080/health

# 2. Model id: must match the OpenCode modelID
curl -s http://127.0.0.1:8080/v1/models | jq -r '.data[].id'

# 3. Chat template and thinking default: expect exactly OK, with no reasoning_content
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":16}' \
  | jq '{content: .choices[0].message.content, reasoning: .choices[0].message.reasoning_content, finish: .choices[0].finish_reason, timings: .timings}'

# 4. Thinking on for one request: expect reasoning_content to be filled
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Is 221 prime? Answer yes or no."}],"max_tokens":2048,"chat_template_kwargs":{"enable_thinking":true}}' \
  | jq '{content: .choices[0].message.content, reasoning_chars: (.choices[0].message.reasoning_content // "" | length)}'

# 5. Tool call: expect finish_reason "tool_calls" and parsed arguments, not text in content
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
       "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city",
       "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":512}' \
  | jq '{finish: .choices[0].finish_reason, calls: .choices[0].message.tool_calls, content: .choices[0].message.content}'

# 6. Placement and memory
docker logs llamacpp 2>&1 | grep -iE 'CUDA0|offload|n_ctx|error|warn' | tail -n 40
nvidia-smi --query-gpu=memory.used,memory.total,power.draw --format=csv
docker stats --no-stream llamacpp
```

Run test 3 twice; on the second run `timings.cache_n` should be above 0, which shows the prompt cache works.
For OpenCode, also run one real session with a multi-line file edit, a file write and a few parallel tool calls before relying on a model, because the open issues above show up there and not in a single curl.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `error: invalid argument: GGML_CUDA_DISABLE_GRAPHS=1` | The `-e` went after the image name and reached `llama-server` as `--escape`. Move it before the image name. |
| `error: invalid argument: --no-mmap` or `--mlock` | Removed flags. Use `-lm none` or `-lm mlock`. |
| Container does not start, CUDA requirement not met | The cuda13 image on a GeForce card with a CUDA 13.2 driver. Use `server-cuda-b12407`. |
| `could not select device driver` or no GPU in the container | NVIDIA Container Toolkit missing or not configured; see [../docs/host-setup.md](../docs/host-setup.md). |
| GPU stops responding after minutes of agent load, Xid 8 in `sudo dmesg` | [#27330](https://github.com/ggml-org/llama.cpp/issues/27330): `GGML_CUDA_DISABLE_GRAPHS=1` is missing. |
| CUDA out of memory at load | Raise `--n-cpu-moe` by 1 or 2, lower `-c`, add `--no-mmproj`, or unset `-ngl` and `--n-cpu-moe` and let `--fit` place experts. |
| Context smaller than asked | `--fit` lowered it because `-c` was unset or 0. Set `-c` explicitly. |
| Container exits with code 137 | Container memory cap reached. Raise `--memory` and `--memory-swap` together, or lower `--cache-ram` or the experts in RAM. |
| Empty answer, `finish_reason: length` | Thinking used the whole `max_tokens`. Check `--reasoning off` and that no client sends `enable_thinking: true`, or set `--reasoning-budget`. |
| HTTP 400 `invalid type for "enable_thinking"` | The value was sent as a string. Send a JSON boolean. |
| HTTP 400 `exceed_context_size_error` | The conversation outgrew `-c`. Raise `-c` or lower OpenCode `limit.context`. |
| Tool call appears as text in `content` | Template problem ([#30118](https://github.com/ggml-org/llama.cpp/issues/30118) for Qwen3.6). Check `--jinja` was not disabled, try another uploader's GGUF or `--chat-template-file`. |
| HTTP 400 `Pattern must start with ^ and end with $` | An old llama.cpp build. Use b12407. |
| Every OpenCode turn re-prefills the whole context | `timings.cache_n` near 0: `--ctx-checkpoints 0` was set, or [#30357](https://github.com/ggml-org/llama.cpp/issues/30357) whitespace mismatch. |
| Gibberish output | A self-built CUDA 13.2 binary with IQ quants ([#28581](https://github.com/ggml-org/llama.cpp/issues/28581)), a stale GGUF that needed re-download, or try `-ctk bf16 -ctv bf16`. |
| Slow decode with an offloaded model | Expected: bound by DDR5 bandwidth. Move fewer blocks to RAM, compare q8_0 with f16 KV, and measure ([benchmarking.md](benchmarking.md)). |
| Port 8080 reachable from the network | The port was published without `127.0.0.1:`. Docker bypasses ufw. |

## Sources

- llama.cpp at tag b12418: [tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md), [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/b12418/common/arg.cpp), [tools/server/server-common.cpp](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/server-common.cpp), [common/chat.cpp](https://github.com/ggml-org/llama.cpp/blob/b12418/common/chat.cpp), [common/hf-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/b12418/common/hf-cache.cpp), [docs/moe-cache.md](https://github.com/ggml-org/llama.cpp/blob/b12418/docs/moe-cache.md), [tools/fit-params/README.md](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/fit-params/README.md), [ggml-cuda/CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/b12418/ggml/src/ggml-cuda/CMakeLists.txt).
- Images: [.devops/cuda.Dockerfile](https://github.com/ggml-org/llama.cpp/blob/b12418/.devops/cuda.Dockerfile), [.devops/tools.sh](https://github.com/ggml-org/llama.cpp/blob/b12418/.devops/tools.sh), [.github/workflows/docker.yml](https://github.com/ggml-org/llama.cpp/blob/b12418/.github/workflows/docker.yml), [GHCR package](https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp).
- Releases: [b12418](https://github.com/ggml-org/llama.cpp/releases/tag/b12418), [v1.0.0](https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0).
- Issues and PRs: [#27330](https://github.com/ggml-org/llama.cpp/issues/27330), [#30357](https://github.com/ggml-org/llama.cpp/issues/30357), [#32316](https://github.com/ggml-org/llama.cpp/issues/32316), [#30118](https://github.com/ggml-org/llama.cpp/issues/30118), [#29655](https://github.com/ggml-org/llama.cpp/issues/29655), [#29947](https://github.com/ggml-org/llama.cpp/issues/29947), [#27623](https://github.com/ggml-org/llama.cpp/issues/27623), [#25859](https://github.com/ggml-org/llama.cpp/issues/25859), [#28581](https://github.com/ggml-org/llama.cpp/issues/28581), [PR #26939](https://github.com/ggml-org/llama.cpp/pull/26939), [PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802), [OpenCode #58019](https://github.com/anomalyco/opencode/issues/58019).
- NVIDIA: [Container Toolkit, specialized configurations](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html), [CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
- Guides and scores: [ggml-org gpt-oss guide](https://github.com/ggml-org/llama.cpp/discussions/15396), [Artificial Analysis Qwen3.8-27B](https://artificialanalysis.ai/models/qwen3-8-27b).
- Unsloth docs (vendor docs for their own quants): [Dynamic 3.0 GGUFs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs), [Qwen3.6](https://unsloth.ai/docs/models/qwen3.6), [Gemma 4 QAT](https://unsloth.ai/docs/models/gemma-4/qat), [Qwen3-Coder-Next](https://unsloth.ai/docs/models/qwen3-coder-next), [gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune).
- Source catalog and evidence classes: [../docs/sources.md](../docs/sources.md).

## Maintenance

Before relying on this document, re-check:

- The newest `server-cuda-bNNNNN` tag on GHCR and whether `server-cuda` still points at it; move the pin forward only after re-running the smoke tests.
- Whether [#27330](https://github.com/ggml-org/llama.cpp/issues/27330) is fixed; until it is, keep `GGML_CUDA_DISABLE_GRAPHS=1`.
- The states of the issues in the Known issues table, especially [PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802) (q8_0 KV speed) and [#30357](https://github.com/ggml-org/llama.cpp/issues/30357) (prefix cache).
- Flag names and defaults against `llama-server --help` of the pinned image, because flags are renamed and removed between builds (`--no-mmap` and `--mlock` were).
- Whether a newer cuda13 image still lacks a GeForce brand clause, and whether the driver has moved past CUDA 13.2.
- After the first run on this laptop, replace the memory-cap and `--n-cpu-moe` estimates here and in [models.md](models.md) with measured values and record them in [benchmarking.md](benchmarking.md).
