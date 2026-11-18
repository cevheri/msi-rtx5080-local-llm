# Local LLMs on the MSI Raider 18 Max HX (RTX 5080 Laptop)

Last verified: 2026-11-18.

This repository is the knowledge base for running open-weight LLMs in Docker on one laptop, an MSI Raider 18 Max HX A2WI with an RTX 5080 Laptop GPU (16 GB), with llama.cpp, vLLM and SGLang, and for using them from OpenCode as a coding agent.
Use Qwen3.6-35B-A3B on llama.cpp as the default: it measured 122 tok/s with 128k context.
Switch to Qwen3.8-27B (36 tok/s, 64k context) when a task needs the strongest model; it keeps rank 1 on quality evidence.
vLLM is the second path, for the smaller models that fit fully in 16 GB of VRAM.
GPU access from containers works on this laptop since 2026-11-15, and all ten picks were measured on llama.cpp on 2026-11-16, nine of them completing an OpenCode tool session: Qwen3.8-27B at 36 tok/s and K2 Horizon MoVA at 37 tok/s with 64k context, Qwen3.6-35B-A3B at 122 tok/s, Gemma 4 26B-A4B at 148 tok/s and Qwen3-Coder-Next at 72 tok/s with 128k, K2 Horizon 7B at 67 tok/s with 80k, Groot-19B-A3B at 170 tok/s and Nemotron 3.5 Bolt at 74 tok/s with 128k, and gpt-oss-20b at 166 tok/s; the chat-only DeepSeek-R1-0528-Qwen3-8B measured 71 tok/s and failed the OpenCode tool session when tools were enabled for a test, and Nemotron 3.5 Lightning, an honorable mention, measured 144 tok/s with 128k ([llamacpp/models.md](llamacpp/models.md)).
On Ollama 0.40.16, the five small models from the user's LibreDB Studio list (qwen3.5:9b, gemma4:12b, ornith:9b, ministral-3:8b and qwen2.5:7b) fit fully on the GPU at 99-119 tok/s and passed the same OpenCode session ([docs/agent-harness-evidence.md](docs/agent-harness-evidence.md#try-here-first)).
On vLLM v0.33.0, gpt-oss-20b was measured on 2026-11-16, and Mellum2.1, Qwen3.5-9B, Gemma 4 12B, Gemma 4 E4B, K2 Horizon 7B and Groot-19B-A3B on 2026-11-17 ([research/2026-11-17/](research/2026-11-17/README.md)), while DeepSeek-R1-0528-Qwen3-8B failed there on its tokenizer and runs since 2026-11-18 through llmctl's tokenizer overlay at 72.5 tok/s with 64k (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18) ([research/2026-11-18/](research/2026-11-18/vllm-deepseek-r1-0528-qwen3-8b-cold-start/README.md)).
On llama.cpp, the GGUF builds of Qwen3.5-9B, Gemma 4 12B, Gemma 4 E4B and Mellum2.1 ran with 128k context at 66.0, 77.1, 151.3 and 188.2 tok/s (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17) ([llamacpp/models.md](llamacpp/models.md#gguf-builds-of-the-vllm-and-sglang-models)).
SGLang v0.5.23 is the third engine: Qwen3.5-9B NVFP4, gpt-oss-20b, Gemma 4 12B QAT W4A16 and Qwen3.6-35B-A3B NVFP4 ran on it with 128k context and K2 Horizon 7B AWQ-INT4 and Gemma 4 E4B QAT W4A16 with 64k, at 93.7, 124.6, 70.4, 151.3, 89.6 and 141.3 tok/s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17 and 2026-11-18) ([sglang/models.md](sglang/models.md)).
One of them gains materially there: Qwen3.6-35B-A3B with `--runtime sglang`, 151.3 tok/s through Paged Experts against 122.5 on llama.cpp, 24% faster (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-16; [record](research/2026-11-16/llamacpp-qwen3.6-35b-a3b/README.md)).
Gemma 4 E4B decodes 141.3 tok/s on SGLang against 45.2 on vLLM in eager mode (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [record](research/2026-11-17/vllm-gemma-4-e4b/README.md)), but its llama.cpp build is faster still, at 151.3 tok/s with twice the context (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17; [record](research/2026-11-17/llamacpp-gemma-4-e4b/README.md)), so it and the other four are no material speed gain over the fastest other recipe of the same model.
Figures for every other model and runtime are still estimates or third-party numbers until they are measured here.

## Quick start

`./llmctl` starts one measured recipe by name in Docker: it checks the machine first, applies the rules below, waits until the server answers, and points the installed clients at it.
It runs one model server at a time, published on 127.0.0.1 only: llama.cpp answers on http://127.0.0.1:8080/v1, vLLM on http://127.0.0.1:8000/v1 and SGLang on http://127.0.0.1:30000/v1.
`up` refuses while another server runs, and `up --replace` stops that one first.

Each recipe belongs to llama.cpp, vLLM or SGLang.
Without `--runtime`, a name starts on llama.cpp when it has a llama.cpp recipe, else on vLLM, else on SGLang; every SGLang recipe shares its name with another engine's, so SGLang always needs `--runtime sglang`, and for a model that ships on more than one engine, `--runtime llamacpp`, `--runtime vllm` or `--runtime sglang` chooses; since the GGUF recipes of `gemma-4-12b`, `gemma-4-e4b`, `mellum2.1-12b` and `qwen3.5-9b`, those four names start on llama.cpp unless `--runtime vllm` is given, or `--runtime sglang` for the three that ship an SGLang recipe (`gemma-4-12b`, `gemma-4-e4b` and `qwen3.5-9b`; `mellum2.1-12b` ships none).
For `qwen3.5-9b` that trades speed for precision: its Q8_0 build decodes 66.0 tok/s on llama.cpp (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17; [record](research/2026-11-17/llamacpp-qwen3.5-9b/README.md)) against 90.8 for the NVFP4 build on vLLM (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [record](research/2026-11-17/vllm-qwen3.5-9b/README.md)) and 93.7 on SGLang (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [record](research/2026-11-17/sglang-qwen3.5-9b/README.md)), so `--runtime vllm` or `--runtime sglang` is the faster choice for it.

A model is named by its ID or an alias.
The [runtime matrix](https://cevheri.github.io/msi-rtx5080-local-llm/) shows every model on every engine with its speed, context, client results and start command; GitHub Actions builds it from this repository on every push to `main`.
Decode speeds are measured on this laptop (measured on the RTX 5080 Laptop, llama.cpp `server-cuda-b12407`, vLLM v0.33.0 or SGLang v0.5.23), a "-" after "Yes" marks a recipe with no measured speed, and "No | - | -" means the model has no recipe for that engine.

<!-- recipe table: written by python3 -m launcher.site readme from recipes/; do not edit by hand -->

| Model ID | Aliases | llama.cpp | Context | Decode tok/s | vLLM | Context | Decode tok/s | SGLang | Context | Decode tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| `deepseek-r1-0528-qwen3-8b` | `deepseek-r1` | Yes | 64k | 70.6 | Yes | 64k | 72.5 | No | - | - |
| `gemma-4-12b` | `gemma4-12b` | Yes | 128k | 77.1 | Yes | 128k | 69.5 | Yes | 128k | 70.4 |
| `gemma-4-26b-a4b` | `gemma4-26b` | Yes | 128k | 148.2 | No | - | - | No | - | - |
| `gemma-4-e4b` | `gemma4-e4b` | Yes | 128k | 151.3 | Yes | 64k | 45.2 | Yes | 64k | 141.3 |
| `gpt-oss-20b` | `gpt-oss` | Yes | 128k | 166.2 | Yes | 64k | 83.5 | Yes | 128k | 124.6 |
| `groot-19b-a3b` | `groot` | Yes | 128k | 169.8 | Yes | 128k | 193.4 | No | - | - |
| `k2-horizon-7b` | `k2-7b` | Yes | 80k | 66.9 | Yes | 64k | 94.8 | Yes | 64k | 89.6 |
| `k2-horizon-mova-36b` | `mova`, `k2-mova` | Yes | 64k | 37.2 | No | - | - | No | - | - |
| `mellum2.1-12b` | `mellum` | Yes | 128k | 188.2 | Yes | 128k | 257.2 | No | - | - |
| `nemotron-3.5-bolt` | `bolt` | Yes | 128k | 73.5 | No | - | - | No | - | - |
| `nemotron-3.5-lightning` | `lightning` | Yes | 128k | 143.6 | No | - | - | No | - | - |
| `qwen3-coder-next` | `coder-next` | Yes | 128k | 71.5 | No | - | - | No | - | - |
| `qwen3.5-9b` | `qwen3.5` | Yes | 128k | 66 | Yes | 128k | 90.8 | Yes | 128k | 93.7 |
| `qwen3.6-35b-a3b` | `qwen3.6` | Yes | 128k | 122.5 | No | - | - | Yes | 128k | 151.3 |
| `qwen3.8-27b` | `qwen3.8` | Yes | 64k | 36.4 | No | - | - | No | - | - |

<!-- end of recipe table -->

The vLLM recipe of `deepseek-r1-0528-qwen3-8b` runs through llmctl's tokenizer overlay, a folder that names the right tokenizer class for vLLM v0.33.0 ([vllm/models.md](vllm/models.md#deepseek-r1-0528-qwen3-8b-chat-only)).

```bash
./llmctl list                                # every recipe, its evidence, what is downloaded, and what runs
./llmctl show qwen3.6                        # the effective recipe and the docker command up would run
./llmctl pull qwen3.6                        # asks, then downloads the model files and the image if missing
./llmctl up qwen3.6                          # starts on llama.cpp, the default engine for this name
./llmctl up gpt-oss --runtime vllm --replace # stops the running server, then starts gpt-oss-20b on vLLM
./llmctl up qwen3.8 --set-default            # also takes over client defaults that point elsewhere
./llmctl up qwen3.5 --runtime sglang --replace  # stops the running server, then starts Qwen3.5-9B on SGLang
./llmctl up groot --runtime vllm --dry-run   # prints the docker command and the client changes, starts nothing
./llmctl status                              # the running server, its health and each client's entry
./llmctl test                                # a reply, a tool call and a decode speed against the server
./llmctl logs -f                             # follows the server's log
./llmctl down                                # stops the server; the client entries stay
LLMCTL_CLIENTS=opencode,pi ./llmctl up qwen3.6  # updates only OpenCode and Pi
```

Every `up` moves a client default that already points at llmctl to the running model, and leaves any other default as it is; `--set-default` also takes over those other defaults and runs `hermes profile use llmctl` ([docs/llmctl.md](docs/llmctl.md#defaults-and-the-follow-rule)).

After `./llmctl up`, run the model from a client; `up` prints these lines for the running recipe, where `<served name>` is `local` on llama.cpp and the recipe's served name, such as `gpt-oss-20b`, on vLLM and SGLang.

| Client | What to run |
|---|---|
| OpenCode | `/models` (or `ctrl+x m`) in the OpenCode screen, or `opencode run --model llamacpp-docker/qwen3.6-35b-a3b "your prompt"` (`vllm-docker/<id>` on vLLM and `sglang-docker/<id>` on SGLang, where DeepSeek-R1 has no OpenCode entry on vLLM) |
| Hermes Agent | `hermes -p llmctl` |
| Pi | `pi --model llmctl/<served name>`, with `--no-tools` for the chat-only DeepSeek-R1 |
| DeepSeek Harness (dsh) | `dsh web`, then pick `<id> on <engine> (llmctl)` in its model picker; while llmctl holds dsh's default, `dsh web` alone or `dsh --profile headless "your prompt"` |
| Kimi Code CLI | `kimi -m llmctl/<served name>`, or `kimi` and `kimi -p "your prompt"` when it is Kimi's default; `kimi -p` needs Kimi's default model even with `-m`, which `--set-default` sets; for the chat-only DeepSeek-R1, only `kimi -p` with the agent file that `up` names |

OpenCode needs the provider config from [docs/opencode.md](docs/opencode.md#the-validated-config) installed once; llmctl then changes only its default model line, while it writes the Hermes, Pi, dsh and Kimi entries itself.
For the chat-only DeepSeek-R1, `up` prints a tool-free route per client instead: Pi with `--no-tools`, Kimi `kimi -p` with `--agent-file` once it is Kimi's default, and dsh a `--patch` with the rows in [docs/clients.md](docs/clients.md#chat-only-recipes).

[docs/llmctl.md](docs/llmctl.md) has every command, flag and check, and [docs/clients.md](docs/clients.md) the entries llmctl writes for each client.

## Machine

| Item | Value |
|---|---|
| Laptop | MSI Raider 18 Max HX A2WI (board MS-1824, BIOS E1824IMS.70E) |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU, 16303 MiB, compute capability 12.0 (sm_120), 896 GB/s, 140 W power limit |
| Driver | 595.99.02, open kernel module, CUDA 13.2 |
| CPU | Intel Core Ultra 9 290HX Plus, 24 cores |
| RAM | 61 GiB usable, about 40 GiB free with the desktop running |
| Disk | 2 TB NVMe; the model cache lives under /home |
| Software | Ubuntu 26.04.1, Docker 29.9.3, Compose v5.7.1, OpenCode v2.3.0, Hermes Agent v0.23.1, Pi 1.5.2, DeepSeek Harness 0.2.2-rc.3, Kimi Code CLI 2.5.3 |
| NVIDIA Container Toolkit | 1.20.1, installed 2026-11-15; CDI, the `nvidia` Docker runtime and a CUDA kernel in a container verified ([install record](research/2026-11-15/probes/toolkit-install.md)) |
| Not installed | Ollama and llama.cpp on the host (both run as containers instead) |

These values were observed with nvidia-smi, lscpu, free, lsblk and docker info on 2026-11-15.
The laptop model and BIOS version were read from DMI (`/sys/class/dmi/id`) on 2026-11-16.
The bandwidth figure is NVIDIA's laptop specification ([NVIDIA](https://www.nvidia.com/en-us/geforce/laptops/compare/)).

## Documents

| Path | What it covers |
|---|---|
| [docs/llmctl.md](docs/llmctl.md) | `./llmctl`: start a measured recipe by name, the client entries it writes, the rules it enforces, the recipe format, overrides for another machine |
| [docs/clients.md](docs/clients.md) | Hermes Agent v0.23.1, Pi 1.5.2, DeepSeek Harness 0.2.2-rc.3 and Kimi Code CLI 2.5.3 with the local servers: the entries llmctl writes, thinking per type, the gateway and sticky-profile side effects, dsh's home patch and default, Kimi's `[providers.llmctl]` and `[models."llmctl/<served name>"]` tables and `default_model`, silent fallbacks |
| [docs/model-selection.md](docs/model-selection.md) | Ranked picks with evidence, capability lists, new candidates, exclusions and what to measure first |
| [docs/host-setup.md](docs/host-setup.md) | NVIDIA Container Toolkit, CDI, Compose GPU syntax, localhost-only ports, memory caps, laptop power, model cache |
| [docs/opencode.md](docs/opencode.md) | OpenCode v2.3.0 config with three providers, llama.cpp, vLLM and SGLang (schema-validated), thinking variants, context budget, known issues, remote trials |
| [vllm/README.md](vllm/README.md) | vLLM v0.33.0 on sm_120: image, run template, memory budget, flags, parsers, quantization paths, known issues, smoke tests |
| [vllm/models.md](vllm/models.md) | vLLM recipes for the models that fit, and the models that do not |
| [vllm/benchmarking.md](vllm/benchmarking.md) | Benchmarking with `vllm bench serve`, scenarios and the results template |
| [sglang/README.md](sglang/README.md) | SGLang v0.5.23 on sm_120: image, run template, memory budget and the post-ready check, the model overlay, grammar requests and NCCL, flags, parsers, quantization paths, Paged Experts, known issues, smoke tests |
| [sglang/models.md](sglang/models.md) | SGLang recipes for the models that run, and the models that do not |
| [sglang/benchmarking.md](sglang/benchmarking.md) | The SGLang measurement method, post-ready margins, results and cold starts without the network |
| [llamacpp/README.md](llamacpp/README.md) | llama.cpp `server-cuda-b12407`: CUDA-graph workaround, run template, MoE expert offload, KV quantization, thinking, prompt cache, known issues |
| [llamacpp/models.md](llamacpp/models.md) | llama.cpp recipes for all ten picks and for the GGUF builds of Qwen3.5-9B, Gemma 4 12B, Gemma 4 E4B and Mellum2.1 |
| [llamacpp/benchmarking.md](llamacpp/benchmarking.md) | `llama-bench`, server metrics, offload sweeps and the results template |
| [docs/community-benchmarks.md](docs/community-benchmarks.md) | Third-party measurements and calculators, labeled desktop or laptop and measured or estimated |
| [docs/sources.md](docs/sources.md) | Source catalog, evidence classes and the 19 reviewed user-provided sources |
| [docs/agent-harness-evidence.md](docs/agent-harness-evidence.md) | First-hand tool-loop results from the user's LibreDB Studio harness (measured on the M5 Max), what transfers to this laptop, and which models to try first |
| [research/2026-11-15/](research/2026-11-15/README.md) | Raw research records behind these documents (discovery, verification, source reviews, ranking, writer and reviewer reports, the no-toolkit GPU probe) |
| [research/2026-11-16/](research/2026-11-16/README.md) | Measurement records from this laptop: gpt-oss-20b on llama.cpp and vLLM, Qwen3.8-27B, Qwen3.6-35B-A3B and Gemma 4 26B-A4B with their tuning variants, K2 Horizon 7B with an f16 KV check, K2 Horizon MoVA, Qwen3-Coder-Next with offload and KV variants, Nemotron 3.5 Bolt, Groot-19B-A3B, DeepSeek-R1-0528-Qwen3-8B, Nemotron 3.5 Lightning, five Ollama 0.40.16 cross-checks of LibreDB Studio models, the launch plans for the next eight models, and the LibreDB Studio review |
| [research/2026-11-17/](research/2026-11-17/README.md) | Measurement records from this laptop: on vLLM, Mellum2.1, K2 Horizon 7B, Qwen3.5-9B, Gemma 4 12B, Gemma 4 E4B and Groot-19B-A3B measured, and DeepSeek-R1-0528-Qwen3-8B failed on its tokenizer; the llmctl clients e2e over every recipe with Hermes, Pi and dsh, and with OpenCode for every recipe that has an OpenCode entry (`llmctl-dsh-e2e/`), the first e2e run's summary and cited records (`llmctl-dsh-e2e-run1/`), the dsh Web UI checklist (`llmctl-dsh-web/`), the Kimi Code CLI stub evidence behind the Kimi entry (`llmctl-kimi-stub/`), the e2e run with Kimi Code as the fifth client with its strace check and TUI checklist (`llmctl-kimi-e2e/`), and SGLang v0.5.23's first runs on this laptop by hand, the hands-on run, measurement A, the design's offline probes and measurement B (`sglang-runtime/`), with their per-model records (`sglang-qwen3.5-9b/`, `sglang-k2-horizon-7b/`, `sglang-gpt-oss-20b/`, `sglang-gemma-4-e4b/`, `sglang-gemma-4-12b/`, `sglang-mellum2.1-12b/`, `sglang-groot-19b-a3b/` and `sglang-deepseek-r1-0528-qwen3-8b/`), and measurement B's vLLM records of gpt-oss-20b without a network (`vllm-gpt-oss-20b-offline/`) and of the DeepSeek-R1-0528-Qwen3-8B tokenizer retry (`vllm-deepseek-r1-0528-qwen3-8b-retry/`), and measurement C's lane report (`sglang-runtime/measure-c/`), its gpt-oss-20b pinned-store check (`sglang-gpt-oss-20b/measure-c/`) and its llama.cpp records of the four GGUF drafts (`llamacpp-gemma-4-12b/`, `llamacpp-gemma-4-e4b/`, `llamacpp-mellum2.1-12b/` and `llamacpp-qwen3.5-9b/`) |
| [research/2026-11-18/](research/2026-11-18/README.md) | Measurement records from this laptop: measurement C's runs dated 2026-11-18, SGLang v0.5.23 wave 3 by hand with Qwen3.6-35B-A3B measured on Paged Experts and Nemotron 3.5 Lightning, Gemma 4 26B-A4B and K2 Horizon MoVA failed (`sglang-<id>/`), the vLLM DeepSeek-R1-0528-Qwen3-8B cold start with llmctl's tokenizer folder (`vllm-deepseek-r1-0528-qwen3-8b-cold-start/`), and measurement D, K2 Horizon 7B at fraction 0.88 and the margins of Qwen3.5-9B, Gemma 4 12B and Gemma 4 E4B with `PYTORCH_ALLOC_CONF` and `TIKTOKEN_ENCODINGS_BASE` added (`sglang-<id>/measure-d/`, with its lane report in `research/2026-11-17/sglang-runtime/measure-d/`), and measurement E's flushed reading of Qwen3.6-35B-A3B (its `measure-e/` folder); the llmctl clients e2e over all 29 recipes on the three runtimes with five clients and the lead's checks after it (`llmctl-sglang-e2e/`) |

## Current picks

| Rank | Model | Runtime and context (estimate) | Thinking | Tools |
|---|---|---|---|---|
| 1 | Qwen3.8-27B | llama.cpp, all on the GPU, 64k (measured 36 tok/s, 14.4 GiB peak) | Toggle | Yes |
| 2 | Qwen3.6-35B-A3B | llama.cpp, experts partly in RAM with a GPU expert cache, 128k (measured 122 tok/s, 14.8 GiB peak) | Toggle | Yes |
| 3 | K2 Horizon MoVA 36B A4B | llama.cpp, all routed experts in RAM with a GPU expert cache, 64k (measured 37 tok/s, 14.5 GiB peak) | Always on | Yes |
| 4 | K2 Horizon 7B | llama.cpp, all on the GPU, 80k (measured 67 tok/s, 14.7 GiB peak) | Always on | Yes |
| 5 | Gemma 4 26B-A4B IT | llama.cpp, experts partly in RAM with a GPU expert cache, 128k (measured 148 tok/s, 14.7 GiB peak) | Toggle, off by default | Yes |
| 6 | Qwen3-Coder-Next | llama.cpp, experts mostly in RAM with a GPU expert cache, 128k (measured 72 tok/s, 14.7 GiB peak) | No | Yes |
| 7 | NVIDIA Nemotron 3.5 Bolt 74B-A9B | llama.cpp, experts partly in RAM with a GPU expert cache, 128k (measured 74 tok/s, 14.7 GiB peak) | Toggle | Yes |
| 8 | Groot-19B-A3B-261023 | llama.cpp all on the GPU, 128k (measured 170 tok/s, 13.5 GiB peak), or vLLM NVFP4 | Toggle | Yes |
| 9 | gpt-oss-20b | llama.cpp 128k (measured 166 tok/s); vLLM only in eager mode at half speed (measured) | Always on | Yes |
| 10 | DeepSeek-R1-0528-Qwen3-8B | llama.cpp or Ollama, chat only, 64k (measured 71 tok/s, 13.0 GiB peak) | Always on | No |

The ranking, checkpoints, sizes, evidence and caveats are in [docs/model-selection.md](docs/model-selection.md).
Ranks 7 and 8 rest on vendor-reported scores, and rank 10 is there only because the DeepSeek family was required.

## Rules for every setup here

- Pin image tags: `vllm/vllm-openai:v0.33.0`, `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407` and `lmsysorg/sglang:v0.5.23-runtime`.
- Do not use SGLang's moving tags `latest`, `dev` or `dev-cu13`, nor `nightly-cu134*`, which is built for CUDA 13.4 (driver branch R615), so 595.99 would rely on minor version compatibility ([sglang/README.md](sglang/README.md#image-and-versions)).
- Do not use `server-cuda13`: its image requires CUDA 13.5 or newer, and the driver provides CUDA 13.2 ([llamacpp/README.md](llamacpp/README.md)).
- Pass `-e GGML_CUDA_DISABLE_GRAPHS=1` to every llama.cpp container, before the image name, because CUDA graphs hang RTX 50 laptop GPUs ([llama.cpp #27330](https://github.com/ggml-org/llama.cpp/issues/27330)).
- Publish ports on 127.0.0.1 only, because Docker-published ports bypass ufw, which is active here ([Docker docs](https://docs.docker.com/engine/network/packet-filtering-firewalls/)).
- Give every model container a memory cap (`--memory` with an equal `--memory-swap`), because the `heavy` wrapper caps only the docker CLI, not the container ([docs/host-setup.md](docs/host-setup.md)).
- Use no restart policies: start a server on demand, run one at a time, and stop it when done, so the dGPU can power down.
- Start toggle models with thinking off on the server and turn it on per session when a task needs it ([docs/opencode.md](docs/opencode.md)).
- For a MoE model with experts in RAM, load with `-lm none` and give the VRAM left at the peak to `--moe-cache-mib`, keeping about 500 MiB free: on Qwen3.6 this doubled prefill and raised decode from 67 to 122 tok/s, and on Gemma 4 and Nemotron 3.5 Lightning moving whole blocks' VRAM into the cache added another 5% (measured, [llamacpp/README.md](llamacpp/README.md#moe-expert-offload)).
- Measure q8_0 KV against f16 at depth, not only at a short context: on Qwen3-Coder-Next q8_0 was 13% faster at a short context, because it leaves more VRAM for the expert cache, but 17% slower after a 101k prompt (measured, [llamacpp/README.md](llamacpp/README.md#kv-cache-quantization-and-context)).

## Next steps

1. Done on 2026-11-15: the full NVIDIA Container Toolkit 1.20.1 is installed and GPU access from containers is verified ([docs/host-setup.md](docs/host-setup.md)).
2. Done on 2026-11-16 for all ten ranks ([research/2026-11-16/](research/2026-11-16/README.md)); next come the remaining measurements listed in [docs/model-selection.md](docs/model-selection.md#what-to-measure-first).
3. Record results in the templates in [llamacpp/benchmarking.md](llamacpp/benchmarking.md), [vllm/benchmarking.md](vllm/benchmarking.md) and [sglang/benchmarking.md](sglang/benchmarking.md), replace estimates with measured numbers, and update the "Last verified" lines.

## Conventions

- Every load-bearing claim links to its source.
- Evidence labels: "(estimate)" for arithmetic, "(vendor-reported)" for self-reported scores, and "(measured on <hardware>, <runtime>)" for measurements.
- Each document starts with a "Last verified" date; update it whenever you re-check its facts.
- Each sentence sits on its own line, so a one-sentence change is a one-line diff.

## Maintenance

Versions in this area change every few weeks.
Before acting on a document, re-check the image tags, issue states and model cards it cites.
Re-run the judge panel when a strong new model with independent agentic scores appears.
The raw records are kept in [research/2026-11-15/](research/2026-11-15/README.md) (research runs), [research/2026-11-16/](research/2026-11-16/README.md), [research/2026-11-17/](research/2026-11-17/README.md) and [research/2026-11-18/](research/2026-11-18/README.md) (measurements); do not edit them by hand.
Before the repository went public on 2026-11-16, a script replaced local paths in the records with `$HOME`, `$SCRATCH` and `$SESSION`, masked one GPU UUID and changed nothing else.

## License

The documentation and the measurement records are licensed under CC BY 4.0 ([LICENSE](LICENSE)); the scripts (`.sh`, `.py`, `.js`), `llmctl`, `launcher/` and `recipes/` are licensed under MIT ([LICENSE-CODE](LICENSE-CODE)).
Third-party material that is quoted or linked here keeps its own license.
