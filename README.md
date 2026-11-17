# Local LLMs on the MSI Raider 18 Max HX (RTX 5080 Laptop)

Last verified: 2026-11-17.

This repository is the knowledge base for running open-weight LLMs in Docker on one laptop, an MSI Raider 18 Max HX A2WI with an RTX 5080 Laptop GPU (16 GB), with llama.cpp and vLLM, and for using them from OpenCode as a coding agent.
Use Qwen3.6-35B-A3B on llama.cpp as the default: it measured 122 tok/s with 128k context.
Switch to Qwen3.8-27B (36 tok/s, 64k context) when a task needs the strongest model; it keeps rank 1 on quality evidence.
vLLM is the second path, for the smaller models that fit fully in 16 GB of VRAM.
GPU access from containers works on this laptop since 2026-11-15, and all ten picks were measured on llama.cpp on 2026-11-16, nine of them completing an OpenCode tool session: Qwen3.8-27B at 36 tok/s and K2 Horizon MoVA at 37 tok/s with 64k context, Qwen3.6-35B-A3B at 122 tok/s, Gemma 4 26B-A4B at 148 tok/s and Qwen3-Coder-Next at 72 tok/s with 128k, K2 Horizon 7B at 67 tok/s with 80k, Groot-19B-A3B at 170 tok/s and Nemotron 3.5 Bolt at 74 tok/s with 128k, and gpt-oss-20b at 166 tok/s; the chat-only DeepSeek-R1-0528-Qwen3-8B measured 71 tok/s and failed the OpenCode tool session when tools were enabled for a test, and Nemotron 3.5 Lightning, an honorable mention, measured 144 tok/s with 128k ([llamacpp/models.md](llamacpp/models.md)).
On Ollama 0.40.16, the five small models from the user's LibreDB Studio list (qwen3.5:9b, gemma4:12b, ornith:9b, ministral-3:8b and qwen2.5:7b) fit fully on the GPU at 99-119 tok/s and passed the same OpenCode session ([docs/agent-harness-evidence.md](docs/agent-harness-evidence.md#try-here-first)).
On vLLM v0.33.0, gpt-oss-20b was measured on 2026-11-16, and Mellum2.1, Qwen3.5-9B, Gemma 4 12B, Gemma 4 E4B, K2 Horizon 7B and Groot-19B-A3B on 2026-11-17 ([research/2026-11-17/](research/2026-11-17/README.md)), while DeepSeek-R1-0528-Qwen3-8B failed there on its tokenizer.
Figures for every other model and runtime are still estimates or third-party numbers until they are measured here.

## Quick start

Start a measured model by name with [llmctl](docs/llmctl.md), which applies the rules below for you and points OpenCode, Hermes Agent and Pi at it; `--set-default` also makes it each installed client's default:

```bash
./llmctl list
./llmctl pull qwen3.6
./llmctl up qwen3.6 --set-default
./llmctl down
```

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
| [docs/opencode.md](docs/opencode.md) | OpenCode v2.3.0 provider config (schema-validated), thinking variants, context budget, known issues, remote trials |
| [vllm/README.md](vllm/README.md) | vLLM v0.33.0 on sm_120: image, run template, memory budget, flags, parsers, quantization paths, known issues, smoke tests |
| [vllm/models.md](vllm/models.md) | vLLM recipes for the models that fit, and the models that do not |
| [vllm/benchmarking.md](vllm/benchmarking.md) | Benchmarking with `vllm bench serve`, scenarios and the results template |
| [llamacpp/README.md](llamacpp/README.md) | llama.cpp `server-cuda-b12407`: CUDA-graph workaround, run template, MoE expert offload, KV quantization, thinking, prompt cache, known issues |
| [llamacpp/models.md](llamacpp/models.md) | llama.cpp recipes for all ten picks |
| [llamacpp/benchmarking.md](llamacpp/benchmarking.md) | `llama-bench`, server metrics, offload sweeps and the results template |
| [docs/community-benchmarks.md](docs/community-benchmarks.md) | Third-party measurements and calculators, labeled desktop or laptop and measured or estimated |
| [docs/sources.md](docs/sources.md) | Source catalog, evidence classes and the 19 reviewed user-provided sources |
| [docs/agent-harness-evidence.md](docs/agent-harness-evidence.md) | First-hand tool-loop results from the user's LibreDB Studio harness (measured on the M5 Max), what transfers to this laptop, and which models to try first |
| [research/2026-11-15/](research/2026-11-15/README.md) | Raw research records behind these documents (discovery, verification, source reviews, ranking, writer and reviewer reports, the no-toolkit GPU probe) |
| [research/2026-11-16/](research/2026-11-16/README.md) | Measurement records from this laptop: gpt-oss-20b on llama.cpp and vLLM, Qwen3.8-27B, Qwen3.6-35B-A3B and Gemma 4 26B-A4B with their tuning variants, K2 Horizon 7B with an f16 KV check, K2 Horizon MoVA, Qwen3-Coder-Next with offload and KV variants, Nemotron 3.5 Bolt, Groot-19B-A3B, DeepSeek-R1-0528-Qwen3-8B, Nemotron 3.5 Lightning, five Ollama 0.40.16 cross-checks of LibreDB Studio models, the launch plans for the next eight models, and the LibreDB Studio review |
| [research/2026-11-17/](research/2026-11-17/README.md) | vLLM measurement records from this laptop: Mellum2.1, K2 Horizon 7B, Qwen3.5-9B, Gemma 4 12B, Gemma 4 E4B and Groot-19B-A3B measured, and DeepSeek-R1-0528-Qwen3-8B failed on its tokenizer; the llmctl clients e2e over every recipe with Hermes, Pi and dsh, and with OpenCode for every recipe that has an OpenCode entry (`llmctl-dsh-e2e/`), the first e2e run's summary and cited records (`llmctl-dsh-e2e-run1/`), the dsh Web UI checklist (`llmctl-dsh-web/`), and the Kimi Code CLI stub evidence behind the Kimi entry (`llmctl-kimi-stub/`) |

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

- Pin image tags: `vllm/vllm-openai:v0.33.0` and `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407`.
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
3. Record results in the templates in [llamacpp/benchmarking.md](llamacpp/benchmarking.md) and [vllm/benchmarking.md](vllm/benchmarking.md), replace estimates with measured numbers, and update the "Last verified" lines.

## Conventions

- Every load-bearing claim links to its source.
- Evidence labels: "(estimate)" for arithmetic, "(vendor-reported)" for self-reported scores, and "(measured on <hardware>, <runtime>)" for measurements.
- Each document starts with a "Last verified" date; update it whenever you re-check its facts.
- Each sentence sits on its own line, so a one-sentence change is a one-line diff.

## Maintenance

Versions in this area change every few weeks.
Before acting on a document, re-check the image tags, issue states and model cards it cites.
Re-run the judge panel when a strong new model with independent agentic scores appears.
The raw records are kept in [research/2026-11-15/](research/2026-11-15/README.md) (research runs), [research/2026-11-16/](research/2026-11-16/README.md) and [research/2026-11-17/](research/2026-11-17/README.md) (measurements); do not edit them by hand.
Before the repository went public on 2026-11-16, a script replaced local paths in the records with `$HOME`, `$SCRATCH` and `$SESSION`, masked one GPU UUID and changed nothing else.

## License

The documentation and the measurement records are licensed under CC BY 4.0 ([LICENSE](LICENSE)); the scripts (`.sh`, `.py`, `.js`), `llmctl`, `launcher/` and `recipes/` are licensed under MIT ([LICENSE-CODE](LICENSE-CODE)).
Third-party material that is quoted or linked here keeps its own license.
