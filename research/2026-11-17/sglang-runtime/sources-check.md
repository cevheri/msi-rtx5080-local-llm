# SGLang v0.5.23 sources: the independent check, 2026-11-18

Last verified: 2026-11-18.

This file is the independent check of the sources behind the SGLang runtime, step 3 of [Adding a new source](../../../docs/sources.md#adding-a-new-source): a second agent re-fetched every source and checked each claim that the knowledge base takes from it.
It covers the source list of the design's documents section (D16), the claims that the design, the lead's decisions and [measurement A](measure-a/README.md) take from those sources, and every claim of the research report's sections 2 to 4, 6 and 7 that the SGLang documents will carry and that rests on a source rather than on a run on this laptop.
The research lanes were the first read of these sources, and the research report's section 2.7 was their first check; this file is the second, done by a second agent on 2026-11-18 between 00:28 and 00:41 (+03:00).
The documents cite a claim through its ID below, and only a claim whose verdict starts with `Holds`.

## How the check worked

- Every file and page was fetched again with `curl -fsSL` into a new, empty folder `$SCRATCH/sglang-sources-<random>/` and read only by `python3 -I` scripts kept in `$SCRATCH/sglang/sources/`; nothing fetched was run.
- SGLang files came from `https://raw.githubusercontent.com/sgl-project/sglang/v0.5.23/<path>`, located in the tag's tree listing (`git/trees/v0.5.23?recursive=1`, not truncated), and the whole tree from the tag's source archive, searched member by member without unpacking it.
- The tag resolves to the annotated tag object `7df5dd09bc2f0e7e1509a1f7c2a40dff00ad8a01` and the commit `297e54fd105e974cdf401e5b4d20253a22c16f90`, which the source archive's header names too, so every file below is that commit's.
- The release, the pull requests, the issues and the comparisons of merge commits with the tag came from the GitHub REST API without a token, the image tags from the Docker Hub v2 API, the checkpoint sizes from the Hugging Face model API with `?blobs=true`, FlashInfer and openai-harmony from their tags `v0.7.2` and `v0.0.4`, and NVIDIA's table from its CUDA Toolkit release notes page, turned into text with `html.parser`.
- A quote holds when it is in the fetched text at the cited place, with backticks and the source's line wrapping ignored, and the claim says no more than the text; a claim about code holds when the cited lines do what the claim says; a line number given below is the fetched file's.
- A verdict is `Holds`, `Holds with corrections: <correction>` or `Does not hold: <why>`, followed by `; checked <how>`.
- The rows keep the labels of the documents: a pull request or issue opened by an outside contributor is a third-party report, a merged one by a project collaborator is vendor-reported.

## Result

Of 110 claims, 105 hold and 5 hold with corrections (SG-06, SG-43, SG-82, SG-85 and SG-102); none fails.
The corrections are:

- SG-06: `nightly-cu134*` is built for CUDA 13.4, whose corresponding driver branch is R615, but NVIDIA states that CUDA 13.x applications run on drivers from 580 on under minor version compatibility, so driver 595.99 is not certain to refuse it; the documents should say "built for CUDA 13.4 (driver branch R615), so 595.99 would rely on minor version compatibility" instead of "needs an R615 driver", and the rule not to use it stands.
- SG-43: the default-fraction sum in the report omits the decode graph's 16 MiB (`max_bs` 8 x 2); 512 + 3,072 + 128 + 16 + 336 gives the 4,064 MiB it states.
- SG-82: the 16.6 tok/s of PR #42557 is `--cpu-offload-gb 3` serving the MoE Qwen3-30B-A3B in int4 on a 16 GB desktop RTX 5070 Ti, not a dense model, so a document that cites it for the dense Qwen3.8-27B names that model and card.
- SG-85: PR #45520 is an open pull request by an outside contributor, so its quote is a third-party report, not vendor-reported.
- SG-102: the Nemotron 3.5 Bolt BF16 checkpoint is 139.49 GiB of safetensors, not about 138 GiB.

## Release, tag and image

### [SGLang v0.5.23 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.23) and [tag](https://github.com/sgl-project/sglang/tree/v0.5.23)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-01 | v0.5.23 is the newest stable release, tag commit 297e54fd of 2026-10-29, GitHub release published 2026-11-02 (report §2.2; LD2; the lead's re-check) | Holds; checked the release record (`published_at` 2026-11-02T22:14:10Z, not a prerelease), the twelve newest releases (none newer on 2026-11-18), and the tag ref, tag object and commit (committer date 2026-10-29T05:22:02Z) |
| SG-02 | Releases were published on 2026-09-05, 09-18, 10-02, 10-19 and 11-02, so v0.5.24 is due (report §2.2, inferred) | Holds; checked the release list: v0.5.19 to v0.5.23 on exactly those dates, and no v0.5.24 on 2026-11-18 |
| SG-03 | The release notes name the CUDA 13 image with "docker pull lmsysorg/sglang:v0.5.23" (report §2.2) | Holds; checked the release body, row "NVIDIA (CUDA 13)", line 38 |

### [Docker Hub tag record of v0.5.23-runtime](https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/v0.5.23-runtime), with the [v0.5.23 record](https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/v0.5.23) and the [tag list](https://hub.docker.com/r/lmsysorg/sglang/tags)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-04 | `lmsysorg/sglang:v0.5.23-runtime` has the index digest `sha256:3ce6da3e5ea0fee2dac6a23b528cb294fedd31ca74a55a4511f709c5e01e8652` and is 14.47 GB compressed (report §1, §2.2; LD2; the design's D10 and D16) | Holds; checked the tag record: `digest` as stated, `full_size` 14,470,475,457 bytes (the amd64 image; arm64 is 14,390,948,038), last pushed 2026-10-30 |
| SG-05 | `v0.5.23` without `-runtime` is 16.11 GB compressed (report §2.2) | Holds; checked the tag record: `full_size` 16,111,480,510 bytes, digest `sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f` |
| SG-06 | Do not use `latest`, `dev` or `dev-cu13`, which move, nor `nightly-cu134*`, which is CUDA 13.4 and needs an R615 driver (report §2.2; D16 for README.md's rules) | Holds with corrections: CUDA 13.4 is the corresponding driver branch R615 (SG-97), while NVIDIA's note says CUDA 13.x applications run on drivers from 580 on under minor version compatibility, so the documents say that 595.99 would rely on minor version compatibility, not that it needs R615; checked the tag list: `latest` and `latest-runtime` now carry v0.5.23's two digests, `dev` and `dev-cu13` were pushed on 2026-11-17, `nightly-cu134` daily (67 tags, the newest 2026-11-17), and CUDA 13.4 is read from the tag name, not from the image |

### [docker/Dockerfile at v0.5.23](https://github.com/sgl-project/sglang/blob/v0.5.23/docker/Dockerfile)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-07 | The Dockerfile builds only CUDA 13.0.3: "ARG CUDA_VERSION=13.0.3" (report §2.2; D16 for docs/host-setup.md) | Holds; checked line 1, and the release workflow passes `--build-arg CUDA_VERSION=13.0.3` for both architectures (SG-10) |
| SG-08 | The runtime stage copies the full image's packages: "COPY --from=framework_final /opt/sglang/lib/python3.12/site-packages" (report §2.2) | Holds; checked line 785 |
| SG-09 | The Dockerfile's own default leaves the FlashInfer JIT cache out, which the local lane read (report §2.7) | Holds; checked line 20, `ARG INSTALL_FLASHINFER_JIT_CACHE=0`, and the install step at line 294 that runs only for `1` |

### [Release workflow _docker-build-and-publish.yml at v0.5.23](https://github.com/sgl-project/sglang/blob/v0.5.23/.github/workflows/_docker-build-and-publish.yml)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-10 | The release builds pass "--build-arg INSTALL_FLASHINFER_JIT_CACHE=1" (report §2.2, §2.7) | Holds; checked lines 122 (amd64) and 201 (arm64), each in a `docker buildx build` whose `--target` is the caller's `docker_target`, "framework or runtime" (line 10) |

## Documentation at tag v0.5.23

### [README.md](https://github.com/sgl-project/sglang/blob/v0.5.23/README.md)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-11 | "SGLang is an open-source inference framework for large language, vision-language, and diffusion models, optimized for agentic workloads, large-scale serving, and RL rollouts." (report §2.1) | Holds; checked line 21 |
| SG-12 | The README lists "RTX 30/40/50 series" among the supported GPUs (report §2.3) | Holds; checked line 55, the NVIDIA row of the hardware table |

### [docs/get-started/install.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/get-started/install.mdx)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-13 | "CUDA 12 images (`-cu12` / `-cu129`) are no longer published", and v0.5.19 was the last CUDA 12 release (report §2.2) | Holds; checked line 22, which names `lmsysorg/sglang:v0.5.19-cu129` as the last CUDA 12 tag |
| SG-14 | The runtime image is a smaller image of the same stable release (report §2.2, preferring `-runtime`) | Holds; checked line 38: "A smaller image of the same stable release, with inference dependencies and fewer development utilities" |

### [docs/get-started/quickstart.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/get-started/quickstart.mdx)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-15 | `sglang serve` is the documented entrypoint, while `python3 -m sglang.launch_server` still works with a warning (report §2.4; the design's container command) | Holds; checked lines 25 and 42, `sglang serve MODEL_PATH --host 0.0.0.0 --port 30000`, with no `launch_server` on the page; the warning itself is the hands-on lane's measured log line |

### [docs/advanced_features/server_arguments.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/server_arguments.mdx)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-16 | `--disable-cuda-graph` is deprecated in favour of the per-phase flags, and `--disable-prefill-cuda-graph` and `--disable-decode-cuda-graph` exist (report §2.5; D5 rules 6 and 7; the spec's research risk 13) | Holds; checked the rows at lines 2705 ("Deprecated. Use --cuda-graph-backend-decode=disabled and/or --cuda-graph-backend-prefill=disabled."), 2711 and 2717 ("Disable the decode-phase CUDA graph. Convenience for --cuda-graph-backend-decode=disabled.") |
| SG-17 | Both parser flags default to none and accept `auto`: "Use `auto` to detect the parser from the model's chat template." (report §5.1 via D5 rule 10; lane-docs 27) | Holds; checked the rows of `--reasoning-parser` (line 1165) and `--tool-call-parser` (line 1171), default `None` |
| SG-18 | `--config` reads a YAML file that may carry any option, and "CLI arguments will override config file values." (D5 rule 2) | Holds; checked line 14 and the row at line 3386: "Read CLI options from a config file. Must be a YAML file with configuration options." |
| SG-19 | `--sampling-defaults` defaults to `model`, which takes the defaults from `generation_config.json`, and `openai` ignores it (D5 rule 8; A §M8) | Holds; checked the row at line 1183, choices `openai` and `model`, default `model` |
| SG-20 | There is no plain `fp8` KV cache type; FP8 KV is `fp8_e4m3` (report §2.5) | Holds; checked the row at line 365: choices `auto`, `fp8_e5m2`, `fp8_e4m3`, `bf16`, `bfloat16`, `nvfp4`, `fp4_mx_block16` |
| SG-21 | `--paged-experts-num-resident` is required with `--enable-paged-experts` and takes an integer, with no `auto` in v0.5.23 (D5 rule 11; lane-docs 30) | Holds; checked the rows at lines 2279 and 2285: "Required with `--enable-paged-experts`; at least the router's top-k.", type int, default `None` |
| SG-22 | `--watchdog-timeout` defaults to 300 s: "If a forward batch takes longer than this, the server will crash to prevent hanging." (lane-docs 35, for sglang/README.md's troubleshooting) | Holds; checked the row at line 753 |
| SG-23 | `--preferred-sampling-params` is a JSON of sampling settings and promises nothing about chat requests (A §M8, where it never reached a chat request) | Holds; checked the row at line 1315: "json-formatted sampling settings that will be returned in /get_model_info" |
| SG-24 | Left unset, `--mem-fraction-static` is computed from the GPU memory and a reserve (report §2.6) | Holds; checked the row at line 445: "When unset, it is computed as `(GPU memory - reserved memory) / GPU memory`" |
| SG-25 | Per-request `chat_template_kwargs` win over `--default-chat-template-kwargs` (report §6.1, §6.3) | Holds; checked the row at line 1141: "For ordinary template keys, explicit per-request values take precedence." |
| SG-26 | `--swa-full-tokens-ratio` defaults to 0.8, and `--cpu-offload-gb` and `--max-mamba-cache-size` exist (report §2.5) | Holds; checked the rows at lines 565 (default `0.8`), 2406 and 2025 |

### [docs/advanced_features/paged_experts.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/paged_experts.mdx)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-27 | Paged Experts keeps K experts per MoE layer on the GPU and pages the rest from pinned host memory; "Computation stays on the GPU." (report §2.5) | Holds; checked lines 4 and 19 |
| SG-28 | MXFP4 and ModelOpt NVFP4 experts page only with `--moe-runner-backend marlin`, ModelOpt NVFP4 running as W4A16; GPTQ, AWQ and compressed-tensors W4A16 go through Marlin, unquantized, FP8 and INT8 through Triton (D5 rule 11; lane-docs 30) | Holds; checked the table at lines 52 to 63 and line 65, "Other quantization methods and MoE runners are rejected at startup." |
| SG-29 | Tensor, expert, pipeline and data parallelism, fused shared experts (use `--disable-shared-experts-fusion`) and speculative decoding are rejected at startup (report §4, §8) | Holds; checked lines 66 to 69 |
| SG-30 | The host store holds every expert and "is page-locked, so the host must be able to pin that much memory" (report §3.3 and §4 for the memory caps) | Holds; checked lines 73 to 74 |
| SG-31 | Each resident expert costs `per_expert_bytes x num_moe_layers` of the static budget, for example K = 92 of 128 on a 16 GB GPU at `--mem-fraction-static 0.88` (report §4 notes) | Holds; checked lines 131 to 140, where the page writes the multiplication sign |
| SG-32 | GSM8K scores 0.96 to 0.97 for gpt-oss-20b paged with 16 of 32 experts resident against 0.975 unpaged, and "Accuracy is unchanged" (report §3.3) | Holds; checked lines 153 to 155 |
| SG-108 | v0.5.23 has no disk tier and no host-memory cap for Paged Experts, so its only options are `--enable-paged-experts` and `--paged-experts-num-resident`, without `auto` (the design's Non-goals, which cite report §2.5 and lane-docs 32; report §4's runtime coverage) | Holds; checked lines 9 and 73 of this page (the host store holds all E experts in pinned host memory, with no other tier), the only two Paged Experts rows of server_arguments.mdx (lines 2280 and 2286), the only two Paged Experts fields of [arg_groups/fields/exec_.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/fields/exec_.py) (lines 999 and 1007), [layers/moe/paged_experts/store.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/layers/moe/paged_experts/store.py) line 30, which allocates the store as CPU tensors, and a scan of every text file of the source archive (as in SG-77), where `paged_experts_enable_disk`, `paged_experts_host_memory` and a `num_resident` with `auto` have no hit in either the underscore or the hyphen spelling |

### [docs/advanced_features/tool_parser.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/tool_parser.mdx)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-33 | `--tool-call-parser` alone turns tool calling on, and SGLang has no `--enable-auto-tool-choice` (report §2.5; D5 client checks; D16 for docs/opencode.md) | Holds; checked line 119, "Note that `--tool-call-parser` defines the parser used to interpret responses.", no `enable-auto-tool-choice` on the page or among the flag rows of server_arguments.mdx, and no such field in `server_args.py` |
| SG-34 | "Tool choice is fully supported with the **Xgrammar backend**, which is the default grammar backend" (lane-docs 40; report §3.4 for the grammar path of a forced choice) | Holds; checked line 475 |

### [docs/advanced_features/separate_reasoning.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/separate_reasoning.mdx)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-35 | Replies carry the chain of thought as `reasoning_content`: "`reasoning_content`: The content of the CoT." (report §6.2; D16 for docs/opencode.md) | Holds; checked line 125 |
| SG-36 | The `deepseek-r1` reasoning parser "Supports all variants (R1, R1-0528, R1-Distill)" (report §6.4) | Holds; checked the table row at line 36 |

## Source files at tag v0.5.23

Each heading links the file at the tag; the IDs follow the order in which the documents meet them; SG-108 (in the documentation above), SG-109 and SG-110 were added to their sources' tables after the first pass and stand out of that order.

### [python/sglang/srt/arg_groups/fields/serving.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/fields/serving.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-37 | The server binds 127.0.0.1:30000 by default, so a container passes `--host 0.0.0.0 --port 30000` (report §2.4; D6) | Holds; checked lines 65 and 66, `host ... = "127.0.0.1"` and `port ... = 30000` |

### [python/sglang/srt/arg_groups/fields/schedule.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/fields/schedule.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-38 | `swa_full_tokens_ratio` falls back to 0.8: "fallback=0.8" (report §2.5) | Holds; checked lines 150 to 162 |

### [python/sglang/srt/arg_groups/fields/model.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/fields/model.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-39 | `--model` is an alias of `--model-path` (D5 rule 1) | Holds; checked lines 51 to 57, `aliases=["--model"]` at line 55 |

### [python/sglang/srt/arg_groups/fields/observability.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/fields/observability.py) and [entrypoints/openai/serving_base.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/entrypoints/openai/serving_base.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-40 | `--log-requests-level` 1 is the lowest level that logs sampling parameters, and level 2 also logs the raw OpenAI request (A §M8) | Holds; checked observability.py lines 40 to 46 (levels 0 to 3, default 2) and serving_base.py line 96, which logs the received OpenAI request at level 2 and above |
| SG-110 | SGLang's OpenTelemetry tracing is behind `--enable-trace`, which is off by default (report §7) | Holds; checked observability.py line 164 (`enable_trace: A[bool, "Enable opentelemetry trace"] = False`) and line 171 (`--otlp-traces-endpoint`, "Config opentelemetry collector endpoint if --enable-trace is set", default `localhost:4317`), and the `--enable-trace` row of server_arguments.mdx at line 996, default `False` |

### [python/sglang/srt/server_args.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/server_args.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-41 | The parser is `argparse.ArgumentParser(prog="sglang serve")` with Python's default abbreviation matching, so a prefix such as `--context-len` reaches `--context-length` (D5 rule 3, plan ruling 5) | Holds; checked line 679 and that the file never sets `allow_abbrev`; the abbreviation behaviour is Python's default (inferred from the code) |

### [python/sglang/srt/arg_groups/memory_hook.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/memory_hook.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-42 | Below 20 GiB SGLang uses "# (chunked_prefill_size 2k, max_bs 8)", so `--chunked-prefill-size` defaults to 2,048 (report §2.5, §2.6) | Holds; checked lines 90 to 101 |
| SG-43 | Left unset, the fraction is about 0.75: SGLang reserves 512 + 2,048 x 1.5 + 128 MiB plus 8 MiB for each of the 42 logged prefill graph shapes, 4,064 of 16,303 MiB, and about 0.77 without the prefill graph (report §2.6) | Holds with corrections: the sum also holds the decode graph's `max_bs` x 2 = 16 MiB (lines 364 to 368), without which the terms give 4,048 MiB; with it, 4,064 MiB and a fraction of round((16,303 - 4,064) / 16,303, 3) = 0.751, and 0.771 without the 336 MiB of prefill graphs; checked lines 296 to 328 and 357 to 386 |

### [python/sglang/srt/mem_cache/kv_cache_configurator.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/kv_cache_configurator.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-44 | SGLang sizes the KV pool as "currently-free GPU memory minus the non-static runtime slack (pre_model_load_memory * (1 - mem_fraction_static))" (report §2.6; D16 for sglang/README.md#memory-budget) | Holds; checked lines 2990 to 2993, the comment of `_profile_available_bytes`, wrapped over two lines |

### [python/sglang/srt/entrypoints/http_server.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/entrypoints/http_server.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-45 | `/health` returns 503 while the server status is Starting: "if _global_state.tokenizer_manager.server_status == ServerStatus.Starting:" (report §2.4) | Holds; checked line 735 in `health_generate`, which serves `/health` and `/health_generate` (lines 719 to 721) |
| SG-46 | `/ready` and `/health` both wait for `ServerStatus.Up`, which the server sets after its own warm-up request (A §M7; LD5 revised; D8) | Holds; checked lines 714 to 717 (`/ready` answers 200 only when `is_ready()`, which is `ServerStatus.Up` and not paused, `tokenizer_manager.py` lines 728 to 734) and line 2502, where a passed warm-up sets `ServerStatus.Up` |
| SG-47 | Once up, `/health` sleeps a second and generates one token, while `/ready` only reads the status (A §M7) | Holds; checked lines 751 to 786; the generation runs because `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` defaults to true (`environ.py` line 344) |
| SG-48 | `/v1/models` answers from the tokenizer manager without looking at the server status, so it is not a readiness signal (report §2.4) | Holds; checked `available_models`, lines 1993 to 2005 |
| SG-49 | `/get_server_info` is deprecated for `/server_info`, and both carry the server's state (A §M7; LD5 revised; D8) | Holds; checked lines 853 to 863, the warning "Endpoint '/get_server_info' is deprecated and will be removed in a future version." |
| SG-50 | For a vision model the server's own warm-up is a chat request, and a failed warm-up stops the server (A §M8 step 3, §M6 step 4) | Holds; checked line 2402 (`request_name = "/v1/chat/completions"` for a VLM), lines 2490 to 2502 and 2531 to 2535 (`kill_process_tree` after "Initialization failed. warmup error") |
| SG-51 | The HTTP server builds the Responses API object in its start-up and keeps serving chat completions when that fails, logging "OpenAI Responses API (/v1/responses) disabled" (A, other findings) | Holds; checked lines 395 to 416, whose comment names "the gpt-oss harmony vocab download" as such a failure |

### [python/sglang/srt/utils/auth.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/utils/auth.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-52 | "Health/readiness/metrics endpoints are always allowed" even with an API key (report §7) | Holds; checked lines 92 to 95 |

### [python/sglang/srt/environ.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/environ.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-53 | gpt-oss web search needs "EXA_API_KEY is the vendor BYOK credential", which llmctl never sets, and no environment variable carries the server's API key (report §7; lane-docs 34) | Holds; checked lines 805 and 808 (`EXA_API_KEY = EnvStr(None, secret=True)`), the only `API_KEY` in the file |
| SG-54 | The FA3 kernel from the Hugging Face hub is off by default (report §7, from the docs lane) | Holds; checked line 1496, `SGLANG_USE_SGL_FA3_KERNEL = EnvBool(True)`, which `kernels/ops/attention/flash_attention_v3.py` lines 30 to 38 read to load the sgl-kernel build instead of `kernels-community/sgl-flash-attn3` |

### [python/sglang/srt/layers/sampler.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/layers/sampler.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-55 | The sampler runs the token-id `all_reduce` when `SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars`, so no flag skips it while a grammar is active, and the variable forces it on for every batch (A §M6 step 4; D4; LD6 revised) | Holds; checked line 81, which reads the bare name `SYNC_TOKEN_IDS_ACROSS_TP` (so the runtime file's name is the one the sampler reads, not `environ.py`'s separate `SGLANG_SYNC_TOKEN_IDS_ACROSS_TP` at line 790), and lines 715 to 730 |

### [python/sglang/srt/entrypoints/engine.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/entrypoints/engine.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-56 | For a single-node start the engine sets `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0`, and no SGLang flag sizes NCCL's buffers (A §M6 step 4) | Holds; checked lines 1898 to 1915, which write 0 for both when they are unset and neither symmetric memory nor NVLS is enabled |

### [python/sglang/srt/managers/tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/tokenizer_manager.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-57 | Preferred sampling parameters are merged under the request's: `{**self.preferred_sampling_params, **obj.sampling_params}` (A §M8) | Holds; checked line 1550 |

### [python/sglang/srt/entrypoints/openai/protocol.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/entrypoints/openai/protocol.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-58 | Chat sampling follows "Priority: user value > model generation_config > OpenAI defaults", so a request's own fields win (report §8; A §M8; D7) | Holds; checked lines 1189 to 1202 and the five `get_param` fields after them |
| SG-59 | The defaults are temperature 1.0, top_p 1.0, top_k -1, min_p 0.0 and repetition_penalty 1.0 (A §M8; D7's effective values) | Holds; checked lines 1062 to 1068 |
| SG-60 | Chat messages and stream deltas carry `reasoning_content` and no `reasoning` field (report §3.6, §6.2; lane-docs 42) | Holds; checked lines 762, 1296 and 1357 (`reasoning_content: Optional[str] = None`), while `reasoning` appears only on the Responses API's types (lines 1958 and 2270) |
| SG-61 | Any `reasoning_effort` other than `none` turns `enable_thinking` on at request validation, and explicit `chat_template_kwargs` keys of the request still win (report §6.3; D11's Kimi wording) | Holds; checked lines 1137 to 1150: `thinking = effort != "none"`, then `setdefault` of `thinking` and `enable_thinking` |
| SG-62 | `separate_reasoning` and `stream_reasoning` default to true (lane-docs 42) | Holds; checked lines 1018 and 1019 |
| SG-109 | Chat and completion requests reach the tokenizer manager with temperature already set, so `--preferred-sampling-params` cannot fill it (A's finding on the overlay, citing `protocol.py:386-397`) | Holds; checked lines 386 to 397, the `CompletionRequest` defaults (temperature 1.0 at line 386 to top_k -1 at line 397), which [serving_completions.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/entrypoints/openai/serving_completions.py) line 152 passes on as `request.temperature`; for chat, `ChatCompletionRequest.temperature` defaults to `None` (line 945) and `to_sampling_params` fills it at lines 1197 to 1213 from the request, then the generation config, then 1.0, so a document cites lines 386 to 397 for completion requests and lines 945 and 1197 to 1213 for chat requests |

### [python/sglang/srt/entrypoints/openai/serving_chat.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/entrypoints/openai/serving_chat.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-63 | A request with tools to a recipe without `--tool-call-parser` gets HTTP 200 and any call as text in `content` (report §6.4, inferred, not run; D11's chat-only wording) | Holds, as an inference that no run confirms; checked lines 1080 to 1092 (`_tool_call_parsing_active` is false without a parser) and that neither serving_chat.py, serving_base.py nor protocol.py refuses tools when no parser is set |

### [python/sglang/srt/entrypoints/openai/serving_responses.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/entrypoints/openai/serving_responses.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-64 | For `model_type` `gpt_oss`, `OpenAIServingResponses.__init__` calls `get_stop_tokens_for_assistant_actions()`, which loads the Harmony encoding (A, other findings; D4's `TIKTOKEN_ENCODINGS_BASE`) | Holds; checked lines 213 to 224 |

### [python/sglang/srt/model_loader/weight_utils.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/model_loader/weight_utils.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-65 | With `HF_HUB_OFFLINE=1` and a snapshot path, downloads are skipped: `local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE` and "skipping download" (report §7) | Holds; checked lines 336, 703 and 742, and line 609 |

### [python/sglang/srt/utils/hf_transformers/common.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/utils/hf_transformers/common.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-66 | A snapshot without `generation_config.json` loads with no defaults, because the loader catches the error (A §M8 step 4; D7) | Holds; checked lines 717 to 723, which catch `FileNotFoundError` and `OSError` and go on without a generation config; `utils/hf_transformers_utils.py`, which A cited, is a shim that re-exports this package |

### [python/sglang/srt/configs/model_config.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/configs/model_config.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-67 | SGLang reads only `repetition_penalty`, `temperature`, `top_k`, `top_p` and `min_p` from `generation_config.json`, and only while `--sampling-defaults` is `model` (A §M8; D5's `[server] sampling`; LD8 revised) | Holds; checked lines 2210 to 2240 |

### [python/sglang/srt/sampling/sampling_params.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/sampling/sampling_params.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-68 | SGLang's bounds: temperature finite and non-negative, top_p above 0 and at most 1, min_p 0 to 1, top_k -1 or at least 1, repetition_penalty above 0 and at most 2 (D5's sampling table) | Holds; checked lines 249 to 279, where -1 becomes the whole vocabulary before the check |

### [python/sglang/srt/function_call/parser_names.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/function_call/parser_names.py) and [parser/reasoning_parser_names.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/parser/reasoning_parser_names.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-69 | The tool-call parsers `qwen3_coder`, `gpt-oss`, `k2_horizon`, `gemma4` and `hermes` and the reasoning parsers `qwen3`, `gpt-oss`, `k2_horizon`, `gemma4`, `deepseek-r1` and `nemotron_3` exist in v0.5.23 (report §6.1; D5's parser table) | Holds; checked both lists, 44 tool-call and 36 reasoning parser names |
| SG-70 | Three names differ from vLLM's: `gpt-oss` for `openai` and `openai_gptoss`, `deepseek-r1` for `deepseek_r1`, `nemotron_3` for `nemotron_v3` (report §6.1, inferred from both registries) | Holds; checked vLLM v0.33.0's [tool_parsers/__init__.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/tool_parsers/__init__.py) (`openai`, line 194) and [reasoning/__init__.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/reasoning/__init__.py) (`deepseek_r1`, `openai_gptoss`, `nemotron_v3` at lines 23, 71 and 143), which hold none of SGLang's three spellings |

### [python/sglang/srt/models/xllm.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/xllm.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-71 | `K2HorizonForCausalLM` is native: `EntryClass = [XllmForCausalLM, K2HorizonForCausalLM]`, "Inference-only xLLM K2MoE and MoVA models compatible with HF weights." (report §2.7) | Holds; checked lines 27 and 1868 |
| SG-72 | The native class serves MoVA but "supports only compressed-tensors quantized model weights", so the GPTQ build of K2 Horizon MoVA is refused (report §4) | Holds; checked lines 652 to 669, `_validate_mova_config`, whose message reads "Native xLLM/K2 Horizon serving supports only compressed-tensors quantized model weights" |

### [python/sglang/srt/models/gemma4_mm.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/gemma4_mm.py) and [gemma4_unified.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/gemma4_unified.py)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-73 | Gemma 4 E4B's audio encoder is built with the checkpoint's quantization config, and only "if getattr(config, "audio_config", None) is not None:" (report §3.5; the `{"audio_config": null}` override that B measured) | Holds; checked gemma4_mm.py lines 213 to 216 |
| SG-74 | Gemma 4 12B uses the encoder-free class, "Encoder-free unified Gemma4 (text + vision + audio).", with `self.audio_tower = None` (report §3.5) | Holds; checked gemma4_unified.py lines 131 and 179 |

### [python/sglang/kernels/aot/CMakeLists.txt](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/kernels/aot/CMakeLists.txt) and the tree at the tag

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-75 | The ahead-of-time kernels are built with "-gencode=arch=compute_120a,code=sm_120a" (report §2.3) | Holds; checked lines 222 and 509 |
| SG-76 | The tree holds tuned MoE configs for the RTX 4090 and the RTX PRO 6000 Blackwell but none for a GeForce RTX 50 GPU (report §2.3, inferred) | Holds; checked the 589 `device_name=` config files of the tree listing, whose only GeForce device is `NVIDIA_GeForce_RTX_4090` |
| SG-77 | The tree has no usage telemetry: only streaming usage statistics, OpenTelemetry tracing and FastAPI telemetry that SGLang switches off (report §7, inferred from absence) | Holds; checked every text file of the v0.5.23 source archive, that is all 11,425 regular files in every folder, not only `python/`, less the 78 that hold a NUL byte, so 11,347 files read whole: `posthog`, `DO_NOT_TRACK` and `report_usage` have no hit; `usage_stats` occurs only as `continuous_usage_stats`; `analytics` only in the CI runner-fleet reports (`scripts/ci/utils/` and two `.github/workflows/` job monitors) and the model gateway README's note on stored conversations; `telemetry`, apart from the `opentelemetry` packages and `fastapi_telemetry_disabled_kwargs` (`observability/trace.py` lines 276 to 278, which switch FastAPI's own telemetry off), only in comments, docs, tests, CI dashboards, the diffusion web UI's on-page panel and the NIXL storage backend's `captureTelemetry` agent option (`mem_cache/storage/nixl/hicache_nixl.py` line 129), none of which sends usage data anywhere; the tracing is SG-110; an earlier pass had read only the 5,473 files under `python/` that end in `.py`, `.sh`, `.toml`, `.cfg`, `.txt`, `.yml`, `.yaml` or `.json` |

## Pull requests

### [#48487 "[Fix] sampler: skip the token-id all-reduce when the TP sync group has one rank"](https://github.com/sgl-project/sglang/pull/48487)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-78 | The fix of the grammar crash was merged on 2026-11-06, after the v0.5.23 tag, and no release contains it (report §1; LD2; D4; open point 1), vendor-reported | Holds; checked `merged_at` 2026-11-06T21:58:58Z, the comparison of its merge commit `98d7f49e` with v0.5.23 (diverged: the tag lacks it), and the release list (no release after v0.5.23 on 2026-11-18); the body says a one-rank group still creates "the group's NCCL communicator on its first collective" |

### [#43302 "fix(sampler): skip identity all-reduce when tp_sync_group has a single rank"](https://github.com/sgl-project/sglang/pull/43302)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-79 | "the final-token-id `all_reduce` is gated only on `SYNC_TOKEN_IDS_ACROSS_TP` (env, default off) or non-empty `sampling_info.grammars`. It does not check the size of the sync group." (report §3.4), third-party report, open | Holds; checked body line 3, state open, opened by an outside contributor |

### [#42557 "[Feat] paged-experts: Serve MoE models whose weights do not fit VRAM"](https://github.com/sgl-project/sglang/pull/42557)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-80 | Paged Experts was merged on 2026-10-19 and is in v0.5.23; `--cpu-offload-gb` "streams *every* offloaded weight to the GPU on every forward pass, whatever the routing" (report §4; lane-docs 29) | Holds; checked `merged_at` 2026-10-19T15:10:05Z, the comparison of merge commit `8e7601ae` with v0.5.23 (the tag is 634 commits ahead and 0 behind) and body line 9 |
| SG-81 | Measured on a desktop RTX 5070 Ti (16 GB, PCIe 5.0 x16): "205.7 vs 194.4 tok/s single-stream decode" against tuned llama.cpp on Qwen3-30B-A3B int4, and "llama.cpp has the lower time-to-first-token on short prompts." (report §4 notes), vendor-reported | Holds; checked body lines 23, 25 and 97 |
| SG-82 | `--cpu-offload-gb` gave 16.6 tok/s with 3 GB offloaded on a 16 GB card, cited for the dense Qwen3.8-27B (report §4) | Holds with corrections: the figure is SGLang `--cpu-offload-gb 3` serving Qwen3-30B-A3B int4 (GPTQ, an MoE model) on the 16 GB desktop RTX 5070 Ti, so a document names that model and card beside it; checked the results table, body lines 112 to 117 |
| SG-83 | With CUDA graphs off, the first Marlin compile happens on the first request and "can exceed the scheduler watchdog" (lane-docs 35) | Holds; checked body line 158 |

### [#45787 "[Feat] paged-experts: AWQ, compressed-tensors, MXFP4, NVFP4, W8A8 and block-INT8 experts"](https://github.com/sgl-project/sglang/pull/45787)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-84 | The MXFP4 and NVFP4 Paged Experts formats were merged on 2026-10-25 and are in v0.5.23 (lane-docs 30 and 31), vendor-reported | Holds; checked `merged_at` 2026-10-25T15:26:25Z, the comparison of merge commit `2d3de826` with v0.5.23 (the tag is 121 commits ahead and 0 behind), and the body's table rows for MXFP4 and ModelOpt NVFP4 with `--moe-runner-backend marlin` |

### [#45520 "[Fix] [NVIDIA] moe: Pick a supported runner for auto-detected ModelOpt NVFP4 MoE on SM110/SM120"](https://github.com/sgl-project/sglang/pull/45520)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-85 | "ModelOpt NVFP4 MoE checkpoints crash at warmup on SM120 with the default `--moe-runner-backend auto`.", which covers `nvidia/Qwen3.6-35B-A3B-NVFP4` (report §2.3; D5 rule 11), vendor-reported, open | Holds with corrections: the pull request is open and comes from an outside contributor (author association NONE), so the documents label it a third-party report; checked body line 3, which names `nvidia/Qwen3.6-35B-A3B-NVFP4`, and state open |

## Issues

All eight were open on 2026-11-18 and were opened by outside contributors, so each quote is a third-party report.

### [#40094 "Prefill CUDA graph reserves ~1.8 GB and starves quantized-KV long-context prefill on small cards (no auto-disable rule looks at free VRAM)"](https://github.com/sgl-project/sglang/issues/40094)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-86 | "Prefill CUDA graph reserves ~1.8 GB and starves quantized-KV long-context prefill on small cards" (report §2.3; D16's Known issues) | Holds; checked the title, opened 2026-09-18, and body line 19 (1.81 GB over 51 shapes, measured upstream on 0.5.19) |

### [#49209 "[Bug] sm_120 (RTX 50-series consumer Blackwell) cannot run any model out-of-the-box with CUDA 13 wheel install"](https://github.com/sgl-project/sglang/issues/49209)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-87 | pip installs on RTX 5080 and 5090 "cannot serve any model without manual workarounds" (report §2.3; D16's Known issues) | Holds; checked body line 14, which says this of "the standard uv pip install sglang CUDA 13 wheel install", opened 2026-11-09 |

### [#48172 "[Bug] --tool-call-parser auto finds no parser on Qwen2.5-0.5B, Qwen3 2507 and Ling-mini-2.0, so tool calls come back as text"](https://github.com/sgl-project/sglang/issues/48172)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-88 | "`--tool-call-parser auto` finds no parser for templates that use the plain JSON `<tool_call>` format" and logs "--tool-call-parser=auto specified but could not detect tool-call parser from chat template. Disabling tool-call parser." (report §5.1; D5 rule 10) | Holds; checked body lines 11 and 14, opened 2026-11-03 |

### [#41747 "[Bug] gpt-oss: `tool_choice: "required"` returns raw harmony control tokens as assistant content"](https://github.com/sgl-project/sglang/issues/41747)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-89 | A forced tool choice on gpt-oss "comes back as HTTP 200 with `finish_reason: "stop"`, no `tool_calls`" (report §3.6; D13; D15) | Holds; checked body line 11; the title names `tool_choice: "required"` |

### [#44931 "[Bug] gpt-oss on /v1/chat/completions drops replayed reasoning: the template reads `thinking`, SGLang only forwards `reasoning_content`"](https://github.com/sgl-project/sglang/issues/44931)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-90 | "the template reads `thinking`, SGLang only forwards `reasoning_content`", and `reasoning` is dropped at request validation (report §3.6, §6.3) | Holds; checked the title, which holds the first quote, and body line 23 |

### [#40739 "[Bug] `qwen3_coder` non-streaming parser drops visible text after a valid tool call"](https://github.com/sgl-project/sglang/issues/40739)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-91 | "Non-streaming returns `content="before"`" (report §8), in the `qwen3_coder` parser | Holds; checked body line 21 and the title, "`qwen3_coder` non-streaming parser drops visible text after a valid tool call" |

### [#42791 "[Bug] gemma4 tool-call parser hangs forever on a stray "\]" in an array argument"](https://github.com/sgl-project/sglang/issues/42791)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-92 | The `gemma4` tool-call parser loop "blocks the whole server" (report §8) | Holds; checked body line 5 and the title, "gemma4 tool-call parser hangs forever on a stray "]" in an array argument" |

### [#48749 "[Bug] Checkpoint FP8 KV-cache scales are ignored: attention built without quant_config (GLM-4.5, Llama 4, Qwen3-MoE, GPT-OSS, HunYuan, HY3, K-EXAONE, LFM2)"](https://github.com/sgl-project/sglang/issues/48749)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-93 | Checkpoint KV scales are ignored for GPT-OSS, Qwen3-MoE and others, and "v0.5.23 has the same code" (report §8) | Holds; checked the title, the table rows at body lines 17 and 18 (Qwen3-MoE with Mellum, and GPT-OSS) and line 24 |

## FlashInfer, openai-harmony, NVIDIA and llama.cpp

### [FlashInfer v0.7.2 flashinfer/jit/cubin_loader.py](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2/flashinfer/jit/cubin_loader.py)

The image ships FlashInfer 0.7.2 (the hands-on lane's in-container check), so upstream's file at the tag stands for the image's copy; A read the image's copy offline and cited lines 221 to 226, which match.

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-94 | A missing kernel artifact is downloaded from "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/": "If the file is missing or its SHA-256 doesn't match, it is downloaded from" that repository (report §7) | Holds; checked lines 35 to 38 and the docstring of `get_artifact` at lines 207 to 214 |
| SG-95 | `FLASHINFER_NO_DOWNLOAD` makes a missing artifact raise instead of downloading it (report §7; A §M15; D4; LD15 revised) | Holds; checked lines 221 to 227 (`raise RuntimeError` with "refusing to download") and lines 229 to 232, the download after it |

### [openai-harmony v0.0.4 src/tiktoken_ext/public_encodings.rs](https://github.com/openai/harmony/blob/v0.0.4/src/tiktoken_ext/public_encodings.rs)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-96 | openai-harmony 0.0.4 fetches `o200k_base.tiktoken` from `https://openaipublic.blob.core.windows.net/encodings/` into a temporary-folder cache unless `TIKTOKEN_ENCODINGS_BASE` is set, and offline it fails with "error downloading or loading vocab file" (A, other findings; D4; LD15 revised) | Holds; checked `Cargo.toml` line 3 (`version = "0.0.4"`) and public_encodings.rs lines 29, 56 to 57, 124 to 137 (the variable's folder, else a download), 243 to 244 and 462 to 473 (`std::env::temp_dir()` joined with `tiktoken-rs-cache` unless `TIKTOKEN_RS_CACHE_DIR` is set), and A's offline record [harmony-check.txt](measure-a/logs/harmony-offline/harmony-check.txt), which shows that error without the variable and a 0.08 s load with it |

### [NVIDIA CUDA Toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-97 | NVIDIA lists "CUDA 13.2 R595" and "CUDA 13.0 R580", so driver 595.99 runs the CUDA 13.0.3 image without minor version compatibility (report §2.2; D16 for docs/host-setup.md) | Holds; checked Table 2 "CUDA Toolkit and Corresponding Driver Branch" (13.5 R620, 13.4 R615, 13.3 R610, 13.2 R595, 13.1 R590, 13.0 R580) and its note that CUDA 13.x applications run on drivers from 580 on under minor version compatibility |

### [llama.cpp b12407 src/llama-arch.cpp](https://github.com/ggml-org/llama.cpp/blob/b12407/src/llama-arch.cpp)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-107 | The pinned llama.cpp b12407 knows the `qwen35`, `gemma4` and `mellum` architectures (report §4 notes), inferred from source | Holds; checked lines 45, 65 and 174 |

## Hugging Face model records

Each size below is the sum of the `.safetensors` or the named `.gguf` file sizes in `https://huggingface.co/api/models/<repo>?blobs=true` (or `/revision/<sha>` for a pinned revision) on 2026-11-18, in GiB of 2^30 bytes (vendor-reported).
The SG-106 row reads a model card's README instead of the API record.

### The four llama.cpp GGUF picks: [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf), [google/gemma-4-E4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf), [JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF) and [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-98 | The four llama.cpp GGUF picks total 32,538,157,024 bytes, 30.3 GiB: [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf) 6.50 GiB at 29d0977, [google/gemma-4-E4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf) 4.80 GiB at 4b4a2c1, [JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF) Q6_K 10.13 GiB at 4159d68, [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) Q8_0 8.87 GiB at 3885219 (report §4, §7; D16 for docs/host-setup.md) | Holds; checked the four records at their full revisions: 6,975,879,296, 5,154,941,280, 10,879,834,400 and 9,527,502,048 bytes, which sum to the stated total |

### [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-99 | [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) is 21.82 GiB at 1355db6 (report §4) | Holds; checked the record: revision `1355db6a052410cfd62085d94b58866fd0f2c3c5`, 21.82 GiB of safetensors (21.85 GiB with the other files) |

### [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-100 | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) is 20.08 GiB at bee7596 (report §4) | Holds; checked the record: revision `bee7596271d1495f6992ae224aefde4410e816b8`, 20.08 GiB of safetensors (20.10 GiB with the other files) |

### The two Paged Experts candidates: [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) and [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-101 | The two Paged Experts candidates add 41.9 GiB of downloads (report §7; LD9; D16 for docs/host-setup.md) | Holds; checked 21.82 + 20.08 = 41.90 GiB of safetensors, 41.95 GiB with every file of both repositories |

### [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16) and [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-102 | Nemotron 3.5 Bolt has no public NVFP4 build (HTTP 401) and its BF16 checkpoint is about 138 GiB (report §4) | Holds with corrections: `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4` answers 401, but [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16) holds 149,774,331,136 bytes of safetensors, 139.49 GiB (149,795,738,013 bytes, 139.51 GiB, with every file); checked both records |

### [Qwen/Qwen3-Coder-Next-FP8](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8) and [nvidia/Qwen3-Coder-Next-NVFP4](https://huggingface.co/nvidia/Qwen3-Coder-Next-NVFP4)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-103 | Qwen3-Coder-Next has no public NVFP4 build (HTTP 401), and the official FP8 build is 74.86 GiB (report §4) | Holds; checked [Qwen/Qwen3-Coder-Next-FP8](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8), 74.86 GiB of safetensors, and that `nvidia/Qwen3-Coder-Next-NVFP4` answers 401; the report does not name the NVFP4 repository it asked for, so this check asked for that one |

### [IFM/K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-104 | K2 Horizon MoVA in BF16 is 69.75 GiB (report §4) | Holds; checked [IFM/K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B), 69.75 GiB of safetensors; the report names no repository, so this check used the publisher's |

### [Intel/gemma-4-26B-A4B-it-int4-AutoRound](https://huggingface.co/Intel/gemma-4-26B-A4B-it-int4-AutoRound) and [grevinden/qwen3.8-27b-w4a16-autoround-rtx3090](https://huggingface.co/grevinden/qwen3.8-27b-w4a16-autoround-rtx3090)

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-105 | The int4 file of Gemma 4 26B-A4B is 14.31 GiB, and the smallest Qwen3.8-27B W4A16 is 15.59 GiB (report §4) | Holds; checked [Intel/gemma-4-26B-A4B-it-int4-AutoRound](https://huggingface.co/Intel/gemma-4-26B-A4B-it-int4-AutoRound), 14.31 GiB of safetensors, and [grevinden/qwen3.8-27b-w4a16-autoround-rtx3090](https://huggingface.co/grevinden/qwen3.8-27b-w4a16-autoround-rtx3090), 15.59 GiB, the builds vllm/models.md names |

### The Mellum2.1 card, [JetBrains/Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking) README.md

| ID | Claim (where it is used) | Verdict |
|---|---|---|
| SG-106 | The Mellum2.1 card gives the vLLM pair `--reasoning-parser qwen3` and `--tool-call-parser hermes` and no SGLang command (report §6.1) | Holds; checked the card's README at [JetBrains/Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking), lines 243 to 250 |

## Not checked here

- Every figure measured on this laptop (start logs, traces, API shapes): the archive reviews of the hands-on, A and B records cover them, and these rows cite only the sources.
- The vLLM issue #66264, which the research report cites in section 2.5's flags table and section 4's runtime coverage, and OpenCode v2.3.0's `vllm.ts`, which the design and report section 5.2 cite for docs/opencode.md: they are vLLM and OpenCode sources that docs/sources.md already lists, outside the SGLang source list of D16.

## Left out

The fetched files stay in `$SCRATCH/sglang-sources-<random>/` and the parsing scripts in `$SCRATCH/sglang/sources/`: they are vendor code, vendor pages and API records, which the rows above cite by URL and line at their tags, so none of them enters the repository.
