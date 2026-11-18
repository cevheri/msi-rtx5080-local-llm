# Project instructions

This folder is the knowledge base, and later the Docker setup, for running open-weight LLMs on this laptop with llama.cpp, vLLM and SGLang and using them from OpenCode.
Read README.md first: it indexes the documents and lists the rules that apply to every setup.

## Facts to keep in mind

- GPU: RTX 5080 Laptop, 16 GB, sm_120; driver 595.99 offers CUDA 13.2, so containers built for CUDA 13.3 or newer may be refused or rely on minor-version compatibility.
- RAM: about 40 GiB is free while the desktop runs; size `--n-cpu-moe` and container memory caps from llamacpp/models.md.
- ufw is active and Docker-published ports bypass it, so publish on 127.0.0.1 only.
- The `heavy` wrapper does not reach containers, so every model container needs `--memory` with an equal `--memory-swap` (Compose `mem_limit` and `memswap_limit`).
- The NVIDIA Container Toolkit 1.20.1 is installed (2026-11-15): use `--device nvidia.com/gpu=all`; `--gpus all` and `--runtime=nvidia` also work.
- First measurements (2026-11-16, gpt-oss-20b): llama.cpp 166 tok/s at 128k context; vLLM started only in eager mode at 83.5 tok/s, because CUDA reports 15.47 GiB in total and vLLM's weights and reserve were larger than estimated.
  Treat the other vLLM context estimates as optimistic until measured.
- llama.cpp measurements (2026-11-16): Qwen3.8-27B 36 tok/s at 64k; Qwen3.6-35B-A3B 122 tok/s at 128k, but only with `-lm none` and `--moe-cache-mib` (67 tok/s without).
  llama.cpp's CUDA context and buffers took 0.7-0.8 GiB, so budget VRAM against 15,841 MiB minus about 112 MiB for the desktop.
  For an offloaded MoE model, give spare VRAM to `--moe-cache-mib` before lowering `--n-cpu-moe`.
  Measure q8_0 KV against f16 at depth: on Qwen3-Coder-Next (256-wide heads, GQA 8) q8_0 lost 40% of decode by a 101k prompt against 18% for f16, until llama.cpp PR #31802 lands.
- SGLang measurements (2026-11-17 and 2026-11-18): `lmsysorg/sglang:v0.5.23-runtime` (CUDA 13.0.3) serves Qwen3.5-9B, Gemma 4 12B, gpt-oss-20b and Qwen3.6-35B-A3B at 128k and K2 Horizon 7B and Gemma 4 E4B at 64k, the two MoE models gpt-oss-20b and Qwen3.6-35B-A3B only with Paged Experts; every SGLang recipe sets `--mem-fraction-static` (at most 0.90) and starts only with `--runtime sglang`, because llama.cpp and vLLM come first.
  Qwen3.6-35B-A3B decodes 151.3 tok/s on SGLang against 122.5 on llama.cpp, while Mellum2.1, Groot-19B-A3B, DeepSeek-R1-0528-Qwen3-8B, Nemotron 3.5 Lightning, Gemma 4 26B-A4B and K2 Horizon MoVA do not run on v0.5.23 (sglang/models.md).
  Paged Experts pins every expert in host memory, and the image's PyTorch rounds each pinned allocation above 1 MiB up to a power of two unless `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` (inferred from its source), which the runtime file sets (gpt-oss-20b's store fell from 19.19 to 10.37 GiB, measured).
- SGLang v0.5.23's first grammar-constrained request allocates NCCL memory (the fix, #48487, is unreleased), so its runtime file sets `SYNC_TOKEN_IDS_ACROSS_TP=1` and `NCCL_MAX_P2P_NCHANNELS=1`, and llmctl stops a start that leaves less free GPU memory than `min_free_at_ready_mib` (768 MiB) after ready.

## Working on the knowledge base

- Before relying on a version, image tag, issue state or score, re-verify it from the cited primary source and update the "Last verified" line of the document you touch.
- Keep the evidence labels: "(estimate)", "(vendor-reported)" and "(measured on <hardware>, <runtime>)".
- Record measurements from this laptop in the results tables of llamacpp/benchmarking.md, vllm/benchmarking.md and sglang/benchmarking.md, with the date, image tag, flags and power limit.
- A new source gets a review and an independent check before use, and an entry in docs/sources.md.
- Ranking changes go into docs/model-selection.md first; then update llamacpp/models.md, vllm/models.md, sglang/models.md and the OpenCode config in docs/opencode.md, and re-validate that config against https://opencode.ai/config.json.
- Never copy network details (IP addresses, MAC addresses, SSIDs, hostnames, user names) from the DGX Spark runbook that the documents cite.
- This folder is a public git repository (https://github.com/cevheri/msi-rtx5080-local-llm): commit with the GitHub noreply address set in the local git config, keep `.remember/` out of it, and replace local paths in new records with `$HOME` and `$SCRATCH` before a commit.

## Running things

- Run one model server at a time; host-side heavy commands still go through `heavy`.
- Start models with `./llmctl up <model>` ([docs/llmctl.md](docs/llmctl.md)); a change in `recipes/` goes with its document section and OpenCode entry in the same change, or `python3 -m unittest discover -s launcher/tests -t .` fails.
- Ollama runs as `ollama/ollama:0.40.16` with its models in the `ollama-models` volume; a fresh container compiles CUDA kernels for about 25 s before its first answer, so keep `/root/.nv` in a volume too (docs/opencode.md, Ollama notes).
- Anything that needs sudo is the user's to run: `sudo` through the `!` prefix has no terminal and fails, so hand over a script for a separate terminal (or `pkexec`), and never pipe `sudo apt-get` through `tee` (it stalled once).
- Ask the user before installs, Docker restarts or large image and model downloads.
