# Model selection

Last verified: 2026-11-18.

Use Qwen3.6-35B-A3B on llama.cpp as the default, the user's choice on 2026-11-16 after it measured 122 tok/s with 128k context.
Switch to Qwen3.8-27B on llama.cpp when a task needs the strongest model: it keeps rank 1 on quality evidence, at 36 tok/s and 64k context.
The new K2 Horizon models (MoVA 36B A4B and 7B) are the next two picks: both have independent agentic scores, but both cap out around 64k and always think.
All ten picks run on llama.cpp `server-cuda-b12407`; vLLM v0.33.0 is a secondary path with options for Groot and K2 Horizon 7B (DeepSeek-R1-0528-Qwen3-8B runs there only through llmctl's tokenizer overlay, at 72.5 tok/s with 64k (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18), [record](../research/2026-11-18/vllm-deepseek-r1-0528-qwen3-8b-cold-start/README.md), and llama.cpp stays its primary path), and on 2026-11-16 it ran gpt-oss-20b only in eager mode at half the llama.cpp speed (measured).
On 2026-11-17 vLLM ran Groot NVFP4 at 193 tok/s with 128k over runs of 780 to 829 tokens and K2 Horizon 7B AWQ-INT4 at 95 tok/s with 64k over full 1,500-token runs, the latter only at `--gpu-memory-utilization 0.94`, while DeepSeek-R1-0528-Qwen3-8B started but garbled every reply because of its tokenizer class; it now runs through llmctl's tokenizer overlay, measured 2026-11-18 ([vllm/models.md](../vllm/models.md#deepseek-r1-0528-qwen3-8b-chat-only)).
All ten ranks were measured on this laptop on 2026-11-16, all on llama.cpp: Qwen3.8-27B decodes at 36 tok/s with 64k context, K2 Horizon MoVA at 37 tok/s with 64k, Qwen3.6-35B-A3B at 122 tok/s and Gemma 4 26B-A4B at 148 tok/s with 128k once their RAM-resident experts are cached on the GPU, K2 Horizon 7B at 67 tok/s with 80k, Qwen3-Coder-Next at 72 tok/s with 128k (59 tok/s after a 101k prompt), Groot at 170 tok/s with 128k all in VRAM, Nemotron 3.5 Bolt at 74 tok/s with 128k, gpt-oss-20b at 166 tok/s, and the chat-only DeepSeek-R1-0528-Qwen3-8B at 71 tok/s.
Every other context and speed figure below is an estimate or a third-party number unless it carries a measured label, and each pick without one needs a smoke test before use.

## Decision summary

The ranking is the Borda aggregate of three judges (quality, practicality, coverage), with one forced swap so that the DeepSeek family is represented.
Ranks 1 to 9 are Borda ranks 1 to 9; ties keep the order of the aggregate.
Borda rank 10 was Nanbeige4.3-33B (3 points, quality judge only), which moves to the [honorable mentions](#honorable-mentions) so that DeepSeek-R1-0528-Qwen3-8B (1 point) can take rank 10.
That rank 10 is a chat and reasoning model only: it has no tool calling and is not for OpenCode agent mode.
The default model is a separate, practical choice and does not change the ranking: since 2026-11-16 it is rank 2, Qwen3.6-35B-A3B, which measured 3.4 times the decode speed of rank 1 with twice its context ([llamacpp/models.md](../llamacpp/models.md#2-qwen36-35b-a3b)).
The required families are covered: Qwen (ranks 1, 2, 6), Gemma (5), Nemotron (7), gpt-oss (9) and DeepSeek (10).
SGLang v0.5.23 is the third engine and changes neither the ranking nor the default: it serves six recipes, Qwen3.5-9B, K2 Horizon 7B, gpt-oss-20b, Gemma 4 12B, Gemma 4 E4B and Qwen3.6-35B-A3B, and one of them is materially faster than the fastest other recipe of the same model, the default Qwen3.6-35B-A3B, while the smoke-test model Gemma 4 E4B, which is not an agent pick, is faster on SGLang than on vLLM but slower than its llama.cpp GGUF build ([sglang/models.md](../sglang/models.md)).
Gemma 4 E4B decodes 141.3 tok/s on SGLang (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [record](../research/2026-11-17/sglang-gemma-4-e4b/README.md)) against 45.2 tok/s on vLLM in eager mode (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [record](../research/2026-11-17/vllm-gemma-4-e4b/README.md)), both with 64k context, and 151.3 tok/s with 128k on llama.cpp (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17; [record](../research/2026-11-17/llamacpp-gemma-4-e4b/README.md)).
Qwen3.6-35B-A3B decodes 151.3 tok/s on SGLang through Paged Experts (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18; [record](../research/2026-11-18/sglang-qwen3.6-35b-a3b/README.md)) against 122.5 tok/s on llama.cpp (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-16; [record](../research/2026-11-16/llamacpp-qwen3.6-35b-a3b/README.md)), 24% faster with `--runtime sglang`; the default stays the llama.cpp recipe by the owner's choice, and the ranking does not change.

Runtime rules that apply to every pick are documented in [llamacpp/README.md](../llamacpp/README.md) and [vllm/README.md](../vllm/README.md); the short version is below.

- Primary runtime: llama.cpp image `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407` (CUDA 12.8.1), with `-np 1` and `--jinja`.
- Every llama.cpp command on this laptop passes `docker -e GGML_CUDA_DISABLE_GRAPHS=1` before the image name, because the RTX 50 Laptop CUDA-graph hang ([llama.cpp #27330](https://github.com/ggml-org/llama.cpp/issues/27330)) reproduces on driver 595.99.02; placed after the image, llama-server reads `-e` as `--escape` and refuses to start.
- Avoid `server-cuda13-b12407`: its `NVIDIA_REQUIRE_CUDA` is `cuda>=13.5` while driver 595.99.02 offers CUDA 13.2, so the container is likely refused (inferred).
- `--n-cpu-moe N` moves the experts of blocks 0 to N-1 to system RAM and counts all blocks, not only MoE layers.
- vLLM v0.33.0 has a budget of 14.23 GiB at the default `--gpu-memory-utilization` 0.92 (0.92 of the 15.47 GiB that CUDA reports, measured; the earlier estimate was 14.65 GiB), and none of the four highest-ranked models fits it; vLLM expert offload is silently undone for Marlin weights ([vLLM #66264](https://github.com/vllm-project/vllm/issues/66264)).
- The NVIDIA Container Toolkit with CDI must be installed first ([host-setup.md](host-setup.md)), and OpenCode configuration lives in [opencode.md](opencode.md).

## Runtime coverage

Every model of the [README recipe table](../README.md#quick-start) has one cell per engine, and each cell starts with one of three words.
`Runs:` gives the shipped recipe's context, its decode speed and the date of that measurement (measured on the MSI Raider 18 Max HX with the engine of the column), a link to the recipe's section, and the result of the llmctl clients e2e of 2026-11-18 over every shipped recipe (measured on the MSI Raider 18 Max HX, llama.cpp b12407, vLLM v0.33.0 and SGLang v0.5.23, 2026-11-18; [summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [README](../research/2026-11-18/llmctl-sglang-e2e/README.md)).
A failed e2e session is named by its client step in the summary (`opencode`, `hermes-z`, `hermes-chat`, `pi`, `dsh-headless` or `kimi`), with its reason and the class that the e2e README's [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows) give it.
`Failed:` gives the error or failed check of a measured run, its date and its record, and `Does not fit:` gives the arithmetic of [vllm/models.md](../vllm/models.md#models-that-do-not-fit-in-vllm) or [sglang/models.md](../sglang/models.md#models-that-do-not-run-on-sglang), which is an estimate.
The vLLM cells keep the 14.65 GiB budget of the earlier estimate, and the measured 14.23 GiB of the runtime rules above only makes each shortfall larger.

The [model guide](https://cevheri.github.io/msi-rtx5080-local-llm/) reads this table: each `Runs:` cell gives a model page an engine with its recipe's facts and client results, and each `Failed:` or `Does not fit:` cell gives the reason under Why, its text up to the date or the `from [` of its source with the cell's first link, so such a cell needs a link to its record or its arithmetic.
Write a cell in plain text, code spans and links to files of this repository or to https pages, because the page build checks every cell whole and stops on any other Markdown.
It also stops on text that GitHub shows as plain text: `*`, `~`, `<`, `>`, a backslash, an HTML entity, a bare URL or e-mail address and an `_` at the edge of a word outside a code span, so put such text in a code span.
`python3 -m unittest discover -s launcher/tests -t .` checks every cell and names the row, engine and column of a stop, so run it before you push a change to this table.
The page shows a failed e2e session as a known issue when its cell names the session's step in a code span after the `e2e <date>:` of that session's date, and as Failed until then.

| Model ID | llama.cpp b12407 | vLLM v0.33.0 | SGLang v0.5.23 |
|---|---|---|---|
| `deepseek-r1-0528-qwen3-8b` | Runs: 64k, 70.6 tok/s, measured 2026-11-16 ([recipe](../llamacpp/models.md#10-deepseek-r1-0528-qwen3-8b-chat-and-reasoning-only)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) | Runs: 64k, 72.5 tok/s through llmctl's tokenizer overlay, measured 2026-11-18 ([recipe](../vllm/models.md#deepseek-r1-0528-qwen3-8b-chat-only)), with no OpenCode entry, because OpenCode v2.3.0 sends `tool_choice` `auto`, which vLLM refuses with HTTP 400 without a tool parser; e2e 2026-11-18: `opencode` failed with that HTTP 400 before the entry was removed, and every other session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) | Failed: out of GPU memory in the BF16 `lm_head`, because SGLang creates the BF16 weights before it quantizes them to FP8 (inferred from the source), measured 2026-11-17 ([record](../research/2026-11-17/sglang-deepseek-r1-0528-qwen3-8b/README.md)) |
| `gemma-4-12b` | Runs: 128k, 77.1 tok/s, measured 2026-11-17 ([recipe](../llamacpp/models.md#gemma-4-12b-it-qat-q4_0)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) | Runs: 128k, 69.5 tok/s, measured 2026-11-17 ([recipe](../vllm/models.md#gemma-4-12b-it-qat-w4a16)); e2e 2026-11-18: `dsh-headless` failed on goal calls without a goal (`GOAL_NOT_FOUND`), model behaviour in a session that passed on 2026-11-17 ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) | Runs: 128k, 70.4 tok/s, measured 2026-11-17 ([recipe](../sglang/models.md#gemma-4-12b-it-qat-w4a16)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) |
| `gemma-4-26b-a4b` | Runs: 128k, 148.2 tok/s, measured 2026-11-16 ([recipe](../llamacpp/models.md#5-gemma-4-26b-a4b-it)); e2e 2026-11-18: `dsh-headless` failed on a `justification` argument to `write` that dsh refused, as on 2026-11-17 ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) | Does not fit: the smallest checkpoint holds 13.24 GiB of language model (14.31 GiB file), which leaves a KV pool of about 0.21 GiB, about 11.8k tokens with FP8 KV, in the 14.65 GiB budget (estimate), from [vllm/models.md](../vllm/models.md#models-that-do-not-fit-in-vllm) | Failed: the Marlin MoE runner, the only runner that pages its int4 experts, supports only gated SiLU and stopped the decode-graph capture on Gemma's GELU, measured 2026-11-18 ([record](../research/2026-11-18/sglang-gemma-4-26b-a4b/README.md)) |
| `gemma-4-e4b` | Runs: 128k, 151.3 tok/s, measured 2026-11-17 ([recipe](../llamacpp/models.md#gemma-4-e4b-it-qat-q4_0)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) | Runs: 64k, 45.2 tok/s, measured 2026-11-17 ([recipe](../vllm/models.md#smoke-test-model-gemma-4-e4b-it)); e2e 2026-11-18: `dsh-headless` failed on `write` calls with a `justification` argument and reads before the write, as on 2026-11-17 ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) | Runs: 64k, 141.3 tok/s, measured 2026-11-17 ([recipe](../sglang/models.md#gemma-4-e4b-it-qat-w4a16)); e2e 2026-11-18: `dsh-headless` failed on `write` calls with a `justification` argument, the same model's failure on its new SGLang recipe ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) |
| `gpt-oss-20b` | Runs: 128k, 166.2 tok/s, measured 2026-11-16 ([recipe](../llamacpp/models.md#9-gpt-oss-20b)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) | Runs: 64k, 83.5 tok/s, measured 2026-11-16 ([recipe](../vllm/models.md#gpt-oss-20b)); e2e 2026-11-18: `hermes-z` left no `hello.py` in the work folder, model behaviour in a session that passed on 2026-11-17 ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) | Runs: 128k, 124.6 tok/s, measured 2026-11-17 ([recipe](../sglang/models.md#gpt-oss-20b)); e2e 2026-11-18: `opencode` ended after an empty turn without the read, which the lead reproduced and which is now a [known issue](../sglang/README.md#known-issues) of SGLang's `gpt-oss` tool parser ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) |
| `groot-19b-a3b` | Runs: 128k, 169.8 tok/s, measured 2026-11-16 ([recipe](../llamacpp/models.md#8-groot-19b-a3b-261023)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) | Runs: 128k, 193.4 tok/s, measured 2026-11-17 ([recipe](../vllm/models.md#groot-19b-a3b-261023-nvfp4)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) | Failed: an `AttributeError` on `group_size` while preparing the NVFP4 linears for the FP4 Marlin kernel, after the weights loaded, measured 2026-11-17 ([record](../research/2026-11-17/sglang-groot-19b-a3b/README.md)) |
| `k2-horizon-7b` | Runs: 80k, 66.9 tok/s, measured 2026-11-16 ([recipe](../llamacpp/models.md#4-k2-horizon-7b)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) | Runs: 64k, 94.8 tok/s, measured 2026-11-17 ([recipe](../vllm/models.md#k2-horizon-7b-awq-int4)); e2e 2026-11-18: `hermes-z` (a refused overwrite outside the work folder), `hermes-chat` and `pi` (HTTP 400 `Assistant message is missing a thinking field`) failed as on 2026-11-17, and `dsh-headless` on the same HTTP 400, model behaviour in a session that passed on 2026-11-17 ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) | Runs: 64k, 89.6 tok/s, measured 2026-11-18 ([recipe](../sglang/models.md#k2-horizon-7b-awq-int4)); e2e 2026-11-18: `hermes-z` (a refused overwrite outside the work folder) and `hermes-chat` (HTTP 400 for a missing thinking field) failed, the same model's failures on its new SGLang recipe ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) |
| `k2-horizon-mova-36b` | Runs: 64k, 37.2 tok/s, measured 2026-11-16 ([recipe](../llamacpp/models.md#3-k2-horizon-mova-36b-a4b)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) | Does not fit: the smallest checkpoint, a GPTQ-Int4 build, is 20.25 GiB against the 14.65 GiB budget (estimate), from [vllm/models.md](../vllm/models.md#models-that-do-not-fit-in-vllm) | Failed: SGLang refused the AWQ-INT4 build before loading, because it supports only unquantized MoVA weights, and the BF16 model is 69.75 GiB (estimate), measured 2026-11-18 ([record](../research/2026-11-18/sglang-k2-horizon-mova-36b/README.md)) |
| `mellum2.1-12b` | Runs: 128k, 188.2 tok/s, measured 2026-11-17 ([recipe](../llamacpp/models.md#mellum21-12b-a25b-thinking-q6_k)); e2e 2026-11-18: `kimi` retried one request after a thinking-only reply, a new recipe's model behaviour ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) | Runs: 128k, 257.2 tok/s, measured 2026-11-17 ([recipe](../vllm/models.md#mellum21-12b-a25b-thinking-awq-int4)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) | Failed: a `RuntimeError` on tensor sizes 288 against 28 two seconds into loading, because SGLang's WNA16 MoE loader expects the zero points transposed (inferred from the source), measured 2026-11-17 ([record](../research/2026-11-17/sglang-mellum2.1-12b/README.md)) |
| `nemotron-3.5-bolt` | Runs: 128k, 73.5 tok/s, measured 2026-11-16 ([recipe](../llamacpp/models.md#7-nvidia-nemotron-35-bolt-74b-a9b)); e2e 2026-11-18: `pi` failed on reads before the write, as on 2026-11-17, and `kimi` answered without a `Read` call, model behaviour in a session that passed on 2026-11-17 ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) | Does not fit: the 24GB build is 22.43 GiB against the 14.65 GiB budget, and its IQ2_XXS and Q8_0 blocks need b12x kernels that the image does not install (estimate), from [vllm/models.md](../vllm/models.md#models-that-do-not-fit-in-vllm) | Does not fit: no public NVFP4 build answers, and the BF16 checkpoint is 139.5 GiB, far above 16 GB of VRAM plus the about 40 GiB of free RAM (estimate), from [sglang/models.md](../sglang/models.md#models-that-do-not-run-on-sglang) |
| `nemotron-3.5-lightning` | Runs: 128k, 143.6 tok/s, measured 2026-11-16 ([recipe](../llamacpp/models.md#nemotron-35-lightning-30b-a3b-fallback-if-bolt-does-not-fit)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) | Does not fit: the NVFP4 build is 20.08 GiB and the community W4A16 build 16.60 GiB, both over the 14.65 GiB budget (estimate), from [vllm/models.md](../vllm/models.md#models-that-do-not-fit-in-vllm) | Failed: the NVFP4 build loaded and held 128k at 133.3 tok/s, but no request with tools produced a tool call in any thinking mode, and 20 fixed prompts scored 10 of 20, measured 2026-11-18 ([record](../research/2026-11-18/sglang-nemotron-3.5-lightning/README.md)) |
| `qwen3-coder-next` | Runs: 128k, 71.5 tok/s, measured 2026-11-16 ([recipe](../llamacpp/models.md#6-qwen3-coder-next)); e2e 2026-11-18: `hermes-z` left no `hello.py` in the work folder, model behaviour in a session that passed on 2026-11-17 ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) | Does not fit: the smallest checkpoint is 40.53 GiB, far over the 14.65 GiB budget (estimate), from [vllm/models.md](../vllm/models.md#models-that-do-not-fit-in-vllm) | Does not fit: no public NVFP4 build answers, the official FP8 build is 74.86 GiB, and an int4 build would pin about 38 GiB of experts in host memory against the about 40 GiB of free RAM (estimate), and Intel's AutoRound format is not among the formats Paged Experts accepts (inferred), from [sglang/models.md](../sglang/models.md#models-that-do-not-run-on-sglang) |
| `qwen3.5-9b` | Runs: 128k, 66 tok/s, measured 2026-11-17 ([recipe](../llamacpp/models.md#qwen35-9b-q8_0)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) | Runs: 128k, 90.8 tok/s, measured 2026-11-17 ([recipe](../vllm/models.md#qwen35-9b-nvfp4)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) | Runs: 128k, 93.7 tok/s, measured 2026-11-17 ([recipe](../sglang/models.md#qwen35-9b-nvfp4)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) |
| `qwen3.6-35b-a3b` | Runs: 128k, 122.5 tok/s, measured 2026-11-16 ([recipe](../llamacpp/models.md#2-qwen36-35b-a3b)); e2e 2026-11-18: `hermes-z` and `hermes-chat` aimed `hello.py` outside the work folder, as on 2026-11-17 ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) | Does not fit: the smallest checkpoint holds 18.75 GiB of language model (20.02 GiB file), over the 14.65 GiB budget, and vLLM undoes expert offload for Marlin weights (estimate), from [vllm/models.md](../vllm/models.md#models-that-do-not-fit-in-vllm) | Runs: 128k, 151.3 tok/s through Paged Experts, measured 2026-11-18 ([recipe](../sglang/models.md#qwen36-35b-a3b-nvfp4)); e2e 2026-11-18: `hermes-z` wrote `hello.py` to `/tmp`, the same model's failure on its new SGLang recipe ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)) |
| `qwen3.8-27b` | Runs: 64k, 36.4 tok/s, measured 2026-11-16 ([recipe](../llamacpp/models.md#1-qwen38-27b)); e2e 2026-11-18: every session passed ([summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md)) | Does not fit: the smallest checkpoint is 15.59 GiB, so the weights alone exceed the 14.65 GiB budget (estimate), from [vllm/models.md](../vllm/models.md#models-that-do-not-fit-in-vllm) | Does not fit: the model is dense, so Paged Experts does not apply (inferred), and the smallest W4A16 build, 15.59 GiB, exceeds with its vision tower the 13.55 GiB budget at fraction 0.90 (estimate), from [sglang/models.md](../sglang/models.md#models-that-do-not-run-on-sglang) |

## How the models were selected

The selection came from three research runs on 2026-11-15.

1. Discovery and verification: thirteen discovery lanes (model families, registry and leaderboard sweeps, platform lanes and a completeness critic) produced the candidate list.
   Each of 22 candidates then got a verification record and two adversarial checks (a facts check and a feasibility check against this machine).
   Three judges ranked the verified set independently, the ranks were aggregated with Borda (a judge's rank r gives 11 - r points), and an audit reopened more than 40 references and forced the fixes listed under [changes](#changes-versus-the-earlier-ranking).
2. Source review: the 19 sources the user supplied were each reviewed and then independently re-checked; the corrections are recorded in [sources.md](sources.md) and the third-party numbers in [community-benchmarks.md](community-benchmarks.md).
3. New candidates: nine models surfaced by the source review and by new releases were verified the same way, the judge panel ranked the combined set again with Borda, and the load-bearing GGUF sizes of the new picks were rechecked live through the Hugging Face tree API.

Evidence labels follow [sources.md](sources.md): "independent" means a third party ran the benchmark, "vendor-reported" means the publisher scored its own model, and "(estimate)" means arithmetic from config files, real file sizes and the 896 GB/s spec bandwidth.

### Evidence from the user's agent harness

The user's LibreDB Studio agent harness is the first first-hand tool-reliability evidence for these picks: a model passes only when each of six SQLite tool-calling surfaces is answered five consecutive times, 30/30 in all ([agent-harness-evidence.md](agent-harness-evidence.md)).
It ran on Ollama 0.32.13 and 0.33.0 on an Apple M5 Max (measured on the M5 Max, libredb-studio), so its pass/fail transfers to this laptop and its durations do not.
It confirms tool reliability for rank 1 only as the Ollama Q4_K_M build (`qwen3.8:latest`), not the llama.cpp UD-Q3_K_XL path recommended here.
It confirms rank 2 only with thinking off on agent turns, which supports the `--reasoning off` server default.
It confirms ranks 5 and 9; rank 5 needed an empty-turn retry and rank 9 passed with reasoning suppressed on agent turns.
It covers none of ranks 3, 4, 6, 7, 8 and 10.
It does not test code edits, coding quality or 64k-scale contexts, so it does not reorder the top 10.

## Top 10

Context is the safe target on this laptop (15,841 MiB of the 16,303 MiB are usable by CUDA, about 40 GiB of RAM is free), not the model's native window; it is measured for every rank except 6 and estimated for rank 6.
Sizes are GiB from the Hugging Face API.
Full commands are in [llamacpp/models.md](../llamacpp/models.md) and, for the vLLM paths, [vllm/models.md](../vllm/models.md).

| Rank | Model | Vendor | Release | Params | Runtime | Checkpoint | Size (GiB) | Context (estimate) | Thinking | Tools | License |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Qwen (Alibaba) | 2026-08-14 | 27.8B dense | llama.cpp, all on GPU | [unsloth UD-Q3_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | 12.24 | 64k, q8_0 KV | Toggle, effort xhigh/medium/low | Yes | Apache-2.0 |
| 2 | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Qwen (Alibaba) | 2026-04-16 | 35B / 3B active MoE | llama.cpp, `--n-cpu-moe 25` plus a 3,200 MiB expert cache | [ggml-org Q4_K_M](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF) | 19.13 | 128k, f16 KV | Toggle | Yes | Apache-2.0 |
| 3 | [K2 Horizon MoVA 36B A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B) | IFM | 2026-09-03 | 37.44B / about 4B active MoE | llama.cpp, `--n-cpu-moe 48` plus a 1,536 MiB expert cache | [bartowski Q4_K_M](https://huggingface.co/bartowski/K2-Horizon-MoVA-36B-A4B-GGUF) | 21.14 | 64k, q8_0 KV | Always on, effort high | Yes | Apache-2.0 |
| 4 | [K2 Horizon 7B](https://huggingface.co/IFM/K2-Horizon-7B) | IFM | 2026-09-03 | 9.0B dense | llama.cpp, all on GPU | [bartowski Q8_0](https://huggingface.co/bartowski/K2-Horizon-7B-GGUF) | 8.92 | 80k, q8_0 KV (measured) | Always on, effort high | Yes | Apache-2.0 |
| 5 | [Gemma 4 26B-A4B IT](https://huggingface.co/google/gemma-4-26B-A4B-it) | Google DeepMind | 2026-04-02 | 25.2B / 3.8B active MoE | llama.cpp, `--n-cpu-moe 12` plus a 2,368 MiB expert cache | [Google QAT Q4_0](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf) | 13.45 | 128k, f16 KV (measured) | Toggle, off by default | Yes | Apache-2.0 |
| 6 | [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | Qwen (Alibaba) | 2026-02-03 | 79.7B / 3B active MoE | llama.cpp, `--n-cpu-moe 38` plus a 1,504 MiB expert cache | [unsloth UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) | 35.79 | 128k, f16 KV (measured) | No | Yes | Apache-2.0 |
| 7 | [Nemotron 3.5 Bolt 74B-A9B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) | NVIDIA | 2026-10-15 | 74B / 9B active hybrid MoE | llama.cpp, `--n-cpu-moe 60` plus a 2,816 MiB expert cache | NVIDIA 24GB GGUF | 20.51 | 128k, f16 KV (measured) | Toggle, medium effort option | Yes | OpenMDW-1.1 |
| 8 | [Groot-19B-A3B-261023](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023) | GrootTeam (anonymous) | 2026-10-23 | About 19B / 3B active MoE | llama.cpp, all on GPU; vLLM NVFP4 option | [GrootTeam Q4_K_M](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023-GGUF) | 10.86 | 128k, f16 KV | Toggle | Yes | Apache-2.0 in metadata, no LICENSE file |
| 9 | [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | OpenAI | 2025-08-05 | 21B / 3.6B active MoE | llama.cpp, all on GPU; vLLM at 64k | [ggml-org MXFP4](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) | 11.28 | 128k llama.cpp, 64k vLLM | Always on, effort low/medium/high | Yes | Apache-2.0 |
| 10 | [DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) | DeepSeek | 2025-05-29 | 8.19B dense | llama.cpp, all on GPU | [unsloth Q8_0](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF) | 8.11 | 64k, q8_0 KV | Always on | No (chat and reasoning only) | MIT |

## Per-model notes

Speed bounds below are memory-bandwidth arithmetic (bytes read per token divided by bandwidth) and are labelled inferred; they are upper bounds, not measurements.
The fit estimates were made against 15.92 GiB with 1.4-1.9 GiB assumed for the CUDA context and compute buffers.
Measured on 2026-11-16, CUDA can use 15.47 GiB, the desktop holds about 0.11 GiB, and llama.cpp needed only 0.7-0.8 GiB of context and buffers, so for llama.cpp the two corrections roughly cancel ([llamacpp/models.md](../llamacpp/models.md#at-a-glance)).

### 1. Qwen3.8-27B

Why: Borda 1 with 29 points (quality 1, practicality 2, coverage 1).
It has the strongest independent agentic scores of anything that fits: [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-27b) measured Terminal-Bench 2.1 at 79.8 (xhigh), 65.2 (medium), 67.4 (low) and 49.1 (non-reasoning), with tau3-Banking 48.0 and Intelligence Index 20.1.
The [model card](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/README.md) reports TB2.1 73.0 and SWE-bench Pro 61.7 (vendor-reported).
It is dense and fully in VRAM: measured on 2026-11-16 at 64k with q8_0 KV, it peaked at 14,790 MiB including the desktop, about 1 GiB below the 15,841 MiB that CUDA can use; up to about 88k is untested.
Decode measured 36.4 tok/s at a short context, 34.2 after 8.6k and 29.6 after 32k, against a bound of about 896/13 GB, roughly 65 tok/s ([llamacpp/models.md](../llamacpp/models.md#1-qwen38-27b)).
Thinking: on by default, `reasoning_effort` xhigh (default), medium or low per the [chat template](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja); for OpenCode use medium, through `--reasoning-effort medium` or a variant body, plus a variant with `enable_thinking` false.
Sampling: thinking temperature 1.0, top_p 0.95, top_k 20; non-thinking temperature 0.7, top_p 0.8, top_k 20, presence 1.5 ([card](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/README.md)).
Caveats: AA measured BF16 through the API, so the 3-bit quality loss is unmeasured, and AA TB 4.0 is only 5.6.
The q8_0 KV decode penalty ([#27623](https://github.com/ggml-org/llama.cpp/issues/27623), reported in a comment; the cliff in the issue title was retracted; fix [PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802) open) may slow decode; below 15 tok/s, switch to f16 KV at a shorter context.
Output is verbose (AA counts 86-110k output tokens per task at medium and xhigh).
Hybrid prefix-cache misses ([#30357](https://github.com/ggml-org/llama.cpp/issues/30357)) and tool calls leaking into content ([#30118](https://github.com/ggml-org/llama.cpp/issues/30118)) apply.
vLLM does not fit: the smallest build (AutoRound, 15.59 GiB) exceeds the 14.23 GiB budget, and [Ollama `qwen3.8:27b`](https://ollama.com/library/qwen3.8/tags) (18 GB) spills to the CPU.
Evidence: independent (Artificial Analysis, BF16 via API) plus vendor; memory and speed measured on this laptop.

### 2. Qwen3.6-35B-A3B

Why: Borda 2 with 27 points (quality 3, practicality 1, coverage 2).
It has the best-corroborated independent evidence: NVIDIA's same-harness table in the [Nemotron 3.5 Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/README.md) gives SWE-bench Verified 70.12 and TB2.1 44.38, which agrees with [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-6-35b-a3b) TB2.1 44.9 within one point; AA tau2 is 95.3.
Qwen evaluated SkillsBench through OpenCode ([card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/README.md)).
It is the only top pick that reaches 128k with a cheap KV cache: 20 KiB/token f16, 2.5 GiB at 128k (estimate from [config.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json)).
Measured on 2026-11-16: with `--n-cpu-moe 25`, `-lm none` and a 3,200 MiB expert cache, 10.5 GiB of experts sit in pinned RAM and the GPU peaked at 15,190 MiB at 128k; 262144 would need q8_0 KV (estimate).
It has a measured anchor on other hardware: 42.7 tok/s single stream for the RedHatAI NVFP4 build (measured on the user's DGX Spark, GB10 with 273 GB/s unified memory, vLLM, about 280-token completions; private runbook, no public link).
Decode measured 122.5 tok/s at a short context and 92.6 tok/s after a 107k prompt, with 1,990 tok/s prefill at 8.6k; the recipe without the expert cache gave 67.3 tok/s, so the 40-90 tok/s range inferred earlier was low ([llamacpp/models.md](../llamacpp/models.md#2-qwen36-35b-a3b)).
Thinking: toggle, on by default, `chat_template_kwargs` `{"enable_thinking": false}` per request.
Sampling: precise coding temperature 0.6, top_p 0.95, top_k 20, presence 0; general thinking temperature 1.0, presence 1.5 ([card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/README.md)).
Caveats: AA tau3-Banking is only 9.3; [#30118](https://github.com/ggml-org/llama.cpp/issues/30118) and [#30357](https://github.com/ggml-org/llama.cpp/issues/30357) apply, although neither showed up in the measured OpenCode session; thinking is verbose.
vLLM is not viable: the smallest vLLM checkpoint (Intel AutoRound int4, 18.75 GiB language-model only) exceeds the budget and [#66264](https://github.com/vllm-project/vllm/issues/66264) undoes expert offload.
Fallback: Ollama `qwen3.6:35b-a3b` with `OLLAMA_CONTEXT_LENGTH=65536`.
Evidence: independent (NVIDIA same-harness and Artificial Analysis agree) plus vendor; memory and speed measured on this laptop.

### 3. K2 Horizon MoVA 36B A4B

Why: Borda 3 with 24 points (quality 2, practicality 4, coverage 3).
It has the second-best independent agentic scores among models that fit: [Artificial Analysis](https://artificialanalysis.ai/models/k2-horizon-mova-36b-a4b) TB2.1 56.6, tau3-Banking 32.0 and AutomationBench 26.1, against 44.9 and 9.3 for Qwen3.6.
llama.cpp b12407 supports it natively: the `k2-horizon` arch is in [llama-arch.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/src/llama-arch.cpp) with a dedicated tool parser in [common/parsers/k2-horizon.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/common/parsers/k2-horizon.cpp).
Memory (estimate from the GGUF header): the KV cache is heavy at 192 KiB/token f16, about 6.375 GiB at 64k in q8_0.
`--n-cpu-moe` matches only `ffn_*_exps`, so the 3.96 GiB of `attn_v_exps` stays on the GPU.
Measured on 2026-11-16 with every routed expert in pinned RAM (`--n-cpu-moe 48`) and a 1,536 MiB expert cache at 64k with q8_0 KV: peak 14,848 MiB, container RAM 19.7 GiB, decode 37.2 tok/s at a short context and 26.2 after 30k, prefill about 1,000-1,200 tok/s, and the tool test and an OpenCode session passed ([llamacpp/models.md](../llamacpp/models.md#3-k2-horizon-mova-36b-a4b)).
At high effort it spent a 3,000-token budget on reasoning without answering a short coding request, as K2 Horizon 7B did.
Thinking: always on; set `reasoning_effort` high through `chat_template_kwargs`, as the [card](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/raw/main/README.md) recommends (medium and low are not recommended there).
Sampling: temperature 1.0, top_p 0.95 ([card](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/raw/main/README.md)).
Caveats: context is capped at 64k; always-on high-effort reasoning at temperature 1.0 makes turns long, at a decode speed close to Qwen3.8-27B's; llama.cpp support is about six weeks old; the q8_0 KV decode penalty is unmeasured; AA TB 4.0 is 0; the IFM org is not HF-verified; the strict template rejects malformed tool schemas (a `required` name missing from `properties`), so check MCP tools.
The official IFM Q4_K_M (20.83 GiB) is a possible thinking-off option, unverified.
vLLM does not fit: the smallest build (GPTQ-Int4) is 20.25 GiB ([vLLM v0.33.0 recipe](https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/recipes/models/IFM/K2-Horizon-MoVA-36B-A4B.yaml)), and the [Ollama `ifm/k2-horizon:36b` tag](https://ollama.com/ifm/k2-horizon/tags) in the vendor namespace (22 GB) is not recommended.
Evidence: independent (Artificial Analysis) plus vendor; memory and speed measured on this laptop.

### 4. K2 Horizon 7B

Why: Borda 4 with 20 points (quality 4, practicality 3, coverage 6).
It is the most practical new candidate with independent evidence: [Artificial Analysis](https://artificialanalysis.ai/models/k2-horizon-7b) TB2.1 44.2 is level with Qwen3.6's 44.9, and tau3-Banking 27.6 is far above Qwen3.6's 9.3.
Measured on 2026-11-16 it is dense and fully in VRAM at 81,920 with q8_0 KV (15,008 MiB peak), because `token_embd` stays in host memory; f16 KV fits only to about 40k.
It decoded at 67 tok/s at a short context, 40 after 30k and 28 after 61k, prefilled 4,333 tok/s at 8k, and passed the tool test and an OpenCode session ([llamacpp/models.md](../llamacpp/models.md#4-k2-horizon-7b)).
At high effort it is very verbose: a short coding request spent a 3,000-token budget on reasoning without an answer, so give it the card's 32,768 output tokens.
The bartowski Q8_0 was made on 2026-10-07, after the anti-doom-loop (FTPO) checkpoint was promoted on 2026-10-01, so it likely carries the released weights (inferred).
Thinking: always on; the [card](https://huggingface.co/IFM/K2-Horizon-7B/raw/main/README.md) recommends `reasoning_effort` high and at least 32,768 output tokens.
Sampling: temperature 1.0, top_p 0.95 ([card](https://huggingface.co/IFM/K2-Horizon-7B/raw/main/README.md)).
Caveats: it only matches Qwen3.6 on TB2.1, and Qwen3.6 leads on SciCode (36.6 vs 27.5), LCR (71.7 vs 69.7) and HLE (22.2 vs 18.2) on AA.
Output is long and context is 80k, not 128k.
The vendor's SWE-bench Verified 72.4 is not confirmed independently, the IFM org is not HF-verified, and runtime support is new.
vLLM is a weak alternative: `cyankiwi/K2-Horizon-7B-AWQ-INT4` with FP8 KV and `TRITON_ATTN` reached 65536 only at `--gpu-memory-utilization 0.94`, with 95,168 KV tokens and 94.8 tok/s over full 1,500-token runs (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17), but those are pre-FTPO weights (inferred) and [vLLM #62743](https://github.com/vllm-project/vllm/issues/62743) breaks medium and low effort; support starts at v0.30.0 ([registry at v0.30.0](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/model_executor/models/registry.py), [recipe](https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/recipes/models/IFM/K2-Horizon-7B.yaml)).
Evidence: independent (Artificial Analysis) plus vendor; memory and speed measured on this laptop.

### 5. Gemma 4 26B-A4B IT

Why: Borda 5 with 16 points (quality 7, practicality 5, coverage 5).
It is the best Gemma for this machine, and two independent sources agree: NVIDIA same-harness SWE-bench Verified 57.40 and TB2.1 37.22 ([Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/README.md)) and [AA](https://artificialanalysis.ai/models/gemma-4-26b-a4b) TB2.1 39.0.
The official QAT Q4_0 build ([13.45 GiB](https://huggingface.co/api/models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf?blobs=true)) is non-thinking by default.
On a third-party agentic SQL tool loop it scored 21/25 ([measured via OpenRouter, not local](https://www.reddit.com/r/LocalLLaMA/comments/1s7r9wu/i_tested_as_many_of_the_small_local_and/)).
Measured on 2026-11-16: it does not fit fully in VRAM, because llama.cpp keeps a second copy of the 577.5 MiB embedding on the GPU as the output head, so it runs with 12 of 30 blocks' experts in pinned RAM and a 2,368 MiB expert cache, at 128k with f16 KV.
It decoded at 148 tok/s at a short context and 95 tok/s after a 124k prompt, prefilled 3,716 tok/s, peaked at 15,064 MiB and passed the tool test and an OpenCode session ([llamacpp/models.md](../llamacpp/models.md#5-gemma-4-26b-a4b-it)).
Thinking: off by default; `chat_template_kwargs` `{"enable_thinking": true}` turns it on.
Keep the 2026-07-09 [chat template](https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/chat_template.jinja).
Sampling: temperature 1.0, top_p 0.95, top_k 64 ([card](https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/README.md)).
Caveats: tool reliability is the weak point.
[llama.cpp #29655](https://github.com/ggml-org/llama.cpp/issues/29655) reports malformed multi-line write and edit arguments when streaming, so test OpenCode's edit tool first.
AA tau2 43.6 and tau3 12.0 are low, and results are harness-sensitive: Qwen's scaffold gives SWE-bench Verified 17.4 ([Qwen3.6 card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/README.md)) and KAT's Claude Code harness 35.8 ([KAT card](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev/raw/main/README.md)).
The Ollama default `gemma4:26b` (17 GB) does not fit; use the [`26b-a4b-it-qat` tag](https://registry.ollama.ai/v2/library/gemma4/manifests/26b-a4b-it-qat).
vLLM does not fit usefully: at 0.92 the Intel AutoRound build leaves about 11.8k context.
Evidence: independent (NVIDIA same-harness and Artificial Analysis), contested by other harnesses; memory and speed measured on this laptop.

### 6. Qwen3-Coder-Next

Why: Borda 6 with 15 points (quality 5, practicality 6, coverage 7).
It is the best non-thinking tool caller, so OpenCode turns stay short and it avoids [OpenCode #55378](https://github.com/anomalyco/opencode/issues/55378) (reasoning-only turns that end silently).
Independent [AA](https://artificialanalysis.ai/models/qwen3-coder-next) results are tau2 79.5 and TB2.1 38.2; the vendor reports SWE-bench Verified 70.6-71.3 ([card](https://huggingface.co/Qwen/Qwen3-Coder-Next/raw/main/README.md), [tech report](https://raw.githubusercontent.com/QwenLM/Qwen3-Coder/main/qwen3_coder_next_tech_report.pdf)).
Measured on 2026-11-16 ([llamacpp/models.md](../llamacpp/models.md#6-qwen3-coder-next)): with `--n-cpu-moe 38`, 26.4 GiB of experts in pinned RAM, f16 KV and a 1,504 MiB expert cache it ran at 131072 context, decoding 71.5 tok/s at a short context and 58.6 tok/s after a 101k prompt, with prefill at about 1,000-1,150 tok/s, a 15,076 MiB GPU peak and a 31.9 GiB container under a 35g cap.
The tool test and an OpenCode write-and-read session passed.
q8_0 KV with a larger cache decoded 13% faster at a short context but 17% slower after 101k, so it suits only sessions that stay under about 30k tokens.
Sampling: temperature 1.0, top_p 0.95, top_k 40 ([card](https://huggingface.co/Qwen/Qwen3-Coder-Next/raw/main/README.md)).
Caveats: its container takes about 32 of the roughly 40 GiB of free RAM, so nothing large can run beside it, and the q8_0 depth penalty ([PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802), open) forces the trade above.
Community SQL tool-loop tests scored it 16/25 ([via OpenRouter](https://www.reddit.com/r/LocalLLaMA/comments/1s7r9wu/i_tested_as_many_of_the_small_local_and/)) and 20/25 ([measured on a desktop RTX 5080 with 96 GB RAM, UD-Q5_K_XL](https://www.reddit.com/r/LocalLLaMA/comments/1s9mkm1/benchmarked_18_models_that_i_can_run_on_my_rtx/)); AA TB 4.0 is 0.
UD-IQ4_XS holds about 20.6 GiB of IQ3_S expert tensors ([llamacpp/models.md](../llamacpp/models.md)); [#28581](https://github.com/ggml-org/llama.cpp/issues/28581) affected nvcc 13.2 builds, not the CUDA 12.8.1 image, and the answers on this image were coherent (measured).
vLLM does not fit (the smallest 4-bit is 40.5 GiB), and [Ollama `q4_K_M`](https://ollama.com/library/qwen3-coder-next/tags) (52 GB) is too large.
Evidence: independent (Artificial Analysis) plus vendor; memory and speed measured on this laptop.

### 7. NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB GGUF)

Why: Borda 7 with 15 points (quality 6, practicality 8, coverage 4); it replaces Nemotron 3.5 Lightning as the Nemotron pick.
NVIDIA's same-harness table in the [Bolt card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/raw/main/README.md) gives TB2.1 50.98, against 44.38 for Qwen3.6 and 23.46 for Lightning, plus MCP-Atlas 52.2 and tau3 17.0 (vendor-run); that table's Qwen3.6 TB2.1 (44.38) is about half a point from AA's 44.9.
The single GGUF mixes IQ2_XXS QAT experts, NVFP4 Mamba and Q8_0 for the rest: 20.51 GiB plus a 1.61 GiB mmproj that is not used ([HF tree](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF)).
The KV cache is tiny (8 KiB/token f16, 1.00 GiB at 128k) and the Mamba state is about 162 MiB per sequence, so 128k is cheap.
Measured on 2026-11-16 at 128k with f16 KV, `--n-cpu-moe 60`, `-lm none` and a 2,816 MiB expert cache: peak 15,064 MiB, container RAM 17.8 GiB, decode 73.5 tok/s at a short context and 65.8 after a 104k prompt, prefill about 1,400-1,600 tok/s, and the tool test and an OpenCode session passed after one recovered parallel-call race ([llamacpp/models.md](../llamacpp/models.md#7-nvidia-nemotron-35-bolt-74b-a9b)).
llama.cpp b12407 runs it through the `nemotron_h_moe` arch ([nemotron-h.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/src/models/nemotron-h.cpp)) and the Qwen3-Coder tool handler.
Thinking: on by default; `{"enable_thinking": false}` turns it off and `{"enable_thinking": true, "medium_effort": true}` gives medium effort ([card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/raw/main/README.md)).
Sampling: temperature 1.0, top_p 0.95 ([card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/raw/main/README.md)).
License: [OpenMDW-1.1](https://raw.githubusercontent.com/OpenMDW/OpenMDW/main/1.1/LICENSE.OpenMDW-1.1), permissive with commercial use allowed.
Caveats: the Bolt numbers are NVIDIA-run only, with no SWE-bench Verified score and no independent TB2.1 run; the only independent signals are weak ([Artificial Analysis](https://artificialanalysis.ai/models/nemotron-3-5-bolt-74b-a9b): TB 4.0 0, AutomationBench 15.5, no Intelligence Index).
This machine is below NVIDIA's stated 32 GB+ hardware floor (it ran anyway), NVIDIA's own agent replay is 32% slower than Qwen3.6, decode here is 40% slower than Qwen3.6's, and the [#30118](https://github.com/ggml-org/llama.cpp/issues/30118) risk applies on the Qwen3-Coder path.
vLLM does not fit (22.43 GiB); [Ollama `nemotron-3.5-bolt:74b-a9b`](https://registry.ollama.ai/v2/library/nemotron-3.5-bolt/manifests/74b-a9b) (24.75 GB) is a slower fallback.
Lightning versus Bolt: Nemotron 3.5 Lightning (honorable mention) has first-hand tool-loop evidence, 30/30 in the user's harness with thinking on and only a longer turn limit (measured on the M5 Max, libredb-studio; [agent-harness-evidence.md](agent-harness-evidence.md)), while Bolt's evidence is NVIDIA-run only and this laptop is below NVIDIA's stated 32 GB floor.
On this laptop on 2026-11-16 Lightning measured 143.6 tok/s at 128k and Bolt 73.5 tok/s, and both passed the tool test and an OpenCode session, so Bolt's vendor-reported quality lead costs half the decode speed here.
The ranking is left as aggregated; measuring both on this laptop decides the Nemotron slot (see [What to measure first](#what-to-measure-first)).
Evidence: vendor-run in a harness shared with competitors ([BenchLM](https://benchlm.ai/models/nemotron-3-5-bolt-74b-a9b) aggregates the same claims); memory and speed measured on this laptop.

### 8. Groot-19B-A3B-261023

Why: Borda 8 with 5 points (quality 10, practicality 7).
It is the most practical fast option: fully in VRAM at 128k on llama.cpp, measured on 2026-11-16 at 169.8 tok/s, 112.6 tok/s after a 107k prompt and a 13,822 MiB peak ([llamacpp/models.md](../llamacpp/models.md#8-groot-19b-a3b-261023)), with the same Qwen3-Coder XML tool format and parsers as Qwen3.6 in both pinned runtimes.
It reads about 2.06 GB per token, so its decode bound is very high (inferred).
One third-party run exists ([Blackwall](https://guitaripod.github.io/blackwall/#groot-19b-a3b-nvfp4~medium): tools 100, quality index 78.5), without a Qwen3.6 comparator.
Second path: vLLM v0.33.0 with `GrootTeam/Groot-19B-A3B-261023-NVFP4` and `--language-model-only` (10.99 GiB loaded, weight-only FP4 experts on Marlin), 131072 with FP8 KV, 167,480 KV tokens and 193.4 tok/s with thinking off over runs of 780 to 829 tokens (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17); the hybrid GDN NVFP4 crash under load ([vLLM #54331](https://github.com/vllm-project/vllm/issues/54331)) is a risk, so keep `--enforce-eager` in reserve.
Thinking: toggle, on by default (Qwen3.6-style).
Sampling: precise coding temperature 0.6, top_p 0.95, top_k 20, presence 0; general thinking temperature 1.0, top_p 0.95, top_k 20, presence 1.5; non-thinking temperature 0.7, top_p 0.8, top_k 20, presence 1.5 ([card](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/README.md)).
Caveats: all agentic scores are self-reported, some Qwen3.6 baselines were copied from Qwen's card rather than run, and on the vendor's own chart Groot loses to its base on SWE-bench Verified (31.66 vs 36.27).
The publisher is anonymous, the card metadata says Apache-2.0 but the repo has no LICENSE file, and the pruning method and data are undisclosed.
Community reports describe altered verbatim path text, which is a direct risk for a coding agent: run an A/B against Qwen3.6 before relying on it.
Evidence: vendor-reported plus one third-party run without a comparator; memory and speed measured on this laptop, quality not.

### 9. gpt-oss-20b

Why: Borda 9 with 5 points (practicality 9, coverage 8), and the required gpt-oss member (gpt-oss-120b needs 49-52 GiB of RAM).
It is the most mature model in both pinned runtimes, the only official checkpoint that runs entirely in vLLM, and the fastest: about 205 tok/s tg128 ([measured on a desktop RTX 5080, llama-bench build 6210, MXFP4](https://github.com/ggml-org/llama.cpp/discussions/15396)), which is a different card with 960 GB/s.
That makes it the safe pipeline baseline.
Measured on this laptop on 2026-11-16 with llama.cpp (rank 1 for comparison: Qwen3.8-27B ran at 36.4 tok/s with 64k context in 14.4 GiB, see [llamacpp/models.md](../llamacpp/models.md#1-qwen38-27b)): 166 tok/s decode at a short context, 145 tok/s after 8k and 105 to 108 tok/s after 30k of prompt, 13,360 MiB of GPU memory at 128k context with q8_0 KV, working tool calls, and an OpenCode write-and-read session that succeeded ([llamacpp/models.md](../llamacpp/models.md#9-gpt-oss-20b)).
llama.cpp: the [MXFP4 GGUF](https://huggingface.co/api/models/ggml-org/gpt-oss-20b-GGUF?blobs=true) (11.28 GiB) reaches 131072 with q8_0 KV at about 13.9 GiB VRAM (estimate).
vLLM: the estimate said 65536 needs `--gpu-memory-utilization 0.95`, `--kv-cache-dtype fp8` and `--attention-backend TRITON_ATTN` (1.11 GiB pool, about 97k FP8 tokens), but on 2026-11-16 that start failed, and only utilization 0.965 with `--enforce-eager` and 256 batched tokens started, at 83.5 tok/s (measured), on the Python frontend because the Rust frontend breaks the parser flags ([vLLM #66989](https://github.com/vllm-project/vllm/issues/66989)); details in [vllm/models.md](../vllm/models.md).
Thinking: always on, `reasoning_effort` low, medium (default) or high; it cannot be turned off.
Sampling: temperature 1.0, top_p 1.0 ([card](https://huggingface.co/openai/gpt-oss-20b/raw/main/README.md)).
Caveats: agentic strength is weak on independent measures ([AA](https://artificialanalysis.ai/models/gpt-oss-20b) TB2.1 13.9 at high effort; NVIDIA SWE-bench Verified 52.44 and TB2.1 15.17).
Open Harmony bugs: vLLM [#65770](https://github.com/vllm-project/vllm/issues/65770) and [#58825](https://github.com/vllm-project/vllm/issues/58825), and [llama.cpp #30781](https://github.com/ggml-org/llama.cpp/issues/30781) (HTTP 500 for array parameters without items; that it hits 20b is inferred).
The FlashInfer race [vLLM #61239](https://github.com/vllm-project/vllm/issues/61239) is not fixed in v0.33.0, so force `TRITON_ATTN` with FP8 KV.
It is the oldest pick.
Evidence: independent (NVIDIA same-harness, Artificial Analysis) plus vendor; speed measured on a desktop card only.

### 10. DeepSeek-R1-0528-Qwen3-8B

This is a chat and reasoning model only and is not for OpenCode agent mode.
Why: the required-family swap; it replaces Borda rank 10 (Nanbeige4.3-33B) because the DeepSeek family was otherwise missing, with 1 point (coverage 10).
It is the newest DeepSeek checkpoint that runs on 16 GB: the January 2025 R1 distills are older and also lack tool calling, and DeepSeek V4-Flash (159.6 GB) does not fit.
Measured on 2026-11-16 at 64k with q8_0 KV: peak 13,286 MiB, decode 70.6 tok/s at a short context and 41.1 after 30k, prefill 4,291 tok/s at 8k ([llamacpp/models.md](../llamacpp/models.md#10-deepseek-r1-0528-qwen3-8b-chat-and-reasoning-only)); f16 KV at 64k (17.9 GiB) does not fit.
Thinking: always on; there is no official off switch, and Ollama's think flag is the only toggle.
Tools: none.
The [tokenizer chat template](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/raw/main/tokenizer_config.json) never renders tools, vLLM's [deepseek_v3 tool parser](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/tool_parsers/deepseekv3_tool_parser.py) raises at init, and [Artificial Analysis](https://artificialanalysis.ai/models/deepseek-r1-qwen3-8b) measured tau2 at 0%.
Do not set `capabilities.tools` for it in OpenCode: in a test on 2026-11-16 with tools enabled it wrote the `write` tool's arguments as text instead of calling the tool, although a one-tool probe request did parse.
Sampling: temperature 0.6, top_p 0.95 ([card](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/raw/main/README.md)).
Caveats: it is old and superseded by small Qwen3.x models; on vLLM, `--quantization fp8` is deprecated in v0.33.0 (use `fp8_per_tensor`) and [vLLM #59738](https://github.com/vllm-project/vllm/issues/59738) leaves `deepseek_r1` streaming reasoning deltas undetokenized.
On vLLM v0.33.0 it runs only through llmctl's tokenizer overlay: on 2026-11-17 its first start, with an empty torch.compile cache, failed at the default `--gpu-memory-utilization 0.92`, later starts with the cache warm worked at 0.92, and the image's transformers loaded its tokenizer as a Metaspace `LlamaTokenizer`, so prompts lost their spaces and every reply came back garbled ([record](../research/2026-11-17/vllm-deepseek-r1-0528-qwen3-8b/)).
With the overlay naming `PreTrainedTokenizerFast` and `--kv-cache-memory-bytes` set, it starts cold and warm with llama.cpp's token counts and correct replies at 72.5 tok/s (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18) ([record](../research/2026-11-18/vllm-deepseek-r1-0528-qwen3-8b-cold-start/README.md)); llama.cpp stays its primary path.
Evidence: independent (Artificial Analysis) plus vendor; memory, speed and the tool failure measured on this laptop.

## Capability lists

"Toggle" means thinking can be switched per request with `chat_template_kwargs`; how each runtime exposes it is in [opencode.md](opencode.md).

Thinking and tools:

- Qwen3.8-27B (toggle; effort xhigh/medium/low)
- Qwen3.6-35B-A3B (toggle)
- K2 Horizon MoVA 36B A4B (always on; effort high recommended)
- K2 Horizon 7B (always on; effort high recommended)
- Gemma 4 26B-A4B IT (toggle, off by default)
- NVIDIA Nemotron 3.5 Bolt 74B-A9B (toggle; medium_effort option)
- Groot-19B-A3B-261023 (toggle)
- gpt-oss-20b (always on; effort low/medium/high)
- Nanbeige4.3-33B (toggle; honorable mention, trial only)
- NVIDIA Nemotron 3.5 Lightning 30B-A3B (toggle; honorable mention)
- Gemma 4 12B IT (toggle, off by default; honorable mention)
- GLM-4.7-Flash (toggle; honorable mention)
- Mellum2.1 12B-A2.5B Thinking (toggle; honorable mention)
- KAT-Coder-V2.5-Dev (toggle; honorable mention)
- Qwen3.5-9B (toggle; honorable mention)
- Xing4.0-29B-A4B (toggle; excluded)
- LFM2.5-24B-A2B (always on in the template; excluded)
- Ternary Bonsai 2 27B (toggle; excluded)
- Ornith-1.5-9B (toggle; excluded; not the Ornith-1.0-9B that passed the user's harness)
- Laguna XS 2.1 (toggle, off by default; excluded)

Tools without thinking:

- Qwen3-Coder-Next (non-thinking only)
- Qwen3-Coder-30B-A3B-Instruct (non-thinking only; honorable mention as a tool-calling worker)
- Devstral Small 2 24B (non-thinking only; excluded)
- Every toggle model above with `enable_thinking` false (the OpenCode "none" variant)

Thinking without tools:

- DeepSeek-R1-0528-Qwen3-8B (always thinking, no tool calling)

Neither: none among the verified candidates.

## New candidates assessed in this run

| Model | Verdict | Main reason |
|---|---|---|
| [K2 Horizon MoVA 36B A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/raw/main/README.md) | Top 10, rank 3 | AA TB2.1 56.6 and tau3 32.0, second only to Qwen3.8 among models that fit; llama.cpp only, `--n-cpu-moe 46`, 64-85k context |
| [K2 Horizon 7B](https://huggingface.co/IFM/K2-Horizon-7B/raw/main/README.md) | Top 10, rank 4 | AA TB2.1 44.2 and tau3 27.6 at 9B dense fully in VRAM; it does not match Qwen3.6 on SciCode, LCR or HLE |
| [Nemotron 3.5 Bolt 74B-A9B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/raw/main/README.md) | Top 10, rank 7 | NVIDIA same-harness TB2.1 50.98, tiny KV; vendor-run only, below NVIDIA's 32 GB+ floor |
| [Groot-19B-A3B-261023](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/README.md) | Top 10, rank 8 | Fully in VRAM at 128k with the Qwen3.6 tool format; self-reported scores, anonymous publisher, no LICENSE file |
| [Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/raw/main/README.md) | Honorable mention, trial only | Borda rank 10, displaced by the DeepSeek swap; needs IQ2_M with q4_0 KV and a tokenizer fix |
| [Mellum2.1 12B-A2.5B Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking/raw/main/README.md) | Honorable mention | Fast sub-agent on either runtime; below Qwen3.5-9B in its own table, no independent benchmark |
| [Xing4.0-29B-A4B](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B/raw/main/README.md) | Excluded | Self-reported scores with mislabelled comparators, no dedicated parser at b12407 |
| [LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) | Excluded | Vendor-only quality, silent tool-call loss ([#30490](https://github.com/ggml-org/llama.cpp/issues/30490), closed without a linked fix), licence revenue cap |
| [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/raw/main/README.md) | Excluded | The only independent test regresses against Qwen3.8-27B |

Nanbeige4.3-33B has the highest independent small-model score in this set ([AA](https://artificialanalysis.ai/models/nanbeige4-3-33b) Intelligence Index 20.4, TB 4.0 29.8, AutomationBench 53.1).
On 16 GB it only runs as a dense 33B at IQ2_M with q4_0 KV (about 15.5-15.7 of 15.92 GiB, estimate), with unmeasured quality loss.
b12407 mis-tokenizes its chat prompts even with the bartowski GGUF, so it needs `--override-kv tokenizer.ggml.add_space_prefix=bool:false` (unverified) until [PR #31986](https://github.com/ggml-org/llama.cpp/pull/31986) ships.
It reasons for 158.5k output tokens per AA task, its tool calls route to the Qwen3-Coder handler (untested), and vLLM does not fit (smallest build 19.34 GiB).

Xing4.0-29B-A4B: Qwen's TB2.0 51.5 is relabelled as TB2.1 and the Qwen3.6 SWE-bench Verified is given as 76.0 against Qwen's own 73.4; there is no xing parser in `common/parsers` at b12407, so tool calls rely on the generic auto-parser; there is a public report of failed write and edit tool calls; the official GGUFs predate the 2026-11-05 mainline merge ([PR #31589](https://github.com/ggml-org/llama.cpp/pull/31589)); vLLM does not fit (FP8 30.89 GiB).

LFM2.5-24B-A2B fits fully in VRAM at 131k (QAD-Q4_0 12.65 GiB plus q8_0 KV, about 14.7 GiB, estimate), and vLLM fits through the 12.92 GiB NVFP4A16.
But in its own OpenCode table SWE-bench Verified is 51.0 against 66.2 for Qwen3.6, AA has no page, [BenchLM](https://benchlm.ai/models/lfm2-5-24b-a2b) gives an estimated 34.4, [llama.cpp #30490](https://github.com/ggml-org/llama.cpp/issues/30490) reports tool calls silently lost as Qwen3-Coder-style XML text (closed on 2026-10-26 as completed, with no linked fix or comment, so treat it as unresolved), and the [LFM Open License](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/raw/main/LICENSE) bars commercial use by entities with revenue of USD 10M or more.

Ternary Bonsai 2 27B is fast and fits at 128k with q8_0 KV (Q2_0 7.10 GiB, about 13.1 GiB total, estimate), but the only independent quality test scored it 12/47 against 19/47 for Qwen3.8-27B ([InsiderLLM](https://insiderllm.com/guides/bonsai-2-27b-vs-qwen3-8-27b-rtx-3090/), intent-routing split), every agentic score is Prism's own, the vendor lists looping and malformed tool calls, and the stock image runs only Q2_0, not PQ2_0.

## Honorable mentions

| Model | Role | Reason |
|---|---|---|
| [Nanbeige4.3-33B](https://artificialanalysis.ai/models/nanbeige4-3-33b) | Trial candidate | Borda rank 10 (3 points), displaced by the required DeepSeek swap; re-judge after PR #31986 ships and a smoke test passes |
| [Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12B-it/raw/main/README.md) | vLLM-native Gemma, fast fallback or sub-agent | 3 points; official QAT W4A16 at 9.56 GiB, 128k with FP8 KV at 69.5 tok/s over runs of 885 to 933 tokens (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17); AA TB2.1 27.3; open gemma4 parser bugs remain |
| [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash/raw/main/README.md) | Alternative MoE | 2 points; AA tau2 98.8 and TB-Hard 22.0 but no TB2.1, older (2026-01); unsloth Q4_K_M 17.05 GiB with `--n-cpu-moe 20` (17-18 is too tight because block 0 is dense) |
| [NVIDIA Nemotron 3.5 Lightning 30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/README.md) | Fallback if Bolt does not fit | Previous rank 6; ggml-org Q4_0 17.60 GiB with `--n-cpu-moe 25`, `-lm none` and a 3,200 MiB expert cache, measured on 2026-11-16 at 128k: 143.6 tok/s decode, 128 tok/s after a 104k prompt, 15,106 MiB peak, tool test and OpenCode session passed ([llamacpp/models.md](../llamacpp/models.md#nemotron-35-lightning-30b-a3b-fallback-if-bolt-does-not-fit)); TB2.1 24.58 in the Lightning card's NVIDIA table (23.46 in the Bolt card's table), AA 24.3; first-hand 30/30 in the user's harness with thinking on (measured on the M5 Max, libredb-studio; [agent-harness-evidence.md](agent-harness-evidence.md)), which Bolt lacks |
| [Qwen3-Coder-30B-A3B-Instruct](https://ollama.com/library/qwen3-coder/tags) | Fast non-thinking tool-calling worker or sub-agent | In the user's harness it was the most reliable and among the fastest: 30/30, every cell on its first attempt, 11 s median run (measured on the M5 Max, libredb-studio; [agent-harness-evidence.md](agent-harness-evidence.md)); its coding scores (independent SWE-bench Verified 31.8, TB2.1 13.5) keep it out of the top 10, and here its experts partly spill to RAM (Ollama 19 GB; llama.cpp with `--n-cpu-moe` about 20, estimate), so the Mac's speed will not carry over |
| [Mellum2.1 12B-A2.5B Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking/raw/main/README.md) | Fast sub-agent | Fully in VRAM with 128k on either runtime (GGUF Q6_K 10.13 GiB, vLLM AWQ-INT4 7.15 GiB, 257.2 tok/s with thinking on over runs of 1,171 to 1,500 tokens (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17)); vendor-only, weaker than Qwen3.5-9B in its own table |
| [KAT-Coder-V2.5-Dev](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev/raw/main/README.md) | Qwen3.6 fine-tune | Same runtime as Qwen3.6; in-house SWE-bench Verified 69.4 and TB2.1 41.02, contested; try it only if your own tests beat Qwen3.6 |
| [Qwen3.5-9B](https://huggingface.co/RedHatAI/Qwen3.5-9B-NVFP4) | Conservative vLLM-native 9B | RedHatAI NVFP4 7.42 GiB, 131k at 90.8 tok/s over runs of 676 to 791 tokens (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17), build with `MAX_JOBS=2`; use it if every vendor-only small model is rejected |

## Excluded models

| Model | Reason |
|---|---|
| [Xing4.0-29B-A4B](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B/raw/main/README.md) | Self-reported scores with mislabelled comparators, no xing parser at b12407, a public report of failed write and edit tool calls, GGUFs that predate the mainline merge |
| [LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/raw/main/LICENSE) | Vendor-only quality (OpenCode SWE-bench Verified 51.0 vs 66.2 for Qwen3.6 in its own table), silent tool-call loss ([#30490](https://github.com/ggml-org/llama.cpp/issues/30490), closed without a linked fix), licence revenue cap |
| [Ternary Bonsai 2 27B](https://insiderllm.com/guides/bonsai-2-27b-vs-qwen3-8-27b-rtx-3090/) | The only independent quality test regresses against Qwen3.8-27B (12/47 vs 19/47); all agentic scores are Prism's own; vendor-listed tool-call looping |
| [Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B/raw/main/README.md) | Previous rank 7; no votes this round; every score comes from an anonymous, unverified org, and the NVFP4 hybrid GDN crash ([vLLM #54331](https://github.com/vllm-project/vllm/issues/54331)) applies; Groot and K2 Horizon 7B now cover the fast in-VRAM slot. This exclusion is Ornith-1.5-9B only: the harness passed Ornith-1.0-9B (Ollama repo `ornith`, tag `ornith:9b`) 30/30 (measured on the M5 Max, libredb-studio; [agent-harness-evidence.md](agent-harness-evidence.md)), a different model that has not been judged here |
| [Laguna XS 2.1](https://huggingface.co/poolside/Laguna-XS-2.1) | Vendor table below Qwen3.6 on every benchmark; tool calls untested in llama.cpp |
| [gpt-oss-120b](https://huggingface.co/api/models/ggml-org/gpt-oss-120b-GGUF?blobs=true) | Needs about 49-52 GiB of RAM against about 40 GiB free; weak independent agentic scores |
| [Devstral Small 2 24B](https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json) | Dense, so it reaches only about 32k with q8_0 KV; 64k needs q4_0 KV or 3-bit weights |
| IBM Granite 4.2 8B, Nemotron 3 Nano 4B, [Nemotron 3.5 Bolt E4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16/raw/main/config.json), MiMo-V2.6-Distill-Qwen-9B | Weak or missing agentic scores, a 64k hard cap (Bolt E4B), or a contested tool format (MiMo) |
| Ministral 3 (8B and 14B), Gemma 4 E4B | Coding-agent quality: weak or missing agentic coding scores. Not tool reliability: all three passed the user's harness 30/30 at the compiled defaults, every cell on its first attempt (measured on the M5 Max, libredb-studio; [agent-harness-evidence.md](agent-harness-evidence.md)); Ministral 3 8B is a reasonable fast sub-agent for simple tool loops |
| Qwen2.5 7B, 14B and 32B, Qwen2.5-Coder-14B, QwQ-32B | 32k native context (40k for QwQ-32B) with no YaRN in the Ollama GGUFs, so OpenCode's 64k needs llama.cpp YaRN flags; superseded by the Qwen3.x models; the dense 32B models spill to RAM here. All passed the user's harness, and Qwen2.5-7B is a reasonable fast sub-agent for simple tool loops ([agent-harness-evidence.md](agent-harness-evidence.md)) |
| qwen3:8b and qwen3:14b (Ollama tags) | 40,960-token native context, below OpenCode's 64k minimum; 64k needs YaRN ([agent-harness-evidence.md](agent-harness-evidence.md)) |
| Magistral Small (`magistral:24b`) | Context capped at 40,960 tokens, below OpenCode's 64k ([agent-harness-evidence.md](agent-harness-evidence.md)) |
| Granite 4.1 30B, Cogito v1 32B | Dense; at Q4_K_M about 40-45% of the model would sit in RAM, so they do not fit (estimate, [agent-harness-evidence.md](agent-harness-evidence.md)) |
| Nemotron 3 Nano 30B | Superseded by Nemotron 3.5 Lightning, which has the cleaner harness record ([agent-harness-evidence.md](agent-harness-evidence.md)) |
| LFM3-1.7B | Its own Ollama card advises against agentic coding ([ollama.com/library/lfm3](https://ollama.com/library/lfm3), [agent-harness-evidence.md](agent-harness-evidence.md)) |
| [DeepSeek V4-Flash](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash?blobs=true) | 284B total, 159.6 GB; far above VRAM plus free RAM |
| Qwen3.5-27B, [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/README.md), Gemma 4 31B, Nemotron 3 Super, Mistral Small 4, Ripple-65B, Muse Glimmer 30B, Flint, Ornith-1.5-35B, North Mini Code | Superseded by Qwen3.8-27B, do not fit, or remain unverified discovery leads (carried over from the previous run) |

## What to measure first

All ten ranks were measured on 2026-11-16 (items 3 to 8 below, and Gemma 4, Qwen3-Coder-Next, gpt-oss-20b and DeepSeek-R1 in their own sections); on 2026-11-17 the vLLM honorable mentions Gemma 4 12B, Qwen3.5-9B and Mellum2.1, plus Gemma 4 E4B, were measured on vLLM ([research/2026-11-17/README.md](../research/2026-11-17/README.md)); the rest has not been run on this laptop.
Record each result with the template in [llamacpp/benchmarking.md](../llamacpp/benchmarking.md) or [vllm/benchmarking.md](../vllm/benchmarking.md), and note the power limit.

1. Done on 2026-11-15: the NVIDIA Container Toolkit with CDI is installed and `docker run --device nvidia.com/gpu=all` sees the RTX 5080 Laptop GPU and runs a CUDA kernel ([install record](../research/2026-11-15/probes/toolkit-install.md)).
   Done on 2026-11-16: `server-cuda-b12407` starts with `-e GGML_CUDA_DISABLE_GRAPHS=1` placed before the image name, and no GPU hang occurred across the three model runs.
2. Read the configured DDR5 speed (`dmidecode`), because every expert-offload speed estimate depends on it.
3. Qwen3.8-27B UD-Q3_K_XL: done on 2026-11-16 for VRAM at `-c 65536` with q8_0 KV (14,790 MiB peak), decode (36.4 tok/s) and an OpenCode write-and-read session at `reasoning_effort` medium.
   Still open: decode with q8_0 versus f16 KV (#27623) and an edit task in OpenCode.
   The q8_0 check matters more since Qwen3-Coder-Next, which also has 256-wide heads, lost 40% of its q8_0 decode speed by a 101k prompt (measured on 2026-11-16).
4. Qwen3.6-35B-A3B Q4_K_M at 131072: done on 2026-11-16.
   `--n-cpu-moe 21` left 2.1 GiB free, and the tuned launch with `-lm none` and a 3,200 MiB expert cache decodes at 122.5 tok/s ([llamacpp/models.md](../llamacpp/models.md#2-qwen36-35b-a3b)).
   Still open: how often #30118 tool-call leaks occur over longer OpenCode sessions.
5. K2 Horizon MoVA Q4_K_M: done on 2026-11-16.
   It runs at 64k with q8_0 KV, all routed experts in pinned RAM and a 1,536 MiB expert cache, at 37.2 tok/s, and the OpenCode session passed in 20 s; high-effort turns are long.
6. K2 Horizon 7B Q8_0: done on 2026-11-16.
   The k2-horizon parser passed the tool test and an OpenCode session at 81,920 context, decode is 67 tok/s, and a 3,000-token budget truncated a coding answer, so the OpenCode output limit is now 32768.
7. Nemotron Bolt: done on 2026-11-16.
   131072 starts with `--n-cpu-moe 60` and a 2,816 MiB expert cache, decode is 73.5 tok/s, and the tool-call path works.
8. Groot Q4_K_M: an A/B against Qwen3.6 on the same OpenCode tasks, checking verbatim path fidelity.
   Speed is done (2026-11-16: 169.8 tok/s at 128k, all in VRAM, and a short OpenCode session passed); the quality A/B is still open.
9. A short agentic A/B with thinking on and off across ranks 1 to 4 and 7, to replace the inferred speed and quality ordering with measurements.
10. Rank 1 tool loop on the KB path: repeat a multi-turn tool-loop check (the libredb surfaces or an equivalent) against the llama.cpp UD-Q3_K_XL server, because the harness 30/30 covers only the Ollama Q4_K_M build ([agent-harness-evidence.md](agent-harness-evidence.md)).
11. Measure Nemotron 3.5 Lightning against Bolt on this laptop: the same OpenCode tasks, tool-call path and decode tok/s, to settle the Nemotron slot.
    Both are done for speed and the short tool session (2026-11-16: Lightning 143.6 tok/s, Bolt 73.5 tok/s); a longer OpenCode task set to compare their quality is still open.
12. The cheap Ollama cross-checks listed under "Try here first" in [agent-harness-evidence.md](agent-harness-evidence.md#try-here-first): qwen3.5:9b at 128k, gemma4:12b with thinking off, ornith:9b, ministral-3:8b with q8_0 KV and qwen2.5:7b, each with `OLLAMA_CONTEXT_LENGTH` set explicitly and the Ollama version and model digest recorded.
    Done on 2026-11-16 on Ollama 0.40.16: all five fit fully on the GPU, decoded at 99-119 tok/s and passed the LibreDB `tool_calls` probe, a tool test and a short OpenCode session ([agent-harness-evidence.md](agent-harness-evidence.md#try-here-first)).
    ministral-3:8b fell to 35 tok/s after a 58k prompt, and a fresh Ollama container spends about 25 s compiling CUDA kernels before its first answer ([opencode.md](opencode.md#ollama-notes)).

Each pick also needs a `/v1/models` check and an OpenCode tool-call smoke test before use; the commands are in [llamacpp/README.md](../llamacpp/README.md).

## Changes versus the earlier ranking

The earlier ranking is the first-run synthesis of 2026-11-15, before the new-candidate run.

| Model | Earlier rank | Now | Reason |
|---|---|---|---|
| Qwen3.8-27B | 2 | 1 | Two of three judges put it first on independent AA evidence (TB2.1 79.8 xhigh, 65.2 medium; tau3-Banking 48.0) |
| Qwen3.6-35B-A3B | 1 | 2 | Evidence unchanged, but the AA review added tau3-Banking 9.3; still the 128k pick with the most mature runtime path |
| K2 Horizon MoVA 36B A4B | New | 3 | AA TB2.1 56.6, tau3 32.0 |
| K2 Horizon 7B | New | 4 | AA TB2.1 44.2, tau3 27.6 |
| Gemma 4 26B-A4B IT | 3 | 5 | Evidence unchanged; pushed down by the two K2 models |
| Qwen3-Coder-Next | 5 | 6 | Pushed down by the new entries |
| Nemotron 3.5 Bolt 74B-A9B | New | 7 | Replaces Lightning as the Nemotron pick (TB2.1 50.98 vs 23.46 in the same NVIDIA table in the Bolt card) |
| Groot-19B-A3B-261023 | New | 8 | Takes the fast all-in-VRAM slot that Ornith held |
| gpt-oss-20b | 4 | 9 | Weak independent agentic scores (AA TB2.1 13.9, NVIDIA 15.17); stays as the required gpt-oss member and pipeline baseline |
| DeepSeek-R1-0528-Qwen3-8B | 10 | 10 | Same required-family swap, this time displacing Nanbeige4.3-33B rather than KAT-Coder |
| Nemotron 3.5 Lightning 30B-A3B | 6 | Honorable | Fallback if Bolt does not fit |
| Ornith-1.5-9B | 7 | Excluded | No votes; vendor-only from an unverified org |
| Gemma 4 12B IT | 8 | Honorable | 3 points; the vLLM-native Gemma |
| GLM-4.7-Flash | 9 | Honorable | 2 points |
| Qwen3-Coder-30B-A3B-Instruct | Excluded | Honorable | 30/30 in the user's agent harness, as a fast tool-calling worker; coding scores unchanged ([agent-harness-evidence.md](agent-harness-evidence.md)) |

Audit fixes applied in this document:

- `GGML_CUDA_DISABLE_GRAPHS=1` is a `docker -e` flag placed before the image and applies to every llama.cpp command.
- gpt-oss-20b runs primarily on llama.cpp, with vLLM as an option at 64k using utilization 0.965, `--enforce-eager`, FP8 KV and `TRITON_ATTN`.
- #28581 is no longer given as the reason to avoid the CUDA 13 image; the reason is `cuda>=13.5`; a soft IQ-quant sanity note remains for Qwen3-Coder-Next.
- `--n-cpu-moe` is described as counting blocks from 0.
- The smallest vLLM checkpoint for Qwen3.6 is the 18.75 GiB AutoRound (language model only).
- DeepSeek-R1-0528-Qwen3-8B is the newest DeepSeek that fits, not the only one.
- Qwen3.8 medium effort has a mechanism (`--reasoning-effort medium` or a variant body).
- 65536 is the safe target for Qwen3.8; 88k is an untested estimate.

## Recipes and related documents

- llama.cpp commands, offload values, context and sampling per model: [llamacpp/models.md](../llamacpp/models.md).
- vLLM recipes for the models that fit (gpt-oss-20b, Groot NVFP4, the K2 Horizon 7B AWQ path, DeepSeek-R1-0528-Qwen3-8B) and the models that do not: [vllm/models.md](../vllm/models.md).
- SGLang recipes for the models that run on SGLang v0.5.23 (Qwen3.5-9B, K2 Horizon 7B, gpt-oss-20b with Paged Experts, Gemma 4 12B, Gemma 4 E4B and Qwen3.6-35B-A3B with Paged Experts) and the models that do not: [sglang/models.md](../sglang/models.md).
- Third-party speed and quality numbers, normalized and labelled: [community-benchmarks.md](community-benchmarks.md).
- OpenCode configuration, thinking variants and trying these models remotely through OpenRouter: [opencode.md](opencode.md).
- Evidence classes and the 19 reviewed sources: [sources.md](sources.md).
- First-hand tool-loop results from the user's LibreDB Studio harness: [agent-harness-evidence.md](agent-harness-evidence.md).

## Sources

- [Artificial Analysis](https://artificialanalysis.ai/) model pages: [Qwen3.8-27B](https://artificialanalysis.ai/models/qwen3-8-27b), [Qwen3.6-35B-A3B](https://artificialanalysis.ai/models/qwen3-6-35b-a3b), [K2 Horizon MoVA](https://artificialanalysis.ai/models/k2-horizon-mova-36b-a4b), [K2 Horizon 7B](https://artificialanalysis.ai/models/k2-horizon-7b), [Gemma 4 26B-A4B](https://artificialanalysis.ai/models/gemma-4-26b-a4b), [Qwen3-Coder-Next](https://artificialanalysis.ai/models/qwen3-coder-next), [Nemotron 3.5 Bolt](https://artificialanalysis.ai/models/nemotron-3-5-bolt-74b-a9b), [gpt-oss-20b](https://artificialanalysis.ai/models/gpt-oss-20b), [Nanbeige4.3-33B](https://artificialanalysis.ai/models/nanbeige4-3-33b), [DeepSeek-R1-0528-Qwen3-8B](https://artificialanalysis.ai/models/deepseek-r1-qwen3-8b)
- NVIDIA same-harness tables: [Nemotron 3.5 Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/README.md), [Nemotron 3.5 Bolt card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/raw/main/README.md)
- Model cards: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/README.md), [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/README.md), [K2 Horizon MoVA](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/raw/main/README.md), [K2 Horizon 7B](https://huggingface.co/IFM/K2-Horizon-7B/raw/main/README.md), [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/README.md), [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next/raw/main/README.md), [Groot-19B-A3B-261023](https://huggingface.co/GrootTeam/Groot-19B-A3B-261023/raw/main/README.md), [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/raw/main/README.md), [DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/raw/main/README.md)
- llama.cpp b12407 source: [llama-arch.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/src/llama-arch.cpp), [k2-horizon parser](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/common/parsers/k2-horizon.cpp), [chat.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b12407/common/chat.cpp)
- llama.cpp issues: [#27330](https://github.com/ggml-org/llama.cpp/issues/27330), [#27623](https://github.com/ggml-org/llama.cpp/issues/27623), [#28581](https://github.com/ggml-org/llama.cpp/issues/28581), [#29655](https://github.com/ggml-org/llama.cpp/issues/29655), [#30118](https://github.com/ggml-org/llama.cpp/issues/30118), [#30357](https://github.com/ggml-org/llama.cpp/issues/30357), [#30490](https://github.com/ggml-org/llama.cpp/issues/30490), [#30781](https://github.com/ggml-org/llama.cpp/issues/30781), [PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802), [PR #31986](https://github.com/ggml-org/llama.cpp/pull/31986)
- vLLM v0.33.0: [K2 Horizon recipes](https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/recipes/models/IFM/K2-Horizon-7B.yaml), [#66264](https://github.com/vllm-project/vllm/issues/66264), [#61239](https://github.com/vllm-project/vllm/issues/61239), [#54331](https://github.com/vllm-project/vllm/issues/54331), [#66989](https://github.com/vllm-project/vllm/issues/66989), [#65770](https://github.com/vllm-project/vllm/issues/65770), [#58825](https://github.com/vllm-project/vllm/issues/58825), [#59738](https://github.com/vllm-project/vllm/issues/59738), [#62743](https://github.com/vllm-project/vllm/issues/62743)
- OpenCode: [#55378](https://github.com/anomalyco/opencode/issues/55378)
- First-hand: the user's LibreDB Studio agent harness ([docs/llms/README.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/README.md)), analysed in [agent-harness-evidence.md](agent-harness-evidence.md)
- Third-party runs: [LLM SQL benchmark post](https://www.reddit.com/r/LocalLLaMA/comments/1s7r9wu/i_tested_as_many_of_the_small_local_and/), [desktop RTX 5080 18-model post](https://www.reddit.com/r/LocalLLaMA/comments/1s9mkm1/benchmarked_18_models_that_i_can_run_on_my_rtx/), [ggml-org gpt-oss guide](https://github.com/ggml-org/llama.cpp/discussions/15396), [Blackwall](https://guitaripod.github.io/blackwall/#groot-19b-a3b-nvfp4~medium), [InsiderLLM](https://insiderllm.com/guides/bonsai-2-27b-vs-qwen3-8-27b-rtx-3090/), [BenchLM](https://benchlm.ai/)

## Maintenance

Re-verify before relying on this document:

- Run the measurements under [What to measure first](#what-to-measure-first) and replace the inferred speeds and context targets with measured ones.
- Check whether llama.cpp #27330, #27623, #30118, #30357, PR #31802 and PR #31986, vLLM #66989 and #62743, and OpenCode #55378 have changed state, and whether a newer `server-cuda` image tag exists.
- Recheck the GGUF file sizes through the Hugging Face API if a repo is updated, especially the K2 Horizon, Bolt and Groot repos, which are recent.
- Recheck Artificial Analysis pages, because its index versions change methodology (v5.0 removed several evaluations) and new models appear often.
- Re-run the judge panel when a new candidate with independent agentic scores appears.
