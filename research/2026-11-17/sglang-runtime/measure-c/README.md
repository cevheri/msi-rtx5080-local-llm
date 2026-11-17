# SGLang v0.5.23 measurement C: wave 3 on Paged Experts, the four llama.cpp GGUF recipes, K2 Horizon MoVA and the vLLM DeepSeek-R1 cold start

Last verified: 2026-11-18.

This folder holds measurement C for the lead who adds SGLang as llmctl's third runtime, as the brief `$SCRATCH/prompts/sglang-measure-c.md` asks, with measurement B's records as the template for method, layout and wording.
Every SGLang start used `lmsysorg/sglang:v0.5.23-runtime`, one container at a time named `sglang-measure`, `--device nvidia.com/gpu=all --ipc=host`, `--memory 32g --memory-swap 32g` unless a row says otherwise, `-p 127.0.0.1:30000:30000` (none for `--network none` starts), the Hugging Face cache read-only, no restart policy, the model overlay of measurement A's mechanism (one folder per model under `overlay/`, mounted read-only at `/llmctl/model` with `--model-path /llmctl/model`, its `generation_config.json` carrying the twin recipe's sampling by plan ruling 18), `--served-model-name`, `--host 0.0.0.0 --port 30000`, fraction at most 0.90, `--max-running-requests 1`, `--disable-prefill-cuda-graph`, and B's environment `HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `FLASHINFER_NO_DOWNLOAD=1`, plus `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` from the gpt-oss check on (and `TIKTOKEN_ENCODINGS_BASE` for gpt-oss); warm starts used the volume `sglang-research-cache`.
vLLM starts used `vllm/vllm-openai:v0.33.0` as `launcher/runtimes/vllm.py` `container_spec` builds it, named `vllm-measure`, on 127.0.0.1:8000; the four llama.cpp starts were llmctl's own `llamacpp` container, started with `LLMCTL_CLIENTS= LLMCTL_RECIPES_DIR=$SCRATCH/sglang/measure-c/overrides $SCRATCH/llmctl-main/llmctl up <id> --runtime llamacpp`, the copy of main at 5ad8883.
Every llmctl command of the lane ran as that copy (`$SCRATCH/llmctl-main/llmctl`); the repository was only read.
The enforced GPU power limit read 140 W before 18 starts and 145 W before 10 (`<record>/logs/<run>.meta`).
Labels: "(measured on the MSI Raider 18 Max HX, SGLang v0.5.23)", "llama.cpp b12407" or "vLLM v0.33.0" for runs here, "(inferred from <what>)" for readings of code and logs, "(estimate)" for arithmetic, "(vendor-reported)" for sources.
"Free" means `nvidia-smi` free memory of the whole GPU; times are UTC and dates local (+03:00), so the runs from 21:00 UTC on are dated 2026-11-18.

## Answers for the lead

### Per recipe

| Recipe | Outcome | Shipped settings | `gpu_mib` | `ram_gib` | `memory_gib` | Decode | First token | Pool | Date | K | Draft |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.6-35b-a3b, SGLang | Runs | Paged Experts, Marlin, fraction 0.88, BF16 KV, 8 Mamba slots, 128k, parsers `qwen3_coder` and `qwen3`, `{"enable_thinking":false}`; needs `PYTORCH_ALLOC_CONF` in the runtime `[env]` | 15,266 | 23 | 27 | 151.3 tok/s | 2.417 s at 8,591, 7.006 s at 32,153 tokens | 161,736 BF16 tokens | 2026-11-18 | 88 of 256 | `$SCRATCH/sglang/overrides/sglang/qwen3.6-35b-a3b.toml` |
| nemotron-3.5-lightning, SGLang | Failed the smoke gate: no tool call in any thinking mode, M13 10 of 20 | none | none | none | none | 133.3 tok/s at K 72 (sweep) | 2.101 s at 8,343 | 237,617 at K 72, fraction 0.88 | 2026-11-18 | none | none |
| gemma-4-26b-a4b, SGLang | Failed: `AssertionError: Only gated SiLU/SiTU is supported, got gelu.` in the Marlin MoE runner, after a vision-tower load failure | none | none | none | none | none | none | 160,936 FP8 tokens were allocated at K 56 | 2026-11-18 | none | none |
| k2-horizon-mova-36b, SGLang | Failed: `ValueError: K2 Horizon MoVA supports unquantized bf16/fp16 weights only` | none | none | none | none | none | none | none | 2026-11-18 | none | none |
| deepseek-r1-0528-qwen3-8b, vLLM | Runs cold and warm at 64k | the shipped args plus `--kv-cache-memory-bytes 5153960755`, and llmctl's tokenizer folder with `--tokenizer /llmctl/tokenizer` after `--max-model-len` | none (vLLM) | none | 32 | 72.5 tok/s | 1.410 s at 8,078, 8.580 s at 30,254 | 69,904 FP8 tokens | 2026-11-18 | none | `$SCRATCH/sglang/overrides/vllm/deepseek-r1-0528-qwen3-8b.toml` |
| gemma-4-12b, llama.cpp | Runs | the draft unchanged: 128k, f16 KV | 9,660 | 12.9 | 20 | 77.1 tok/s | 3.416 s at 9,936, 11.617 s at 37,194 (9,915 cached) | 131,072 context | 2026-11-17 | none | `$SCRATCH/sglang/overrides/llamacpp/gemma-4-12b.toml` |
| gemma-4-e4b, llama.cpp | Runs | the draft unchanged: 128k, f16 KV | 5,310 | 7.9 | 20 | 151.3 tok/s | 0.866 s at 9,932, 3.120 s at 37,190 | 131,072 context | 2026-11-17 | none | `$SCRATCH/sglang/overrides/llamacpp/gemma-4-e4b.toml` |
| mellum2.1-12b, llama.cpp | Runs | the draft unchanged: 128k, f16 KV, thinking on by default | 12,562 | 13.2 | 16 | 188.2 tok/s, thinking on | 1.553 s at 9,932, 5.073 s at 37,190 | 131,072 context | 2026-11-17 | none | `$SCRATCH/sglang/overrides/llamacpp/mellum2.1-12b.toml` |
| qwen3.5-9b, llama.cpp | Runs | the draft unchanged: 128k, f16 KV | 12,670 | 15.0 | 16 | 66.0 tok/s | 1.942 s at 8,591, 6.305 s at 32,153 (8,075 cached) | 131,072 context | 2026-11-17 | none | `$SCRATCH/sglang/overrides/llamacpp/qwen3.5-9b.toml` |

All figures are measured on the MSI Raider 18 Max HX with the runtime of the row (SGLang v0.5.23, vLLM v0.33.0, llama.cpp b12407), from the records named below; `gpu_mib` is the GPU peak minus the desktop's 112 MiB, SGLang's `ram_gib` the peak of cgroup anon plus shmem rounded up (D5), llama.cpp's `ram_gib` the `docker stats` peak as the shipped llama.cpp recipes take it, and `memory_gib` the cap the protocol ran under.
The summary lines, as in the drafts' `[evidence]`:
- qwen3.6-35b-a3b: "151.3 tok/s decode with thinking off and 128k context through Paged Experts (88 of 256 experts resident), 161,736-token BF16 pool, 951 MiB free after ready and 503 MiB after a 124,498-token prompt and the first grammar request, 300 s of sustained load without error, 15,378 MiB GPU peak, 22.92 GiB of non-reclaimable host memory including the 16.97 GiB pinned expert store, smoke test and tool call passed".
- deepseek-r1-0528-qwen3-8b (vLLM): "72.5 tok/s decode with 64k context, cold and warm starts with a 4.8 GiB KV cache (69,904 tokens) set by --kv-cache-memory-bytes, 15,036 MiB GPU peak, 1,105 MiB free after ready; with the tokenizer folder the prompts count as on llama.cpp and replies read correctly; tools get HTTP 400 as a chat-only recipe should".
- gemma-4-12b: "77.1 tok/s decode with 128k context, 9,772 MiB GPU peak, 12.9 GiB container RAM, llmctl test, a tool call with its second turn and a json_schema title request passed; OpenCode session not run".
- gemma-4-e4b: "151.3 tok/s decode with 128k context, 5,422 MiB GPU peak, 7.9 GiB container RAM, llmctl test, a tool call with its second turn and a json_schema title request passed; OpenCode session not run".
- mellum2.1-12b: "188.2 tok/s decode with thinking on and 128k context, 12,674 MiB GPU peak, 13.2 GiB container RAM, llmctl test, a tool call with its second turn and a json_schema title request passed; OpenCode session not run".
- qwen3.5-9b: "66.0 tok/s decode with 128k context, 12,782 MiB GPU peak, 15.0 GiB container RAM, llmctl test, a tool call with its second turn and a json_schema title request passed; OpenCode session not run".
- The llama.cpp `ram_gib` figures include the page cache of the mmapped GGUF, which those runs kept active (6.68, 5.00, 10.32 and 9.07 GiB); anonymous memory peaked at 6.25, 2.84, 2.83 and 5.87 GiB, mostly llama-server's host prompt cache (measured).
- B's gpt-oss-20b draft is not edited; with the pinned-store setting its `ram_gib` would be 16 instead of 25 and its `memory_gib` 20 instead of 28, measured below.

Records: [sglang-qwen3.6-35b-a3b/](../../../2026-11-18/sglang-qwen3.6-35b-a3b/measure-c/README.md), [sglang-nemotron-3.5-lightning/](../../../2026-11-18/sglang-nemotron-3.5-lightning/measure-c/README.md), [sglang-gemma-4-26b-a4b/](../../../2026-11-18/sglang-gemma-4-26b-a4b/measure-c/README.md), [sglang-k2-horizon-mova-36b/](../../../2026-11-18/sglang-k2-horizon-mova-36b/measure-c/README.md), [sglang-gpt-oss-20b/](../../sglang-gpt-oss-20b/measure-c/README.md), [vllm-deepseek-r1-0528-qwen3-8b/](../../../2026-11-18/vllm-deepseek-r1-0528-qwen3-8b-cold-start/measure-c.md), [llamacpp-gemma-4-12b/](../../llamacpp-gemma-4-12b/measure-c.md), [llamacpp-gemma-4-e4b/](../../llamacpp-gemma-4-e4b/measure-c.md), [llamacpp-mellum2.1-12b/](../../llamacpp-mellum2.1-12b/measure-c.md), [llamacpp-qwen3.5-9b/](../../llamacpp-qwen3.5-9b/measure-c.md).

### The margin table and the threshold

| Start | Free right after ready | Near-full prompt (tokens, share) | After it | After the first grammar request | After the forced tool request | Lowest in the trace after ready | Drop |
|---|---|---|---|---|---|---|---|
| C gpt-oss-20b pe-round1 | 1,346 | 124,457, 95.0% | 1,036 | 1,036 | 1,036 | 1,022 | 324 |
| C gpt-oss-20b pe-round2-mem20 (20 GiB cap) | 1,347 | 124,459, 95.0% | 1,006 | 1,006 | 1,006 | 1,006 | 341 |
| C qwen3.6-35b-a3b w3-k88-f088 | 951 | 124,498, 95.0% | 503 | 503 | 503 | 465 | 486 |
| C qwen3.6-35b-a3b w3-k88-f088-mem27 (27 GiB cap) | 951 | 124,498, 95.0% | 503 | 503 | 503 | 501 | 450 |
| B qwen3.5-9b m12-a3 | 911 | 124,499, 95.0% | 545 | 545 | 545 | 537 | 374 |
| B k2-horizon-7b b0-a2 | 865 | 62,257, 95.0% | 543 | 543 | 543 | 532 | 333 |
| B gpt-oss-20b b0-a4 | 1,329 | 124,463, 95.0% | 989 | 989 | 989 | 989 | 340 |
| B gpt-oss-20b ld19-mem28 | 1,328 | 124,458, 95.0% | 988 | 988 | 988 | 988 | 340 |
| B gemma-4-e4b m5-a4-swa | 1,407 | 62,244, 95.0% | 1,403 | 1,403 | 1,403 | 1,403 | 4 |
| B gemma-4-12b m2-fp8-b | 1,099 | 124,431, 94.9% | 839 | 839 | 839 | 839 | 260 |

All values are MiB, measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([lane-logs/margins-c.txt](lane-logs/margins-c.txt) and [lane-logs/margins-b-recomputed.txt](lane-logs/margins-b-recomputed.txt), both by `scripts/margins_c.py`, B's own script; B's rows equal B's table).
- The largest drop is now Qwen3.6's 486 MiB, above the 374 MiB that set the threshold; D8's rule over B's and C's recipes together gives `MIN_FREE` = 768 MiB (486 + 256 = 742, rounded up to a multiple of 64) and a ship gate of `F_ready >= 896` MiB (estimate from the measured margins).
- Under that gate Qwen3.6 passes (951 MiB), as do gpt-oss-20b (1,329), Gemma 4 E4B (1,407), Gemma 4 12B (1,099) and Qwen3.5-9B (911, by 15 MiB), while B's K2 Horizon 7B (865 MiB) fails; by plan ruling 12 its fraction would have to drop by 0.02 and be measured again, or the lead keeps 640 MiB and does not ship Qwen3.6 as is (estimate).
- Qwen3.6's lowest point came during M12's load and the second 95% prompt; its first 95% prompt alone took 448 MiB (951 to 503) (measured).

### The slowest SGLang start

- Cold (new empty volume, `--network none`): 351.4 s for Qwen3.6 (m15-netnone-b; 345.6 s in m15-netnone), of which about 260 s was FlashInfer's autotune on the empty volume (measured).
- Warm volume, first start of a model: 233.1 s for Qwen3.6 (m11-k64, its kernels compiled into `sglang-research-cache`); every other start on that volume reached its ready line in 50.7 to 82.2 s (measured).
- Both are below D4's 450 s line, so `start_timeout_s` stays 900 (measured; the rule is the design's).
- The vLLM DeepSeek-R1 starts: 44.2 s cold (with the fix) and 20.3 s warm (measured).

### The pinned store: shared memory, and not in /dev/shm

- B's 1.84 factor is PyTorch 2.14.1's pinned host allocator rounding each paged tensor of the store up to a power of two; the documented `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` keeps every pinned request above 1 MiB at its size, and gpt-oss-20b's store fell from 19.19 to 10.37 GiB at the same decode (123.3 and 124.6 against 125.5 tok/s) and stability (73 requests in 300 s, no error), including under a 20 GiB cap ([sglang-gpt-oss-20b/](../../sglang-gpt-oss-20b/measure-c/README.md)) (measured; the mechanism inferred from the image's source).
- With the setting, the measured stores were 10.37 GiB for gpt-oss-20b, 16.97 GiB for Qwen3.6 (16.875 GiB of checkpoint expert tensors), 16.07 GiB for Nemotron (15.389 GiB) and 11.98 GiB for Gemma 4 26B (11.964 GiB); Nemotron without it took 26.02 GiB (measured).
- The store is shared memory: the scheduler holds it in `/dev/zero (deleted)` shared mappings (`rw-s`), as the CUDA driver's pinned allocations appear, counted as cgroup `shmem` and process `RssShmem` (measured in `logs/<run>.maps.txt`; that `cudaHostAlloc` makes them is inferred).
- It does not sit in `/dev/shm` under `--ipc=host`: the host's `/dev/shm` grew by 29 to 37 MiB at ready in the SGLang starts and fell back after the container was gone, while the store was 10 to 26 GiB (measured in `logs/<run>.devshm.txt`).
- Each SGLang start left one root-owned `sglang_loads_<hex>_<hex>.shm` of 16,396 bytes in `/dev/shm`, and the crashed Gemma K 72 start also a `sem.mp-*` of 32 bytes; C removed exactly those, naming each in the run's `.devshm.txt`, with a throwaway `--ipc=host --network none` container because the files are root's (measured).
- 49 `sglang_loads_*.shm` and 42 `sem.mp-*` files from earlier lanes' starts (2026-11-16 20:54 to 2026-11-17 23:03 local time) were there before the lane and are still there, untouched; at the end `/dev/shm` holds the same 93 files as at the start (measured).
- From the end of start w3-k88-f088 on, `df` counted 15 to 23 MB more in `/dev/shm` than before with no file to account for it (16,519,168 bytes at the end against 1,519,616); deleted files that some process outside the containers holds open would show this way, and a non-root user cannot see whose (measured; the explanation is inferred).
- For llmctl: `ram_gib` must hold the store (Qwen3.6: anon 5.73 plus shmem 16.97 GiB), and `/dev/shm`'s size (30.6 GiB here, half the RAM) does not bound it (measured).

### K2 Horizon MoVA

Failed before loading: the first start stopped on `ValueError: K2Horizon MoVA requires explicit source router GEMM provenance; ... pass --json-model-override-args '{"xllm_source_router_gemm_partitions": 2}'`, and with that override the second on `ValueError: K2 Horizon MoVA supports unquantized bf16/fp16 weights only` (`xllm.py` line 1215); the 69.75 GiB BF16 model does not fit (about 49.4 GiB of routed experts to pin, about 20.4 GiB that stay on the GPU) ([sglang-k2-horizon-mova-36b/](../../../2026-11-18/sglang-k2-horizon-mova-36b/measure-c/README.md)) (measured; the BF16 arithmetic is an estimate).
The download is verified: 24,272,026,244 bytes, all 20 files equal to the API tree by name, size and hash; `/home` kept 182.94 GiB free (measured).

### vLLM DeepSeek-R1 cold start

The failure still holds with the tokenizer folder: cold at the default 0.92, `Available KV cache memory: 3.83 GiB` against the 4.5 GiB that 64k needs.
Chosen setting: `--kv-cache-memory-bytes 5153960755` (4.8 GiB): a cold start (44.2 s) and a warm start (20.3 s) both got 69,904 tokens at 64k, a 15,036 MiB GPU peak and 1,105 MiB free, and the warm start measured 72.5 tok/s, 1.410 and 8.580 s to the first token, llama.cpp's token counts (8, 8,078, 30,254), correct replies, and HTTP 400 for every request with tools ([vllm-deepseek-r1-0528-qwen3-8b/](../../../2026-11-18/vllm-deepseek-r1-0528-qwen3-8b-cold-start/measure-c.md)) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0).
Context 65,536 and output 16,384 are kept; no smaller context was needed.
The tokenizer folder is built as llmctl will ship it (the lead's message): every snapshot file linked at its container path and a real `tokenizer_config.json` naming `PreTrainedTokenizerFast`; it behaved as B's three-entry folder did.

### The four M16 outcomes

All four GGUF drafts run unchanged at 128k with f16 KV and are finalized as `kind = "measured"`: gemma-4-12b 77.1 tok/s and 9,660 MiB, gemma-4-e4b 151.3 tok/s and 5,310 MiB, mellum2.1-12b 188.2 tok/s (thinking on) and 12,562 MiB, qwen3.5-9b 66.0 tok/s and 12,670 MiB; each passed `llmctl test`, a tool call with its second turn, Hermes' title request with a strict `json_schema`, a forced tool request, and thinking per its type (measured on the MSI Raider 18 Max HX, llama.cpp b12407).
`$SCRATCH/llmctl-main/llmctl check` with the finalized drafts printed `ok: 23 recipes (llamacpp 15, vllm 8)` ([lane-logs/llmctl-check-final-llamacpp.txt](lane-logs/llmctl-check-final-llamacpp.txt)).
Notes for Task 17: on llama.cpp a `reasoning_effort` does not turn on a toggle model's thinking (only `enable_thinking` does), Mellum's code request spent its 3,000 tokens on reasoning, and an OpenCode session was not run for any of the four (M14 covers it).

### What contradicted the research report, measurement A or measurement B

- B, LD19 and the brief's estimate rule: the 1.84 factor is not a property of Paged Experts but PyTorch's power-of-two rounding; it is 1.0 for checkpoints whose tensors are already powers of two (Qwen3.6, Gemma 4 26B) and is removed by a documented setting, which halves gpt-oss-20b's store and B's sizing (`ram_gib` 25 to 16, `memory_gib` 28 to 20) (measured).
- B's threshold: `MIN_FREE` 640 MiB came from a 374 MiB drop; Qwen3.6 dropped 486 MiB, so the rule gives 768 MiB and B's K2 Horizon 7B no longer passes the gate (estimate from measured margins).
- Report section 1 ("no material speed gain over the current defaults"): for Qwen3.6-35B-A3B, SGLang with Paged Experts decodes 151.3 tok/s against 122.5 on llama.cpp and answers the 8.6k and 32k prompts in 2.4 and 7.0 s against 4.3 and 13.0 s (measured).
- Report section 4, Qwen3.6 ("about 100 of 256 experts resident at 0.90 with 128k FP8 KV ... Gated DeltaNet with paging is untested upstream"): it runs; v0.5.23 picks FP8 KV from the checkpoint unless told otherwise, and with BF16 KV K 88 at 0.88 holds 128k, because 0.90 leaves under 768 MiB free (measured).
- Report section 4, Nemotron ("about 65 of 128 experts resident and about 15 GiB pinned"): K 72 at 0.88 holds 237,617 tokens and the store is 16.07 GiB, but the served model is wrong (measured).
- Report section 4 and the downloads lane, Gemma 4 26B ("Paged Experts needs a GPTQ, AWQ or compressed-tensors int4 build"): such a build exists and loads with a `quantization_config` override, but v0.5.23's Marlin MoE runner, the only one Paged Experts uses for int4, has no gated-GELU path (measured).
- The downloads lane inferred MoVA would fail on the value experts' `weight_packed` keys; v0.5.23 refuses any quantization config earlier, in the attention layer's constructor (measured).
- Report section 4, SGLang coverage ("at most 11 if ... both downloadable Paged Experts candidates work, and a supported Gemma 4 26B-A4B build turns up"): wave 3 adds one model, so SGLang runs 6 of 15 (measured).
- The downloads lane's llama.cpp `gpu_mib` estimates were upper bounds: 10,240, 9,216, 13,312 and 13,312 MiB against 9,660, 5,310, 12,562 and 12,670 measured (measured).
- Measurement A's cold-start reading held for Qwen3.6: 687 MiB right after a cold ready and 969 MiB after `/flush_cache`, as A found 709 and 1,039 on Qwen3.5-9B (measured).

### End state

- Start: `$SCRATCH/llmctl-main/llmctl status` at 23:17:48 (+03:00) showed gemma-4-26b-a4b on llama.cpp running and ready at 15,008 MiB and the stopped gpt-oss-20b `vllm` container; `down` at 23:17:53 left 112 MiB in use ([lane-logs/llmctl-status-before.txt](lane-logs/llmctl-status-before.txt), [lane-logs/llmctl-down.txt](lane-logs/llmctl-down.txt)) (measured).
- End: `PATH=$HOME/.kimi-code/bin:$PATH $SCRATCH/llmctl-main/llmctl up gemma-4-26b-a4b` at 01:09:06 (+03:00) printed `ready` and no `wrote` line, and `status` read running, health ready, 15,008 MiB of 16,303 MiB, and OpenCode, Hermes, Pi, dsh and Kimi on the running model ([lane-logs/llmctl-up.txt](lane-logs/llmctl-up.txt), [lane-logs/llmctl-status-after.txt](lane-logs/llmctl-status-after.txt)) (measured).
- Containers: llmctl's `llamacpp` running; no `sglang-measure`, `vllm-measure` or download container; the `-measured-2026-11-16` containers untouched; llmctl's stopped `vllm` container (gpt-oss-20b, exited about 8 hours before the lane) is gone, removed by this lane's first `llmctl down --remove` (deviation 1).
- Volumes: every `sglang-measure-cold` and `vllm-measure-cold` volume was removed after its start; `sglang-research-cache` (warm starts added Qwen3.6's, Nemotron's and Gemma's kernels), `vllm-cache`, `flashinfer-cache` and `ollama-models` are kept.
- Disk: `/home` 196,327,817,216 bytes free (182.85 GiB) and `/` 301,188,653,056 bytes (measured with `df -B1` after the restore).
- The repository was not edited, and no git command was run in it.

## Starts

| Time (UTC) | Record | Start | Ready line or poll | Outcome |
|---|---|---|---|---|
| 20:26 | [llamacpp-gemma-4-12b](../../llamacpp-gemma-4-12b/measure-c.md) | m16-a1 | 3.5 s (`llmctl up`) | M16 |
| 20:30 | [llamacpp-gemma-4-e4b](../../llamacpp-gemma-4-e4b/measure-c.md) | m16-a1 | 2.4 s | M16 |
| 20:32 | [llamacpp-mellum2.1-12b](../../llamacpp-mellum2.1-12b/measure-c.md) | m16-a1 | 4.5 s | M16 |
| 20:34 | [llamacpp-qwen3.5-9b](../../llamacpp-qwen3.5-9b/measure-c.md) | m16-a1 | 3.4 s | M16 |
| 20:37 | [sglang-gpt-oss-20b](../../sglang-gpt-oss-20b/measure-c/README.md) | pe-round1 | 54.6 s | Pinned-store setting, full protocol |
| 20:47 | sglang-gpt-oss-20b | pe-round2-mem20 | 50.7 s | The setting under a 20 GiB cap, protocol and M12 |
| 20:56 | [sglang-qwen3.6-35b-a3b](../../../2026-11-18/sglang-qwen3.6-35b-a3b/measure-c/README.md) | m11-k64 | 232.5 s | M11, FP8 KV chosen by the checkpoint |
| 21:00 | sglang-qwen3.6-35b-a3b | m11-bf16-k88 | 60.5 s | M11 |
| 21:02 | sglang-qwen3.6-35b-a3b | m11-bf16-k96 | 56.8 s | M11 |
| 21:04 | sglang-qwen3.6-35b-a3b | m11-bf16-k104 | 57.3 s | M11; pool short of 128k |
| 21:05 | sglang-qwen3.6-35b-a3b | w3-k88-f088 | 57.9 s | Shipped settings, full protocol, M12 |
| 21:14 | sglang-qwen3.6-35b-a3b | w3-k88-f088-mem27 | 59.9 s | Protocol and M12 under 27 GiB |
| 21:22 | sglang-qwen3.6-35b-a3b | m15-netnone | 345.6 s (poll) | M15 |
| 21:28 | [sglang-nemotron-3.5-lightning](../../../2026-11-18/sglang-nemotron-3.5-lightning/measure-c/README.md) | m11-bf16-k64 | 82.2 s | M11; tool test failed |
| 21:31 | sglang-nemotron-3.5-lightning | m11-bf16-k72 | 55.6 s | M11; tool test failed |
| 21:32 | sglang-nemotron-3.5-lightning | m11-bf16-k80 | 57.7 s | M11; pool short of 128k; tool test failed |
| 21:34 | [sglang-gemma-4-26b-a4b](../../../2026-11-18/sglang-gemma-4-26b-a4b/measure-c/README.md) | m11-fp8-k56 | none | Failed while loading (vision MLP repack) |
| 21:35 | sglang-gemma-4-26b-a4b | m11-fp8-k64 | none | The same |
| 21:35 | sglang-gemma-4-26b-a4b | m11-fp8-k72 | none | The same |
| 21:36 | sglang-gemma-4-26b-a4b | m11-fp8-k56-ign | none | Loaded, then failed: no gated-GELU Marlin path |
| 21:39 | sglang-nemotron-3.5-lightning | diag-k72-f088 | 57.7 s | Tool-call diagnostic, smoke, M13 10 of 20 |
| 21:42 | sglang-nemotron-3.5-lightning | diag-k72-f088-noround | 58.6 s | The same without the setting (34 GiB cap) |
| 21:44 | [sglang-k2-horizon-mova-36b](../../../2026-11-18/sglang-k2-horizon-mova-36b/measure-c/README.md) | mova-a1 | none | Failed: router provenance check |
| 21:45 | sglang-k2-horizon-mova-36b | mova-a2 | none | Failed: quantized MoVA refused |
| 21:46 | [vllm-deepseek-r1-0528-qwen3-8b](../../../2026-11-18/vllm-deepseek-r1-0528-qwen3-8b-cold-start/measure-c.md) | cold-r1 | none | Cold, 0.92: 3.83 GiB of KV, refused |
| 21:51 | vllm-deepseek-r1-0528-qwen3-8b | cold-r2-kvbytes | 44.2 s (poll) | Cold with `--kv-cache-memory-bytes`: started |
| 21:57 | vllm-deepseek-r1-0528-qwen3-8b | warm-r3-kvbytes | 20.3 s (poll) | Warm measurement |
| 22:02 | sglang-qwen3.6-35b-a3b | m15-netnone-b | 351.4 s (poll) | M15 again, with the post-flush reading |

All rows are measured on the MSI Raider 18 Max HX ([scripts/starts_table_c.py](scripts/starts_table_c.py)); "ready line" is SGLang's "The server is fired up and ready to roll!" after the `docker run` call, "poll" the readiness poll (`/ready` 200 and the served name, or for vLLM `/health` and `/v1/models`), and for llama.cpp the time `llmctl up` took.
Besides these, offline containers (`--rm --network none --memory 2g` to `4g`, no GPU) read source files from both images, rendered Nemotron's template, and listed volumes ([offline/](offline/README.md)), and one download container ran from 20:18 to 20:56.

## Deviations from the brief

1. llmctl's own rules removed two stopped containers of the owner's: the first `up` removed the `llamacpp` container that `down` had stopped (gemma-4-26b-a4b, "removing the stopped container 'llamacpp' from an earlier llmctl run"), and the first `down --remove` also removed llmctl's stopped `vllm` container (gpt-oss-20b, exited about 8 hours before), because `down --remove` removes every container llmctl started (`launcher/ops.py`, lines 156 to 169); their logs are gone.
2. The MoVA download ran from 20:18 to 20:56 UTC, beside the four llama.cpp runs and the first SGLang starts (network and disk only; each M16 record's meta says so).
3. `llmctl list` ran once, with `LLMCTL_CLIENTS` empty and C's overrides, to view the finalized drafts; it is read-only but not in the brief's command list.
4. From the gpt-oss check on, every SGLang start carried `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`, the brief's conditional; the stores were estimated with the factor it gave (1.096) ([estimates/README.md](estimates/README.md)).
5. Qwen3.6: the K 64 start ran with FP8 KV, which v0.5.23 took from the checkpoint; C stopped chain q1 and swept K 88, 96 and 104 with BF16, and shipped fraction 0.88 because 0.90 left less than the gate (the brief's "at most 0.90" holds).
6. Nemotron failed the smoke gate, so instead of the full protocol, M12, M15 and the cap run C ran two diagnostic starts, one without the setting under the 34 GiB maximum, to show the setting is not the cause.
7. Gemma 4 26B: a fourth start with a `quantization_config` override, to get past the vision-tower failure to a measured verdict on the experts; the K 72 start's command file was deleted by mistake after the start and rewritten with the same helper (it differs from K 64's only in K).
8. MoVA: two starts instead of one, because the first stopped on a configuration check whose message names the override.
9. vLLM: per the lead's message, the tokenizer folder is built as llmctl will ship it (every snapshot file linked, plus the real `tokenizer_config.json`; `--tokenizer` right after `--max-model-len`); B's three-entry folder was first rebuilt, then deleted before any start; the fix tried first, `--kv-cache-memory-bytes`, worked, so two of the three extra starts were used.
10. `/dev/shm` files left by this lane's starts were removed with a throwaway root container (`--ipc=host --network none`), as the files are root's; nothing else was removed.
11. Qwen3.6's M15 ran twice; the second added the `/flush_cache` reading that D8's check takes.
12. M13 was not run for Qwen3.6, because its recipe uses BF16 KV and D14's M13 row applies to FP8 KV recipes.
13. Local paths in the records were replaced with `$HOME` and `$SCRATCH`, and the user and host names with `<user>` and `<host>`, by `scripts/scrub.py`.

## Files

- One folder per runtime and model, each with `cmds/` (one `docker run` or llmctl command file per start, reading `$HOME` and `$SCRATCH`), `logs/` (cut and full logs, 1 s GPU and cgroup traces, state, metadata, process memory, the mappings inside the container at ready, the `/dev/shm` readings), `shapes/` (readiness and `/server_info`, sequence records, probe, api probe, sustained load, M13, summaries) and a `README.md`.
- [scripts/](scripts/): `run_c.sh`, `run_netnone_c.sh`, `run_llamacpp_c.sh` and `run_vllm_cold_c.sh` drive a start; `poll_c.py`, `seq_c.py` (B's `seq_b.py` with a `llamacpp` kind and a `diag` phase), `mkcmd_c.py`, `mkw3.sh`, `trace_c.sh`, `procmem.sh`, `maps_inside.py`, `shm_cleanup.sh`, `expert_bytes.py`, `anonshmem.py`, `w3_table.py`, `m16_summary.py`, `margins_c.py`, `summarize_c.py`, `starts_table_c.py`, `render_template.py`, `make_tokenizer_folder.py` (and the superseded `make_tokenizer_overlay.py`), `dl-hf-c.sh`, `verify-snapshot.py` (the downloads lane's), `scrub.py`, and `chain_q1.sh` to `chain_q4.sh` and `chain_n1.sh`, which queued starts; copied unchanged from B: `make_overlay.py`, `mcommon.py`, `cutlog.py`, `reqlog.py`, `sustained.py`, `api_probe.py`, `llm_probe_chat_only.py`, `nonreclaim.py`, and `llm_probe.py` (sha256 `ee3fdc0a...`, as in the repository); run the Python scripts with `python3 -I`.
- `overlay/`: the model overlays and `*.make.json`; [offline/](offline/README.md): source files and listings read from the images; [estimates/](estimates/README.md): the store estimates and expert bytes; [downloads/](downloads/): the download log and its verification; [overrides/](overrides/): the four llama.cpp drafts as finalized (the starts read the drafts as they were then, kept as each record's `logs/m16-a1.recipe.toml`); [drafts-before/](drafts-before/): the drafts as they were before this lane (the llama.cpp four and B's vLLM DeepSeek draft); [lane-logs/](lane-logs/): llmctl outputs, chain logs and the margin tables.
