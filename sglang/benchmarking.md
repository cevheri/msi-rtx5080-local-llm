# Benchmarking SGLang on this laptop

Last verified: 2026-11-18.

This page holds the method of the SGLang measurements on this laptop, the post-ready margins that set the runtime file's `min_free_at_ready_mib`, the results of each shipped recipe at its shipped settings, and the cold starts without the network.
Every figure is measured on the MSI Raider 18 Max HX with `lmsysorg/sglang:v0.5.23-runtime` unless it says otherwise, and the records are archived under [research/2026-11-17/sglang-runtime/](../research/2026-11-17/sglang-runtime/README.md), one folder per measurement lane, with one folder per model beside it.
The engine's memory model and the reasons behind each rule are in [README.md](README.md); the llama.cpp and vLLM results that the same probe produced are in [llama.cpp benchmarking](../llamacpp/benchmarking.md) and [vLLM benchmarking](../vllm/benchmarking.md).

## Method

Every start ran one container at a time with the GPU otherwise idle except for the desktop's 112 MiB.
The enforced GPU power limit was read before each start of measurements C, D and E (140 or 145 W; each record's `.meta` file) and once during measurement B (145 W; [B's lane README](../research/2026-11-17/sglang-runtime/measure-b/README.md)).
"Free" is `nvidia-smi` free memory of the whole GPU, and "flushed" means read right after `POST /flush_cache` on the idle server, the reading llmctl's [memory check](README.md#readiness-and-the-memory-check) takes.
A 1 s trace of GPU memory, power and the container's cgroup memory (anon, file, shmem, kernel) ran through every start.

The probes, run in this order after the server was ready (measurement B's `seq_b.py` and its successors in each lane's `scripts/`):

1. The ready reading, then, from measurement D on, D8's flush and the flushed reading.
2. A fresh near-full prompt: a random first line and copies of a code paragraph up to 95% of the context, with `max_tokens` 32; in B and C with no flush before it, so it started from the server's state at ready, and from D on right after D8's flush, the state llmctl's check leaves the server in.
3. Hermes' title request, the first grammar-constrained request: a system and a user message, `reasoning_effort` `none` and a strict `json_schema` response format.
4. A forced `tool_choice` request with one tool, the other grammar path.
5. The repository probe, [research/2026-11-16/llm_probe.py](../research/2026-11-16/llm_probe.py), after a `/flush_cache`: three streamed decode runs of up to 1,500 tokens (the median is the decode figure), then an 8k and a 30k prompt, each sent twice to show the prefix cache; the 30k prompt begins with the 8k prompt's paragraphs, so its first-token time compares engines but is not a fresh prefill rate (inferred from the script).
6. The api probe: plain and streamed replies, a tool call and its second turn, `json_schema`, the thinking switches and efforts, and edge shapes such as `"tools": []` (`api_probe.py`).
7. M12, sustained load: 300 s of back-to-back requests with fresh prompts of 200 to 12,000 words and 256 or 512 tokens out, at one request at a time (the hands-on `sustained.py`, [copy in measurement B](../research/2026-11-17/sglang-runtime/measure-b/scripts/sustained.py)).
8. A second fresh near-full prompt after the load.

The margin protocol reads free memory before and after each of steps 2 to 4 and 8 and takes, per start:

- `F_ready`: free memory right after ready, unflushed in measurements B and C and flushed from D on.
- `F_low`: the lowest free memory after that reading, over every reading and every trace sample to the end of the sequence.
- Used after ready: `F_ready - F_low`.

The threshold rule (D8 of the design): `MIN_FREE` is the largest used-after-ready figure over every recipe plus 256 MiB, rounded up to a multiple of 64, where a sum that is already a multiple stays (estimate: the 256 MiB covers kernels that a request shape outside the protocol may load).
A recipe ships only when its own `F_ready` is at least `MIN_FREE + 128` MiB, room for a desktop that holds a little more than 112 MiB (estimate); otherwise its `--mem-fraction-static` drops by 0.02 and it is measured again.

M13 compares FP8 with BF16 KV on 20 fixed prompts at temperature 0: 8 arithmetic, 6 Python-output and 6 tool prompts ([measurement B](../research/2026-11-17/sglang-runtime/measure-b/README.md#m13-fp8-against-bf16-kv)).

A cold start without the network uses a new, empty cache volume, `FLASHINFER_NO_DOWNLOAD=1`, `--network none` and no published port; readiness and one chat request go through `docker exec`, the full log is searched for `download`, `fetch`, `urm.nvidia`, `huggingface.co`, `openaipublic`, `blob.core`, `harmony`, `tiktoken`, `resolve` and `connect`, and the volume is listed and removed afterwards ([measurement B's script](../research/2026-11-17/sglang-runtime/measure-b/scripts/run_netnone_b.sh)).

Measure a new recipe, or an image after a pin bump, with the same sequence, and add its rows to the tables below.

## Post-ready margins

| Date | Recipe and start | Flags | `F_ready` (MiB) | `F_low` (MiB) | Used after ready (MiB) |
|---|---|---|---|---|---|
| 2026-11-17 | qwen3.5-9b, B m12-a3 | The recipe's, with the lane's six variables (no `TIKTOKEN_ENCODINGS_BASE`, no `PYTORCH_ALLOC_CONF`) | 911 | 537 | 374 |
| 2026-11-18 | qwen3.5-9b, D d2-env | The recipe's, with the runtime file's eight variables | 937 flushed (911 unflushed) | 541 | 396 |
| 2026-11-17 | k2-horizon-7b, B b0-a2 | Fraction 0.90 instead of 0.88, six variables | 865 | 532 | 333 |
| 2026-11-18 | k2-horizon-7b, D d1-f088 | The recipe's, eight variables | 1,152 flushed (1,152 unflushed) | 822 | 330 |
| 2026-11-17 | gpt-oss-20b, B b0-a4 | The recipe's, seven variables (no `PYTORCH_ALLOC_CONF`), 32 GiB cap | 1,329 | 989 | 340 |
| 2026-11-17 | gpt-oss-20b, B ld19-mem28 | As b0-a4 under a 28 GiB cap, the cap B proposed for the recipe | 1,328 | 988 | 340 |
| 2026-11-17 | gpt-oss-20b, C pe-round1 | The recipe's, eight variables, 32 GiB cap | 1,346 | 1,022 | 324 |
| 2026-11-17 | gpt-oss-20b, C pe-round2-mem20 | The recipe's, eight variables, the recipe's 20 GiB cap | 1,347 | 1,006 | 341 |
| 2026-11-18 | qwen3.6-35b-a3b, C w3-k88-f088 | Paged Experts with 88 of 256 experts resident, Marlin, fraction 0.88, BF16 KV, 8 Mamba slots, 128k; seven variables (no `TIKTOKEN_ENCODINGS_BASE`), 32 GiB cap | 951 | 465 | 486 |
| 2026-11-18 | qwen3.6-35b-a3b, C w3-k88-f088-mem27 | As w3-k88-f088 under a 27 GiB cap | 951 | 501 | 450 |
| 2026-11-18 | qwen3.6-35b-a3b, E e1-k88-f088-mem27 | As w3-k88-f088-mem27, eight variables | 977 flushed (951 unflushed) | 505 | 472 |
| 2026-11-17 | gemma-4-12b, B m2-fp8-b | Fraction 0.90, `--json-model-override-args '{"audio_config": null}'`, FP8 KV, `--swa-full-tokens-ratio 0.1`, 128k; six variables | 1,099 | 839 | 260 |
| 2026-11-18 | gemma-4-12b, D d2-env | As m2-fp8-b, eight variables | 1,119 flushed (1,099 unflushed) | 841 | 278 |
| 2026-11-17 | gemma-4-e4b, B m5-a4-swa | Fraction 0.85, `--json-model-override-args '{"audio_config": null}'`, `--swa-full-tokens-ratio 0.1`, BF16 KV, 64k; six variables | 1,407 | 1,403 | 4 |
| 2026-11-18 | gemma-4-e4b, D d2-env | As m5-a4-swa, eight variables | 1,923 flushed (1,407 unflushed) | 1,627 | 296 |

All rows are measured on the MSI Raider 18 Max HX, SGLang v0.5.23, on the date given, which is the local date (+03:00) of the start ([measurement B](../research/2026-11-17/sglang-runtime/measure-b/README.md#the-margin-table-the-post-ready-threshold), [measurement C](../research/2026-11-17/sglang-runtime/measure-c/README.md), [measurement D](../research/2026-11-17/sglang-runtime/measure-d/README.md), [measurement E](../research/2026-11-17/sglang-runtime/measure-e/README.md); [margins-bcde.txt](../research/2026-11-17/sglang-runtime/measure-e/lane-logs/margins-bcde.txt) recomputes every row with one script).
The rows are every margin start of measurements B to E at the settings chosen for a recipe, or, for K2 Horizon 7B at fraction 0.90, at the settings it was measured at before; the rule takes the largest drop over all of them.
D's three d2-env starts (Qwen3.5-9B, Gemma 4 12B and Gemma 4 E4B) ran a shorter sequence: the flushed reading, the near-full prompt, the two grammar requests and the repository probe, without M12 or the second near-full prompt ([measurement D](../research/2026-11-17/sglang-runtime/measure-d/README.md)).
"Recipe" names the recipe ID whose settings the start used; a start in B or C has no flushed reading, and a flush can only raise free memory, because it releases cached blocks (inferred from what the flush does).
With both NCCL variables the first grammar request and the forced tool request moved no memory in any start, so every drop is the growth of PyTorch's cache during the near-full prompt plus kernels loaded while serving (measured; the attribution is inferred).
Gemma 4 E4B's drop rose from 4 to 296 MiB only because the flush released 516 MiB of PyTorch's cache that the unflushed prompt in B reused (measured in D; the mechanism is inferred from the readings).

The rule and its result:

- The largest used-after-ready figure is 486 MiB, Qwen3.6-35B-A3B's start w3-k88-f088.
- 486 + 256 = 742 MiB, rounded up to a multiple of 64: `MIN_FREE` = 64 x 12 = 768 MiB, the value of `min_free_at_ready_mib` in `recipes/sglang/_runtime.toml` (estimate from the measured margins).
- Without C's two unflushed Qwen3.6 rows, which E's flushed start replaces, the largest drop is E's 472 MiB, and 472 + 256 = 728 rounds up to the same 768 MiB (estimate).
- C's lowest of 465 MiB against E's flushed 977 MiB would be a drop of exactly 512 MiB; 512 + 256 = 768 is already a multiple of 64 and stays 768 (estimate).
  C's 465 MiB came after the repository and api probes, which E's sequence did not send; the same probes left C's 27 GiB start at 501 MiB (measured).
- The ship gate is 768 + 128 = 896 MiB, and every shipped recipe passes it: Qwen3.5-9B with 937 MiB flushed (911 unflushed), K2 Horizon 7B with 1,152 MiB at fraction 0.88, and gpt-oss-20b with 1,347 MiB unflushed (measured; the gate is an estimate).
- K2 Horizon 7B at fraction 0.90 had 865 MiB and would fail the gate, so its recipe uses 0.88 (measured in B and D).
- The threshold holds only with both NCCL variables of the runtime file; without them the first grammar request takes about 605 MiB after ready ([README.md](README.md#grammar-requests-and-nccl)).

## Results

One row per scenario at each shipped recipe's settings, from the record that its `[evidence] record` names.

| Date | Image | Model and checkpoint | Server flags (non-default) | Context / `token_capacity` | Scenario | Thinking | In / out tokens | First token | Decode tok/s | Peak `memory.used` (MiB) | Power | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-17 | lmsysorg/sglang:v0.5.23-runtime | Qwen3.5-9B, `RedHatAI/Qwen3.5-9B-NVFP4` (NVFP4 W4A16) | Fraction 0.88, 1 running request, 8 Mamba slots, no prefill CUDA graph, BF16 KV; parsers `qwen3_coder` and `qwen3`; six variables | 131072 / 140,981 | Short prompt, up to 1,500 tokens out (3 runs) | Toggle, off (recipe default) | 40 / 664 to 854 | 0.05 s | 93.7 | 15,305 | 145 W limit; 161.0 W in one 1 s sample, 70 C | Every run stopped by itself before 1,500 tokens; [record](../research/2026-11-17/sglang-qwen3.5-9b/measure-b/README.md) |
| 2026-11-17 | same | Qwen3.5-9B | same | 131072 / 140,981 | 8k fresh prompt, then the same prompt again | Toggle, off | 8,591 / 46 | 0.572 s, repeat 0.063 s | | 15,305 | | Prefix cache hit on the repeat |
| 2026-11-17 | same | Qwen3.5-9B | same | 131072 / 140,981 | 30k prompt with the 8k prefix, then again | Toggle, off | 32,153 / 51 | 2.398 s, repeat 0.052 s | | 15,305 | | vLLM v0.33.0: 2.424 s |
| 2026-11-17 | same | Qwen3.5-9B | same | 131072 / 140,981 | Fresh near-full prompt (95%), no flush before it | Toggle, off | 124,499 / 32 | 24.38 s for the whole request | | 15,305 | | 911 to 545 MiB free |
| 2026-11-17 | same | Qwen3.5-9B | same | 131072 / 140,981 | M12: 300 s of load, prompts up to 13,448 tokens | Toggle, off | 66 requests, 256 or 512 out | | 88.0 / 92.0 / 93.7 (min / median / max) | 15,305 | | No error; halves 92.0 and 92.0 |
| 2026-11-18 | lmsysorg/sglang:v0.5.23-runtime | K2 Horizon 7B, `cyankiwi/K2-Horizon-7B-AWQ-INT4` (AWQ-INT4) | Fraction 0.88, 1 running request, no prefill CUDA graph, FP8 KV, `--trust-remote-code`; parsers `k2_horizon`; eight variables | 65536 / 77,988 | Short prompt, up to 1,500 tokens out (3 runs) | Always, high | 36 / 1,286 to 1,500 | 0.024 to 0.042 s | 89.6 | 15,020 | 140 W limit; 147.0 W in 1 s samples, 74 C | B at fraction 0.90: 89.5 tok/s; [record](../research/2026-11-18/sglang-k2-horizon-7b/measure-d/README.md) |
| 2026-11-18 | same | K2 Horizon 7B | same | 65536 / 77,988 | 8k fresh prompt, then the same prompt again | Always, high | 8,085 / 64 | 2.196 s, repeat 0.031 s | | 15,020 | | Prefix cache hit on the repeat |
| 2026-11-18 | same | K2 Horizon 7B | same | 65536 / 77,988 | 30k prompt with the 8k prefix, then again | Always, high | 30,261 / 64 | 9.233 s, repeat 0.056 s | | 15,020 | | vLLM v0.33.0: 10.869 s |
| 2026-11-18 | same | K2 Horizon 7B | same | 65536 / 77,988 | Fresh near-full prompt (95%), after the flush | Always, high | 62,254 / 32 | 33.22 s for the whole request | | 15,020 | | 1,152 to 828 MiB free |
| 2026-11-18 | same | K2 Horizon 7B | same | 65536 / 77,988 | M12: 300 s of load, prompts up to 13,101 tokens | Always, high | 51 requests, 256 or 512 out | | 78.3 / 86.5 / 89.7 (min / median / max) | 15,020 | | No error; halves 88.3 and 86.4 |
| 2026-11-17 | lmsysorg/sglang:v0.5.23-runtime | gpt-oss-20b, `openai/gpt-oss-20b` (MXFP4) | Paged Experts with 24 of 32 experts resident, Marlin MoE runner, fraction 0.90, 1 running request, FP8 KV, `--swa-full-tokens-ratio 0.1`, 1,024-token prefill chunks, no prefill CUDA graph; parsers `gpt-oss`; eight variables; 20 GiB cap | 131072 / 169,853 | Short prompt, 1,500 tokens out (3 runs) | Effort, none sent | 93 / 1,500 | 0.095 to 0.102 s | 124.6 | 14,836 | 140 W limit; 146.1 W in 1 s samples, 73 C | B without `PYTORCH_ALLOC_CONF`: 125.5 tok/s; [record](../research/2026-11-17/sglang-gpt-oss-20b/measure-c/README.md) |
| 2026-11-17 | same | gpt-oss-20b | same | 131072 / 169,853 | 8k fresh prompt, then the same prompt again | Effort, none sent | 8,058 / 64 | 1.472 s, repeat 0.057 s | | 14,836 | | Prefix cache hit on the repeat |
| 2026-11-17 | same | gpt-oss-20b | same | 131072 / 169,853 | 30k prompt with the 8k prefix, then again | Effort, none sent | 30,003 / 64 | 4.935 s, repeat 0.084 s | | 14,836 | | llama.cpp b12407: 4.05 s |
| 2026-11-17 | same | gpt-oss-20b | same | 131072 / 169,853 | Fresh near-full prompt (95%), no flush before it | Effort, none sent | 124,459 / 32 | 44.27 s for the whole request | | 14,836 | | 1,347 to 1,006 MiB free |
| 2026-11-17 | same | gpt-oss-20b | same | 131072 / 169,853 | M12: 300 s of load | Effort, none sent | 73 requests, 256 or 512 out | | 118.0 / 126.2 / 132.9 (min / median / max) | 14,836 | | No error; 17.04 GiB cgroup peak under the 20 GiB cap |

All rows are measured on the MSI Raider 18 Max HX, SGLang v0.5.23, on the date given (the probe output in each record's `shapes/<run>-probe.json`, the sequence in `shapes/<run>-seq.jsonl`, the load in `shapes/<run>-sustained.json`).
"Six variables" are `HF_HUB_OFFLINE`, `HF_MODULES_CACHE`, `MAX_JOBS`, `FLASHINFER_NO_DOWNLOAD`, `SYNC_TOKEN_IDS_ACROSS_TP` and `NCCL_MAX_P2P_NCHANNELS`; "eight variables" add `TIKTOKEN_ENCODINGS_BASE` and `PYTORCH_ALLOC_CONF`, as the runtime file sets them.
Peak `memory.used` is the start's peak over the whole sequence; the first-token times are client-side, and the near-full rows give the whole request's time, which is what the sequence records.
The vLLM and llama.cpp comparisons come from [vLLM benchmarking](../vllm/benchmarking.md#results) and [llama.cpp benchmarking](../llamacpp/benchmarking.md).

## Cold starts without the network

| Date | Recipe and start | `/ready` 200 | Decode-graph capture, cold (warm) | Free after ready | Chat request | Download or fetch lines | Volume afterwards |
|---|---|---|---|---|---|---|---|
| 2026-11-17 | qwen3.5-9b, A m15-netnone | 46.6 s ready line, `/health` 200 at 47.26 s | 10.05 to 10.34 s over A's two cold starts (2.27 to 2.28 s) | not read | 200, `OK` | none | 397 files, removed |
| 2026-11-17 | qwen3.5-9b, B m15-netnone | 48.33 s | 10.01 s (2.38 s) | 607 MiB | 200, `OK` | none | 357 files, 18 MB, removed |
| 2026-11-17 | k2-horizon-7b, B m15-netnone (fraction 0.90) | 137.87 s | 95.26 s (1.21 s) | 865 MiB | 200, `OK` | none | 155 files, 11 MB, removed |
| 2026-11-17 | gpt-oss-20b, B b0-netnone-tik | 253.58 s | 185.09 s (1.72 s) | 1,329 MiB | 200, `OK` | none | 249 files, 23 MB, removed |
| 2026-11-17 | gpt-oss-20b, B b0-netnone-notik, without `TIKTOKEN_ENCODINGS_BASE` | 254.62 s | 185.31 s (1.72 s) | not read | 200, `OK` | one: `OpenAI Responses API (/v1/responses) disabled`, from the failed vocabulary download | removed |
| 2026-11-18 | qwen3.6-35b-a3b, C m15-netnone and m15-netnone-b | 345.46 s and 351.21 s | not read | 687 MiB in both, 969 MiB after the flush in m15-netnone-b | 200, `OK` (m15-netnone) | none | removed |

All rows are measured on the MSI Raider 18 Max HX, SGLang v0.5.23, on the date given, with `--network none`, a new empty `sglang-measure-cold` volume and `FLASHINFER_NO_DOWNLOAD=1` ([measurement A](../research/2026-11-17/sglang-runtime/measure-a/README.md), [measurement B](../research/2026-11-17/sglang-runtime/measure-b/README.md#m15-cold-starts-without-a-network), [Qwen3.6-35B-A3B record](../research/2026-11-18/sglang-qwen3.6-35b-a3b/README.md)); B's starts ran with the lane's variables, without `PYTORCH_ALLOC_CONF`.

- No start reached the network for a model file, a kernel or a vocabulary once `TIKTOKEN_ENCODINGS_BASE` was set: the searched patterns matched only `server_args`, Gloo's `Unable to resolve hostname` warning and the prefill-graph line (measured).
- The cold cost is the first decode-graph capture, which compiles kernels into the empty volume, and for Qwen3.6-35B-A3B about 257 s of FlashInfer's autotune; the warm volume brings later starts back to 27 to 64 s, apart from measurement A's two gpt-oss-20b starts with a network and without `TIKTOKEN_ENCODINGS_BASE`, at 106 and 116 s (measured; [measurement A](../research/2026-11-17/sglang-runtime/measure-a/README.md)).
- The slowest cold start, 351.2 s, is below half of the runtime file's `start_timeout_s` of 900 (measured; the rule of twice the slowest start is the design's).
- A cold start reads less free memory right after ready until a flush, as Qwen3.5-9B's 607 MiB and Qwen3.6-35B-A3B's 687 and 969 MiB show; llmctl's check reads after the flush (measured).
