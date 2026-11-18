# DeepSeek-R1-0528-Qwen3-8B on SGLang v0.5.23 (M4, M9): out of GPU memory while building the model

Last verified: 2026-11-17.

Result: the vLLM recipe's BF16 checkpoint does not start on SGLang v0.5.23 with `--quantization fp8`, because SGLang creates every linear weight in BF16 before it quantizes, and the BF16 model (15.26 GiB) does not fit in the 15.06 GiB that CUDA leaves free (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
M9 (a request with tools to the chat-only recipe) was not run, because no SGLang chat-only recipe loads; no draft recipe is written.

## The start

- Checkpoint: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` at `6e8885a6ff5c1dc5201574c8fd700323f23c25fa`, the vLLM recipe's revision, from the overlay `$SCRATCH/sglang/measure-b/overlay/deepseek-r1-0528-qwen3-8b/`, whose `generation_config.json` holds only the vLLM recipe's temperature 0.6 and top_p 0.95, because the snapshot has no such file.
- Settings as the brief asks: `--quantization fp8`, FP8 KV, fraction 0.90, `--disable-prefill-cuda-graph`, `--max-running-requests 1`, `--reasoning-parser deepseek-r1`, no tool parser, context 65536, and the lane environment ([cmds/m4-a1.sh](cmds/m4-a1.sh)).
- The log auto-detected `reasoning_parser=deepseek-r1, tool_call_parser=deepseekv31` from the template, took the memory baseline with 15.06 GB available, and failed in `Load weight begin` while it was still creating the model: the last frames are `loader.py:309 in _initialize_model`, `models/qwen3.py:582 in __init__`, `vocab_parallel_embedding.py:577 in __init__` and `quantization/unquant.py:520 in create_weights`, that is, the `lm_head` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, [logs/m4-a1.log](logs/m4-a1.log), [logs/m4-a1.full.log](logs/m4-a1.full.log)).
- The container exited 22 s after `docker run`; the 1 s trace never saw more than 424 MiB in use, so the allocations happened between two samples (measured, [logs/m4-a1-trace.csv](logs/m4-a1-trace.csv)).

```text
CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 0 while trying to allocate 1245708288 bytes (free: 962723840, total: 16611278848).
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has a total capacity of 15.47 GiB of which 918.12 MiB is free. Including non-PyTorch memory, this process has 14.46 GiB memory in use. Of the allocated memory 14.16 GiB is allocated by PyTorch, and 1.39 MiB is reserved by PyTorch but unallocated.
```

## The arithmetic that explains it

- For a checkpoint that is not FP8 already, SGLang's FP8 linear method creates the weight in the model's dtype: `weight_dtype = (torch.float8_e4m3fn if is_checkpoint_fp8_serialized else params_dtype)` (`../offline/sglang-src/fp8.py`, lines 653 to 656), and quantizes it only after loading (inferred from the source).
- The BF16 model is 16,388,927,770 bytes, 15.26 GiB, by the vLLM recipe's `size_bytes`; the 14.16 GiB allocated before the `lm_head` are the 36 decoder layers and the embedding, and the `lm_head` is 151,936 x 4,096 x 2 bytes = 1,245,708,288 bytes, exactly the failed allocation (arithmetic from `config.json`, checked against the log).
- 14.16 + 1.16 = 15.32 GiB exceeds the 15.06 GiB free before loading at any `--mem-fraction-static`, so 0.905 cannot help, and no smaller context or KV setting changes a failure that happens before the pool is sized (inferred).
- The way around it in SGLang, `--load-format layered` ("loads weights layer by layer so that one can quantize a layer before loading another"), needs a `load_weights_to_module` method, which in the v0.5.23 image only `models/torch_native_llama.py` has, not `models/qwen3.py` (inferred from an offline search of the image; not run).
- vLLM's `fp8_per_tensor` quantizes while it loads and needs 9.12 GiB (`research/2026-11-17/vllm-deepseek-r1-0528-qwen3-8b/`), which is where the report's "about 9.1 GiB as on vLLM" came from; on SGLang that estimate does not hold.

## What stays open

- The token-count comparison with llama.cpp and the garbled-reply check of M4 were not reached on SGLang; the vLLM retry in [../vllm-deepseek-r1-0528-qwen3-8b/](../../vllm-deepseek-r1-0528-qwen3-8b-retry/measure-b.md) measured both on vLLM.
- An FP8 checkpoint of this model, which SGLang would load at about 9 GiB, is not cached, and this lane downloads nothing; the cached Q8_0 GGUF of the llama.cpp recipe was not tried, because the brief names the BF16 checkpoint.
- M9: SGLang's behaviour for tools sent to a server without `--tool-call-parser` (report section 6.4 infers HTTP 200 with any call left in `content`) stays unmeasured; it matters only once an SGLang chat-only recipe exists.

## Files

- [cmds/m4-a1.sh](cmds/m4-a1.sh): the `docker run` command (reads `$HOME` and `$SCRATCH`).
- [logs/](logs/): the cut log, the full log, the container state, the trace and the run metadata.
