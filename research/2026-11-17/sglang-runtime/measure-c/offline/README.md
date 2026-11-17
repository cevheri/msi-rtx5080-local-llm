# Files read offline from the images (measurement C)

Last verified: 2026-11-17.

Every file here was copied or printed by a throwaway container started with `docker run --rm --network none --memory 2g --memory-swap 2g --user <uid>:<gid> -e HOME=/tmp --entrypoint sh` (or `python3`), no GPU, from `lmsysorg/sglang:v0.5.23-runtime` or `vllm/vllm-openai:v0.33.0`; they are vendor code, read as data and cited by path and line.

| File | From | Used for |
|---|---|---|
| `sglang-src/store.py`, `method.py`, `formats.py`, `runners.py`, `executor.py`, `residency.py`, `__init__.py` | `/sgl-workspace/sglang/python/sglang/srt/layers/moe/paged_experts/` | How the pinned expert store is allocated and repacked |
| `sglang-src/mxfp4.py`, `marlin_utils_fp4.py`, `modelopt_quant.py`, `compressed_tensors_wNa16_moe.py` | `srt/layers/quantization/` | The Marlin layouts (padding) of the paged tensors |
| `torch-src/ATen_core_CachingHostAllocator.h`, `ATen_cuda_CachingHostAllocator.h`, `c10_core_AllocatorConfig.h`, `c10_cuda_CUDAAllocatorConfig.h`, `c10_core_CachingDeviceAllocator.h` | `/opt/sglang/lib/python3.12/site-packages/torch/include/` (PyTorch 2.14.1+cu130) | The pinned host allocator's power-of-two rounding and its thresholds |
| `torch-libc10-pinned-keys.txt` | `torch/lib/libc10.so` of the same image | The configuration keys the image's PyTorch accepts |
| `sglang-alloc-conf-search.txt` | the image's `sglang` package and environment | Whether SGLang or the image sets an allocator variable |
| `sglang-parser-names.txt` | `srt/function_call/parser_names.py`, `srt/parser/reasoning_parser_names.py`, the detector maps | The wave 3 parser names, which equal measurement B's lists |
| `vllm-src/gpu_worker.py` | `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py` | How vLLM sizes the KV cache, and `--kv-cache-memory-bytes` |
