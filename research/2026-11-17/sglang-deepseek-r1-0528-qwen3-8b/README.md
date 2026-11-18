# DeepSeek-R1-0528-Qwen3-8B on SGLang, 2026-11-17

Last verified: 2026-11-18.

Result: failed in measurement B: with `--quantization fp8` SGLang v0.5.23 creates every linear weight of the BF16 checkpoint in BF16 before it quantizes, and the 15.26 GiB BF16 model does not fit in the 15.06 GiB that CUDA leaves free, so the start ran out of GPU memory before loading (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; the mechanism inferred from the source).
No SGLang recipe is drafted, and M9 (a request with tools to a chat-only recipe) was not run, because no SGLang chat-only recipe loads; the hands-on run and measurement A did not run this model.

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` at revision `6e8885a6ff5c1dc5201574c8fd700323f23c25fa`, the vLLM recipe's revision, served as `deepseek-r1-8b`.
Settings of m4-a1 ([measure-b/cmds/m4-a1.sh](measure-b/cmds/m4-a1.sh)): context 65536, `--quantization fp8`, FP8 KV, `--mem-fraction-static 0.90`, `--disable-prefill-cuda-graph`, `--max-running-requests 1`, `--reasoning-parser deepseek-r1` and no tool parser, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only, a model overlay whose `generation_config.json` holds only temperature 0.6 and top_p 0.95 (the vLLM recipe's values), because the snapshot has no such file, and measurement B's environment.
The reports are [measurement B's README for this model](measure-b/README.md) and [measurement B's lane README](../sglang-runtime/measure-b/README.md), where every figure below comes from.

## Measurement B

- The log took the memory baseline with 15.06 GB available and failed while creating the model, in the `lm_head`, with `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB.`; the container exited 22 s after `docker run` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The failed allocation is the BF16 `lm_head`, 151,936 x 4,096 x 2 = 1,244,659,712 bytes, which PyTorch's CUDA caching allocator requested as 1,245,708,288 bytes, rounded up to whole 2 MiB blocks ([measure-b/logs/m4-a1.full.log](measure-b/logs/m4-a1.full.log) line 39; the rounding inferred from the two figures), after 14.16 GiB of BF16 layers, so no fraction, context or KV setting helps (arithmetic from `config.json`, checked against the log).
  Measurement B's report ([measure-b/README.md](measure-b/README.md) line 23), archived unchanged, gives the rounded 1,245,708,288 bytes as the product.
- `--load-format layered` would quantize layer by layer, but the v0.5.23 image implements it only for `models/torch_native_llama.py`, not `models/qwen3.py` (inferred from an offline search of the image; not run).
- On vLLM the same checkpoint loads at 9.12 GiB, because vLLM's `fp8_per_tensor` quantizes while it loads ([vllm-deepseek-r1-0528-qwen3-8b/](../vllm-deepseek-r1-0528-qwen3-8b/README.md)); measurement B's vLLM tokenizer retry is in [vllm-deepseek-r1-0528-qwen3-8b-retry/](../vllm-deepseek-r1-0528-qwen3-8b-retry/README.md).

## Files

| Path | Holds |
|---|---|
| [measure-b/README.md](measure-b/README.md) | Measurement B's report on this model, links rewritten |
| [measure-b/links.diff](measure-b/links.diff) | The link rewrite of that report against the scratch copy |
| [measure-b/cmds/m4-a1.sh](measure-b/cmds/m4-a1.sh) | The `docker run` of the start; it reads `$HOME` and `$SCRATCH` |
| [measure-b/logs/](measure-b/logs/) | The cut and full logs, the `docker run` output, the readiness poll, the container state, the 1 s GPU and cgroup trace and the metadata |
| [measure-b/shapes/m4-a1-ready.json](measure-b/shapes/m4-a1-ready.json) | The readiness record of the start, which never became ready |
| [measure-b/overlay/](measure-b/overlay/) | The overlay's generation config before and after (none in the snapshot), and its link count |
