# Groot-19B-A3B NVFP4 on SGLang v0.5.23 (M3): failed after loading, in the FP4 Marlin preparation

Last verified: 2026-11-17.

Result: the vLLM recipe's NVFP4 checkpoint does not start on SGLang v0.5.23: the weights load, then preparing the weight-only NVFP4 linears for the FP4 Marlin kernel raises `AttributeError: 'CompressedTensorsConfig' object has no attribute 'group_size'` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
No draft recipe is written; Groot stays on vLLM and llama.cpp.

## The start

- Checkpoint: `GrootTeam/Groot-19B-A3B-261023-NVFP4` at `8ff73aca9300f4e9835972f0012ef5edbb016a8b`, the vLLM recipe's revision, from the overlay `$SCRATCH/sglang/measure-b/overlay/groot-19b-a3b/` (temperature 0.6, top_p 0.95, top_k 20, the vLLM recipe's values).
- Settings as the brief asks: fraction 0.90, FP8 KV, `--max-mamba-cache-size 4`, `--disable-prefill-cuda-graph`, `--max-running-requests 1`, parsers `qwen3_coder` and `qwen3`, `--default-chat-template-kwargs '{"enable_thinking":false}'`, context 131072, and the lane environment ([cmds/m3-a1.sh](cmds/m3-a1.sh)).
- The log: FlashInfer attention by default, `Multimodal data loading enabled`, `Using triton_attn as multimodal attention backend` (the vision tower loads), `Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod`, `Using CompressedTensorsW4A16Nvfp4MoE` for the experts, and then `--fp4-gemm-backend flashinfer_cutlass has no bf16 x fp4 GEMM; serving NVFP4 weight-only linears with FP4 Marlin.` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, [logs/m3-a1.log](logs/m3-a1.log)).
- So the FP4 kernel the log names is FP4 Marlin for the weight-only NVFP4 linears, chosen because the `auto` FP4 GEMM backend resolved to `flashinfer_cutlass`, which has no BF16-by-FP4 GEMM; the experts use the W4A16 NVFP4 MoE method (measured, the log lines above).
- The scheduler died 14 s after `Load weight begin`, in `loader.py:1089 in postprocess_weights`, `compressed_tensors.py:1294 in process_weights_after_loading`, `compressed_tensors_w4a16_nvfp4.py:246 in process_weights_after_loading` and `marlin_utils_fp4.py:142 in prepare_nvfp4_layer_for_marlin`; the container exited 36 s after `docker run`, with the GPU at 13,162 MiB at most ([logs/m3-a1.full.log](logs/m3-a1.full.log), [logs/m3-a1-trace.csv](logs/m3-a1-trace.csv)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

```text
AttributeError: 'CompressedTensorsConfig' object has no attribute 'group_size'
```

## Why

- The checkpoint's `quantization_config` is compressed-tensors `mixed-precision`: `group_0` quantizes the attention and linear-attention projections to FP8 W8A8 per tensor, and `group_1` the routed experts, the shared expert and `lm_head` to NVFP4 weight-only (`nvfp4-pack-quantized`, 4-bit float, `group_size` 16, no input activations) (vendor-reported, the cached `config.json`).
- The dense weight-only NVFP4 layers (the shared expert and `lm_head`) take `CompressedTensorsW4A16Nvfp4`, whose `process_weights_after_loading` renames `weight_packed` and calls `prepare_nvfp4_layer_for_marlin(layer)` (`../offline/sglang-src/compressed_tensors_w4a16_nvfp4.py`, lines 240 to 246) (inferred from the source).
- `prepare_nvfp4_layer_for_marlin` reads `layer.quant_config.group_size` whenever the layer has a `quant_config` (`../offline/sglang-src/marlin_utils_fp4.py`, lines 140 to 144), and the compressed-tensors config object has no such attribute, so every compressed-tensors NVFP4 weight-only dense layer fails here on a GPU that takes the Marlin path (inferred from the source and the traceback).
- No flag avoids it: weight-only NVFP4 needs a BF16-by-FP4 GEMM, which the log says `flashinfer_cutlass` lacks, so the dense layers go to Marlin whatever `--fp4-gemm-backend` names (inferred from the log line; other backends not run).
- vLLM v0.33.0 serves this checkpoint at 128k with 10.99 GiB of weights and 193.4 tok/s (`research/2026-11-17/vllm-groot-19b-a3b/`), so the files are fine.

## What the pool would have been

- The GPU held 13,162 MiB when the start died after loading, about 12.4 GiB of weights plus the CUDA context, against the report's estimate of 11.9 GiB (measured peak; the split is an estimate from the CUDA context of about 0.8 GiB that other starts show).
- At fraction 0.90 the static budget is 15.06 x 0.90 = 13.55 GiB, which would leave about 1.15 GiB for KV, 4 Mamba slots (about 60 MiB each) and the 0.10 GiB multimodal reserve, so about 0.8 GiB of FP8 KV, roughly 80k tokens at 10 KiB per token rather than 128k (estimate from `config.json`: 10 full-attention layers, 2 KV heads of 256).

## Files

- [cmds/m3-a1.sh](cmds/m3-a1.sh): the `docker run` command (reads `$HOME` and `$SCRATCH`).
- [logs/](logs/): the cut log, the full log, the container state, the trace and the run metadata.
