# Groot-19B-A3B NVFP4 on SGLang, 2026-11-17

Last verified: 2026-11-17.

Result: failed in measurement B: the vLLM recipe's NVFP4 checkpoint loads its weights on SGLang v0.5.23, then preparing the weight-only NVFP4 linears for the FP4 Marlin kernel raises `AttributeError: 'CompressedTensorsConfig' object has no attribute 'group_size'` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
No SGLang recipe is drafted; the hands-on run and measurement A did not run this model.

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `GrootTeam/Groot-19B-A3B-261023-NVFP4` at revision `8ff73aca9300f4e9835972f0012ef5edbb016a8b`, the vLLM recipe's revision, served as `groot-19b-a3b`.
Settings of m3-a1 ([measure-b/cmds/m3-a1.sh](measure-b/cmds/m3-a1.sh)): context 131072, `--mem-fraction-static 0.90`, FP8 KV, `--max-mamba-cache-size 4`, `--disable-prefill-cuda-graph`, `--max-running-requests 1`, parsers `qwen3_coder` and `qwen3` and `--default-chat-template-kwargs '{"enable_thinking":false}'`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only, a model overlay whose `generation_config.json` carries temperature 0.6, top_p 0.95 and top_k 20 (the vLLM recipe's values), and measurement B's environment.
The reports are [measurement B's README for this model](measure-b/README.md) and [measurement B's lane README](../sglang-runtime/measure-b/README.md), where every figure below comes from.

## Measurement B

- The log chose FlashInfer attention, loaded the vision tower, used `CompressedTensorsW4A16Nvfp4MoE` for the experts and logged `--fp4-gemm-backend flashinfer_cutlass has no bf16 x fp4 GEMM; serving NVFP4 weight-only linears with FP4 Marlin.` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The scheduler died 14 s after `Load weight begin` in `prepare_nvfp4_layer_for_marlin`, and the container exited 36 s after `docker run` with the GPU at 13,162 MiB at most (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- `prepare_nvfp4_layer_for_marlin` reads `layer.quant_config.group_size`, which the compressed-tensors config object lacks, so every compressed-tensors NVFP4 weight-only dense layer fails there on a GPU that takes the Marlin path, and no `--fp4-gemm-backend` avoids it (inferred from the source and the log; other backends not run).
- vLLM v0.33.0 serves this checkpoint at 128k at 193.4 tok/s (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [vllm-groot-19b-a3b/](../vllm-groot-19b-a3b/README.md)), so the files are fine.
- Had it started, fraction 0.90 would have left about 0.8 GiB of FP8 KV, roughly 80k tokens rather than 128k (estimate from `config.json` and the measured peak).

## Files

| Path | Holds |
|---|---|
| [measure-b/README.md](measure-b/README.md) | Measurement B's report on this model, links rewritten |
| [measure-b/links.diff](measure-b/links.diff) | The link rewrite of that report against the scratch copy |
| [measure-b/cmds/m3-a1.sh](measure-b/cmds/m3-a1.sh) | The `docker run` of the start; it reads `$HOME` and `$SCRATCH` |
| [measure-b/logs/](measure-b/logs/) | The cut and full logs, the `docker run` output, the readiness poll, the container state, the 1 s GPU and cgroup trace and the metadata |
| [measure-b/shapes/m3-a1-ready.json](measure-b/shapes/m3-a1-ready.json) | The readiness record of the start, which never became ready |
| [measure-b/overlay/](measure-b/overlay/) | The overlay's generation config before and after, and its link count |
