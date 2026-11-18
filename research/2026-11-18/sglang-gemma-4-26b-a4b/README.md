# Gemma 4 26B-A4B IT QAT AWQ-INT4 on SGLang, 2026-11-18

Last verified: 2026-11-18.

Result: failed in measurement C: with Paged Experts the experts of this int4 build run only through SGLang's Marlin MoE runner, which has no gated-GELU path, so the one start that loaded died capturing its decode graph with `AssertionError: Only gated SiLU/SiTU is supported, got gelu.` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
The three starts before it failed while loading, on the vision tower, which only a `quantization_config` override got past; unpaged the 16.01 GiB checkpoint exceeds the 13.55 GiB that fraction 0.90 gives to weights and KV together (estimate from the shard headers).
No SGLang recipe is drafted; the starts ran by hand with `docker run` before llmctl had an SGLang recipe.

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4` at revision `76373001e106330a6815f237ef00838c7fed9356`, served as `gemma-4-26b-a4b`.
Settings of m11-fp8-k56-ign ([measure-c/cmds/m11-fp8-k56-ign.sh](measure-c/cmds/m11-fp8-k56-ign.sh)): context 131072, `--mem-fraction-static 0.90`, FP8 KV with `--swa-full-tokens-ratio 0.1`, `--max-running-requests 1`, `--disable-prefill-cuda-graph`, `--moe-runner-backend marlin --enable-paged-experts` with 56 of 128 experts resident, `--tool-call-parser gemma4` and `--reasoning-parser gemma4`, and `--json-model-override-args` with the checkpoint's `quantization_config` and a wider `ignore` list, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only, a model overlay whose `generation_config.json` carries temperature 1.0, top_p 0.95 and top_k 64 (the llama.cpp recipe's values), and measurement C's environment with `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`.
The reports are [measurement C's README for this model](measure-c/README.md) and [measurement C's lane README](../../2026-11-17/sglang-runtime/measure-c/README.md), where every figure below comes from.

## Measurement C

- m11-fp8-k56, m11-fp8-k64 and m11-fp8-k72 died while loading (K 56 33 s and K 72 31 s after `docker run`) with `size_n = 8608 is not divisible by tile_n_size = 64` in the Marlin repack of a dense linear layer (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- The cause: v0.5.23 builds Gemma 4's vision tower whatever the config says, and the checkpoint's `ignore` list names the vision projections with a `.linear` suffix that SGLang's fused shard names do not match, so the BF16 vision MLP is treated as W4A16 (inferred from the source and the config).
- m11-fp8-k56-ign, with the override, loaded (10.38 GB of weights, FP8 pools of 160,936 full and 16,093 sliding-window tokens) and died 89 s into the decode-graph capture on the gated-GELU assertion (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- The pinned expert store did build in that start: 11.98 GiB of shared memory for 11.964 GiB of checkpoint expert tensors; anonymous memory peaked at 15.72 GiB while loading and the GPU at 14,306 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- What would make it run is a gated-GELU path in SGLang's Marlin MoE runner and an `ignore` match for the `.linear` vision names; `nvidia/Gemma-4-26B-A4B-NVFP4` would meet the same runner (inferred from the source, not run).

## Files

| Path | Holds |
|---|---|
| [measure-c/README.md](measure-c/README.md) | Measurement C's report on this model, links rewritten |
| [measure-c/links.diff](measure-c/links.diff) | The link rewrite of that report against the scratch copy |
| [measure-c/cmds/](measure-c/cmds/) | The `docker run` of the starts m11-fp8-k56, m11-fp8-k64, m11-fp8-k72 and m11-fp8-k56-ign; they read `$HOME` and `$SCRATCH`, and measurement C rewrote `m11-fp8-k72.sh` after its start with the same helper and arguments |
| [measure-c/logs/](measure-c/logs/) | Per start the cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace, the readiness poll, the state, the metadata and the `/dev/shm` readings |
| [measure-c/shapes/](measure-c/shapes/) | The readiness record of each start, none of which became ready |
| [measure-c/overlay/](measure-c/overlay/) | The overlay's generation config before and after, and its link count |
