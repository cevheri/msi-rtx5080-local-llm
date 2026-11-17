# Gemma 4 26B-A4B IT QAT AWQ-INT4 on SGLang v0.5.23 with Paged Experts (wave 3): fails

Last verified: 2026-11-18.

Result: failed: `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4` does not run on SGLang v0.5.23, because Paged Experts serves compressed-tensors W4A16 experts only through the Marlin MoE runner, which supports gated SiLU and non-gated SiLU or ReLU² while Gemma 4's experts are gated GELU: the server died capturing its decode graph with `AssertionError: Only gated SiLU/SiTU is supported, got gelu.` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
Before that it failed while loading, on the vision tower, which only a `quantization_config` override got past; no draft is written.
Unpaged it does not fit: the 16.01 GiB checkpoint (11.96 GiB of experts, 4.04 GiB of the rest, the BF16 vision tower included) is more than the 13.55 GiB that fraction 0.90 gives to weights and KV together (estimate from the shard headers and report section 2.6's budget).

The four starts used `lmsysorg/sglang:v0.5.23-runtime` with the lane's settings ([../sglang-qwen3.6-35b-a3b/README.md](../../sglang-qwen3.6-35b-a3b/measure-c/README.md) lists them), C's overlay of the pinned snapshot `76373001e106330a6815f237ef00838c7fed9356` (temperature 1.0, top_p 0.95 and top_k 64, the llama.cpp twin's flags), `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`, fraction 0.90, `--max-running-requests 1`, `--disable-prefill-cuda-graph`, Marlin, Paged Experts, FP8 KV with `--swa-full-tokens-ratio 0.1` (the gemma-4-12b draft's KV settings), 128k, parsers `gemma4` ([cmds/](cmds/)).

| Start | Change | Outcome | Where |
|---|---|---|---|
| m11-fp8-k56 (21:34 UTC) | K 56 of 128 | Died 33 s after `docker run`: `RuntimeError: Failed at /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/gemm/marlin/gptq_marlin_repack.cuh:311: size_n = 8608 is not divisible by tile_n_size = 64` | `compressed_tensors_wNa16.py:250` `transform_w_q`, in `process_weights_after_loading` of a dense linear layer |
| m11-fp8-k64 (21:35 UTC) | K 64 | The same error | as above |
| m11-fp8-k72 (21:35 UTC) | K 72 | The same error, 31 s after `docker run`; the chain was stopped after it ([../lane-logs/chain-n1.txt](../../../2026-11-17/sglang-runtime/measure-c/lane-logs/chain-n1.txt)) | as above |
| m11-fp8-k56-ign (21:36 UTC) | K 56 plus `--json-model-override-args` with the checkpoint's `quantization_config` whose `ignore` list is `lm_head`, `re:.*vision_tower.*`, `re:.*embed_vision.*`, `re:.*\.mlp\.(gate\|up\|down)_proj$` and `re:.*\.router\.proj$` | Loaded (`Load weight end. elapsed=16.02 s ... mem usage=10.38 GB`; FP8 pools of 160,936 full and 16,093 sliding-window tokens, 3.07 GB), then died 89 s into the decode-graph capture: `AssertionError: Only gated SiLU/SiTU is supported, got gelu.` | `layers/moe/moe_runner/marlin.py`, lines 142 to 146, called through `paged_experts/formats.py` line 318 |

All rows are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 (`logs/<run>.full.log`, `logs/<run>.log`, `logs/<run>.state.txt`; every container ended with `oom=false`).
- Why the first error: v0.5.23 builds Gemma 4's vision tower whatever the config says (`gemma4_mm.py` builds `Gemma4VisionEncoder` unconditionally, unlike the audio tower) and passes it the checkpoint's quantization config; the checkpoint's `ignore` list names the vision projections with a `.linear` suffix (its `ClippableLinear` wrappers, 27 layers x 7 entries), which SGLang's fused `qkv_proj` and `gate_up_proj` shard names do not match, so SGLang treats the BF16 vision MLP as W4A16 and Marlin cannot repack its 2 x 4,304 = 8,608 output rows (inferred from the source, `compressed_tensors/utils.py` `should_ignore_layer`, and the config; the vision intermediate size 4,304 is the config's).
- Why the second: `formats.py` runs compressed-tensors W4A16 experts through `_ThroughMarlinRunner`, and `fused_experts_none_to_marlin` asserts gated SiLU (or non-gated SiLU or ReLU²) (`../offline/sglang-src/moe_runner_marlin.py`, lines 142 to 149); Gemma 4's `hidden_activation` is `gelu_pytorch_tanh` (inferred from the source and the config).
- The pinned store did build in the third start: cgroup `shmem` reached 11.98 GiB for the checkpoint's 11.964 GiB of expert tensors, so with the pinned-store fix the factor was 1.001 here; anonymous memory peaked at 15.72 GiB while loading, and the GPU at 14,306 MiB (measured).
- The estimate before the first start was 18.1 GiB with the factor 1.096, or 27.0 GiB with B's 1.84, both under 34 GiB ([../estimates/README.md](../../../2026-11-17/sglang-runtime/measure-c/estimates/README.md)) (estimate).
- `/dev/shm`: each start left one root-owned `sglang_loads_*.shm`, and m11-fp8-k72 also a `sem.mp-f1grp8l_` of 32 bytes, which C removed after each container was gone (`logs/<run>.devshm.txt`) (measured).

What would make it run: a Gemma 4 GELU path in SGLang's Marlin MoE runner (or Paged Experts on a runner that has one), and an `ignore` match for the `.linear` vision names; the downloads lane's fallback, `nvidia/Gemma-4-26B-A4B-NVFP4` (17.53 GiB, not downloaded), would meet the same Marlin runner through the ModelOpt NVFP4 format (inferred from `formats.py` and `marlin.py`, not run).
For the matrix: "Failed: Paged Experts' Marlin MoE runner has no gated-GELU path (`Only gated SiLU/SiTU is supported, got gelu`), measured 2026-11-18; unpaged the 16.0 GiB int4 checkpoint exceeds the 13.55 GiB budget (estimate)".

## Files

- [cmds/](cmds/): the four `docker run` files, reading `$HOME` and `$SCRATCH`; `m11-fp8-k72.sh` was deleted by mistake after its start, while the chain was being stopped, and rewritten with the same helper and arguments (it differs from `m11-fp8-k64.sh` only in K).
- `logs/<run>.*`: full and cut logs, the 1 s GPU and cgroup traces, state, metadata and the `/dev/shm` readings.
