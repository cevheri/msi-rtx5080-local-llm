# Mellum2.1-12B AWQ-INT4 on SGLang v0.5.23 (M1): failed while loading the expert weights

Last verified: 2026-11-17.

Result: the vLLM recipe's checkpoint does not load on SGLang v0.5.23, because SGLang's compressed-tensors WNA16 MoE loader expects the asymmetric zero points in the transposed layout (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
No draft recipe is written, and Mellum stays on vLLM (and on the llama.cpp GGUF recipe that measurement C adds).

## The start

- Checkpoint: `cyankiwi/Mellum2.1-12B-A2.5B-Thinking-AWQ-INT4` at `27089ca421c27c0b6289c1317c10ec974a8254d7`, the vLLM recipe's revision, from the overlay `$SCRATCH/sglang/measure-b/overlay/mellum2.1-12b/` (temperature 0.6, top_p 0.95, top_k 20, the vLLM recipe's values).
- Settings as the brief asks: fraction 0.88, BF16 KV (the default dtype), `--swa-full-tokens-ratio 0.1`, `--disable-prefill-cuda-graph`, `--max-running-requests 1`, `--tool-call-parser hermes`, `--reasoning-parser qwen3`, context 131072, and the lane environment ([cmds/m1-a1.sh](cmds/m1-a1.sh)).
- Both parser names exist in v0.5.23: `hermes` in `function_call/parser_names.py` and `qwen3` in `parser/reasoning_parser_names.py` of the image ([../offline/sglang-parser-names.txt](../../sglang-runtime/measure-b/offline/sglang-parser-names.txt)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23 image, offline read).
- The log names the native class and the MoE method: `Hybrid SWA model detected. architectures=['MellumForCausalLM']`, `Attention backend not specified. Use flashinfer backend by default.`, `Using CompressedTensorsWNA16MarlinMoEMethod` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, [logs/m1-a1.log](logs/m1-a1.log)).
- It auto-detected `reasoning_parser=qwen3` and a `qwen` tool parser from the template, with thinking on by default (`default_enabled=True`) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- Two seconds into `Load weight begin`, the scheduler died; the container exited 26 s after `docker run`, and the poller saw no ready server ([logs/m1-a1.state.txt](logs/m1-a1.state.txt)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

The exact error, after a traceback through `models/qwen3_moe.py:1250 in load_weights` and `layers/moe/fused_moe_triton/layer.py:969 in _load_w2` ([logs/m1-a1.full.log](logs/m1-a1.full.log)):

```text
RuntimeError: The size of tensor a (288) must match the size of tensor b (28) at non-singleton dimension 1
```

## The arithmetic that explains it

- The checkpoint is asymmetric INT4 with zero points: `"group_size": 32, "num_bits": 4, "symmetric": false, "zp_dtype": "torch.int8"` in `config.json` (vendor-reported, the cached `config.json`).
- Its tensors for one expert's down projection (input 896, output 2304) are `weight_packed` I32 [2304, 112], `weight_scale` BF16 [2304, 28] and `weight_zero_point` I32 [288, 28], that is [2304 / 8, 896 / 32] (measured on the MSI Raider 18 Max HX, safetensors header of the cached file).
- SGLang's `CompressedTensorsWNA16MoE` creates `w2_weight_zero_point` as `(num_experts, num_groups_w2, hidden_size // packed_factor)`, which is [28, 288] per expert (`../offline/sglang-src/compressed_tensors_wNa16_moe.py`, lines 207 to 217) (inferred from the source).
- So the loader copies a [288, 28] zero-point tensor into a [28, 288] slot and fails on dimension 1 (288 against 28), the error above (inferred from the shapes and the source).
- The other MoE path cannot help: `--moe-runner-backend triton` is refused for this checkpoint, because "The Triton WNA16 MoE backend only supports symmetric INT4 group quantization with group_size=32 or 128 and no actorder." (`../offline/sglang-src/compressed_tensors.py`, lines 930 to 935) (inferred from the source; not run).
- vLLM v0.33.0 loads the same checkpoint (7.2 GiB, 257.2 tok/s, `research/2026-11-17/vllm-mellum2.1-12b/`), so the defect is SGLang's loader, not the files.

## What follows

- Mellum2.1 on SGLang needs either a symmetric compressed-tensors INT4 build, a GPTQ or AWQ build that SGLang's MoE loaders take, or an SGLang release that transposes compressed-tensors zero points; none is cached, and this lane downloads nothing.
- The pool arithmetic of research report section 4 (about 305k BF16 tokens at ratio 0.1) was never reached, so it stays an estimate.

## Files

- [cmds/m1-a1.sh](cmds/m1-a1.sh): the `docker run` command (reads `$HOME` and `$SCRATCH`).
- [logs/](logs/): the cut log (`m1-a1.log`), the full log, the container state, the 1 s GPU and cgroup trace, and the run metadata.
