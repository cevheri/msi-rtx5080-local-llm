# Mellum2.1-12B AWQ-INT4 on SGLang, 2026-11-17

Last verified: 2026-11-17.

Result: failed in measurement B: the vLLM recipe's checkpoint does not load on SGLang v0.5.23, because SGLang's compressed-tensors WNA16 MoE loader expects the asymmetric zero points in the transposed layout (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; the layout inferred from the source).
No SGLang recipe is drafted; the hands-on run and measurement A did not run this model.

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `cyankiwi/Mellum2.1-12B-A2.5B-Thinking-AWQ-INT4` at revision `27089ca421c27c0b6289c1317c10ec974a8254d7`, the vLLM recipe's revision, served as `mellum2.1`.
Settings of m1-a1 ([measure-b/cmds/m1-a1.sh](measure-b/cmds/m1-a1.sh)): context 131072, `--mem-fraction-static 0.88`, BF16 KV, `--swa-full-tokens-ratio 0.1`, `--disable-prefill-cuda-graph`, `--max-running-requests 1`, `--tool-call-parser hermes` and `--reasoning-parser qwen3`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only, a model overlay whose `generation_config.json` carries temperature 0.6, top_p 0.95 and top_k 20 (the vLLM recipe's values), and measurement B's environment.
The reports are [measurement B's README for this model](measure-b/README.md) and [measurement B's lane README](../sglang-runtime/measure-b/README.md), where every figure below comes from.

## Measurement B

- The log names `MellumForCausalLM` as a hybrid SWA model, FlashInfer attention and `CompressedTensorsWNA16MarlinMoEMethod` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Two seconds into `Load weight begin` the scheduler died with `RuntimeError: The size of tensor a (288) must match the size of tensor b (28) at non-singleton dimension 1`, and the container exited 26 s after `docker run` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The checkpoint stores one expert's down-projection zero points as [288, 28], that is [2304 / 8, 896 / 32], while SGLang creates the slot as [28, 288] (measured from the safetensors header; the slot shape inferred from the source).
- `--moe-runner-backend triton` cannot help, because SGLang's Triton WNA16 MoE backend accepts only symmetric INT4 (inferred from the source; not run).
- vLLM v0.33.0 loads the same checkpoint at 257.2 tok/s (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [vllm-mellum2.1-12b/](../vllm-mellum2.1-12b/README.md)), so the defect is SGLang's loader, not the files.

## Files

| Path | Holds |
|---|---|
| [measure-b/README.md](measure-b/README.md) | Measurement B's report on this model, links rewritten |
| [measure-b/links.diff](measure-b/links.diff) | The link rewrite of that report against the scratch copy |
| [measure-b/cmds/m1-a1.sh](measure-b/cmds/m1-a1.sh) | The `docker run` of the start; it reads `$HOME` and `$SCRATCH` |
| [measure-b/logs/](measure-b/logs/) | The cut and full logs, the `docker run` output, the readiness poll, the container state, the 1 s GPU and cgroup trace and the metadata |
| [measure-b/shapes/m1-a1-ready.json](measure-b/shapes/m1-a1-ready.json) | The readiness record of the start, which never became ready |
| [measure-b/overlay/](measure-b/overlay/) | The overlay's generation config before and after, and its link count |
