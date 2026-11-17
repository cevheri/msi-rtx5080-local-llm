# Gemma 4 E4B QAT W4A16 on SGLang, 2026-11-17

Last verified: 2026-11-17.

Result: failed in the hands-on run: SGLang v0.5.23 could not build the model's audio tower from the W4A16 checkpoint, and with the language model alone it found no processor; measurement A did not run this model.

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `google/gemma-4-E4B-it-qat-w4a16-ct` at revision `6cd26aaa2357fb2bad8c51699a7558a4d1a965bb`, served as `gemma-4-e4b`.
Settings of a1 ([handson/cmds/gemma-4-e4b-a1.sh](handson/cmds/gemma-4-e4b-a1.sh)): `--context-length 65536 --mem-fraction-static 0.85 --max-running-requests 1 --disable-prefill-cuda-graph --tool-call-parser gemma4 --reasoning-parser gemma4`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only and `HF_HUB_OFFLINE=1`; a2 ([handson/cmds/gemma-4-e4b-a2.sh](handson/cmds/gemma-4-e4b-a2.sh)) added `--json-model-override-args '{"architectures":["Gemma4ForCausalLM"]}'`.
The cross-model report is [the hands-on README](../sglang-runtime/handson/README.md), where every figure below comes from.

- a1 failed while building the audio tower: `models/gemma4_audio.py:785` to `layers/clippable_linear.py:210`, `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'weight'` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The encoders are built with the checkpoint's quantization config, so the W4A16 linear has packed weights where the clippable layer expects `weight` (inferred from the traceback and `gemma4_mm.py` at tag v0.5.23).
- a2 failed with `ValueError: No processor registered for architecture: ['Gemma4ForCausalLM'].`, because the config still counts as multimodal (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The lane stopped after two attempts, because the other listed fixes (attention backend, CUDA graphs, memory fraction, context) act after model construction, and `--language-model-only` accepts only four other architectures (inferred from the sources at tag v0.5.23).

## Files

| Path | Holds |
|---|---|
| [handson/cmds/](handson/cmds/) | The `docker run` of the hands-on starts a1 and a2 |
| [handson/logs/](handson/logs/) | The cut container logs (`attempt1.log`, `attempt2.log`), start times (`start-*.epoch`) and the 1 s GPU and cgroup trace (`trace.csv`) |

Neither start reached the API probe, so the hands-on lane's `shapes/gemma-4-e4b/` folder was empty and has no counterpart here.
