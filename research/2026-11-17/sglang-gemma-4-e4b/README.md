# Gemma 4 E4B QAT W4A16 on SGLang, 2026-11-17

Last verified: 2026-11-18.

Result: failed in the hands-on run: SGLang v0.5.23 could not build the model's audio tower from the W4A16 checkpoint, and with the language model alone it found no processor; measurement A did not run this model; measured in measurement B, which ran it with two flags.
At measurement B's settings, `--json-model-override-args '{"audio_config": null}'` and `--swa-full-tokens-ratio 0.1` at fraction 0.85, which the shipped settings keep with two variables added, start m5-a4-swa served text at 64k with a 101,791-token pool, held a fresh 62,244-token prompt, both grammar requests and 300 s of load with at least 1,403 MiB free and decoded 141.3 tok/s, and an offline start from an empty cache volume reached `/ready` in 138.1 s; the vision tower stays uninitialized, and with thinking on a short reply can land wholly in `reasoning_content` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
Measurement D measured the shipped settings, m5-a4-swa's with the runtime file's eight variables (`TIKTOKEN_ENCODINGS_BASE` and `PYTORCH_ALLOC_CONF` added), in [research/2026-11-18/sglang-gemma-4-e4b/](../../2026-11-18/sglang-gemma-4-e4b/README.md).

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `google/gemma-4-E4B-it-qat-w4a16-ct` at revision `6cd26aaa2357fb2bad8c51699a7558a4d1a965bb`, served as `gemma-4-e4b`.
Settings of a1 ([handson/cmds/gemma-4-e4b-a1.sh](handson/cmds/gemma-4-e4b-a1.sh)): `--context-length 65536 --mem-fraction-static 0.85 --max-running-requests 1 --disable-prefill-cuda-graph --tool-call-parser gemma4 --reasoning-parser gemma4`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only and `HF_HUB_OFFLINE=1`; a2 ([handson/cmds/gemma-4-e4b-a2.sh](handson/cmds/gemma-4-e4b-a2.sh)) added `--json-model-override-args '{"architectures":["Gemma4ForCausalLM"]}'`.
The cross-model reports are [the hands-on README](../sglang-runtime/handson/README.md) and [measurement B's lane README](../sglang-runtime/measure-b/README.md), and measurement B's report on this model is [measure-b/README.md](measure-b/README.md); every figure below comes from them.

- a1 failed while building the audio tower: `models/gemma4_audio.py:785` to `layers/clippable_linear.py:210`, `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'weight'` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The encoders are built with the checkpoint's quantization config, so the W4A16 linear has packed weights where the clippable layer expects `weight` (inferred from the traceback and `gemma4_mm.py` at tag v0.5.23).
- a2 failed with `ValueError: No processor registered for architecture: ['Gemma4ForCausalLM'].`, because the config still counts as multimodal (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The lane stopped after two attempts, because the other listed fixes (attention backend, CUDA graphs, memory fraction, context) act after model construction, and `--language-model-only` accepts only four other architectures (inferred from the sources at tag v0.5.23).

## Measurement B

- Both starts used a1's settings plus `--json-model-override-args '{"audio_config": null}'`, measurement B's overlay (the snapshot's `generation_config.json` with temperature 1.0, top_p 0.95 and top_k 64, the vLLM recipe's values) and environment (`HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `FLASHINFER_NO_DOWNLOAD=1`) ([measure-b/cmds/m5-a3.sh](measure-b/cmds/m5-a3.sh), [measure-b/cmds/m5-a4-swa.sh](measure-b/cmds/m5-a4-swa.sh)).
- m5-a3, with the override alone, loaded and served, but its pool held 42,413 tokens, short of 64k (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- m5-a4-swa added `--swa-full-tokens-ratio 0.1`: the ready line came 28.0 s after `docker run`, the pool held 101,791 tokens (SWA 10,179), the sampling line read `{'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}`, and 1,407 MiB were free right after ready (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- A fresh 62,244-token prompt took 22.52 s and left 1,403 MiB; the first grammar request and a forced tool call then moved nothing (1,403 MiB) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Decode 141.3 tok/s median in the repository probe, against 45.2 tok/s on vLLM in eager mode (measured on the MSI Raider 18 Max HX, SGLang v0.5.23 and vLLM v0.33.0, 2026-11-17; [vllm-gemma-4-e4b/](../vllm-gemma-4-e4b/README.md)); 1.617 s and 8.632 s to the first token for 9,932 and 37,190 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- M12: 80 requests in 300 s with no error, decode 116.6 to 141.2 tok/s (median 134.05) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Peaks: 14,439 MiB of GPU memory, 146.7 W, 75 C, and 5.73 GiB of non-reclaimable host memory (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The log names 114 weights not initialized from the checkpoint, the quantized layers of all 16 vision-encoder layers among them, so image input cannot work; text answers, tool calls and grammar requests were correct in every probe (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17); that image input fails is inferred, not run.
- With thinking on, the model sometimes answered `391` without a thought block, and the `gemma4` reasoning parser then returned the whole reply as `reasoning_content`, leaving `content` empty (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- M15: from a new empty cache volume with `--network none`, `/ready` answered 200 at 138.05 s, the decode-graph capture took 100.56 s cold against 1.53 s warm, 1,407 MiB were free after ready, a chat request answered `OK`, and no download or fetch line appeared (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

## Files

| Path | Holds |
|---|---|
| [handson/cmds/](handson/cmds/) | The `docker run` of the hands-on starts a1 and a2 |
| [handson/logs/](handson/logs/) | The cut container logs (`attempt1.log`, `attempt2.log`), start times (`start-*.epoch`) and the 1 s GPU and cgroup trace (`trace.csv`) |
| [measure-b/README.md](measure-b/README.md) | Measurement B's report on this model, unchanged (no link needed a rewrite, so there is no `links.diff`) |
| [measure-b/cmds/](measure-b/cmds/) | The `docker run` of measurement B's starts m5-a3, m5-a4-swa and m15-netnone; they read `$HOME` and `$SCRATCH` |
| [measure-b/logs/](measure-b/logs/) | Per start the cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace (with anon, file, shmem and kernel), the readiness poll, the process memory, the request log, the sequence output, the state and the metadata, and for the network-none starts the network namespace, the matched network lines and the volume listing |
| [measure-b/shapes/](measure-b/shapes/) | Readiness and `/server_info` (`-ready.json`, `-ready-server-info.json`), the sequence records (`-seq.jsonl`), the repository probe, the api probe, the sustained load and the summaries; for the network-none starts the `docker exec` poll |
| [measure-b/overlay/](measure-b/overlay/) | The generation config before and after, and the link count, of measurement B's overlay |

Neither hands-on start reached the API probe, so the hands-on lane's `shapes/gemma-4-e4b/` folder was empty and `handson/` has no `shapes/` folder.
