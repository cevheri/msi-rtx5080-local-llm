# The shipped vLLM gpt-oss-20b recipe without a network (step 0, LD15)

Last verified: 2026-11-17.

Result: the shipped vLLM recipe reaches ready without the network and with empty cache volumes, in 33.3 s, and answers a chat request; it needs no environment variable (measured on the MSI Raider 18 Max HX, vLLM v0.33.0).
The vLLM image ships `oss-harmony` 0.0.12, "fork of openai-harmony", which loaded the Harmony encoding with no network in 0.14 s, so vLLM has no counterpart of SGLang's vocabulary download (measured on the MSI Raider 18 Max HX, vLLM v0.33.0 image, offline, [logs/offline-harmony-check.txt](logs/offline-harmony-check.txt)).
No extra start was needed.

## The start

- `recipes/vllm/gpt-oss-20b.toml` mirrored by hand as `launcher/runtimes/vllm.py` `container_spec` and `launcher/command.py` build it: image `vllm/vllm-openai:v0.33.0`, the `[env]` of `recipes/vllm/_runtime.toml` (`HF_HUB_OFFLINE=1`, `VLLM_NO_USAGE_STATS=1`, `MAX_JOBS=2`, `HF_MODULES_CACHE=/tmp/hf_modules`), the Hugging Face cache read-only, `--ipc=host`, `--memory 32g --memory-swap 32g`, `--device nvidia.com/gpu=all`, the snapshot path, `--served-model-name gpt-oss-20b --max-model-len 65536` and the recipe's arguments unchanged ([cmds/netnone.sh](cmds/netnone.sh)).
- Changes for the check: `--network none` and no published port, new empty volumes `vllm-measure-cold` on `/root/.cache/vllm` and `vllm-measure-cold-fi` on `/root/.cache/flashinfer` in place of the runtime's `vllm-cache` and `flashinfer-cache`, no llmctl labels, and the container name `vllm-measure`.
- Readiness was polled every second with `docker exec` (urllib against 127.0.0.1:8000): `/health` answered 200 at 33.26 s after `docker run` and `/v1/models` listed `gpt-oss-20b` at 33.44 s ([shapes/netnone-exec-poll.txt](shapes/netnone-exec-poll.txt)) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0).
- A chat request `Reply with exactly: OK` returned HTTP 200 with `OK`, 81 characters of reasoning and `finish_reason` `stop` (measured on the MSI Raider 18 Max HX, vLLM v0.33.0).
- The log shows the recipe's known start: `Model loading took 13.8 GiB memory and 4.105638 seconds`, `Available KV cache memory: 0.94 GiB`, `GPU KV cache size: 81,504 tokens`, the same figures as the 2026-11-16 record, and 145 MiB of GPU memory free after ready (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, [logs/netnone.full.log](logs/netnone.full.log)).
- A case-insensitive search of the full log for `download`, `fetch`, `urm.nvidia`, `huggingface.co`, `openaipublic`, `blob.core`, `harmony`, `tiktoken`, `resolve` and `connect` matched only the architecture line, the engine configuration line, Gloo's `Unable to resolve hostname` warning and the line that says auto-prefetch is off for EXT4 ([logs/netnone.network-lines.txt](logs/netnone.network-lines.txt)) (measured).
- Afterwards the FlashInfer volume held 2 files (16 KB, a `0.7.2.post1` folder) and the vLLM volume 1 file (12 KB, `modelinfos`); both volumes were removed ([logs/](logs/), `netnone.volume-*.txt`) (measured).

## Offline image checks

- `find / -name '*.tiktoken'` found no vocabulary file in the vLLM image, and `openai-harmony` is not installed; `oss-harmony` 0.0.12 provides the `openai_harmony` module as one compiled library ([logs/offline-image-check.txt](logs/offline-image-check.txt)) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0 image, `--network none`, no GPU).
- vLLM loads the encoding lazily in `vllm/entrypoints/openai/parser/harmony_utils.py` (`get_encoding()` at line 97), and `load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)` returned a `HarmonyEncoding` in 0.14 s with no network (measured).

## Files

- [cmds/netnone.sh](cmds/netnone.sh): the `docker run` command (reads `$HOME`).
- [logs/](logs/): the full log, the matched network lines, the state, the volume listings, the 1 s trace and the two offline checks.
- [shapes/netnone-exec-poll.txt](shapes/netnone-exec-poll.txt): the readiness poll and the chat result.
