# gpt-oss-20b on vLLM without a network, 2026-11-17

Last verified: 2026-11-17.

Result: measured: the shipped vLLM gpt-oss-20b recipe reached ready with `--network none` and new empty cache volumes in 33.3 s and answered a chat request, with no extra environment variable (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
The vLLM image ships `oss-harmony` 0.0.12, which loaded the Harmony encoding without a network in 0.14 s, so vLLM has no counterpart of SGLang's Harmony vocabulary download (measured on the MSI Raider 18 Max HX, vLLM v0.33.0 image, offline, 2026-11-17).
This is measurement B's check of the recipe, started by hand with `docker run`, not by `./llmctl up`, in one start.

Image: `vllm/vllm-openai:v0.33.0`.
Model: `openai/gpt-oss-20b` at revision `6cee5e81ee83917806bbde320786a8fb61efebee`, served as `gpt-oss-20b`.
The start mirrors `recipes/vllm/gpt-oss-20b.toml` as `launcher/runtimes/vllm.py` `container_spec` builds it, with the runtime's `[env]`, the Hugging Face cache read-only, `--ipc=host`, `--memory 32g --memory-swap 32g` and the recipe's arguments unchanged, and differs in four points: `--network none` and no published port, new empty volumes `vllm-measure-cold` and `vllm-measure-cold-fi` in place of `vllm-cache` and `flashinfer-cache`, no llmctl labels, and the container name `vllm-measure` ([cmds/netnone.sh](cmds/netnone.sh)).
Measurement B's own report is [measure-b.md](measure-b.md), where every figure below comes from.

- Readiness, polled every second with `docker exec`: `/health` answered 200 at 33.26 s after `docker run` and `/v1/models` listed `gpt-oss-20b` at 33.44 s; `Reply with exactly: OK` returned HTTP 200 with `OK` (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
- The log shows `Model loading took 13.8 GiB memory`, `Available KV cache memory: 0.94 GiB` and `GPU KV cache size: 81,504 tokens`, the figures of the 2026-11-16 record, and 145 MiB of GPU memory were free after ready (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
- A case-insensitive search of the full log for `download`, `fetch`, `urm.nvidia`, `huggingface.co`, `openaipublic`, `blob.core`, `harmony`, `tiktoken`, `resolve` and `connect` matched only the architecture line, the engine configuration line, Gloo's `Unable to resolve hostname` warning and the line that says auto-prefetch is off for EXT4 (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
- Afterwards the FlashInfer volume held 2 files (16 KB) and the vLLM volume 1 file (12 KB); both volumes were removed (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).

## Files

| Path | Holds |
|---|---|
| [measure-b.md](measure-b.md) | Measurement B's report on this start and the offline image checks; no link needed a rewrite, so there is no `links.diff` |
| [cmds/netnone.sh](cmds/netnone.sh) | The `docker run` command; it reads `$HOME` |
| [logs/](logs/) | The full log, the matched network lines, the network namespace listing, the `docker run` output, the state, the metadata, the volume listings, the 1 s trace and the two offline image checks |
| [shapes/netnone-exec-poll.txt](shapes/netnone-exec-poll.txt) | The readiness poll and the chat result |
