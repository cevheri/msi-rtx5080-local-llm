# gpt-oss-20b on SGLang, 2026-11-17

Last verified: 2026-11-17.

Result: measured in the hands-on run (a1 was refused, a2 and a3 crashed on their first grammar request, a4 with Paged Experts ran) and in measurement A (M6 and M7 at the a4 settings), all started by hand before llmctl had an SGLang recipe.

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `openai/gpt-oss-20b` at revision `6cee5e81ee83917806bbde320786a8fb61efebee`, served as `gpt-oss-20b`.
Working settings (a4, [handson/cmds/gpt-oss-20b-a4.sh](handson/cmds/gpt-oss-20b-a4.sh)): `--context-length 131072 --mem-fraction-static 0.90 --max-running-requests 1 --kv-cache-dtype fp8_e4m3 --swa-full-tokens-ratio 0.1 --chunked-prefill-size 1024 --disable-prefill-cuda-graph --moe-runner-backend marlin --enable-paged-experts --paged-experts-num-resident 24 --tool-call-parser gpt-oss --reasoning-parser gpt-oss`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only and `HF_HUB_OFFLINE=1`.
The cross-model reports are [the hands-on README](../sglang-runtime/handson/README.md) and [measurement A's README](../sglang-runtime/measure-a/README.md), where every figure below comes from.

## Hands-on run

- a1 at 64k and `--mem-fraction-static 0.95` was refused: the weights left 0.60 GB, no memory for the KV cache (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- a2 at 0.97 ran with an 11,896-token pool and a3 with the Marlin MoE runner at 0.95 with a 34,726-token pool; both crashed on the first grammar-constrained request in NCCL with `Failed to CUDA calloc 536870912 bytes` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- a4 with Paged Experts kept 24 of 32 experts per layer on the GPU: weights 11.46 GB, FP8 KV pool 168,443 tokens plus 16,844 SWA tokens, 1.45 GB free, and the ready line 61 s after `docker run` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Decode at a4: 133.8 tok/s over 512-token runs, and a median of 125.5 tok/s in the repository probe over answers of 1,464 to 1,500 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- First-token time at a4 after `/flush_cache`: 1.452 s for 8,058 tokens and 4.893 s for 30,003 tokens; 123,958 tokens took 41.69 s and 0.209 s again (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Host memory at a4: anonymous memory peaked at 13.94 GiB against 5.47 GiB without paging, because the pinned host store holds all 32 experts of every layer, and the cgroup peak with page cache was 28.24 GiB of the 32 GiB cap (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Forced tool choice returned HTTP 200 with `finish_reason` `stop`, no `tool_calls` and raw Harmony text in `content` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

## Measurement A

- Grammar (M6): free memory was 1,406 MiB after ready and 1,426 MiB after `/flush_cache`; the first grammar request took 602 MiB and left 468 MiB after a 32k-token prompt, and the server stayed up (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- Readiness (M7): the ready line came 106.2 s after `docker run` (115.8 s in the second start), `/v1/models` answered 200 at 106.06 s and `/ready` at 107.06 s; `startup_available` read 1.447 GiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).
- The start-up spent 52 and 63 s, against 6 s in the hands-on starts, in the step that builds the Harmony encoding, which fetches `o200k_base.tiktoken` unless `TIKTOKEN_ENCODINGS_BASE` points at the image's copy (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; the download as the cause is inferred from the sources).
- The container's cgroup stayed at 31.88 of 32 GiB from about 44 s after `docker run` to the end of both starts, with no OOM kill (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17).

## Files

| Path | Holds |
|---|---|
| [handson/cmds/](handson/cmds/) | The `docker run` of the hands-on starts a1 to a4 |
| [handson/logs/](handson/logs/) | Cut container logs (`attemptN.log`), container arguments (`*.cmd.json`), a socket table (`attempt4.proc-net-tcp.txt`), start times (`start-*.epoch`) and the 1 s GPU and cgroup trace (`trace.csv`) |
| [handson/shapes/](handson/shapes/) | The API probes of a2, a3 and a4 and the repository probe of a4 |
| [measure-a/cmds/](measure-a/cmds/) | The `docker run` of the M6 o1 and o2 starts |
| [measure-a/logs/](measure-a/logs/) | Per run the cut log, the full log, the trace, the poller and probe output and the container state |
| [measure-a/shapes/](measure-a/shapes/) | M6 records (`*.jsonl`), M7 poll records (`*-m7.json`) and one full `/get_server_info` body |
