# Qwen3.6-35B-A3B Q4_K_M on llama.cpp, 2026-11-16

Three groups of records, all on `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407`, 128k context, f16 KV, thinking off by default.

| Path | Launch | Main result |
|---|---|---|
| [baseline/](baseline/) | The recipe as written in llamacpp/models.md: `--n-cpu-moe 21`, default mmap load, auto threads (4) ([launch](baseline/run-command.sh)) | 67.3 tok/s decode, 934 tok/s prefill at 8.6k, 13,696 MiB peak GPU memory |
| [variants/](variants/) | One change at a time on top of the baseline, each with 3 decodes and one 8.6k prompt ([script](variants/variant.sh), [one-line results](variants/variants.jsonl)) | `-lm none` doubles prefill; the MoE cache with CPU share 0 raises decode to 112-126 tok/s |
| this folder | Tuned: `--n-cpu-moe 25 -lm none -t 8 --moe-cache-mib 3200 --moe-cache-cpu-share 0`, plus `-lv 4` for the cache statistics ([launch](run-command.sh), [script](validation.sh)) | 122.5 tok/s decode, 1,990 tok/s prefill at 8.6k, 50 s to the first token at 107k, 15,190 MiB peak GPU memory, OpenCode session passed |

The tuned folder also holds the per-request thinking check ([thinking-on.json](thinking-on.json)), a code-generation decode ([code-decode.json](code-decode.json)) and the MoE cache statistics at exit (last lines of [server-timings.log](server-timings.log)).
The baseline container's log was removed when the first variant started; its per-request server timings are in [baseline/probe-results.json](baseline/probe-results.json).
