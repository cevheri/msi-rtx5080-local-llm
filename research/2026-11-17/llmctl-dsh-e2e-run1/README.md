# llmctl clients e2e, first full run of 2026-11-17

Last verified: 2026-11-17.

This folder keeps part of the first full run of the llmctl clients e2e on 2026-11-17, which the documents cite next to the second run in [../llmctl-dsh-e2e/](../llmctl-dsh-e2e/README.md).
It ran the same command as the second run, into the folder the second run later used, before K2 Horizon 7B's vLLM recipe gained its `--reasoning-effort-map` and before the driver stopped sending probe B to a chat-only recipe:

```bash
heavy python3 -m launcher.e2e.clients --out research/2026-11-17/llmctl-dsh-e2e --matrix all --set-default-round-trip
```

The driver wrote these files; they are copied unchanged, and do not edit them by hand.
Each record shows the private work folder as `$WORK` and the home folder as `$HOME`.

## Files

- [summary.md](summary.md): the driver's table of the run, 280 Yes and 14 No rows, exit 1.
- [records/](records/): the run's records that the documents cite, with the run's own numbers:
  - K2 Horizon 7B on vLLM: [207 opencode](records/207-13-k2-horizon-7b-vllm-opencode) (passed), [208 hermes-z](records/208-13-k2-horizon-7b-vllm-hermes-z) (passed), [209 hermes-chat](records/209-13-k2-horizon-7b-vllm-hermes-chat) and [210 pi](records/210-13-k2-horizon-7b-vllm-pi) (failed on the chat template's thinking field), [211 dsh-headless](records/211-13-k2-horizon-7b-vllm-dsh-headless) (passed) and [214 probes-A](records/214-13-k2-horizon-7b-vllm-probes-A) (HTTP 400 `Unsupported reasoning_effort 'none'`).
  - DeepSeek-R1-0528-Qwen3-8B on vLLM: [277 up](records/277-18-deepseek-r1-0528-qwen3-8b-vllm-up) (ready at the shipped utilization), [290 probes-A](records/290-18-deepseek-r1-0528-qwen3-8b-vllm-probes-A) (a garbled title) and [291 probes-B](records/291-18-deepseek-r1-0528-qwen3-8b-vllm-probes-B) (HTTP 400 for a request with tools).

The run's other records, and its lead-check notes, are not kept; the second run's folder holds the full record set and the lead checks.
