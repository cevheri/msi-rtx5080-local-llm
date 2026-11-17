# Mellum2.1 12B-A2.5B Thinking AWQ-INT4 on vLLM, 2026-11-17

Result: measured

All numbers below are measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17.
Image: `vllm/vllm-openai:v0.33.0`.
Model: `cyankiwi/Mellum2.1-12B-A2.5B-Thinking-AWQ-INT4` at revision `27089ca421c27c0b6289c1317c10ec974a8254d7`.
The shipped recipe started on the first attempt with no override, so there is no override diff ([attempts.md](attempts.md)).
Arguments that started ([server-args.json](server-args.json)): `--served-model-name mellum2.1 --max-model-len 131072 --max-num-seqs 1 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'`, with vLLM's default `--gpu-memory-utilization` 0.92.
The log reports max-model-len 131072, model loading at 7.2 GiB, 6.37 GiB of KV cache and a GPU KV cache size of 426,946 tokens (3.26x concurrency at 131,072 tokens) ([server-startup.log](server-startup.log)).
`./llmctl up` returned its `ready:` line about 120 s after the command started, and the server logged `Application startup complete` 108 s after the container started, including 12 s of torch.compile and 64 s of CUDA graph capture ([up-attempt-1.txt](up-attempt-1.txt)).
Decode speed is 257.2 tok/s, the median of three runs capped at 1,500 tokens with thinking on (`{"chat_template_kwargs": {"enable_thinking": true}}`, the recipe's default) ([probe-results.json](probe-results.json)).
Two of the three decode runs stopped early, at 1,337 and 1,171 tokens, so the median run is the 1,337-token one.
Time to the first token was 0.895 s for a fresh 9,932-token prompt and 0.032 s for its repeat, and 4.311 s for a fresh 37,190-token prompt (0.073 s repeat).
`./llmctl test` exited 0: reply ok ('OK'), tool call ok (`get_weather {"city": "Paris"}`), decode 272.5 tok/s over 502 tokens ([llmctl-test.txt](llmctl-test.txt)).
Thinking check, toggle type ([thinking-check.json](thinking-check.json)): off (`enable_thinking: false`) returned HTTP 200, no reasoning and the content "42".
Thinking check on (`enable_thinking: true`) returned HTTP 200, 173 characters in `message.reasoning` and non-empty content.
GPU memory peaked at 14,816 MiB in the trace ([gpu-trace.csv](gpu-trace.csv)).
The enforced power limit was 145 W ([power-limit.txt](power-limit.txt)); during the decode runs the GPU drew about 143 W at 56 to 60 C.

Files:

- [README.md](README.md): this summary.
- [attempts.md](attempts.md): the start attempt and its log line.
- [up-attempt-1.txt](up-attempt-1.txt): the output of `./llmctl up mellum2.1-12b --runtime vllm`.
- [server-startup.log](server-startup.log): the container log up to `Application startup complete`.
- [server-args.json](server-args.json): the container arguments.
- [llmctl-test.txt](llmctl-test.txt): the output of `./llmctl test` and its exit code.
- [probe-results.json](probe-results.json): the output of `research/2026-11-16/llm_probe.py`.
- [thinking-check.json](thinking-check.json): one request per thinking level.
- [gpu-trace.csv](gpu-trace.csv): nvidia-smi memory, power and temperature every 2 s.
- [power-limit.txt](power-limit.txt): the power limit and the enforced power limit.
