# Gemma 4 12B IT QAT W4A16 on vLLM, 2026-11-17

Result: measured

Image: `vllm/vllm-openai:v0.33.0`.
Model: `google/gemma-4-12B-it-qat-w4a16-ct` at revision `1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee`, from the shipped recipe `recipes/vllm/gemma-4-12b.toml`.
The shipped recipe started on the first of three allowed attempts, with no override ([attempts.md](attempts.md), [server-args.json](server-args.json)).
vLLM used its default `--gpu-memory-utilization` of 0.92 and raised `max_num_batched_tokens` from 2048 to 2496 for the multimodal prefix-LM model ([server-startup.log](server-startup.log)).
Max model length 131,072, `Available KV cache memory: 5.37 GiB`, and a GPU KV cache of 366,455 tokens, 2.80x concurrency at 131k (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
Model loading took 8.28 GiB of GPU memory in 5.7 s (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
About 84 s passed from the container start to the `ready:` line of `./llmctl up` ([up-attempt-1.txt](up-attempt-1.txt)) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
Decode is 69.5 tok/s, the median of three streamed runs with thinking off (`{"chat_template_kwargs": {"enable_thinking": false}}`, the recipe's default) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
The three decode runs stopped on their own after 885, 933 and 928 tokens, short of the 1,500-token cap, so the median covers about 900 tokens per run ([probe-results.json](probe-results.json)).
The 8k prompt (9,936 tokens) took 3.88 s to the first token fresh and 0.043 s repeated, and the 30k prompt (37,194 tokens) took 17.18 s fresh and 0.091 s repeated (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
`./llmctl test` exited 0: reply ok (`OK`), tool call ok (`get_weather {"city": "Paris"}`), decode 69.8 tok/s over 340 tokens ([llmctl-test.txt](llmctl-test.txt)).
Thinking check ([thinking-check.json](thinking-check.json)): `enable_thinking: false` gave HTTP 200, no `reasoning` and the answer `391`.
With `enable_thinking: true` it gave HTTP 200, 260 characters of `message.reasoning` and the answer `391`.
A request without a thinking field, as Hermes sends it, gave HTTP 200 and no reasoning, so the server's default is thinking off.
The GPU memory peak in the trace is 15,822 MiB during the startup profiling, 14,660 MiB while serving and 15,016 MiB after the 30k prompt ([gpu-trace.csv](gpu-trace.csv)) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
The enforced power limit is 145 W ([power-limit.txt](power-limit.txt)), and during the decode runs the GPU drew about 134 to 136 W at 53 to 61 C (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).

Files:

- [attempts.md](attempts.md): the start attempts.
- [up-attempt-1.txt](up-attempt-1.txt): the output of `./llmctl up gemma-4-12b --runtime vllm`.
- [server-args.json](server-args.json): the container arguments that started.
- [server-startup.log](server-startup.log): the server log up to `Application startup complete`.
- [llmctl-test.txt](llmctl-test.txt): the output of `./llmctl test`.
- [probe-results.json](probe-results.json): the output of `research/2026-11-16/llm_probe.py`.
- [thinking-check.json](thinking-check.json): the thinking off, on and default requests.
- [gpu-trace.csv](gpu-trace.csv): memory, power and temperature every 2 s.
- [power-limit.txt](power-limit.txt): the power limit and the enforced power limit.
