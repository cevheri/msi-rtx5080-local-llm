# Gemma 4 E4B IT QAT W4A16 on vLLM, 2026-11-17

Result: measured

Image: `vllm/vllm-openai:v0.33.0`.
Model: `google/gemma-4-E4B-it-qat-w4a16-ct` at revision `6cd26aaa2357fb2bad8c51699a7558a4d1a965bb`, from the shipped recipe `recipes/vllm/gemma-4-e4b.toml`.
The shipped recipe did not start: twice the torch.compile autotune of the profile run ran out of GPU memory asking for 5.25 GiB, the size of the per-layer embedding table ([attempts.md](attempts.md)).
The third of three allowed attempts started with an override that adds `--enforce-eager` to the recipe's `[server] args` and changes nothing else ([override.diff](override.diff), [server-args.json](server-args.json)).
For the shipped recipe the change is one flag: `"--language-model-only", "--max-num-seqs", "2", "--enforce-eager", "--enable-auto-tool-choice",`; the other lines of the diff are the override's own header and evidence summary.
vLLM used its default `--gpu-memory-utilization` of 0.92, max-num-batched-tokens 2048, the TRITON_ATTN backend and an auto (bf16) KV cache ([server-startup.log](server-startup.log)).
Max model length 65,536, `Available KV cache memory: 3.67 GiB`, and a GPU KV cache of 204,533 tokens, 3.12x concurrency at 65,536 (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
Model loading took 10.13 GiB of GPU memory in 2.8 s (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
About 26 s passed from the container start to the `ready:` line of `./llmctl up` ([up-attempt-3.txt](up-attempt-3.txt)) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
Decode is 45.2 tok/s, the median of three streamed runs with thinking off (`{"chat_template_kwargs": {"enable_thinking": false}}`, the recipe's default) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
The three decode runs stopped on their own after 1,130, 969 and 1,190 tokens, short of the 1,500-token cap, at 45.2, 45.2 and 45.1 tok/s ([probe-results.json](probe-results.json)).
Eager mode has no CUDA graphs, which likely costs decode speed on this small model (estimate).
The 8k prompt (9,932 tokens) took 1.52 s to the first token fresh and 0.063 s repeated, and the 30k prompt (37,190 tokens) took 7.90 s fresh and 0.109 s repeated (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
`./llmctl test` exited 0: reply ok (`OK`), tool call ok (`get_weather {"city": "Paris"}`), decode 45.2 tok/s over 369 tokens ([llmctl-test.txt](llmctl-test.txt)).
Thinking check ([thinking-check.json](thinking-check.json)): `enable_thinking: false` gave HTTP 200, no `reasoning` and the answer `391`.
With `enable_thinking: true` it gave HTTP 200, 308 characters of `message.reasoning` and the answer `391`.
A request without a thinking field, as Hermes sends it, gave HTTP 200 and no reasoning, so the server's default is thinking off.
The GPU memory peak in the trace is 15,044 MiB, reached once the server was up and held through the probe; the failed attempts peaked at 11,522 MiB ([gpu-trace.csv](gpu-trace.csv)) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
The enforced power limit is 145 W ([power-limit.txt](power-limit.txt)), and during the decode runs the GPU drew about 87 to 91 W at 52 to 60 C, well below the limit; the long prompts reached 147 W at 70 C (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).

Files:

- [attempts.md](attempts.md): the start attempts.
- [attempt-1-error.log](attempt-1-error.log), [attempt-2-error.log](attempt-2-error.log): the key lines of the two failed starts.
- [up-attempt-1.txt](up-attempt-1.txt), [up-attempt-2.txt](up-attempt-2.txt), [up-attempt-3.txt](up-attempt-3.txt): the output of `./llmctl up gemma-4-e4b --runtime vllm` per attempt.
- [override.diff](override.diff): the shipped recipe against the override that started.
- [server-args.json](server-args.json): the container arguments that started.
- [server-startup.log](server-startup.log): the server log up to `Application startup complete`.
- [llmctl-test.txt](llmctl-test.txt): the output of `./llmctl test`.
- [probe-results.json](probe-results.json): the output of `research/2026-11-16/llm_probe.py`.
- [thinking-check.json](thinking-check.json): the thinking off, on and default requests.
- [gpu-trace.csv](gpu-trace.csv): memory, power and temperature every 2 s.
- [power-limit.txt](power-limit.txt): the power limit and the enforced power limit.
