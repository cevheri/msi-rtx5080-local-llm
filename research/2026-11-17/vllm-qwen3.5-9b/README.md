# Qwen3.5-9B NVFP4 on vLLM, 2026-11-17

Result: measured

Image: `vllm/vllm-openai:v0.33.0`.
Model: `RedHatAI/Qwen3.5-9B-NVFP4` at revision `a4df85a20fb7e54740f564f15af0c2f5d9c6616c`, from the recipe `qwen3.5-9b`.
The shipped recipe started on the first attempt with no override, so there is no override diff ([attempts.md](attempts.md)).
Arguments that started ([server-args.json](server-args.json)): `--served-model-name qwen3.5-9b --max-model-len 131072 --language-model-only --max-num-seqs 1 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --default-chat-template-kwargs {"enable_thinking":false} --override-generation-config {"temperature":0.6,"top_p":0.95,"top_k":20}`, with vLLM's default `--gpu-memory-utilization` of 0.92.
The log shows max-model-len 131072, `Model loading took 7.48 GiB memory`, `Available KV cache memory: 4.71 GiB` and `GPU KV cache size: 147,663 tokens` (1.13x concurrency at 131,072 tokens) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
`./llmctl up` reached its ready line 121 s after it started, including the removal of a stopped container, torch.compile and CUDA graph capture; vLLM's engine init took 94.09 s (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
Decode speed: median 90.8 tok/s over three streamed answers of 676, 791 and 788 tokens, with thinking off as sent by `{"chat_template_kwargs": {"enable_thinking": false}}` (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
The probe's decode prompt produced about 700 to 800 tokens, not 1,500, because the model ended its answer there.
First-token time: 0.592 s for a fresh 8,591-token prompt and 0.065 s for the same prompt again, and 2.424 s for a fresh 32,153-token prompt (0.119 s repeated) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
`./llmctl test` exited 0: reply `OK`, tool call `get_weather {"city": "Paris"}`, decode 90.0 tok/s over 410 tokens ([llmctl-test.txt](llmctl-test.txt)).
Thinking check, toggle type with default off ([thinking-check.json](thinking-check.json)): `enable_thinking: false` returned HTTP 200 with no reasoning, `enable_thinking: true` returned HTTP 200 with 521 characters of reasoning, and a request without the field returned no reasoning, as the server default sets.
GPU memory peak: 13,726 MiB during startup warmup, and 13,064 to 13,340 MiB while serving, from [gpu-trace.csv](gpu-trace.csv) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
Power: `power.limit` reads `[N/A]` and the enforced limit is 145 W ([power-limit.txt](power-limit.txt)); during the decode runs the GPU drew 128.6 to 131.7 W at 57 to 61 C (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).

Files:

- [attempts.md](attempts.md): the start attempt with its flags and log line.
- [up-attempt-1.txt](up-attempt-1.txt): the output of `heavy ./llmctl up qwen3.5-9b --runtime vllm`.
- [server-startup.log](server-startup.log): the vLLM log up to `Application startup complete`.
- [server-args.json](server-args.json): the container's arguments from `docker inspect`.
- [llmctl-test.txt](llmctl-test.txt): the output and exit code of `./llmctl test`.
- [probe-results.json](probe-results.json): the output of `research/2026-11-16/llm_probe.py`.
- [thinking-check.json](thinking-check.json): one request per thinking level.
- [gpu-trace.csv](gpu-trace.csv): memory, power and temperature every 2 s.
- [power-limit.txt](power-limit.txt): the GPU power limits.
