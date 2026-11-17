# K2 Horizon 7B AWQ-INT4 on vLLM, 2026-11-17

Result: measured

All numbers below are measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17.
Image: `vllm/vllm-openai:v0.33.0`, started with `./llmctl up k2-horizon-7b --runtime vllm` from a pristine export of main 9a2c43a.
Model: `cyankiwi/K2-Horizon-7B-AWQ-INT4` at revision `530d5ceccd8ea660ceaa2a9f3a4bcbb68ba44291`.
The shipped recipe failed to start because 4.5 GiB of KV cache was needed for 65,536 tokens and only 4.44 GiB was available at vLLM's default `--gpu-memory-utilization` of 0.92 ([attempts.md](attempts.md)).
The server started on the second of two attempts with one change in an override: `--gpu-memory-utilization 0.94` added to `[server] args`, context unchanged ([override.diff](override.diff), [server-args.json](server-args.json)).
The other lines of the diff are the override's generated header comment and evidence summary, not flag changes.
The started server has `--max-model-len 65536` and a GPU KV cache of 95,168 tokens (1.45x concurrency at 64k) in 6.54 GiB, after `Model loading took 7.72 GiB memory` ([server-startup.log](server-startup.log)).
The container reached `Application startup complete` about 27 s after it started (23:22:17 to 23:22:44 UTC in the container log), with the torch.compile cache already filled by attempt 1.
Decode: 94.8 tok/s, the median of three 1,500-token runs (94.8, 94.6 and 94.8 tok/s), sent as the default request with no thinking field, so the model thinks at the server's `reasoning_effort` high ([probe-results.json](probe-results.json)).
Time to first token: 2.23 s for a fresh 8,085-token prompt and 0.037 s when repeated, 10.87 s for a fresh 30,261-token prompt and 0.089 s when repeated.
`./llmctl test` exited 0: reply ok ('OK'), tool call ok (get_weather with city Paris), decode 95.2 tok/s over 512 tokens ([llmctl-test.txt](llmctl-test.txt)).
Thinking check (type always, one level): the default request returned HTTP 200 with a 164-character `message.reasoning` and the content "391", as expected ([thinking-check.json](thinking-check.json)).
vLLM reports `reasoning_tokens: 0` in the usage of every request although reasoning text comes back, so that usage field is not a usable thinking signal here.
GPU memory peak: 15,384 MiB during the 30k prompt, 15,168 MiB from startup until then ([gpu-trace.csv](gpu-trace.csv), which also covers the failed attempt 1).
Power: enforced limit 145 W ([power-limit.txt](power-limit.txt)); the draw during the decode runs was 136 to 142 W at 54 to 63 C.

Files:

- [attempts.md](attempts.md): both start attempts with their exact log lines.
- [up-attempt-1.txt](up-attempt-1.txt) and [server-attempt-1.log](server-attempt-1.log): the failed first start and its full container log.
- [up-attempt-2.txt](up-attempt-2.txt), [server-startup.log](server-startup.log) and [server-args.json](server-args.json): the start that passed.
- [override.diff](override.diff): the shipped recipe against the override that started.
- [llmctl-test.txt](llmctl-test.txt), [probe-results.json](probe-results.json) and [thinking-check.json](thinking-check.json): the checks.
- [gpu-trace.csv](gpu-trace.csv) and [power-limit.txt](power-limit.txt): GPU memory, power and temperature every 2 s, and the power limit.
- [effort-map-check.txt](effort-map-check.txt): a check by hand after the first e2e run of 2026-11-17, with `--reasoning-effort-map '{"none":"low","minimal":"low","xhigh":"high","max":"high"}'` added to the launch, showing that Hermes' title request with `reasoning_effort` "none" got HTTP 200 and a title that parses while the default and high-effort requests still returned reasoning.
