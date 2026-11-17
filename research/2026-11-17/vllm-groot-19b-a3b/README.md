# Groot-19B-A3B-261023 NVFP4 on vLLM, 2026-11-17

Result: measured

The image was `vllm/vllm-openai:v0.33.0`.
The model was `GrootTeam/Groot-19B-A3B-261023-NVFP4` at revision `8ff73aca9300f4e9835972f0012ef5edbb016a8b`, as the shipped recipe pins it.
The shipped recipe started on the first of three allowed attempts, so no override was used ([attempts.md](attempts.md), [server-args.json](server-args.json)).
The arguments that started were `--max-model-len 131072 --language-model-only --max-num-seqs 1 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking":false}' --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'` with vLLM's default `--gpu-memory-utilization` of 0.92 and CUDA graphs on.
The log shows max model len 131072, a KV cache of 1.87 GiB holding 167,480 tokens (1.28x concurrency at 131,072 tokens) and model loading of 10.99 GiB (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
vLLM used the MARLIN NvFp4 MoE backend and warned that the GPU has no native FP4 support in this path, so it ran weight-only FP4 through Marlin.
The server was ready 75 s after `./llmctl up` started, including 16.57 s of torch.compile (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
Decode ran at 193.4 tok/s as the median of three runs with thinking off, sent as `{"chat_template_kwargs": {"enable_thinking": false}}` (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
The three decode runs asked for 1,500 tokens but the model stopped after 780 to 829 tokens ([probe-results.json](probe-results.json)).
An 8,591-token prompt took 0.861 s to the first token fresh and 0.069 s repeated, and a 32,153-token prompt took 3.675 s fresh (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
`./llmctl test` exited 0 with reply ok, tool call ok (`get_weather {"city": "Paris"}`) and 187.7 tok/s decode ([llmctl-test.txt](llmctl-test.txt)).
The thinking check matched the toggle type: off gave no reasoning, on gave 228 characters in `message.reasoning`, and a request without a thinking field gave none, each with HTTP 200 and a non-empty reply ([thinking-check.json](thinking-check.json)).
The GPU memory peak in the trace was 14,812 MiB (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
The enforced power limit was 145 W, and during the decode runs the GPU drew 138 to 140 W at 54 to 57 C (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).

Files:

- [up-attempt-1.txt](up-attempt-1.txt): output of `heavy ./llmctl up groot-19b-a3b --runtime vllm`.
- [attempts.md](attempts.md): the start attempt.
- [server-startup.log](server-startup.log): container log up to `Application startup complete`.
- [server-args.json](server-args.json): the container arguments.
- [llmctl-test.txt](llmctl-test.txt): output of `./llmctl test`.
- [probe-results.json](probe-results.json): `research/2026-11-16/llm_probe.py` results.
- [thinking-check.json](thinking-check.json): one request per thinking level.
- [gpu-trace.csv](gpu-trace.csv): nvidia-smi trace every 2 s of memory, power and temperature.
- [power-limit.txt](power-limit.txt): power limit and enforced power limit.
