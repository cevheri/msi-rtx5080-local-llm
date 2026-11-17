# DeepSeek-R1-0528-Qwen3-8B on vLLM, 2026-11-17

Result: failed: the server starts and decodes, but vLLM loads the tokenizer as a Metaspace LlamaTokenizer, so prompts lose their spaces and every reply comes back as raw byte-level BPE text

All numbers below are measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17.
Image: `vllm/vllm-openai:v0.33.0`, started with `./llmctl up deepseek-r1-0528-qwen3-8b --runtime vllm` from a pristine export of main 9a2c43a.
Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` at revision `6e8885a6ff5c1dc5201574c8fd700323f23c25fa`, quantized on load with `--quantization fp8_per_tensor`.
The shipped recipe failed to start because 4.5 GiB of KV cache was needed for 65,536 tokens and only 3.83 GiB was available at vLLM's default `--gpu-memory-utilization` of 0.92 ([attempts.md](attempts.md)).
The server started on the second of three allowed attempts with one change in an override: `--gpu-memory-utilization 0.965` added at the start of `[server] args`, context unchanged ([override.diff](override.diff), [server-args.json](server-args.json)).
The other lines of the diff are the override's generated header comment and evidence summary, not flag changes.
The third attempt was not used, because no `[server]` argument of vLLM v0.33.0 fixes the tokenizer class (see the tokenizer defect below).
The started server has `--max-model-len 65536` and a GPU KV cache of 80,064 tokens (1.22x concurrency at 64k) in 5.5 GiB, after `Model loading took 9.12 GiB memory` ([server-startup.log](server-startup.log)).
The container reached `Application startup complete` about 22 s after it started (23:45:22 to 23:45:44 UTC in the container log), with the torch.compile cache already filled by attempt 1; `./llmctl up` printed its `ready:` line 23 s after it was called.

Tokenizer defect ([tokenizer-check.json](tokenizer-check.json)):
The model's `tokenizer_config.json` names `LlamaTokenizerFast`, while its `tokenizer.json` is a byte-level BPE (Qwen3) tokenizer.
Inside the container, transformers 5.20.2 loads it through `AutoTokenizer` as `LlamaTokenizer` with a Metaspace pre-tokenizer, which encodes "Say hello world." as 4 tokens that decode to "Sayhelloworld.".
Loading the same folder with `PreTrainedTokenizerFast` keeps the byte-level tokenizer and round-trips the sentence, so the files are intact and the class choice is the cause.
Through the API, chat content and reasoning (streamed and not) and `/v1/completions` text all contain the byte-level markers U+0120 for a space and U+010A for a newline, for example the content `ĊTheĠsumĠofĠ17ĠandĠ25ĠisĠ42.`; the model also echoed the prompt without spaces in its reasoning.
The recipe's `[server] args` cannot point vLLM at a corrected tokenizer folder, because llmctl mounts only the Hugging Face cache, read-only, and `--tokenizer-mode` offers no mode that overrides the class.

Decode: 72.1 tok/s, the median of three 1,500-token runs (64.0, 72.1 and 72.1 tok/s), sent as the default request with no thinking field; all 1,499 tokens after the first were counted as reasoning ([probe-results.json](probe-results.json)).
The first run was slower because the GPU drew only about 92 W for its first 10 s before rising to 138 W.
Token throughput does not depend on the tokenizer defect, but the prompt token counts do: the 8k and 30k prompts were 6,903 and 25,845 tokens instead of about 8,000 and 30,000, because the Metaspace tokenizer merges words.
Time to first token: 1.15 s for a fresh 6,903-token prompt and 0.049 s when repeated, 6.62 s for a fresh 25,845-token prompt and 0.089 s when repeated.
The shipped probe `research/2026-11-16/llm_probe.py` stopped with HTTP 400 at its tool-call step and wrote no results ([probe-attempt-1.txt](probe-attempt-1.txt)), which is expected for a chat-only recipe; the numbers above come from a copy that records the tool step's error and goes on, with no other change ([probe-chat-only.diff](probe-chat-only.diff)).
The recorded tool step error is `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`.
`./llmctl test` exited 0: reply ok, tool call skipped because the recipe is chat only, decode 53.1 tok/s over 512 tokens ([llmctl-test.txt](llmctl-test.txt)).
That reply check passed on the content `'ĊOK'`, where `Ċ` is the byte-level newline marker, so it does not catch the tokenizer defect.
Thinking check (type always, one level): the default request returned HTTP 200 with a 4,106-character `message.reasoning`, no `reasoning_content`, and a non-empty content that holds the answer 42 in byte-level markers ([thinking-check.json](thinking-check.json)).
vLLM reports `reasoning_tokens` here (1,168 of 1,193 completion tokens), unlike the k2-horizon-7b record.
GPU memory peak: 15,750 MiB during the 30k prompt, 15,454 to 15,474 MiB from startup until then ([gpu-trace.csv](gpu-trace.csv)).
Power: enforced limit 140 W ([power-limit.txt](power-limit.txt)); the draw during the decode runs was 138 to 141 W at 59 to 64 C after the first 10 s at about 92 W and 55 C.

Files:

- [attempts.md](attempts.md): both start attempts with their exact log lines.
- [up-attempt-1.txt](up-attempt-1.txt), [server-attempt-1.log](server-attempt-1.log) and [gpu-trace-attempt-1.csv](gpu-trace-attempt-1.csv): the failed first start, its full container log and its partial GPU trace; `up-attempt-1.txt` is empty because the command that wrote it was stopped.
- [up-attempt-2.txt](up-attempt-2.txt), [server-startup.log](server-startup.log) and [server-args.json](server-args.json): the start that passed.
- [override.diff](override.diff): the shipped recipe against the override that started.
- [llmctl-test.txt](llmctl-test.txt), [probe-results.json](probe-results.json), [probe-attempt-1.txt](probe-attempt-1.txt), [probe-chat-only.diff](probe-chat-only.diff) and [thinking-check.json](thinking-check.json): the checks.
- [tokenizer-check.json](tokenizer-check.json): the tokenizer defect, in the container and through the API.
- [gpu-trace.csv](gpu-trace.csv) and [power-limit.txt](power-limit.txt): GPU memory, power and temperature every 2 s from attempt 2 on, and the power limit.
