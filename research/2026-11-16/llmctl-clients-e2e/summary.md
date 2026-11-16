# llmctl clients e2e

| Recipe | Step | Pass | Reasons | Seconds |
|---|---|---|---|---|
| 01-qwen3.8-27b-llamacpp | up | Yes |  | 6 |
| 01-qwen3.8-27b-llamacpp | gateway | Yes |  | 0 |
| 01-qwen3.8-27b-llamacpp | status | Yes |  | 1 |
| 01-qwen3.8-27b-llamacpp | config | Yes |  | 0 |
| 01-qwen3.8-27b-llamacpp | hermes-config-model | Yes |  | 0 |
| 01-qwen3.8-27b-llamacpp | hermes-config-providers | Yes |  | 0 |
| 01-qwen3.8-27b-llamacpp | hermes-config-model_overrides | Yes |  | 0 |
| 01-qwen3.8-27b-llamacpp | hermes-config-gateway | Yes |  | 0 |
| 01-qwen3.8-27b-llamacpp | opencode | Yes |  | 15 |
| 01-qwen3.8-27b-llamacpp | hermes-z | Yes |  | 12 |
| 01-qwen3.8-27b-llamacpp | hermes-chat | Yes |  | 10 |
| 01-qwen3.8-27b-llamacpp | pi | No | tool_execution_end write has isError True | 23 |
| 01-qwen3.8-27b-llamacpp | probes | Yes |  | 5 |
| 02-k2-horizon-7b-llamacpp | up | Yes |  | 3 |
| 02-k2-horizon-7b-llamacpp | status | Yes |  | 1 |
| 02-k2-horizon-7b-llamacpp | config | Yes |  | 0 |
| 02-k2-horizon-7b-llamacpp | hermes-config-model | Yes |  | 0 |
| 02-k2-horizon-7b-llamacpp | hermes-config-providers | Yes |  | 0 |
| 02-k2-horizon-7b-llamacpp | hermes-config-model_overrides | Yes |  | 0 |
| 02-k2-horizon-7b-llamacpp | hermes-config-gateway | Yes |  | 0 |
| 02-k2-horizon-7b-llamacpp | opencode | Yes |  | 13 |
| 02-k2-horizon-7b-llamacpp | hermes-z | No | hello.py is missing or lacks def add and add(2, 3) | 6 |
| 02-k2-horizon-7b-llamacpp | hermes-chat | Yes |  | 8 |
| 02-k2-horizon-7b-llamacpp | pi | Yes |  | 5 |
| 02-k2-horizon-7b-llamacpp | probes | Yes |  | 12 |
| 03-gpt-oss-20b-vllm | up | Yes |  | 27 |
| 03-gpt-oss-20b-vllm | status | Yes |  | 1 |
| 03-gpt-oss-20b-vllm | config | Yes |  | 0 |
| 03-gpt-oss-20b-vllm | hermes-config-model | Yes |  | 0 |
| 03-gpt-oss-20b-vllm | hermes-config-providers | Yes |  | 0 |
| 03-gpt-oss-20b-vllm | hermes-config-model_overrides | Yes |  | 0 |
| 03-gpt-oss-20b-vllm | hermes-config-gateway | Yes |  | 0 |
| 03-gpt-oss-20b-vllm | opencode | Yes |  | 17 |
| 03-gpt-oss-20b-vllm | hermes-z | Yes |  | 4 |
| 03-gpt-oss-20b-vllm | hermes-chat | Yes |  | 7 |
| 03-gpt-oss-20b-vllm | pi | Yes |  | 6 |
| 03-gpt-oss-20b-vllm | probes | Yes |  | 1 |
| 04-qwen3.6-35b-a3b-llamacpp | up | Yes |  | 9 |
| 04-qwen3.6-35b-a3b-llamacpp | status | Yes |  | 1 |
| 04-qwen3.6-35b-a3b-llamacpp | config | Yes |  | 0 |
| 04-qwen3.6-35b-a3b-llamacpp | hermes-config-model | Yes |  | 0 |
| 04-qwen3.6-35b-a3b-llamacpp | hermes-config-providers | Yes |  | 0 |
| 04-qwen3.6-35b-a3b-llamacpp | hermes-config-model_overrides | Yes |  | 0 |
| 04-qwen3.6-35b-a3b-llamacpp | hermes-config-gateway | Yes |  | 0 |
| 04-qwen3.6-35b-a3b-llamacpp | opencode | Yes |  | 9 |
| 04-qwen3.6-35b-a3b-llamacpp | hermes-z | No | hello.py is missing or lacks def add and add(2, 3) | 4 |
| 04-qwen3.6-35b-a3b-llamacpp | hermes-chat | No | tool_result write_file has is_error true; hello.py is missing or lacks def add and add(2, 3) | 4 |
| 04-qwen3.6-35b-a3b-llamacpp | pi | Yes |  | 4 |
| 04-qwen3.6-35b-a3b-llamacpp | probes | Yes |  | 1 |
| round-trip | up | Yes |  | 1 |
| round-trip | opencode | Yes |  | 5 |
| round-trip | hermes-z | No | hello.py is missing or lacks def add and add(2, 3) | 4 |
| round-trip | pi | Yes |  | 3 |
| round-trip | restore | Yes |  | 0 |
| - | down | Yes |  | 1 |
