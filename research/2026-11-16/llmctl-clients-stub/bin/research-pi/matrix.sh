#!/usr/bin/env bash
# Thinking matrix: each scenario runs the same headless write-and-read prompt against the stub.
set -u
R=$SCRATCH/clients/research-pi
C=$R/configs
PROMPT="Create hello.py containing print('hello'), then read hello.py back and reply with its exact contents."
FLAGS=(--mode json --no-session --no-approve --no-context-files --offline --tools read,write)
LLAMA='{"mode":"tools","model_id":"local","reasoning_field":"reasoning_content"}'
VLLM_GPT='{"mode":"tools","model_id":"gpt-oss-20b","reasoning_field":"reasoning"}'
VLLM_GROOT='{"mode":"tools","model_id":"groot-19b-a3b","reasoning_field":"reasoning"}'
NOREASON='{"mode":"tools","model_id":"local","reasoning_field":null}'
run() { "$R/run_pi.sh" "$@"; }
run M-toggle-off        $C/toggle "$LLAMA" "${FLAGS[@]}" --model llmctl/local --thinking off "$PROMPT"
run M-toggle-medium     $C/toggle "$LLAMA" "${FLAGS[@]}" --model llmctl/local --thinking medium "$PROMPT"
run M-toggle-default    $C/toggle "$LLAMA" "${FLAGS[@]}" --model llmctl/local "$PROMPT"
run M-toggle-high       $C/toggle "$LLAMA" "${FLAGS[@]}" --model llmctl/local --thinking high "$PROMPT"
run M-toggle-suffix-off $C/toggle "$LLAMA" "${FLAGS[@]}" --model llmctl/local:off "$PROMPT"
run M-bolt-off          $C/bolt "$LLAMA" "${FLAGS[@]}" --model llmctl/local --thinking off "$PROMPT"
run M-bolt-medium       $C/bolt "$LLAMA" "${FLAGS[@]}" --model llmctl/local --thinking medium "$PROMPT"
run M-bolt-high         $C/bolt "$LLAMA" "${FLAGS[@]}" --model llmctl/local --thinking high "$PROMPT"
run M-effort-off        $C/effort-none-low-medium "$LLAMA" "${FLAGS[@]}" --model llmctl/local --thinking off "$PROMPT"
run M-effort-low        $C/effort-none-low-medium "$LLAMA" "${FLAGS[@]}" --model llmctl/local --thinking low "$PROMPT"
run M-effort-medium     $C/effort-none-low-medium "$LLAMA" "${FLAGS[@]}" --model llmctl/local --thinking medium "$PROMPT"
run M-effort-high       $C/effort-none-low-medium "$LLAMA" "${FLAGS[@]}" --model llmctl/local --thinking high "$PROMPT"
run M-always-default    $C/always-on "$LLAMA" "${FLAGS[@]}" --model llmctl/local "$PROMPT"
run M-always-off        $C/always-on "$LLAMA" "${FLAGS[@]}" --model llmctl/local --thinking off "$PROMPT"
run M-always7b-default  $C/always-on-7b "$LLAMA" "${FLAGS[@]}" --model llmctl/local "$PROMPT"
run M-gptoss-low        $C/gpt-oss-vllm "$VLLM_GPT" "${FLAGS[@]}" --model llmctl/gpt-oss-20b --thinking low "$PROMPT"
run M-gptoss-medium     $C/gpt-oss-vllm "$VLLM_GPT" "${FLAGS[@]}" --model llmctl/gpt-oss-20b --thinking medium "$PROMPT"
run M-gptoss-high       $C/gpt-oss-vllm "$VLLM_GPT" "${FLAGS[@]}" --model llmctl/gpt-oss-20b --thinking high "$PROMPT"
run M-gptoss-off        $C/gpt-oss-vllm "$VLLM_GPT" "${FLAGS[@]}" --model llmctl/gpt-oss-20b --thinking off "$PROMPT"
run M-grootvllm-off     $C/toggle-vllm "$VLLM_GROOT" "${FLAGS[@]}" --model llmctl/groot-19b-a3b --thinking off "$PROMPT"
run M-grootvllm-medium  $C/toggle-vllm "$VLLM_GROOT" "${FLAGS[@]}" --model llmctl/groot-19b-a3b --thinking medium "$PROMPT"
run M-nothink-default   $C/no-thinking "$NOREASON" "${FLAGS[@]}" --model llmctl/local "$PROMPT"
run M-nothink-high      $C/no-thinking "$NOREASON" "${FLAGS[@]}" --model llmctl/local --thinking high "$PROMPT"
run M-chatonly-notools  $C/chat-only '{"mode":"text","model_id":"local","reasoning_field":"reasoning_content"}' --mode json --no-session --no-approve --no-context-files --offline --no-tools --model llmctl/local "Say hello."
run M-baseline-nocompat $C/baseline-nocompat "$LLAMA" "${FLAGS[@]}" --model llmctl/local "$PROMPT"
