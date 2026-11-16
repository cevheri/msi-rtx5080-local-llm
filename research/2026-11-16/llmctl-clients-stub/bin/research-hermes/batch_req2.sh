#!/bin/bash
S=$SCRATCH/clients/research-hermes
C=$S/configs/cases; P="Create hello.py that prints hello, then read it back and show its contents"
R=("$S/bin/run_case.sh" req2)
for n in r02-mo-full r03-mo-bare r04-mo-bare-effort-medium r05-ctk-on r06-ctk-off r07-bolt-medium r08-extra-maxtok r09-model-maxtok r10-effort-none r11-override-low r16-supports-tools-false r17-nostream r18-stale-base-url r19-ctx-32k r21-ctk-on-keep-effort; do
  "${R[@]}" $n $C/$n.yaml llamacpp local 131072 "" -z "$P" -t file
done
"${R[@]}" r12-baseline-for-flag $C/r12-baseline-for-flag.yaml llamacpp local 131072 "" -z "$P" -t file --reasoning high
"${R[@]}" r13-mo-bare-for-flag $C/r13-mo-bare-for-flag.yaml llamacpp local 131072 "" -z "$P" -t file --reasoning high
"${R[@]}" r14-vllm-gptoss $C/r14-vllm-gptoss.yaml vllm gpt-oss-20b 65536 "" -z "$P" -t file
"${R[@]}" r15-vllm-effort-none $C/r15-vllm-effort-none.yaml vllm gpt-oss-20b 65536 "--reject-effort none" -z "$P" -t file
"${R[@]}" r20-vllm-effort-high $C/r20-vllm-effort-high.yaml vllm gpt-oss-20b 65536 "" -z "$P" -t file
"${R[@]}" r22-platform-toolsets-empty $C/r22-platform-toolsets-empty.yaml llamacpp local 131072 "" -z "$P"
"${R[@]}" r23-baseline-no-t $C/r12-baseline-for-flag.yaml llamacpp local 131072 "" -z "$P"
