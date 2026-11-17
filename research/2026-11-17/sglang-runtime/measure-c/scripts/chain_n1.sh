#!/usr/bin/env bash
# Chain n1, run by chain q4 once the Qwen3.6 starts are done: the Nemotron 3.5 Lightning K sweep with BF16 KV (K 64, 72,
# 80) and then the Gemma 4 26B-A4B K sweep with FP8 KV and SWA ratio 0.1 (K 56, 64, 72), probe only, as B's M10.
set -uo pipefail
: "${SCRATCH:?}"
C=$SCRATCH/sglang/measure-c
N=$C/sglang-nemotron-3.5-lightning
G=$C/sglang-gemma-4-26b-a4b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1 ; do sleep 5; done
echo "chain n1 start $(date -u +%T)"
for K in 64 72 80; do
  bash $C/scripts/run_c.sh $N m11-bf16-k$K $N/cmds/m11-bf16-k$K.sh sglang nemotron-3.5-lightning --context 131072 --thinking toggle --phases probe \
    --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
  echo "nemotron m11 bf16 K=$K done $(date -u +%T)"
done
for K in 56 64 72; do
  bash $C/scripts/run_c.sh $G m11-fp8-k$K $G/cmds/m11-fp8-k$K.sh sglang gemma-4-26b-a4b --context 131072 --thinking toggle --phases probe \
    --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
  echo "gemma m11 fp8 K=$K done $(date -u +%T)"
done
echo "chain n1 end $(date -u +%T)"
