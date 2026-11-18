#!/usr/bin/env bash
# Chain q1: the Qwen3.6-35B-A3B K sweep (K 64, 72, 80; probe only, as B's M10), one start after another, once no
# measurement container runs. Needs SCRATCH in the environment.
set -uo pipefail
: "${SCRATCH:?}"
C=$SCRATCH/sglang/measure-c
Q=$C/sglang-qwen3.6-35b-a3b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1; do sleep 5; done
echo "chain q1 start $(date -u +%T)"
for K in 64 72 80; do
  bash $C/scripts/run_c.sh $Q m11-k$K $Q/cmds/m11-k$K.sh sglang qwen3.6-35b-a3b --context 131072 --thinking toggle --phases probe \
    --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
  echo "m11 K=$K done $(date -u +%T)"
done
echo "chain q1 end $(date -u +%T)"
