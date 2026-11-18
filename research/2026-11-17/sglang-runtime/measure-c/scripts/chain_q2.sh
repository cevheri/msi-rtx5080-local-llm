#!/usr/bin/env bash
# Chain q2: the Qwen3.6-35B-A3B K sweep with BF16 KV (K 88, 96, 104; probe only, as B's M10), after chain q1's K 64 start
# (FP8 KV, chosen by the checkpoint's quantization config) is gone. Needs SCRATCH in the environment.
set -uo pipefail
: "${SCRATCH:?}"
C=$SCRATCH/sglang/measure-c
Q=$C/sglang-qwen3.6-35b-a3b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1 || pgrep -f "run_c.sh .*m11-k64" >/dev/null; do sleep 5; done
echo "chain q2 start $(date -u +%T)"
for K in 88 96 104; do
  bash $C/scripts/run_c.sh $Q m11-bf16-k$K $Q/cmds/m11-bf16-k$K.sh sglang qwen3.6-35b-a3b --context 131072 --thinking toggle --phases probe \
    --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
  echo "m11 bf16 K=$K done $(date -u +%T)"
done
echo "chain q2 end $(date -u +%T)"
