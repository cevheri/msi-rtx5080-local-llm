#!/usr/bin/env bash
# Chain 7, after chain 6: M2 Gemma 4 12B with BF16 KV at SWA ratio 0.06, the full sequence with M13. Needs SCRATCH.
set -uo pipefail
: "${SCRATCH:?}"
B=$SCRATCH/sglang/measure-b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1 || pgrep -f chain6.sh >/dev/null; do sleep 5; done
echo "chain7 start $(date -u +%T)"
G=$B/sglang-gemma-4-12b
bash $B/scripts/run_b.sh $G m2-bf16 $G/cmds/m2-bf16.sh sglang gemma-4-12b --context 131072 --thinking toggle \
  --phases nearfull,grammar,probe,smoke,sustained,nearfull2,m13 --grammar-max-tokens 2048 \
  --extra '{"chat_template_kwargs": {"enable_thinking": false}}' --m13-max-tokens 1024
echo "m2 bf16 done $(date -u +%T)"
echo "chain7 end $(date -u +%T)"
