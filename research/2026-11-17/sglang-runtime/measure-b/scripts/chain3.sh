#!/usr/bin/env bash
# Chain 3: the Gemma 4 E4B follow-up at the shipping settings (SWA ratio 0.1) and M2 Gemma 4 12B with FP8 KV, after
# chain 2. Needs SCRATCH in the environment.
set -uo pipefail
: "${SCRATCH:?}"
B=$SCRATCH/sglang/measure-b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1 || pgrep -f chain2.sh >/dev/null; do sleep 5; done
echo "chain3 start $(date -u +%T)"
E=$B/sglang-gemma-4-e4b
bash $B/scripts/run_b.sh $E m5-a4-swa $E/cmds/m5-a4-swa.sh sglang gemma-4-e4b --context 65536 --thinking toggle \
  --phases nearfull,grammar,probe,smoke,sustained,nearfull2 --grammar-max-tokens 2048 \
  --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
echo "e4b a4 done $(date -u +%T)"
G=$B/sglang-gemma-4-12b
bash $B/scripts/run_b.sh $G m2-fp8 $G/cmds/m2-fp8.sh sglang gemma-4-12b --context 131072 --thinking toggle \
  --phases nearfull,grammar,probe,smoke,sustained,nearfull2,m13 --grammar-max-tokens 2048 \
  --extra '{"chat_template_kwargs": {"enable_thinking": false}}' --m13-max-tokens 1024
echo "m2 fp8 done $(date -u +%T)"
echo "chain3 end $(date -u +%T)"
