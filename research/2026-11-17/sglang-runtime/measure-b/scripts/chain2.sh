#!/usr/bin/env bash
# Chain 2: M5 (Gemma 4 E4B retry, one attempt) and M1 (Mellum2.1-12B), one after another, after chain 1's containers
# are gone. Needs SCRATCH in the environment.
set -uo pipefail
: "${SCRATCH:?}"
B=$SCRATCH/sglang/measure-b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1 || pgrep -f chain1.sh >/dev/null; do sleep 5; done
echo "chain2 start $(date -u +%T)"
E=$B/sglang-gemma-4-e4b
bash $B/scripts/run_b.sh $E m5-a3 $E/cmds/m5-a3.sh sglang gemma-4-e4b --context 65536 --thinking toggle \
  --phases nearfull,grammar,probe,smoke,sustained,nearfull2 --grammar-max-tokens 2048 \
  --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
echo "m5 done $(date -u +%T)"
M=$B/sglang-mellum2.1-12b
bash $B/scripts/run_b.sh $M m1-a1 $M/cmds/m1-a1.sh sglang mellum2.1 --context 131072 --thinking toggle \
  --phases nearfull,grammar,probe,smoke,sustained,nearfull2 --grammar-max-tokens 4096 \
  --extra '{"chat_template_kwargs": {"enable_thinking": true}}'
echo "m1 done $(date -u +%T)"
echo "chain2 end $(date -u +%T)"
