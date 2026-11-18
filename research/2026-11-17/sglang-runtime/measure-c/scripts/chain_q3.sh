#!/usr/bin/env bash
# Chain q3: the Qwen3.6-35B-A3B full protocol at K 88, fraction 0.88, BF16 KV (B's shipping-recipe sequence: a 95%
# prompt, the grammar requests, the probe, the api probe, 300 s of load and a second 95% prompt), after chain q2.
set -uo pipefail
: "${SCRATCH:?}"
C=$SCRATCH/sglang/measure-c
Q=$C/sglang-qwen3.6-35b-a3b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1 || pgrep -f chain_q2.sh >/dev/null; do sleep 5; done
echo "chain q3 start $(date -u +%T)"
bash $C/scripts/run_c.sh $Q w3-k88-f088 $Q/cmds/w3-k88-f088.sh sglang qwen3.6-35b-a3b --context 131072 --thinking toggle \
  --phases nearfull,grammar,probe,smoke,sustained,nearfull2 --grammar-max-tokens 2048 \
  --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
echo "w3-k88-f088 done $(date -u +%T)"
echo "chain q3 end $(date -u +%T)"
