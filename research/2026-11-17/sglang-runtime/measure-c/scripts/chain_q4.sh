#!/usr/bin/env bash
# Chain q4: Qwen3.6-35B-A3B under the chosen 27 GiB cap (the protocol of B's ld19-mem28) and the M15 cold start without
# a network on a new empty volume, after chain q3; then the Nemotron sweep (chain n1).
set -uo pipefail
: "${SCRATCH:?}"
C=$SCRATCH/sglang/measure-c
Q=$C/sglang-qwen3.6-35b-a3b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1 || pgrep -f chain_q3 >/dev/null; do sleep 5; done
echo "chain q4 start $(date -u +%T)"
bash $C/scripts/run_c.sh $Q w3-k88-f088-mem27 $Q/cmds/w3-k88-f088-mem27.sh sglang qwen3.6-35b-a3b --context 131072 --thinking toggle \
  --phases nearfull,grammar,probe,sustained,nearfull2 --grammar-max-tokens 2048 \
  --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
echo "w3-k88-f088-mem27 done $(date -u +%T)"
bash $C/scripts/run_netnone_c.sh $Q m15-netnone $Q/cmds/m15-netnone.sh sglang qwen3.6-35b-a3b sglang-measure-cold
echo "m15 qwen3.6 done $(date -u +%T)"
echo "chain q4 end $(date -u +%T)"
bash $C/scripts/chain_n1.sh
