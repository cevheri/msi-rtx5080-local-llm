#!/usr/bin/env bash
# Chain d1: K2 Horizon 7B at fraction 0.88 with the full protocol of B's shipping recipes (D8's flushed reading, a 95%
# prompt, the grammar requests, the probe, the api probe, 300 s of load and a second 95% prompt), once no measurement
# container exists. Needs SCRATCH and HOME in the environment.
set -uo pipefail
: "${SCRATCH:?}"
M=$SCRATCH/sglang/measure-d
K=$M/sglang-k2-horizon-7b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1; do sleep 5; done
echo "chain d1 start $(date -u +%T)"
bash $M/scripts/run_d.sh $K d1-f088 $K/cmds/d1-f088.sh sglang k2-horizon-7b --context 65536 --thinking always \
  --phases flushready,nearfull,grammar,probe,smoke,sustained,nearfull2 --grammar-max-tokens 2048
echo "d1-f088 done $(date -u +%T)"
echo "chain d1 end $(date -u +%T)"
