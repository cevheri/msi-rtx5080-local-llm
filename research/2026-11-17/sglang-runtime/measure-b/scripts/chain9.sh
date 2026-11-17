#!/usr/bin/env bash
# Chain 9, after chain 8: M15 cold start of Gemma 4 12B at its shipping settings (FP8 KV, ratio 0.1). Needs SCRATCH.
set -uo pipefail
: "${SCRATCH:?}"
B=$SCRATCH/sglang/measure-b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1 || pgrep -f chain8.sh >/dev/null; do sleep 5; done
echo "chain9 start $(date -u +%T)"
R=$B/sglang-gemma-4-12b
bash $B/scripts/run_netnone_b.sh $R m15-netnone $R/cmds/m15-netnone.sh sglang gemma-4-12b sglang-measure-cold
echo "m15 gemma-4-12b done $(date -u +%T)"
echo "chain9 end $(date -u +%T)"
