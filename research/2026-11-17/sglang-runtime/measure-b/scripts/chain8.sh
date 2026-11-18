#!/usr/bin/env bash
# Chain 8, after chain 7: LD19 check of gpt-oss-20b at K 24 under a 28 GiB cap, with the margin protocol and M12.
set -uo pipefail
: "${SCRATCH:?}"
B=$SCRATCH/sglang/measure-b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1 || pgrep -f chain7.sh >/dev/null; do sleep 5; done
echo "chain8 start $(date -u +%T)"
G=$B/sglang-gpt-oss-20b
bash $B/scripts/run_b.sh $G ld19-mem28 $G/cmds/ld19-mem28.sh sglang gpt-oss-20b --context 131072 --thinking effort \
  --phases nearfull,grammar,probe,sustained,nearfull2 --grammar-max-tokens 2048
echo "ld19 mem28 done $(date -u +%T)"
echo "chain8 end $(date -u +%T)"
