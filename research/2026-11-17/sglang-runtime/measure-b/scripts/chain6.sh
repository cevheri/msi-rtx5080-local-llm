#!/usr/bin/env bash
# Chain 6, after chain 5: M15 cold starts with no network and a new empty volume for K2 Horizon 7B, Qwen3.5-9B and
# Gemma 4 E4B, each volume listed and removed by run_netnone_b.sh. Needs SCRATCH.
set -uo pipefail
: "${SCRATCH:?}"
B=$SCRATCH/sglang/measure-b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1 || pgrep -f chain5.sh >/dev/null; do sleep 5; done
echo "chain6 start $(date -u +%T)"
for m in k2-horizon-7b qwen3.5-9b gemma-4-e4b; do
  case $m in k2-horizon-7b) s=k2-horizon-7b;; qwen3.5-9b) s=qwen3.5-9b;; gemma-4-e4b) s=gemma-4-e4b;; esac
  R=$B/sglang-$m
  bash $B/scripts/run_netnone_b.sh $R m15-netnone $R/cmds/m15-netnone.sh sglang $s sglang-measure-cold
  echo "m15 $m done $(date -u +%T)"
done
echo "chain6 end $(date -u +%T)"
