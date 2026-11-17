#!/usr/bin/env bash
# Write one wave 3 docker run file with mkcmd_c.py: the model's overlay, the lane environment plus
# PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1 (the pinned-store fix measured on gpt-oss-20b), Paged Experts with
# K resident experts and Marlin, fraction 0.90 (or FRACTION from the environment), one running request, no prefill graph, and the model's own flags.
# Usage: mkw3.sh <id> <run> <K> <memory> <comment> [--network-none --volume NAME:/root/.cache/sglang] -- [model flags]
set -euo pipefail
: "${SCRATCH:?}"
C=$SCRATCH/sglang/measure-c
id=$1; run=$2; K=$3; mem=$4; comment=$5; shift 5
extra=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do extra+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
case $id in
  qwen3.6-35b-a3b) served=qwen3.6-35b-a3b; ctx=131072;;
  nemotron-3.5-lightning) served=nemotron-3.5-lightning; ctx=131072;;
  gemma-4-26b-a4b) served=gemma-4-26b-a4b; ctx=131072;;
  k2-horizon-mova-36b) served=k2-horizon-mova-36b; ctx=65536;;
  *) echo "unknown id $id"; exit 2;;
esac
[ -n "${CTX:-}" ] && ctx=$CTX
mkdir -p "$C/sglang-$id/cmds"
python3 -I "$C/scripts/mkcmd_c.py" --kind sglang --out "$C/sglang-$id/cmds/$run.sh" --comment "$comment" \
  --overlay "$id" --served "$served" --context "$ctx" --memory "$mem" \
  --env PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1 "${extra[@]}" -- \
  --mem-fraction-static "${FRACTION:-0.90}" --max-running-requests 1 --disable-prefill-cuda-graph \
  --moe-runner-backend marlin --enable-paged-experts --paged-experts-num-resident "$K" "$@"
echo "$C/sglang-$id/cmds/$run.sh"
