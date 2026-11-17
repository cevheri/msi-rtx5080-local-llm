#!/usr/bin/env bash
# Chain d2: the margin protocol only (D8's flushed reading, a 95% prompt, the first grammar request, a forced tool
# request) and the repository probe, once each, for Qwen3.5-9B, Gemma 4 12B and Gemma 4 E4B at B's draft settings with
# all eight variables of the runtime file, one start after another, once no measurement container exists.
# Needs SCRATCH and HOME in the environment.
set -uo pipefail
: "${SCRATCH:?}"
M=$SCRATCH/sglang/measure-d
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1 || pgrep -f chain_d1.sh >/dev/null; do sleep 5; done
echo "chain d2 start $(date -u +%T)"
for m in qwen3.5-9b gemma-4-12b gemma-4-e4b; do
  case $m in qwen3.5-9b|gemma-4-12b) ctx=131072;; gemma-4-e4b) ctx=65536;; esac
  R=$M/sglang-$m
  bash $M/scripts/run_d.sh $R d2-env $R/cmds/d2-env.sh sglang $m --context $ctx --thinking toggle \
    --phases flushready,nearfull,grammar,probe --grammar-max-tokens 2048 \
    --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
  echo "$m d2-env done $(date -u +%T)"
done
echo "chain d2 end $(date -u +%T)"
