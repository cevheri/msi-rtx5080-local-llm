#!/usr/bin/env bash
# Chain e1: one warm start of measurement C's shipped Qwen3.6-35B-A3B settings (K 88, Marlin, fraction 0.88, BF16 KV,
# 8 Mamba slots, 128k, the runtime file's eight variables, the 27 GiB cap, the volume sglang-research-cache), then in this
# order: the free memory right after ready without a flush (the poller), D8's POST /flush_cache and the free memory after
# it (seq_d.py's flushready), a fresh prompt of about 95% of the context, Hermes' title request (the first grammar
# request), a forced tool request, 300 s of sustained load with the hands-on sustained.py, and a second near-full prompt,
# with the 1 s GPU and cgroup trace throughout; once no measurement container exists. Needs SCRATCH and HOME in the
# environment.
set -uo pipefail
: "${SCRATCH:?}"
E=$SCRATCH/sglang/measure-e
Q=$E/sglang-qwen3.6-35b-a3b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1; do sleep 5; done
echo "chain e1 start $(date -u +%T)"
bash $E/scripts/run_e.sh $Q e1-k88-f088-mem27 $Q/cmds/e1-k88-f088-mem27.sh sglang qwen3.6-35b-a3b --context 131072 \
  --thinking toggle --phases flushready,nearfull,grammar,sustained,nearfull2 --grammar-max-tokens 2048 \
  --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
echo "e1-k88-f088-mem27 done $(date -u +%T)"
echo "chain e1 end $(date -u +%T)"
