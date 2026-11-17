#!/usr/bin/env bash
# Chain 1: the gpt-oss-20b network-none starts (with and without TIKTOKEN_ENCODINGS_BASE), the vLLM gpt-oss-20b
# network-none start, and the M10 Paged Experts sweep (K 16, 20, 28), one after another; waits for the GPU to be free
# of measurement containers first. Needs SCRATCH in the environment.
set -uo pipefail
: "${SCRATCH:?}"
B=$SCRATCH/sglang/measure-b
G=$B/sglang-gpt-oss-20b; V=$B/vllm-gpt-oss-20b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1; do sleep 5; done
echo "chain1 start $(date -u +%T)"
bash $B/scripts/run_netnone_b.sh $G b0-netnone-tik $G/cmds/b0-netnone-tik.sh sglang gpt-oss-20b sglang-measure-cold
echo "netnone-tik done $(date -u +%T)"
bash $B/scripts/run_netnone_b.sh $G b0-netnone-notik $G/cmds/b0-netnone-notik.sh sglang gpt-oss-20b sglang-measure-cold2
echo "netnone-notik done $(date -u +%T)"
bash $B/scripts/run_netnone_b.sh $V netnone $V/cmds/netnone.sh vllm gpt-oss-20b vllm-measure-cold vllm-measure-cold-fi
echo "vllm netnone done $(date -u +%T)"
for K in 16 20 28; do
  bash $B/scripts/run_b.sh $G m10-k$K $G/cmds/m10-k$K.sh sglang gpt-oss-20b --context 131072 --thinking effort --phases probe
  echo "m10 K=$K done $(date -u +%T)"
done
echo "chain1 end $(date -u +%T)"
