#!/usr/bin/env bash
# Chain 4, after chain 3: M3 Groot, M4 DeepSeek-R1 on SGLang (with the token-count check and the M9 tools request),
# the LD11 vLLM DeepSeek-R1 retry 1, M13 K2 Horizon 7B with BF16 KV, and M12 Qwen3.5-9B. Needs SCRATCH.
set -uo pipefail
: "${SCRATCH:?}"
B=$SCRATCH/sglang/measure-b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1 || pgrep -f chain3.sh >/dev/null; do sleep 5; done
echo "chain4 start $(date -u +%T)"
G=$B/sglang-groot-19b-a3b
bash $B/scripts/run_b.sh $G m3-a1 $G/cmds/m3-a1.sh sglang groot-19b-a3b --context 131072 --thinking toggle \
  --phases nearfull,grammar,probe,smoke,sustained,nearfull2 --grammar-max-tokens 2048 \
  --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
echo "m3 done $(date -u +%T)"
D=$B/sglang-deepseek-r1-0528-qwen3-8b
bash $B/scripts/run_b.sh $D m4-a1 $D/cmds/m4-a1.sh sglang deepseek-r1-8b --context 65536 --thinking always --chat-only \
  --phases nearfull,grammar,tokcheck,probe,smoke,sustained,nearfull2 --grammar-max-tokens 4096
echo "m4 done $(date -u +%T)"
V=$B/vllm-deepseek-r1-0528-qwen3-8b
bash $B/scripts/run_b.sh $V ld11-r1 $V/cmds/ld11-r1.sh vllm deepseek-r1-8b --context 65536 --thinking always --chat-only \
  --phases tokcheck,probe,smoke
echo "ld11 r1 done $(date -u +%T)"
K=$B/sglang-k2-horizon-7b
bash $B/scripts/run_b.sh $K m13-bf16 $K/cmds/m13-bf16.sh sglang k2-horizon-7b --context 32768 --thinking always \
  --phases probe,m13 --m13-max-tokens 4096
echo "m13 k2 bf16 done $(date -u +%T)"
Q=$B/sglang-qwen3.5-9b
bash $B/scripts/run_b.sh $Q m12-a3 $Q/cmds/m12-a3.sh sglang qwen3.5-9b --context 131072 --thinking toggle \
  --phases nearfull,grammar,probe,smoke,sustained,nearfull2 --grammar-max-tokens 2048 \
  --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
echo "m12 qwen done $(date -u +%T)"
echo "chain4 end $(date -u +%T)"
