#!/usr/bin/env bash
# Chain 5, after chain 4: LD11 vLLM retry 2 (tokenizer overlay), M2 Gemma 4 12B attempt 2 (audio_config null, FP8 KV),
# and M10 K 25 for gpt-oss-20b. Needs SCRATCH.
set -uo pipefail
: "${SCRATCH:?}"
B=$SCRATCH/sglang/measure-b
while docker inspect sglang-measure >/dev/null 2>&1 || docker inspect vllm-measure >/dev/null 2>&1 || pgrep -f chain4.sh >/dev/null; do sleep 5; done
echo "chain5 start $(date -u +%T)"
V=$B/vllm-deepseek-r1-0528-qwen3-8b
bash $B/scripts/run_b.sh $V ld11-r2 $V/cmds/ld11-r2.sh vllm deepseek-r1-8b --context 65536 --thinking always --chat-only \
  --phases tokcheck,probe,smoke
echo "ld11 r2 done $(date -u +%T)"
G=$B/sglang-gemma-4-12b
bash $B/scripts/run_b.sh $G m2-fp8-b $G/cmds/m2-fp8-b.sh sglang gemma-4-12b --context 131072 --thinking toggle \
  --phases nearfull,grammar,probe,smoke,sustained,nearfull2,m13 --grammar-max-tokens 2048 \
  --extra '{"chat_template_kwargs": {"enable_thinking": false}}' --m13-max-tokens 1024
echo "m2 fp8 b done $(date -u +%T)"
O=$B/sglang-gpt-oss-20b
bash $B/scripts/run_b.sh $O m10-k25 $O/cmds/m10-k25.sh sglang gpt-oss-20b --context 131072 --thinking effort --phases probe
echo "m10 k25 done $(date -u +%T)"
echo "chain5 end $(date -u +%T)"
