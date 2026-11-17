#!/usr/bin/env bash
# E2 extra: replay of earlier assistant turns in a continued print-mode session (-c), vLLM K2 Horizon 7B entry.
# <variant>: "reasoning" (first reply streams `reasoning`) or "plain" (first reply has no reasoning).
set -u
EXP=$(cd "$(dirname "$0")/.." && pwd)
V=$1; H=$EXP/homes/e2c-$V; WD=$EXP/work/e2c-$V
rm -rf "$H" "$WD"; mkdir -p "$H/.kimi-code" "$H/.cache" "$WD"
{ echo 'default_model = "llmctl/k2-horizon-7b"'; echo; cat $EXP/entries/vllm/k2-horizon-7b.toml; } > "$H/.kimi-code/config.toml"; chmod 600 "$H/.kimi-code/config.toml"
R1='{"type": "text", "text": "PONG-ONE", "reasoning": "First answer, thinking briefly.", "field": "reasoning"}'
[ "$V" = plain ] && R1='{"type": "text", "text": "PONG-ONE"}'
for step in 1 2; do
  C=e2c-$V-$step
  cat > $EXP/runs/$C.stub.json <<EOS
{"record": "$EXP/runs/$C/records.jsonl", "canned_reasoning": "First answer, thinking briefly.", "rules": [], "default": $( [ $step = 1 ] && echo "$R1" || echo '{"type": "text", "text": "PONG-TWO", "reasoning": "Second.", "field": "reasoning"}' )}
EOS
  extra=(); [ $step = 2 ] && extra=(-c)
  HOMEDIR=$H WORKDIR=$WD BPORT=8000 STUBCFG=$EXP/runs/$C.stub.json TO=120 heavy bash $EXP/bin/case.sh $C kimi -p "Reply with the stub word, turn $step." --output-format stream-json "${extra[@]}"
done
