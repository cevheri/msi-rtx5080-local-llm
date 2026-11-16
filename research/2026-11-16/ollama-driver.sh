#!/usr/bin/env bash
# Measures the remaining Ollama tags one at a time, each after its pull has finished.
set -uo pipefail
B=$SCRATCH/bench
PULL=$SESSION/tasks/bwzwyg9st.output
run() {  # key tag ctx kv long extra oc
  local key=$1 tag=$2
  echo "=== $key: waiting for the pull"
  timeout 3600 bash -c "until grep -q '^$tag rc=' $PULL; do sleep 10; done" || { echo "=== $key: pull wait timed out"; return; }
  if ! grep -q "^$tag rc=0" "$PULL"; then echo "=== $key: pull failed: $(grep "^$tag rc=" "$PULL")"; return; fi
  echo "=== $key: measuring"
  bash "$B/measure-ollama.sh" "$@" > "$B/runs/ollama-$key.out" 2>&1
  echo "=== $key: exit $?"
  grep -E '^load |100% GPU|CPU|native tool probe|^probe exit|^opencode exit|^tool_use|^answer|^GPU peak|^container peak|^ok:|^tool:|^decode:|^long ' "$B/runs/ollama-$key.out" | cut -c1-220
  bash "$B/save-ollama.sh" "$key" > /dev/null 2>&1 && echo "=== $key: saved"
}
run gemma4-12b gemma4:12b 65536 f16 "8000,30000,48000" '{"reasoning_effort": "none"}' 'ollama-local/gemma4-12b#none'
run ornith-9b ornith:9b 131072 f16 "8000,30000,100000" '{}' 'ollama-local/ornith-9b'
run ministral-3-8b ministral-3:8b 65536 q8_0 "8000,30000,55000" '{}' 'ollama-local/ministral-3-8b'
run qwen2.5-7b qwen2.5:7b 32768 f16 "8000,24000" '{}' 'ollama-local/qwen2.5-7b'
echo "=== all done"
