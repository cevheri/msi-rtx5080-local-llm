#!/usr/bin/env bash
# Usage: ready_times.sh <handson-dir>  -- seconds from docker run to SGLang's ready log line, per start.
H=$1
while read -r model log tag; do
  f=$H/logs/$model/$log.log; s=$(cat "$H/logs/$model/start-$tag.epoch")
  r=$(grep -m1 "fired up" "$f" | sed -E 's/^\[([0-9-]+ [0-9:]+)\].*/\1/')
  lw=$(grep -m1 "Load weight end" "$f" | sed -E 's/^\[([0-9-]+ [0-9:]+)\].*/\1/')
  if [ -n "$r" ]; then
    echo "$model $log: weights loaded +$(( $(date -u -d "$lw" +%s) - ${s%.*} )) s, ready line +$(( $(date -u -d "$r" +%s) - ${s%.*} )) s"
  else
    echo "$model $log: no ready line; last error: $(grep -E 'Error|error:' "$f" | grep -v -E 'CUDACachingAllocator|ErrorWrapper' | tail -1 | cut -c1-160)"
  fi
done <<'LIST'
qwen3.5-9b attempt1 a1
qwen3.5-9b attempt2 a2
qwen3.5-9b attempt3 a3
qwen3.5-9b run4-sustained a4
gpt-oss-20b attempt1 a1
gpt-oss-20b attempt2 a2
gpt-oss-20b attempt3 a3
gpt-oss-20b attempt4 a4
k2-horizon-7b attempt1 a1
k2-horizon-7b attempt2 a2
gemma-4-e4b attempt1 a1
gemma-4-e4b attempt2 a2
LIST
