#!/usr/bin/env bash
# Usage: trace.sh <container-name> <out.csv> [max-seconds]
# Every second: GPU memory, power, temperature, utilization, and the container's cgroup memory (current, anon, file, peak).
name=$1; out=$2; max=${3:-3600}
echo "epoch,gpu_mem_used_mib,gpu_power_w,gpu_temp_c,gpu_util_pct,cg_current_b,cg_anon_b,cg_file_b,cg_peak_b" > "$out"
end=$(( $(date +%s) + max )); id=
while [ "$(date +%s)" -lt "$end" ]; do
  [ -z "$id" ] && id=$(docker inspect --format '{{.Id}}' "$name" 2>/dev/null)
  g=$(nvidia-smi --query-gpu=memory.used,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  cg=/sys/fs/cgroup/system.slice/docker-$id.scope
  if [ -n "$id" ] && [ -r "$cg/memory.current" ]; then
    cur=$(cat "$cg/memory.current"); peak=$(cat "$cg/memory.peak")
    anon=$(awk '$1=="anon"{print $2}' "$cg/memory.stat"); file=$(awk '$1=="file"{print $2}' "$cg/memory.stat")
  else cur=; peak=; anon=; file=; id=; fi
  echo "$(date +%s.%N | cut -c1-14),$g,$cur,$anon,$file,$peak" >> "$out"
  sleep 1
done
