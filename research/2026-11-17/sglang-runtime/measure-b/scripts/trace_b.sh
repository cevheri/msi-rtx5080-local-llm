#!/usr/bin/env bash
# Usage: trace_b.sh <container-name> <out.csv> [max-seconds]
# Every second: GPU memory used and free, power, temperature, utilization, and the container's cgroup memory:
# memory.current, memory.peak and the memory.stat fields anon, file, shmem, file_mapped, unevictable,
# active_file, inactive_file and kernel (bytes). Based on the hands-on trace.sh, with the shared-memory split added
# for LD19 (the non-reclaimable part is anon plus shmem, since the container has no swap).
name=$1; out=$2; max=${3:-3600}
echo "epoch,gpu_mem_used_mib,gpu_mem_free_mib,gpu_power_w,gpu_temp_c,gpu_util_pct,cg_current_b,cg_peak_b,cg_anon_b,cg_file_b,cg_shmem_b,cg_file_mapped_b,cg_unevictable_b,cg_active_file_b,cg_inactive_file_b,cg_kernel_b" > "$out"
end=$(( $(date +%s) + max )); id=
while [ "$(date +%s)" -lt "$end" ]; do
  [ -z "$id" ] && id=$(docker inspect --format '{{.Id}}' "$name" 2>/dev/null)
  g=$(nvidia-smi --query-gpu=memory.used,memory.free,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  cg=/sys/fs/cgroup/system.slice/docker-$id.scope
  if [ -n "$id" ] && [ -r "$cg/memory.current" ]; then
    cur=$(cat "$cg/memory.current"); peak=$(cat "$cg/memory.peak")
    stat=$(awk '$1=="anon"{a=$2} $1=="file"{f=$2} $1=="shmem"{s=$2} $1=="file_mapped"{m=$2} $1=="unevictable"{u=$2} $1=="active_file"{af=$2} $1=="inactive_file"{inf=$2} $1=="kernel"{k=$2} END{printf "%s,%s,%s,%s,%s,%s,%s,%s", a,f,s,m,u,af,inf,k}' "$cg/memory.stat")
  else cur=; peak=; stat=",,,,,,,"; id=; fi
  echo "$(date +%s.%N | cut -c1-14),$g,$cur,$peak,$stat" >> "$out"
  sleep 1
done
