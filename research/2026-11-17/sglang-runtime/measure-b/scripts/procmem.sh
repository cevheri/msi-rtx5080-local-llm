#!/usr/bin/env bash
# Usage: procmem.sh <container-name> <label>   -- prints one line per process of the container's cgroup with its
# command name and the /proc/<pid>/status memory fields that show pinned and locked host memory (VmPin, VmLck) next to
# VmRSS, RssAnon, RssFile and RssShmem, plus the cgroup's memory.stat totals; used for the Paged Experts host store.
name=$1; label=$2
id=$(docker inspect --format '{{.Id}}' "$name" 2>/dev/null) || { echo "$label: no container"; exit 0; }
cg=/sys/fs/cgroup/system.slice/docker-$id.scope
echo "== $label $(date -u +%H:%M:%S)"
for pid in $(cat "$cg/cgroup.procs" 2>/dev/null); do
  [ -r "/proc/$pid/status" ] || continue
  awk -v pid="$pid" '/^Name:/{n=$2} /^VmRSS:/{r=$2} /^RssAnon:/{a=$2} /^RssFile:/{f=$2} /^RssShmem:/{s=$2} /^VmLck:/{l=$2} /^VmPin:/{p=$2}
    END{if (r+0 > 102400) printf "pid=%s name=%s VmRSS_kB=%s RssAnon_kB=%s RssFile_kB=%s RssShmem_kB=%s VmLck_kB=%s VmPin_kB=%s\n", pid, n, r, a, f, s, l, p}' "/proc/$pid/status"
done
awk '$1=="anon"||$1=="file"||$1=="shmem"||$1=="unevictable"||$1=="file_mapped"{printf "cgroup %s=%.3f GiB\n", $1, $2/1073741824}' "$cg/memory.stat"
printf "cgroup memory.current=%.3f GiB memory.peak=%.3f GiB\n" "$(echo "$(cat $cg/memory.current)/1073741824" | bc -l)" "$(echo "$(cat $cg/memory.peak)/1073741824" | bc -l)"
