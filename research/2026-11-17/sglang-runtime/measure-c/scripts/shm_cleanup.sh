#!/usr/bin/env bash
# After a container with --ipc=host is gone: compare /dev/shm with the listing taken before the start and remove only the
# root-owned SGLang files (sglang_loads_*.shm, sem.mp-*) that appeared during this run, naming each in the record;
# the files are root's, so a throwaway --ipc=host container removes them. Anything else that appeared is only reported.
# Usage: shm_cleanup.sh <before-listing-file> <record-file>
set -uo pipefail
before=$1; rec=$2
now=$(ls -1 /dev/shm)
new=$(comm -13 <(sort "$before") <(echo "$now" | sort))
{
  echo "== /dev/shm files that appeared during the run and are still there after the container is gone, $(date -u +%H:%M:%S)"
  if [ -z "$new" ]; then echo "none"; fi
  ours=()
  for f in $new; do
    ls -la "/dev/shm/$f"
    case "$f" in
      sglang_loads_*.shm|sem.mp-*) [ "$(stat -c %U "/dev/shm/$f")" = root ] && ours+=("/dev/shm/$f");;
    esac
  done
  if [ "${#ours[@]}" -gt 0 ]; then
    docker run --rm --ipc=host --network none --memory 256m --memory-swap 256m --entrypoint rm \
      lmsysorg/sglang:v0.5.23-runtime -v -- "${ours[@]}"
    echo "rm exit=$?"
  fi
  df -B1 /dev/shm
} >> "$rec" 2>&1
