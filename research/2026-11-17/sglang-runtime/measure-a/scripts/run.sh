#!/usr/bin/env bash
# One measured start: start the M7 poller and the docker run together, trace GPU and cgroup memory every second,
# run the probe command once all four endpoints answer 200, then save the logs, remove the container and cut the log.
# Usage: run.sh <model> <run> <cmd-file> <keep-full: yes|no> [probe command and arguments...]
# Needs M (the measure-a folder) and SCRATCH in the environment; the probe command sees BASE and RUN_OUT.
set -uo pipefail
model=$1; run=$2; cmd=$3; keep=$4; shift 4
: "${M:?}" "${SCRATCH:?}"
H=$SCRATCH/sglang/handson
logs=$M/logs/$model; shapes=$M/shapes/$model
mkdir -p "$logs" "$shapes"
name=sglang-measure
if docker inspect "$name" >/dev/null 2>&1; then echo "refusing: a container named $name exists"; exit 9; fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$used" -gt 600 ]; then echo "refusing: $used MiB of GPU memory in use before the start"; exit 9; fi
echo "gpu_used_before_mib=$used" > "$logs/$run.meta"
t0=$(date +%s.%N); echo "$t0" > "$logs/$run.t0"
python3 -I "$M/scripts/poll_ready.py" --t0 "$t0" --container "$name" --out "$shapes/$run-m7.json" > "$logs/$run.poll.txt" 2>&1 &
poller=$!
bash "$cmd" > "$logs/$run.docker-run.txt" 2>&1
echo "docker_run_exit=$?" >> "$logs/$run.meta"
bash "$H/scripts/trace.sh" "$name" "$logs/$run-trace.csv" 3600 > /dev/null 2>&1 &
tracer=$!
wait "$poller"; ready=$?
echo "poller_exit=$ready" >> "$logs/$run.meta"
cat "$logs/$run.poll.txt"
if [ "$ready" = 0 ] && [ "$#" -gt 0 ]; then
  BASE=http://127.0.0.1:30000 RUN_OUT="$shapes/$run" "$@" 2>&1 | tee "$logs/$run.probe.txt"
fi
sleep 2
docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}' "$name" > "$logs/$run.state.txt" 2>&1
docker logs "$name" > "$logs/$run.full.log" 2>&1
docker rm -f "$name" > /dev/null 2>&1
kill "$tracer" 2>/dev/null; wait "$tracer" 2>/dev/null
python3 -I "$M/scripts/cutlog.py" "$logs/$run.full.log" "$logs/$run.log"
if grep -q -E "Receive OpenAI: obj=|Using default chat sampling params" "$logs/$run.full.log"; then
  python3 -I "$M/scripts/reqlog.py" "$logs/$run.full.log" "$logs/$run.reqlog.txt"
fi
if [ "$keep" != yes ]; then rm -f "$logs/$run.full.log"; fi
cat "$logs/$run.state.txt"
grep -E "fired up|Load weight end|KV Cache is allocated|available_gpu_mem|Using default chat sampling|Auto-detected template" "$logs/$run.log" | cut -c1-300
