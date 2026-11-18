#!/usr/bin/env bash
# One measured start for measurement E: measurement D's run_d.sh with E's script folder (seq_d.py unchanged). It starts
# the readiness poller and the docker run together, traces GPU and cgroup memory every second, runs seq_d.py once the
# server is ready, then saves the logs, removes the container and cuts the log; /dev/shm readings (df -B1 and ls -la)
# before the start, at ready and after the container is gone, and the GPU's enforced power limit before the start, as in C.
# Usage: run_e.sh <record-dir> <run> <cmd-file> <sglang|vllm> <served-name> [seq_d.py arguments...]
# With no seq_d.py arguments the start is only polled (readiness, memory at ready) and removed.
# Needs SCRATCH and HOME in the environment; the cmd file reads them too.
set -uo pipefail
dir=$1; run=$2; cmd=$3; kind=$4; served=$5; shift 5
: "${SCRATCH:?}"
S=$SCRATCH/sglang/measure-e/scripts
logs=$dir/logs; shapes=$dir/shapes
mkdir -p "$logs" "$shapes"
if [ "$kind" = sglang ]; then name=sglang-measure; port=30000; else name=vllm-measure; port=8000; fi
for other in sglang-measure vllm-measure llamacpp-measure llamacpp sglang vllm; do
  if [ "$(docker inspect --format '{{.State.Status}}' "$other" 2>/dev/null)" = running ]; then echo "refusing: container $other is running"; exit 9; fi
done
for other in sglang-measure vllm-measure; do
  if docker inspect "$other" >/dev/null 2>&1; then echo "refusing: a container named $other exists"; exit 9; fi
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$used" -gt 600 ]; then echo "refusing: $used MiB of GPU memory in use before the start"; exit 9; fi
echo "gpu_used_before_mib=$used" > "$logs/$run.meta"
echo "enforced_power_limit=$(nvidia-smi --query-gpu=enforced.power.limit --format=csv,noheader | tr -d ' ')" >> "$logs/$run.meta"
echo "cmd=$(basename "$cmd")" >> "$logs/$run.meta"
{ echo "== before the start $(date -u +%H:%M:%S)"; df -B1 /dev/shm; ls -la /dev/shm; } > "$logs/$run.devshm.txt" 2>&1
ls -1 /dev/shm > "$logs/$run.devshm-before.list"
t0=$(date +%s.%N); echo "$t0" > "$logs/$run.t0"
python3 -I "$S/poll_c.py" --t0 "$t0" --container "$name" --served "$served" --kind "$kind" \
  --base "http://127.0.0.1:$port" --timeout 1800 --out "$shapes/$run-ready.json" > "$logs/$run.poll.txt" 2>&1 &
poller=$!
bash "$cmd" > "$logs/$run.docker-run.txt" 2>&1
echo "docker_run_exit=$?" >> "$logs/$run.meta"
bash "$S/trace_c.sh" "$name" "$logs/$run-trace.csv" 7200 > /dev/null 2>&1 &
tracer=$!
wait "$poller"; ready=$?
echo "poller_exit=$ready" >> "$logs/$run.meta"
cat "$logs/$run.poll.txt"
if [ "$ready" = 0 ]; then
  bash "$S/procmem.sh" "$name" "at ready" > "$logs/$run.procmem.txt" 2>&1
  docker exec -i "$name" python3 -I - < "$S/maps_inside.py" > "$logs/$run.maps.txt" 2>&1
  { echo "== at ready $(date -u +%H:%M:%S)"; df -B1 /dev/shm; ls -la /dev/shm; } >> "$logs/$run.devshm.txt" 2>&1
  if [ "$#" -gt 0 ]; then
    python3 -I "$S/seq_d.py" --base "http://127.0.0.1:$port" --kind "$kind" --container "$name" \
      --model "$served" --out-dir "$shapes" --run "$run" "$@" > "$logs/$run.seq.txt" 2>&1
    echo "seq_exit=$?" >> "$logs/$run.meta"
    tail -n 4 "$logs/$run.seq.txt" | cut -c1-400
  fi
  bash "$S/procmem.sh" "$name" "at the end" >> "$logs/$run.procmem.txt" 2>&1
fi
sleep 2
docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}' "$name" > "$logs/$run.state.txt" 2>&1
docker logs "$name" > "$logs/$run.full.log" 2>&1
docker rm -f "$name" > /dev/null 2>&1
kill "$tracer" 2>/dev/null; wait "$tracer" 2>/dev/null
sleep 3
{ echo "== after the container is gone $(date -u +%H:%M:%S)"; df -B1 /dev/shm; ls -la /dev/shm; } >> "$logs/$run.devshm.txt" 2>&1
bash "$S/shm_cleanup.sh" "$logs/$run.devshm-before.list" "$logs/$run.devshm.txt"
if [ "$kind" = sglang ]; then
  python3 -I "$S/cutlog.py" "$logs/$run.full.log" "$logs/$run.log"
  if grep -q -E "Receive OpenAI: obj=|Using default chat sampling params" "$logs/$run.full.log"; then
    python3 -I "$S/reqlog.py" "$logs/$run.full.log" "$logs/$run.reqlog.txt" > /dev/null
  fi
  grep -E "fired up|Load weight end|KV Cache is allocated|SWA|Mamba Cache|available_gpu_mem|Using default chat sampling|Auto-detected template|KV pool holds|paged|Paged|Error|error" "$logs/$run.log" | cut -c1-300 | head -40
else
  grep -E "Model loading took|Available KV cache|GPU KV cache size|Maximum concurrency|Application startup complete|Error|error|tokenizer|Tokenizer" "$logs/$run.full.log" | cut -c1-300 | head -40
fi
cat "$logs/$run.state.txt"
