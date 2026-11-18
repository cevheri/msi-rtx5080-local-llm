#!/usr/bin/env bash
# One measured start for measurement B, after measurement A's run.sh: start the readiness poller and the docker run
# together, trace GPU and cgroup memory every second, run seq_b.py once the server is ready, then save the logs,
# remove the container and cut the log.
# Usage: run_b.sh <record-dir> <run> <cmd-file> <sglang|vllm> <served-name> [seq_b.py arguments...]
# With no seq_b.py arguments the start is only polled (readiness, memory at ready) and removed.
# Needs SCRATCH and HOME in the environment; the cmd file reads them too.
set -uo pipefail
dir=$1; run=$2; cmd=$3; kind=$4; served=$5; shift 5
: "${SCRATCH:?}"
B=$SCRATCH/sglang/measure-b
logs=$dir/logs; shapes=$dir/shapes
mkdir -p "$logs" "$shapes"
if [ "$kind" = sglang ]; then name=sglang-measure; port=30000; else name=vllm-measure; port=8000; fi
for other in sglang-measure vllm-measure; do
  if docker inspect "$other" >/dev/null 2>&1; then echo "refusing: a container named $other exists"; exit 9; fi
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$used" -gt 600 ]; then echo "refusing: $used MiB of GPU memory in use before the start"; exit 9; fi
echo "gpu_used_before_mib=$used" > "$logs/$run.meta"
echo "cmd=$(basename "$cmd")" >> "$logs/$run.meta"
t0=$(date +%s.%N); echo "$t0" > "$logs/$run.t0"
python3 -I "$B/scripts/poll_b.py" --t0 "$t0" --container "$name" --served "$served" --kind "$kind" \
  --base "http://127.0.0.1:$port" --out "$shapes/$run-ready.json" > "$logs/$run.poll.txt" 2>&1 &
poller=$!
bash "$cmd" > "$logs/$run.docker-run.txt" 2>&1
echo "docker_run_exit=$?" >> "$logs/$run.meta"
bash "$B/scripts/trace_b.sh" "$name" "$logs/$run-trace.csv" 5400 > /dev/null 2>&1 &
tracer=$!
wait "$poller"; ready=$?
echo "poller_exit=$ready" >> "$logs/$run.meta"
cat "$logs/$run.poll.txt"
if [ "$ready" = 0 ]; then
  bash "$B/scripts/procmem.sh" "$name" "at ready" > "$logs/$run.procmem.txt" 2>&1
  if [ "$#" -gt 0 ]; then
    python3 -I "$B/scripts/seq_b.py" --base "http://127.0.0.1:$port" --kind "$kind" --container "$name" \
      --model "$served" --out-dir "$shapes" --run "$run" "$@" > "$logs/$run.seq.txt" 2>&1
    echo "seq_exit=$?" >> "$logs/$run.meta"
    tail -n 4 "$logs/$run.seq.txt" | cut -c1-400
  fi
  bash "$B/scripts/procmem.sh" "$name" "at the end" >> "$logs/$run.procmem.txt" 2>&1
fi
sleep 2
docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}' "$name" > "$logs/$run.state.txt" 2>&1
docker logs "$name" > "$logs/$run.full.log" 2>&1
docker rm -f "$name" > /dev/null 2>&1
kill "$tracer" 2>/dev/null; wait "$tracer" 2>/dev/null
if [ "$kind" = sglang ]; then
  python3 -I "$B/scripts/cutlog.py" "$logs/$run.full.log" "$logs/$run.log"
  if grep -q -E "Receive OpenAI: obj=|Using default chat sampling params" "$logs/$run.full.log"; then
    python3 -I "$B/scripts/reqlog.py" "$logs/$run.full.log" "$logs/$run.reqlog.txt" > /dev/null
  fi
  grep -E "fired up|Load weight end|KV Cache is allocated|SWA|Mamba Cache|available_gpu_mem|Using default chat sampling|Auto-detected template|KV pool holds|paged|Paged|Error|error" "$logs/$run.log" | cut -c1-300 | head -40
else
  grep -E "Model loading took|Available KV cache|GPU KV cache size|Maximum concurrency|Application startup complete|Error|error|tokenizer|Tokenizer" "$logs/$run.full.log" | cut -c1-300 | head -40
fi
cat "$logs/$run.state.txt"
