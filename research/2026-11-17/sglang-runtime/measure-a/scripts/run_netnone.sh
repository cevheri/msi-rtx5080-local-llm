#!/usr/bin/env bash
# M15 start without a network: no published port, so /health is polled every second with docker exec inside the
# container (urllib against 127.0.0.1:30000), then one chat request is sent the same way; logs are saved, the
# container is removed and the log is cut. Usage: run_netnone.sh <model> <run> <cmd-file> <served-name>
set -uo pipefail
model=$1; run=$2; cmd=$3; served=$4
: "${M:?}" "${SCRATCH:?}"
H=$SCRATCH/sglang/handson
logs=$M/logs/$model; shapes=$M/shapes/$model
mkdir -p "$logs" "$shapes"
name=sglang-measure
if docker inspect "$name" >/dev/null 2>&1; then echo "refusing: a container named $name exists"; exit 9; fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$used" -gt 600 ]; then echo "refusing: $used MiB of GPU memory in use before the start"; exit 9; fi
echo "gpu_used_before_mib=$used" > "$logs/$run.meta"
probe='import sys, urllib.request, urllib.error
try:
    r = urllib.request.urlopen("http://127.0.0.1:30000" + sys.argv[1], timeout=5)
    print(r.status)
except urllib.error.HTTPError as e:
    print(e.code)
except Exception as e:
    print(type(e).__name__)'
chat='import json, sys, urllib.request
body = json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "Reply with exactly: OK"}], "max_tokens": 16}).encode()
req = urllib.request.Request("http://127.0.0.1:30000/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
r = urllib.request.urlopen(req, timeout=120)
d = json.loads(r.read())
print(json.dumps({"status": r.status, "content": d["choices"][0]["message"]["content"][:40], "finish_reason": d["choices"][0]["finish_reason"], "usage": d.get("usage", {}).get("completion_tokens")}))'
t0=$(date +%s.%N); echo "$t0" > "$logs/$run.t0"
bash "$cmd" > "$logs/$run.docker-run.txt" 2>&1
echo "docker_run_exit=$?" >> "$logs/$run.meta"
bash "$H/scripts/trace.sh" "$name" "$logs/$run-trace.csv" 3600 > /dev/null 2>&1 &
tracer=$!
last=; ready=no
: > "$shapes/$run-exec-poll.txt"
while :; do
  now=$(date +%s.%N); el=$(echo "$now - $t0" | bc)
  st=$(docker inspect --format '{{.State.Status}}' "$name" 2>/dev/null || echo missing)
  if [ "$st" != running ]; then echo "container $st after $el s" | tee -a "$shapes/$run-exec-poll.txt"; break; fi
  code=$(docker exec "$name" python3 -c "$probe" /health 2>&1 | tail -1)
  if [ "$code" != "$last" ]; then printf 'at %.2f s /health %s\n' "$el" "$code" | tee -a "$shapes/$run-exec-poll.txt"; last=$code; fi
  if [ "$code" = 200 ]; then ready=yes; break; fi
  if [ "$(echo "$el > 900" | bc)" = 1 ]; then echo "timeout" | tee -a "$shapes/$run-exec-poll.txt"; break; fi
  sleep 1
done
echo "ready=$ready" >> "$logs/$run.meta"
if [ "$ready" = yes ]; then
  for p in /ready /v1/models /get_server_info; do echo "$p $(docker exec "$name" python3 -c "$probe" "$p" 2>&1 | tail -1)"; done | tee -a "$shapes/$run-exec-poll.txt"
  echo "chat $(docker exec "$name" python3 -c "$chat" "$served" 2>&1 | tail -1)" | tee -a "$shapes/$run-exec-poll.txt"
  docker exec "$name" sh -c 'cat /proc/net/tcp /proc/net/tcp6 2>/dev/null | wc -l; ip -o addr 2>/dev/null || cat /proc/net/dev' > "$logs/$run.netns.txt" 2>&1
fi
sleep 2
docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}} finished={{.State.FinishedAt}} network={{.HostConfig.NetworkMode}}' "$name" > "$logs/$run.state.txt" 2>&1
docker logs "$name" > "$logs/$run.full.log" 2>&1
docker rm -f "$name" > /dev/null 2>&1
kill "$tracer" 2>/dev/null; wait "$tracer" 2>/dev/null
python3 -I "$M/scripts/cutlog.py" "$logs/$run.full.log" "$logs/$run.log"
cat "$logs/$run.state.txt"
grep -E "fired up|Load weight end|KV Cache is allocated|available_gpu_mem" "$logs/$run.log" | cut -c1-300
