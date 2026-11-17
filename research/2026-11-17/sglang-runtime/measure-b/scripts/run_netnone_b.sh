#!/usr/bin/env bash
# A start without a network (M15 and the gpt-oss checks), after measurement A's run_netnone.sh: no published port, so
# the readiness path is polled every second with docker exec inside the container (urllib against 127.0.0.1), then the
# served name in /v1/models is checked and one chat request is sent the same way; the logs are saved, the container is
# removed, the log is cut, and the cache volume is listed and removed.
# Usage: run_netnone_b.sh <record-dir> <run> <cmd-file> <sglang|vllm> <served-name> <volume> [<volume2>]
set -uo pipefail
dir=$1; run=$2; cmd=$3; kind=$4; served=$5; vol=$6; vol2=${7:-}
: "${SCRATCH:?}"
B=$SCRATCH/sglang/measure-b
logs=$dir/logs; shapes=$dir/shapes
mkdir -p "$logs" "$shapes"
if [ "$kind" = sglang ]; then name=sglang-measure; port=30000; ready_path=/ready; py=python3
else name=vllm-measure; port=8000; ready_path=/health; py=python3; fi
for other in sglang-measure vllm-measure; do
  if docker inspect "$other" >/dev/null 2>&1; then echo "refusing: a container named $other exists"; exit 9; fi
done
for v in $vol $vol2; do
  if docker volume inspect "$v" >/dev/null 2>&1; then echo "refusing: volume $v exists; it must be new and empty"; exit 9; fi
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$used" -gt 600 ]; then echo "refusing: $used MiB of GPU memory in use before the start"; exit 9; fi
echo "gpu_used_before_mib=$used" > "$logs/$run.meta"
echo "cmd=$(basename "$cmd") volume=$vol $vol2" >> "$logs/$run.meta"
probe='import sys, urllib.request, urllib.error
try:
    r = urllib.request.urlopen("http://127.0.0.1:'$port'" + sys.argv[1], timeout=5)
    print(r.status)
except urllib.error.HTTPError as e:
    print(e.code)
except Exception as e:
    print(type(e).__name__)'
models='import json, sys, urllib.request
d = json.loads(urllib.request.urlopen("http://127.0.0.1:'$port'/v1/models", timeout=10).read())
print(sys.argv[1] in [m.get("id") for m in d.get("data", [])])'
chat='import json, sys, urllib.request
body = json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "Reply with exactly: OK"}], "max_tokens": 256}).encode()
req = urllib.request.Request("http://127.0.0.1:'$port'/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
r = urllib.request.urlopen(req, timeout=300)
d = json.loads(r.read())
m = d["choices"][0]["message"]
print(json.dumps({"status": r.status, "content": (m.get("content") or "")[:40], "reasoning_chars": len(m.get("reasoning_content") or m.get("reasoning") or ""), "finish_reason": d["choices"][0]["finish_reason"], "completion_tokens": d.get("usage", {}).get("completion_tokens")}))'
t0=$(date +%s.%N); echo "$t0" > "$logs/$run.t0"
bash "$cmd" > "$logs/$run.docker-run.txt" 2>&1
echo "docker_run_exit=$?" >> "$logs/$run.meta"
bash "$B/scripts/trace_b.sh" "$name" "$logs/$run-trace.csv" 3600 > /dev/null 2>&1 &
tracer=$!
last=; ready=no
: > "$shapes/$run-exec-poll.txt"
while :; do
  now=$(date +%s.%N); el=$(echo "$now - $t0" | bc)
  st=$(docker inspect --format '{{.State.Status}}' "$name" 2>/dev/null || echo missing)
  if [ "$st" != running ]; then echo "container $st after $el s" | tee -a "$shapes/$run-exec-poll.txt"; break; fi
  code=$(docker exec "$name" $py -c "$probe" "$ready_path" 2>&1 | tail -1)
  if [ "$code" != "$last" ]; then printf 'at %.2f s %s %s\n' "$el" "$ready_path" "$code" | tee -a "$shapes/$run-exec-poll.txt"; last=$code; fi
  if [ "$code" = 200 ]; then
    listed=$(docker exec "$name" $py -c "$models" "$served" 2>&1 | tail -1)
    if [ "$listed" = True ]; then ready=yes; printf 'at %.2f s served name listed in /v1/models\n' "$(echo "$(date +%s.%N) - $t0" | bc)" | tee -a "$shapes/$run-exec-poll.txt"; break; fi
  fi
  if [ "$(echo "$el > 900" | bc)" = 1 ]; then echo "timeout" | tee -a "$shapes/$run-exec-poll.txt"; break; fi
  sleep 1
done
echo "ready=$ready" >> "$logs/$run.meta"
if [ "$ready" = yes ]; then
  echo "free_mib_after_ready=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')" | tee -a "$shapes/$run-exec-poll.txt"
  echo "chat $(docker exec "$name" $py -c "$chat" "$served" 2>&1 | tail -1)" | tee -a "$shapes/$run-exec-poll.txt"
  docker exec "$name" sh -c 'cat /proc/net/dev' > "$logs/$run.netns.txt" 2>&1
fi
sleep 2
docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}} finished={{.State.FinishedAt}} network={{.HostConfig.NetworkMode}}' "$name" > "$logs/$run.state.txt" 2>&1
docker logs "$name" > "$logs/$run.full.log" 2>&1
docker rm -f "$name" > /dev/null 2>&1
kill "$tracer" 2>/dev/null; wait "$tracer" 2>/dev/null
for v in $vol $vol2; do
  docker run --rm --network none --memory 1g --memory-swap 1g -v "$v:/v:ro" --entrypoint sh "$( [ "$kind" = sglang ] && echo lmsysorg/sglang:v0.5.23-runtime || echo vllm/vllm-openai:v0.33.0 )" \
    -c 'echo "files: $(find /v -type f | wc -l)"; du -sh /v; du -sh /v/* /v/.[!.]* 2>/dev/null' > "$logs/$run.volume-$v.txt" 2>&1
  docker volume rm "$v" > /dev/null && echo "removed volume $v" >> "$logs/$run.meta"
done
if [ "$kind" = sglang ]; then python3 -I "$B/scripts/cutlog.py" "$logs/$run.full.log" "$logs/$run.log"; fi
grep -i -n -E "download|fetch|urm\.nvidia|huggingface\.co|openaipublic|blob\.core|harmony|tiktoken|HarmonyError|Responses API|resolve|connect" "$logs/$run.full.log" | cut -c1-300 > "$logs/$run.network-lines.txt"
cat "$logs/$run.state.txt"; cat "$shapes/$run-exec-poll.txt"; wc -l < "$logs/$run.network-lines.txt"
