#!/usr/bin/env bash
# Usage: wait_ready.sh <container-name> <port> <timeout-s> <start-epoch>
# Polls /health every 2 s; exits when it answers 200, when the container stops, or at the timeout.
name=$1; port=$2; timeout=$3; start=$4
while :; do
  now=$(date +%s.%N | cut -c1-14)
  state=$(docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' "$name" 2>/dev/null || echo missing)
  case "$state" in running*) ;; *) echo "STOPPED after $(echo "$now - $start" | bc) s: $state"; exit 2;; esac
  code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$port/health" || true)
  if [ "$code" = "200" ]; then echo "READY after $(echo "$now - $start" | bc) s (/health 200)"; exit 0; fi
  if [ "$(echo "$now - $start > $timeout" | bc)" = "1" ]; then echo "TIMEOUT after $timeout s, last /health $code"; exit 3; fi
  sleep 2
done
