#!/usr/bin/env bash
# usage: web-stop.sh: stops every dsh web started by web-start.sh (pattern 'dsh web --port 318')
pids=$(pgrep -f "^node .*/dsh web --port 318" || true)
if [ -z "$pids" ]; then echo "no dsh web running"; exit 0; fi
echo "stopping: $pids"; kill -TERM $pids || true
for i in $(seq 1 20); do pgrep -f "^node .*/dsh web --port 318" >/dev/null || { echo "stopped"; exit 0; }; sleep 0.5; done
echo "still running; KILL"; pkill -KILL -f "^node .*/dsh web --port 318" || true
