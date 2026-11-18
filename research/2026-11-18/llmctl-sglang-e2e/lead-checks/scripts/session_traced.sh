#!/usr/bin/env bash
# Run one OpenCode session through heavy under strace, tracing only the four socket calls that carry its HTTP traffic,
# so the requests it sends to the model server and the replies can be rebuilt (strace_http.py); then save the
# server's log lines of that session.
# Usage: session_traced.sh <work folder> <out folder> <model> <prompt>
set -u
work=$1
out=$2
model=$3
prompt=$4
mkdir -p "$out/strace"
start=$(date -u +%FT%T.%3NZ)
echo "$start" > "$out/start-utc.txt"
cd "$work" || exit 1
heavy strace -f -ff -tt -yy -s 100000000 -e trace=sendto,recvfrom,sendmsg,recvmsg -o "$out/strace/t" \
  opencode run --standalone --archived --auto --format json --model "$model" "$prompt" \
  < /dev/null > "$out/events.jsonl" 2> "$out/stderr.txt"
code=$?
end=$(date -u +%FT%T.%3NZ)
echo "$code" > "$out/exit.txt"
echo "$end" > "$out/end-utc.txt"
docker logs --timestamps --since "$start" --until "$end" sglang > "$out/server.log" 2>&1
python3 -I "$(dirname "$0")/strace_http.py" 30000 "$out/http" "$out"/strace/t.* > "$out/http.txt" 2>&1
echo "traced session exit $code"
