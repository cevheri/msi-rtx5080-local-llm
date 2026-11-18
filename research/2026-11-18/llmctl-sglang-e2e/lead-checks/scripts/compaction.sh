#!/usr/bin/env bash
# Check 2 (spec D15, research risk 14): one OpenCode session on sglang-docker/qwen3.5-9b through heavy, in a work
# folder of the repository's Markdown files, then the server's log lines of that session.
set -u
work=$1
out=$2
mkdir -p "$out"
prompt='Read every Markdown file in this folder and its subfolders, one file at a time, with the read tool. After reading each file, write a one-line summary of that file. When every file has its summary, write a final line that says DONE. Do not run any shell commands.'
printf '%s\n' "$prompt" > "$out/prompt.txt"
start=$(date -u +%FT%T.%3NZ)
echo "$start" > "$out/start-utc.txt"
cd "$work" || exit 1
heavy opencode run --standalone --archived --auto --format json -m sglang-docker/qwen3.5-9b "$prompt" \
  < /dev/null > "$out/events.jsonl" 2> "$out/stderr.txt"
code=$?
end=$(date -u +%FT%T.%3NZ)
echo "$code" > "$out/exit.txt"
echo "$end" > "$out/end-utc.txt"
docker logs --timestamps --since "$start" --until "$end" sglang > "$out/server.log" 2>&1
echo "compaction session exit $code"
