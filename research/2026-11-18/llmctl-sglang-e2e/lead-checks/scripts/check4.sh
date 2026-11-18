#!/usr/bin/env bash
# Check 4: start gpt-oss-20b on SGLang with no client, then run the e2e's OpenCode session three times, each in a
# fresh work folder laid out as the driver lays it out (launcher/e2e/clients.py: _session_argv, work, step_env).
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$HOME/projects/llm
out=$1
permissions=$2  # an opencode.json written as the driver writes it
mkdir -p "$out"
"$here/stamp_run.sh" "$out/up.log" env LLMCTL_CLIENTS= ./llmctl up gpt-oss-20b --runtime sglang --replace
grep -q ' ready: ' "$out/up.log" || { echo "check4: no ready line"; exit 1; }
task=$(python3 -I -B -c "import sys; sys.path.insert(0, '$repo'); from launcher.e2e import clients; print(clients.TASK)")
printf '%s\n' "$task" > "$out/task.txt"
export PATH="$HOME/.kimi-code/bin:$PATH"  # driver_env puts the Kimi bin folder first
for run in 1 2 3; do
  work="$out/run-$run/25-gpt-oss-20b-sglang-opencode"
  mkdir -p "$work"
  cp "$permissions" "$work/opencode.json"
  "$here/session_traced.sh" "$work" "$out/run-$run/session" sglang-docker/gpt-oss-20b "$task"
  python3 -I -B "$here/outcome.py" "$repo" "$out/run-$run/session/events.jsonl" "$work" \
    "$(cat "$out/run-$run/session/exit.txt")" > "$out/run-$run/outcome.txt" 2>&1
  echo "run $run: $(cat "$out/run-$run/outcome.txt")"
done
echo "check4 sessions done"
