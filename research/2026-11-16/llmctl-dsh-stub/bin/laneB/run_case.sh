#!/usr/bin/env bash
# Run one headless dsh case against the configured route.
#   run_case.sh <case> <patch.yml> <task> [timeout_s] [extra dsh args...]
# Each case gets a fresh DSH_HOME (mktemp -d), a fresh work folder (cd into it, so PWD and cwd
# agree), an empty DSH_AGENTS_HOME, and DSH_TELEMETRY_DISABLED=1. Output lands in $LB/runs/<case>/.
set -euo pipefail
LB=$SCRATCH/laneB.GwYhns
case_name=$1; patch=$2; task=$3; limit=${4:-120}; shift 4 || shift $#
out=$LB/runs/$case_name
rm -rf "$out"; mkdir -p "$out"
home=$(mktemp -d -p "$LB/homes" "h-$case_name.XXXXXX")
agents=$(mktemp -d -p "$LB/homes" "agents-$case_name.XXXXXX")
work=$(mktemp -d -p "$LB/work" "w-$case_name.XXXXXX")
cp "$patch" "$out/patch.yml"
echo "$home" > "$out/home.txt"; echo "$work" > "$out/work.txt"
cd "$work"
if [ -n "${SETUP_SH:-}" ]; then bash -c "$SETUP_SH"; fi
export DSH_HOME="$home" DSH_AGENTS_HOME="$agents" DSH_TELEMETRY_DISABLED=1 PWD="$work"
start=$(date +%s.%N)
set +e
if [ -n "${TASK_FILE:-}" ]; then
  timeout --kill-after=10 "$limit" heavy dsh --profile headless --patch "$out/patch.yml" "$@" --json - \
    < "$TASK_FILE" > "$out/stdout.jsonl" 2> "$out/stderr.txt"
else
  timeout --kill-after=10 "$limit" heavy dsh --profile headless --patch "$out/patch.yml" "$@" --json "$task" \
    > "$out/stdout.jsonl" 2> "$out/stderr.txt"
fi
code=$?
set -e
end=$(date +%s.%N)
python3 -I -c "import sys; print(f'exit {sys.argv[1]} elapsed {float(sys.argv[3])-float(sys.argv[2]):.1f}s')" "$code" "$start" "$end" | tee "$out/meta.txt"
