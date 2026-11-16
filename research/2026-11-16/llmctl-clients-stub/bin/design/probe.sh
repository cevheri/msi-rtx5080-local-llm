#!/bin/bash
# Host side: seed profile llmctl in a fresh overlay with <config>, run one command in the sandbox,
# then report the overlay writes. usage: probe.sh <overlay> <config|-> <command...>
set -uo pipefail
D=$SCRATCH/clients/design
O="$1"; CFG="$2"; shift 2
RUN="$D/runs/$O"; mkdir -p "$RUN/work"
if [ "$CFG" != "-" ]; then mkdir -p "$D/ovl/$O/upper/profiles/llmctl"; install -m 600 "$CFG" "$D/ovl/$O/upper/profiles/llmctl/config.yaml"; fi
timeout 300 "$D/bin/hxo" "$O" "" "$RUN/work" "$D/bin/inside_probe.sh" "$RUN" -- "$@"
printf '%s\n' "$*" >"$RUN/command.txt"
python3 -I "$D/bin/upper_report.py" "$D/ovl/$O/upper" --max 400 >"$RUN/upper-report.txt"
echo "run=$RUN exit=$(cat $RUN/exit.txt) seconds=$(cat $RUN/seconds.txt)"
