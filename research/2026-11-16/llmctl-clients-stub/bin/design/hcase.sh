#!/bin/bash
# Host side: seed profile llmctl in overlay <overlay> with <config>, run one hermes command against the stub.
# usage: hcase.sh <overlay> <name> <config|-> <flavor> <model> <ctx> "<stub args>" <hermes args...>
set -uo pipefail
D=$SCRATCH/clients/design
O="$1"; N="$2"; CFG="$3"; FLAVOR="$4"; MODEL="$5"; CTX="$6"; STUBARGS="$7"; shift 7
PD="$D/ovl/$O/upper/profiles/llmctl"; mkdir -p "$PD"
if [ "$CFG" != "-" ]; then install -m 600 "$CFG" "$PD/config.yaml"; fi
RUN="$D/runs/$O/$N"; rm -rf "$RUN"; mkdir -p "$RUN/work"
start=$(date +%s.%N)
timeout 600 "$D/bin/hxo" "$O" "" "$RUN/work" "$D/bin/scenario.sh" "$RUN" "$FLAVOR" "$MODEL" "$CTX" "$STUBARGS" -- hermes "$@"
rc=$?
end=$(date +%s.%N)
printf '%s\n' "hermes $*" >"$RUN/command.txt"
echo "run=$RUN rc=$rc seconds=$(python3 -c "print(round($end-$start,1))") requests=$(ls "$RUN/captures" 2>/dev/null | wc -l) hello=$(cat $RUN/work/hello.py 2>/dev/null | head -c 40)"
