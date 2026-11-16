#!/bin/bash
# Host side. Seeds profile <profile> in overlay <overlay> with <config.yaml>, then runs one hermes
# command in the overlay sandbox against the stub.
# usage: run_case.sh <overlay> <profile> <config.yaml|-> <flavor> <model> <ctx> "<stub args>" <hermes args...>
#   config "-" keeps whatever config the profile already has.
# The run folder is $S/runs/<overlay>/<profile>[-N]; hello.py is expected in its work/ folder.
set -uo pipefail
S=$SCRATCH/clients/research-hermes
O="$1"; P="$2"; CFG="$3"; FLAVOR="$4"; MODEL="$5"; CTX="$6"; STUBARGS="$7"; shift 7
PD="$S/ovl/$O/upper/profiles/$P"
mkdir -p "$PD"
if [ "$CFG" != "-" ]; then install -m 600 "$CFG" "$PD/config.yaml"; fi
RUN="$S/runs/$O/$P"; n=1
while [ -e "$RUN" ]; do n=$((n+1)); RUN="$S/runs/$O/$P-$n"; done
mkdir -p "$RUN/work"
start=$(date +%s.%N)
timeout 600 "$S/bin/hxo" "$O" "" "$RUN/work" "$S/bin/scenario.sh" "$RUN" "$FLAVOR" "$MODEL" "$CTX" "$STUBARGS" -- hermes -p "$P" "$@"
rc=$?
end=$(date +%s.%N)
printf '%s\n' "hermes -p $P $*" >"$RUN/command.txt"
echo "run=$RUN rc=$rc seconds=$(python3 -c "print(round($end-$start,1))") requests=$(ls "$RUN/captures" 2>/dev/null | wc -l)"
