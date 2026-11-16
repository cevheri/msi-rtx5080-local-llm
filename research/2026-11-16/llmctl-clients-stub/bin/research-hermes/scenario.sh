#!/bin/bash
# Runs INSIDE a sandbox (hx/hxo). Starts the stub, runs one command, stops the stub.
# usage: scenario.sh <run-dir> <flavor> <model> <ctx> "<extra stub args>" -- <command...>
# Writes into <run-dir>: captures/ (one JSON per request), stub.log, cmd.out, cmd.err, exit.txt,
# home-writes.txt (anything written under the tmpfs $HOME outside the ~/.hermes overlay).
set -uo pipefail
S=$SCRATCH/clients/research-hermes
RUN="$1"; FLAVOR="$2"; MODEL="$3"; CTX="$4"; EXTRA="$5"; shift 5
[ "${1:-}" = "--" ] && shift
PORT=8080; [ "$FLAVOR" = vllm ] && PORT=8000
mkdir -p "$RUN/captures"
# shellcheck disable=SC2086
coproc STUBP { exec python3 -I "$S/bin/stub_server.py" --port "$PORT" --flavor "$FLAVOR" --model "$MODEL" \
  --ctx "$CTX" --capture-dir "$RUN/captures" $EXTRA 2>"$RUN/stub.log"; }
STUB_PID=$STUBP_PID
if ! read -r -t 15 line <&"${STUBP[0]}" || [ "$line" != READY ]; then
  echo "stub did not start" >&2; cat "$RUN/stub.log" >&2; exit 99
fi
"$@" >"$RUN/cmd.out" 2>"$RUN/cmd.err" </dev/null
RC=$?
echo "$RC" >"$RUN/exit.txt"
kill "$STUB_PID" 2>/dev/null
wait "$STUB_PID" 2>/dev/null
find $HOME -xdev -mindepth 1 2>/dev/null | sort >"$RUN/home-writes.txt"
exit "$RC"
