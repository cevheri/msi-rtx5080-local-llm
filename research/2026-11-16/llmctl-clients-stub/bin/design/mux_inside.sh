#!/bin/bash
# INSIDE the sandbox: start the stub on :8080, run one turn in the non-standalone profile "plain" (which starts the
# default host gateway, the multiplexer), wait 40 s for one rescan, then copy the default home's gateway_state.json.
set -uo pipefail
RUN="$1"; mkdir -p "$RUN/captures"
D=$SCRATCH/clients/design
coproc STUBP { exec python3 -I "$D/bin/stub_server.py" --port 8080 --flavor llamacpp --model local --ctx 131072 --capture-dir "$RUN/captures" 2>"$RUN/stub.log"; }
read -r -t 15 line <&"${STUBP[0]}"
hermes -p plain -z "Say hello." >"$RUN/plain.out" 2>"$RUN/plain.err"; echo $? >"$RUN/plain.exit"
python3 -c "import json;d=json.load(open('$HOME/.hermes/gateway_state.json'));print('t0', d.get('served_profiles'), d.get('gateway_state'))" >"$RUN/state.txt" 2>&1
sleep 40
python3 -c "import json;d=json.load(open('$HOME/.hermes/gateway_state.json'));print('t40', d.get('served_profiles'), d.get('gateway_state'), d.get('multiplex_standalone_reason'))" >>"$RUN/state.txt" 2>&1
cp $HOME/.hermes/gateway_state.json "$RUN/gateway_state.json" 2>/dev/null
ps -eo pid,ppid,etimes,args | grep -v "bwrap\|ps -eo" >"$RUN/ps.txt"
grep -h -i "broken\|llmctl\|standalone\|served" $HOME/.hermes/logs/agent.log $HOME/.hermes/logs/gateway*.log 2>/dev/null | tail -30 >"$RUN/log-grep.txt"
