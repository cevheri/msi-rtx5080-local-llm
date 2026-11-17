#!/usr/bin/env bash
# usage: item4.sh <case> <recipe> <runtime> [llmctl up args...]
# Opens a TUI session (case.sh under heavy) on the running model; after its first answer runs
# ./llmctl up <args> outside the sandbox, copies the real config.toml into W by a rename for <recipe> <runtime>,
# and lets the session send its next turn.
set -u
S=$SCRATCH
REPO=$HOME/projects/llm
W=$(cat $S/kimi/tui/W.path); C=$1; RECIPE=$2; RT=$3; shift 3
R=$S/kimi/tui/runs/$C
rm -rf "$W/net"
TO=1800 heavy bash $S/kimi/tui/bin/case.sh "$R" "$W" $S/kimi/tui/steps/item4-open-session.json > $S/kimi/tui/up/$C.case.txt 2>&1 &
CP=$!
for i in $(seq 900); do [ -e "$W/net/ready" ] && break; kill -0 $CP 2>/dev/null || { echo "case ended early"; exit 1; }; sleep 1; done
[ -e "$W/net/ready" ] || { echo "no ready"; kill $CP; exit 1; }
echo "ready at $(date -u +%H:%M:%S)"
cd $REPO
for c in llamacpp vllm; do docker logs --since "$(cat "$R/start-utc")" "$c" 2>&1 | grep -E 'launch_slot_|prompt eval time|       eval time|total time|stop processing|POST /v1/chat/completions' > "$R/server-log-before-up-$c.txt"; done
date -u +%Y-%m-%dT%H:%M:%SZ > "$R/up-start-utc"
PATH=$HOME/.kimi-code/bin:$PATH ./llmctl "$@" > $S/kimi/tui/up/$C.up.txt 2>&1
echo "llmctl $* rc=$?" | tee "$R/up-rc"
python3 -I $S/kimi/tui/bin/prep.py $REPO "$RECIPE" "$RT" "$W/kimi" | tee "$R/prep.txt"
touch "$W/net/go"
wait $CP
cat $S/kimi/tui/up/$C.case.txt | tail -2
