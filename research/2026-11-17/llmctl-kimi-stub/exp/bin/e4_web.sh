#!/usr/bin/env bash
# Inside the sandbox: start `kimi web`, optionally call set_default for an alias through its API, then shut it down.
# usage: e4_web.sh <run dir> <url-encoded alias or "-">
set -u
K=$KIMI_BIN
R=$1; ALIAS_ENC=$2
cp "$KIMI_CODE_HOME/config.toml" "$R/config.at-start.toml"
$K web --no-open --port 58627 > "$R/web.out" 2> "$R/web.err" &
WP=$!
for i in $(seq 300); do grep -q 'Token:' "$R/web.out" 2>/dev/null && break; sleep 0.1; done
TOKEN=$(sed 's/\x1b\[[0-9;]*m//g' "$R/web.out" | sed -n 's/.*Token: *//p' | head -1 | tr -d '[:space:]')
echo "token length ${#TOKEN}"
sleep 3   # let the startup catalog refresh run
cp "$KIMI_CODE_HOME/config.toml" "$R/config.after-start.toml"
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:58627/api/v1/models > "$R/models.json"; echo "models rc $?"
if [ "$ALIAS_ENC" != - ]; then
  curl -s -X POST -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:58627/api/v1/models/${ALIAS_ENC}:set_default" > "$R/set_default.json"; echo "set_default rc $?"
  sleep 2
fi
cp "$KIMI_CODE_HOME/config.toml" "$R/config.after-call.toml"
curl -s -X POST -H "Authorization: Bearer $TOKEN" http://127.0.0.1:58627/api/v1/shutdown > "$R/shutdown.json"; echo "shutdown rc $?"
for i in $(seq 100); do kill -0 $WP 2>/dev/null || break; sleep 0.1; done
kill $WP 2>/dev/null; wait $WP; echo "web exit $?"
cp "$KIMI_CODE_HOME/config.toml" "$R/config.after-exit.toml"
