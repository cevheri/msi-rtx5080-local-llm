#!/usr/bin/env bash
# Inside the sandbox: serve <dir> on 127.0.0.1:47400 (python http.server, logs to stderr), then run the rest.
# usage: e4_registry.sh <dir> <cmd...>   ("kimi" means the installed binary)
set -u
D=$1; shift
python3 -I -m http.server 47400 --bind 127.0.0.1 --directory "$D" &
HP=$!
for i in $(seq 50); do ss -ltn | grep -q "127.0.0.1:47400 " && break; sleep 0.05; done
args=(); for a in "$@"; do [ "$a" = kimi ] && args+=("$KIMI_BIN") || args+=("$a"); done
"${args[@]}"
RC=$?
kill $HP 2>/dev/null
exit $RC
