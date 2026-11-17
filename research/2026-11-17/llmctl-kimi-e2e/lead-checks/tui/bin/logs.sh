#!/usr/bin/env bash
# usage: logs.sh <run dir> <container>: the request lines of the server log since the session started
R=$1; C=$2
docker logs --since "$(cat "$R/start-utc")" --until "$(cat "$R/end-utc")" "$C" 2>&1 \
  | grep -E 'launch_slot_|prompt eval time|       eval time|total time|stop processing|POST /v1/chat/completions' > "$R/server-log-$C.txt"
wc -l < "$R/server-log-$C.txt"
