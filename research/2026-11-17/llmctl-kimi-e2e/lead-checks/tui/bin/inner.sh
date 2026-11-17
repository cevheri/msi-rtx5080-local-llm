#!/usr/bin/env bash
# Inside bwrap: socats 127.0.0.1:8080 and :8000 -> /tmp/kn/s<port>.sock, then the command under timeout.
# usage: inner.sh <run dir> <timeout-s> cmd...
set -u
R=$1; TO=$2; shift 2
SP=()
for p in 8080 8000; do
  socat TCP-LISTEN:$p,bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:/tmp/kn/s$p.sock & SP+=($!)
done
for i in $(seq 50); do [ "$(ss -ltn 2>/dev/null | grep -cE '127.0.0.1:(8080|8000) ')" = 2 ] && break; sleep 0.05; done
timeout "$TO" "$@"
RC=$?
kill "${SP[@]}" 2>/dev/null
exit $RC
