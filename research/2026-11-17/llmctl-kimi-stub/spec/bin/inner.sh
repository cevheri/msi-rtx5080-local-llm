#!/usr/bin/env bash
# Inside bwrap: optional socat 127.0.0.1:<bport> -> /tmp/kn/s.sock, then the command under timeout (and strace).
# usage: inner.sh <bport|0> <strace 0|1> <run dir> <timeout-s> cmd...
set -u
BP=$1; ST=$2; R=$3; TO=$4; shift 4
SP=
if [ "$BP" != 0 ]; then
  socat TCP-LISTEN:"$BP",bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:/tmp/kn/s.sock & SP=$!
  for i in $(seq 50); do ss -ltn 2>/dev/null | grep -q "127.0.0.1:$BP " && break; sleep 0.05; done
fi
T0=$(date +%s.%N)
if [ "$ST" = 1 ]; then
  strace -f -s 256 -e trace=connect,sendto,sendmsg -o "$R/strace.txt" timeout "$TO" "$@"
else
  timeout "$TO" "$@"
fi
RC=$?
T1=$(date +%s.%N)
echo "$T0 $T1" > "$R/time"
[ -n "$SP" ] && kill "$SP" 2>/dev/null
exit $RC
