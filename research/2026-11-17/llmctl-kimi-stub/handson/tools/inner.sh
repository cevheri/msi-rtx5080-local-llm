#!/usr/bin/env bash
# Runs inside the bwrap sandbox (no network namespace access to the host).
# usage: inner.sh <port> <sock> <strace-out> <timeout-s> kimi-args...
set -u
P=$1; SOCK=$2; ST=$3; TO=$4; shift 4
socat TCP-LISTEN:$P,bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:$SOCK &
SP=$!
for i in $(seq 50); do (exec 3<>/dev/tcp/127.0.0.1/$P) 2>/dev/null && break; sleep 0.1; done
strace -f -s 256 -e trace=connect,sendto -o "$ST" timeout "$TO" $HOME/.kimi-code/bin/kimi "$@"
RC=$?
kill $SP 2>/dev/null
exit $RC
