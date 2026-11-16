#!/bin/bash
# Runs INSIDE the hxo sandbox. Runs one command under strace (process starts and socket connects),
# then records the processes still alive 3 s later and what landed in the tmpfs home.
# usage: inside_probe.sh <run-dir> -- <command...>
set -uo pipefail
RUN="$1"; shift; [ "${1:-}" = "--" ] && shift
mkdir -p "$RUN"
start=$(date +%s.%N)
strace -f -qq -e trace=connect,execve -e signal=none -o "$RUN/strace.txt" "$@" >"$RUN/cmd.out" 2>"$RUN/cmd.err" </dev/null
echo $? >"$RUN/exit.txt"
end=$(date +%s.%N)
python3 -c "print(round($end-$start,2))" >"$RUN/seconds.txt"
sleep 3
ps -eo pid,ppid,etimes,args >"$RUN/ps-after.txt"
find $HOME -xdev -mindepth 1 -not -path '$HOME/.hermes*' 2>/dev/null | sort >"$RUN/home-writes.txt"
