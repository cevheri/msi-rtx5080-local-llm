#!/usr/bin/env bash
# Run a command in the repository with every output line prefixed by its UTC time; the exit status goes last.
# Usage: stamp_run.sh <log file> <command...>
set -u
log=$1
shift
cd $HOME/projects/llm || exit 1
{
  printf '%s $ %s\n' "$(date -u +%H:%M:%S.%3N)" "$*"
  "$@" 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date -u +%H:%M:%S.%3N)" "$line"; done
  rc=${PIPESTATUS[0]}
  printf '%s exit %s\n' "$(date -u +%H:%M:%S.%3N)" "$rc"
} >> "$log"
