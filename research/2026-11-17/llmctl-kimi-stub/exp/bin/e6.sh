#!/usr/bin/env bash
# E6: wall time and bytes written for kimi --version and kimi provider list --json, fresh and warm cache.
set -u
EXP=$(cd "$(dirname "$0")/.." && pwd)
ENTRY=$EXP/entries/llamacpp/gemma-4-26b-a4b.toml
OUT=$EXP/runs/e6-summary.jsonl; : > "$OUT"
one() { # <case> <home> <label> <cmd...>
  local c=$1 h=$2 label=$3; shift 3
  local before after
  before=$(python3 -I $EXP/bin/tree_bytes.py "$h/.kimi-code" "$h/.cache")
  HOMEDIR=$h STRACE=${STRACE:-0} heavy bash $EXP/bin/case.sh "$c" "$@" >/dev/null
  after=$(python3 -I $EXP/bin/tree_bytes.py "$h/.kimi-code" "$h/.cache")
  python3 -I -c 'import json,sys; c,label,b,a,t,x=sys.argv[1:]; t0,t1=map(float,open(t).read().split()); print(json.dumps({"case":c,"label":label,"exit":int(open(x).read()),"seconds":round(t1-t0,3),"before":json.loads(b),"after":json.loads(a)}))' \
    "$c" "$label" "$before" "$after" "$EXP/runs/$c/time" "$EXP/runs/$c/exit" >> "$OUT"
}
for i in 1 2 3; do
  h=$EXP/homes/e6-version-$i; rm -rf "$h"; mkdir -p "$h/.kimi-code" "$h/.cache"
  one e6-version-fresh-$i "$h" "version fresh" kimi --version
  one e6-version-warm-$i "$h" "version warm" kimi --version
  h=$EXP/homes/e6-list-$i; rm -rf "$h"; mkdir -p "$h/.kimi-code" "$h/.cache"
  cp "$ENTRY" "$h/.kimi-code/config.toml"; chmod 600 "$h/.kimi-code/config.toml"
  one e6-list-fresh-$i "$h" "list fresh" kimi provider list --json
  one e6-list-warm-$i "$h" "list warm" kimi provider list --json
done
# strace runs (no timing claims): network check
h=$EXP/homes/e6-strace; rm -rf "$h"; mkdir -p "$h/.kimi-code" "$h/.cache"; cp "$ENTRY" "$h/.kimi-code/config.toml"
STRACE=1 one e6-strace-version "$h" "strace version" kimi --version
STRACE=1 one e6-strace-list "$h" "strace list" kimi provider list --json
# empty home: no config.toml at all (Kimi writes its template)
h=$EXP/homes/e6-empty; rm -rf "$h"; mkdir -p "$h/.kimi-code" "$h/.cache"
STRACE=1 one e6-empty-list "$h" "list empty home" kimi provider list --json
cat "$OUT"
