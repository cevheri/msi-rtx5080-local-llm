#!/usr/bin/env bash
# usage: run.sh <label> <kimi-args...>   (one attempt; writes runs/<label>/{stdout,stderr,exit,strace})
set -u
H=$SCRATCH/kimi/handson
P=47303; L=$1; shift
R=$H/runs/$L; W=$H/work/$L
mkdir -p "$R" "$W"
echo "$L" > $H/shapes/.tag
printf '%q ' "$@" > "$R/argv"
heavy bwrap \
  --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --tmpfs $HOME \
  --ro-bind $HOME/.kimi-code/bin/kimi $HOME/.kimi-code/bin/kimi \
  --ro-bind "$H/tools" "$H/tools" \
  --bind "$H/home" "$H/home" --bind "$W" "$W" --bind "$R" "$R" --bind "$H/net" /tmp/kn \
  --unshare-net --unshare-ipc --unshare-uts --new-session --die-with-parent \
  --chdir "$W" --clearenv \
  --setenv HOME "$H/home" \
  --setenv KIMI_CODE_HOME "$H/home/.kimi-code" \
  --setenv KIMI_CODE_CACHE_DIR "$H/home/.cache/kimi-code" \
  --setenv XDG_CACHE_HOME "$H/home/.cache" --setenv XDG_CONFIG_HOME "$H/home/.config" \
  --setenv XDG_DATA_HOME "$H/home/.local/share" --setenv XDG_STATE_HOME "$H/home/.local/state" \
  --setenv TMPDIR /tmp --setenv PATH /usr/bin:/bin --setenv TERM dumb --setenv NO_COLOR 1 --setenv LANG C.UTF-8 \
  --setenv KIMI_DISABLE_TELEMETRY 1 --setenv KIMI_CODE_NO_AUTO_UPDATE 1 --setenv KIMI_CLI_NO_AUTO_UPDATE 1 \
  --setenv KIMI_CODE_REGION_MARKER off \
  bash "$H/tools/inner.sh" $P /tmp/kn/llm.sock "$R/strace.txt" 600 "$@" \
  > "$R/stdout" 2> "$R/stderr"
echo $? > "$R/exit"
echo "$L exit $(cat $R/exit)"
