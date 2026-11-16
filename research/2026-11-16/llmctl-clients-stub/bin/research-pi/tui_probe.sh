#!/usr/bin/env bash
# Start the interactive TUI under a pty for a few seconds with no input, then stop it, and report
# whether settings.json changed. Usage: tui_probe.sh <scenario> <settings-file>
set -u
R=$SCRATCH/clients/research-pi
scenario=$1; settings_src=$2
run=$R/runs/$scenario
rm -rf "$run"; mkdir -p "$run"/{agent,home,work,tmp,xdg/config,xdg/data,xdg/cache,xdg/state}
port=$(cat $R/stub/port)
sed "s/__PORT__/$port/g" $R/configs/toggle/models.json > "$run/agent/models.json"
cp "$settings_src" "$run/agent/settings.json"
cp "$run/agent/settings.json" "$run/settings-before.json"
cd "$run/work"
inner="stty cols 120 rows 40; exec $HOME/.pi/agent/bin/pi --no-session --no-approve --offline --model llmctl/local"
timeout -s INT 8 env -i HOME="$run/home" TMPDIR="$run/tmp" PI_CODING_AGENT_DIR="$run/agent" \
  XDG_CONFIG_HOME="$run/xdg/config" XDG_DATA_HOME="$run/xdg/data" XDG_CACHE_HOME="$run/xdg/cache" XDG_STATE_HOME="$run/xdg/state" \
  PATH="$HOME/.nvm/versions/node/v24.22.1/bin:/usr/local/bin:/usr/bin:/bin" TERM=xterm-256color LANG=C.UTF-8 \
  PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0 \
  script -qfec "$inner" "$run/typescript.txt" < <(sleep 10) > /dev/null 2>&1
echo "exit=$?" > "$run/exit_code"
cat "$run/exit_code"
if cmp -s "$run/settings-before.json" "$run/agent/settings.json"; then echo "settings.json: unchanged"; else echo "settings.json: CHANGED"; echo "--- before"; cat "$run/settings-before.json"; echo; echo "--- after"; cat "$run/agent/settings.json"; echo; fi
ls -la "$run/agent" "$run/home"
