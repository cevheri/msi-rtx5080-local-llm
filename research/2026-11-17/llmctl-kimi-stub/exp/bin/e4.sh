#!/usr/bin/env bash
# E4: Kimi's config writer in private homes, starting from writer/f0-template-plus-llmctl.toml.
set -u
EXP=$(cd "$(dirname "$0")/.." && pwd)
W=$EXP/writer; B=$EXP/bin
fresh() { rm -rf "$1"; mkdir -p "$1/.kimi-code" "$1/.cache"; cp "$2" "$1/.kimi-code/config.toml"; chmod 600 "$1/.kimi-code/config.toml"; }
keep() { cp "$1" "$W/$2"; stat -c '%a %s' "$1" > "$W/$2.stat"; }
step=${1:-all}
if [ "$step" = a ] || [ "$step" = all ]; then
  H=$EXP/homes/e4a; fresh "$H" "$W/f0-template-plus-llmctl.toml"; ino0=$(stat -c %i "$H/.kimi-code/config.toml")
  HOMEDIR=$H TO=120 heavy bash $B/case.sh e4a-web-set-default bash $B/e4_web.sh $EXP/runs/e4a-web-set-default 'llmctl%2Flocal'
  echo "inode before $ino0 after $(stat -c %i "$H/.kimi-code/config.toml")" >> $EXP/runs/e4a-web-set-default/inode.txt
  keep "$H/.kimi-code/config.toml" a-web-set-default.toml
fi
if [ "$step" = b ] || [ "$step" = all ]; then
  H=$EXP/homes/e4b; fresh "$H" "$W/f0-template-plus-llmctl.toml"
  EXTRA_RO=$EXP/registry HOMEDIR=$H TO=120 heavy bash $B/case.sh e4b-provider-add bash $B/e4_registry.sh $EXP/registry/v1 kimi provider add http://127.0.0.1:47400/api.json
  keep "$H/.kimi-code/config.toml" b-provider-add.toml
  # d: kimi web start (startup refresh, scope all) with the registry changed to v2; llmctl's static provider must not move
  EXTRA_RO=$EXP/registry HOMEDIR=$H TO=120 heavy bash $B/case.sh e4d-web-start-refresh bash $B/e4_registry.sh $EXP/registry/v2 bash $B/e4_web.sh $EXP/runs/e4d-web-start-refresh -
  keep "$H/.kimi-code/config.toml" d-web-start-refresh.toml
  # c: remove the dummy provider
  HOMEDIR=$H TO=120 heavy bash $B/case.sh e4c-provider-remove kimi provider remove dummy
  keep "$H/.kimi-code/config.toml" c-provider-remove.toml
fi
if [ "$step" = a3 ]; then
  H=$EXP/homes/e4a3; fresh "$H" "$W/f2-default-at-top-plus-other.toml"
  HOMEDIR=$H TO=120 heavy bash $B/case.sh e4a3-web-switch-default bash $B/e4_web.sh $EXP/runs/e4a3-web-switch-default 'other%2Fm'
  keep "$H/.kimi-code/config.toml" a3-web-switch-default.toml
fi
