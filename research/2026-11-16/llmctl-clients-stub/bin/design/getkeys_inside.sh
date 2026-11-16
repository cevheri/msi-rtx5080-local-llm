#!/bin/bash
# INSIDE the sandbox: print each llmctl block key as Hermes resolves it (JSON), one file per key.
OUT="$1"; mkdir -p "$OUT"
for k in model providers.llmctl model_overrides gateway platform_toolsets _config_version; do
  hermes -p llmctl config get "$k" --json >"$OUT/$k.json" 2>"$OUT/$k.err"; echo $? >"$OUT/$k.exit"
done
