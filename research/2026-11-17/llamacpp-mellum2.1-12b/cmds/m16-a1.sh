#!/usr/bin/env bash
# M16 m16-a1: the draft $SCRATCH/sglang/measure-c/overrides/llamacpp/mellum2.1-12b.toml started through the copy of main, no client.
set -euo pipefail
LLMCTL_CLIENTS= LLMCTL_RECIPES_DIR="$SCRATCH/sglang/measure-c/overrides" "$SCRATCH/llmctl-main/llmctl" up mellum2.1-12b --runtime llamacpp
