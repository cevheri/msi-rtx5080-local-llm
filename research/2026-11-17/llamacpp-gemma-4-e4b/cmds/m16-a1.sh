#!/usr/bin/env bash
# M16 m16-a1: the draft $SCRATCH/sglang/measure-c/overrides/llamacpp/gemma-4-e4b.toml started through the copy of main, no client.
set -euo pipefail
LLMCTL_CLIENTS= LLMCTL_RECIPES_DIR="$SCRATCH/sglang/measure-c/overrides" "$SCRATCH/llmctl-main/llmctl" up gemma-4-e4b --runtime llamacpp
