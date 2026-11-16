#!/usr/bin/env bash
# End-to-end check of llmctl with the default model: up, status, test, an OpenCode session without -m, down.
set -u
WT=$HOME/projects/llm/.worktrees/llmctl
OUT=$SCRATCH/llmctl/e2e/vllm-gpt-oss-20b
mkdir -p "$OUT/proj"
cd "$WT" || exit 1

step() { echo "== $* (at $(date +%T))"; }

step "llmctl up gpt-oss --runtime vllm"
S=$(date +%s)
./llmctl up gpt-oss --runtime vllm > "$OUT/up.txt" 2>&1
echo "up exit=$? after $(( $(date +%s) - S )) s"
cat "$OUT/up.txt"

step "llmctl status"
./llmctl status > "$OUT/status.txt" 2>&1
echo "status exit=$?"
cat "$OUT/status.txt"

step "llmctl test"
./llmctl test > "$OUT/test.txt" 2>&1
echo "test exit=$?"
cat "$OUT/test.txt"

cd "$WT" || exit 1
step "llmctl down"
./llmctl down > "$OUT/down.txt" 2>&1
echo "down exit=$?"
cat "$OUT/down.txt"
docker ps -a --filter label=io.llmctl.managed=true --format '{{.Names}} {{.State}}'
nvidia-smi --query-gpu=memory.used --format=csv,noheader
