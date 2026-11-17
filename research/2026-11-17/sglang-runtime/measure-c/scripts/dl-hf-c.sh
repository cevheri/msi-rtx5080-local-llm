#!/usr/bin/env bash
# Measurement C's one download, in the downloads lane's form ($SCRATCH/sglang/scripts/downloads/dl-hf.sh): the hf CLI of
# the pinned vLLM image as the host user, HOME=/tmp, HF_HOME=/hf, the token only through --env-file, an 8g memory cap.
# Before and after: df -B1 of the model cache's filesystem (/home) and of /; it stops when /home or / would fall below
# 150 GiB (161,061,273,600 bytes) after the download, and deletes nothing.
# Usage: dl-hf-c.sh <label> <repo> <full-revision> <bytes>; needs SCRATCH in the environment.
set -uo pipefail
: "${SCRATCH:?}"
label="$1"; repo="$2"; rev="$3"; bytes="$4"
C=$SCRATCH/sglang/measure-c
log="$C/downloads/hf-$label.log"
floor=161061273600
[[ "$rev" =~ ^[0-9a-f]{40}$ ]] || { echo "revision must be a full commit hash" | tee -a "$log"; exit 2; }
{
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) disk before:"
  df -B1 "$HOME/.cache/huggingface" /
} >> "$log"
root=$(df -B1 --output=avail / | tail -1 | tr -d ' ')
cache=$(df -B1 --output=avail "$HOME/.cache/huggingface" | tail -1 | tr -d ' ')
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) need $bytes B; /home (cache) avail $cache B, after $((cache - bytes)) B; / avail $root B, after $((root - bytes)) B; floor $floor B" >> "$log"
if (( cache - bytes < floor || root - bytes < floor )); then
  echo "STOP: the download would leave less than 150 GiB free; nothing downloaded, nothing deleted" | tee -a "$log"; exit 9
fi
start=$(date +%s)
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) docker run --rm --name dl-$label --user \$(id -u):\$(id -g) -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_DISABLE_PROGRESS_BARS=1 --env-file \$HOME/.config/hf/hf.env -v \$HOME/.cache/huggingface:/hf --memory 8g --memory-swap 8g --entrypoint hf vllm/vllm-openai:v0.33.0 download $repo --revision $rev" >> "$log"
docker run --rm --name "dl-$label" --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf \
  -e HF_HUB_DISABLE_PROGRESS_BARS=1 --env-file "$HOME/.config/hf/hf.env" \
  -v "$HOME/.cache/huggingface:/hf" --memory 8g --memory-swap 8g \
  --entrypoint hf vllm/vllm-openai:v0.33.0 download "$repo" --revision "$rev" >> "$log" 2>&1
rc=$?
end=$(date +%s)
{
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) exit $rc after $((end - start)) s; disk after:"
  df -B1 "$HOME/.cache/huggingface" /
} >> "$log"
exit $rc
