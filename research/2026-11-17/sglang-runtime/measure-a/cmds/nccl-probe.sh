#!/usr/bin/env bash
# One-rank NCCL probe (no model server): the first all_reduce on a one-rank torch NCCL group, as the sampler runs it.
# Usage: nccl-probe.sh <variant>...   (variants are defined in scripts/nccl_probe.py)
set -euo pipefail
M=${M:?set M to the measure-a folder}
docker run --rm --name sglang-measure \
  --device nvidia.com/gpu=all --ipc=host \
  --memory 32g --memory-swap 32g \
  --network none \
  -v "$M/scripts:/probe:ro" \
  --entrypoint python3 \
  lmsysorg/sglang:v0.5.23-runtime \
  -I /probe/nccl_probe.py --only "$@"
