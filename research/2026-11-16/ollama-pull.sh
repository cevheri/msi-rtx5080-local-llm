#!/usr/bin/env bash
# Pulls the pinned Ollama image and the five LibreDB tags into the ollama-models volume,
# in a container without a GPU, pinned to E-cores 16-23, with no published port.
set -uo pipefail
S=$(date +%s)
docker pull ollama/ollama:0.40.16 2>&1 | tail -2
echo "image pulled after $(( $(date +%s) - S )) s"
docker rm -f ollama-pull >/dev/null 2>&1
docker run -d --name ollama-pull --cpuset-cpus 16-23 --memory 4g --memory-swap 4g \
  -v ollama-models:/root/.ollama ollama/ollama:0.40.16 >/dev/null || { echo "docker run failed"; exit 1; }
timeout 60 bash -c 'until docker exec ollama-pull ollama --version >/dev/null 2>&1 && docker exec ollama-pull ollama list >/dev/null 2>&1; do sleep 1; done' || { echo "ollama-pull did not start"; docker logs ollama-pull 2>&1 | tail -10; exit 1; }
docker exec ollama-pull ollama --version
for tag in qwen3.5:9b gemma4:12b ornith:9b ministral-3:8b qwen2.5:7b; do
  T=$(date +%s)
  docker exec ollama-pull ollama pull "$tag" > /dev/null 2>&1
  rc=$?
  echo "$tag rc=$rc after $(( $(date +%s) - T )) s"
done
docker exec ollama-pull ollama list
docker stop -t 10 ollama-pull >/dev/null && docker rm ollama-pull >/dev/null
echo "all pulls done after $(( $(date +%s) - S )) s"
