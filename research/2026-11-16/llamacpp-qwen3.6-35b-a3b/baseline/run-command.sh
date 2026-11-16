# Exact launch used on 2026-11-16 for the baseline (recipe from llamacpp/models.md, plus --user,
# the /hf cache mount and -m instead of -hf, because the file was pre-downloaded with the hf CLI)
docker rm -f llamacpp 2>/dev/null
docker run -d --name llamacpp \
  --user "$(id -u):$(id -g)" \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_HOME=/hf \
  -v "$HOME/.cache/huggingface:/hf" \
  -p 127.0.0.1:8080:8080 \
  --memory 20g --memory-swap 20g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -m /hf/hub/models--ggml-org--Qwen3.6-35B-A3B-GGUF/snapshots/52f9f9b8052fb9c8b69bba4a95eda0f319ac3d2a/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --alias local -ngl all --n-cpu-moe 21 -fa on -c 131072 -np 1 --jinja \
  --reasoning off \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0
