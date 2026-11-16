# Exact launch used on 2026-11-16 (recipe from llamacpp/models.md, plus --user and the /hf cache mount)
docker run -d --name llamacpp \
  --user "$(id -u):$(id -g)" \
  --device nvidia.com/gpu=all \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -e HF_TOKEN \
  -e HF_HOME=/hf \
  -v "$HOME/.cache/huggingface:/hf" \
  -p 127.0.0.1:8080:8080 \
  --memory 16g --memory-swap 16g \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf ggml-org/gpt-oss-20b-GGUF:MXFP4 \
  --alias local \
  -ngl all -fa on -c 131072 -ctk q8_0 -ctv q8_0 -np 1 --jinja \
  --reasoning-effort medium \
  --temp 1.0 --top-p 1.0
