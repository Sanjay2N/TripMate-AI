#!/usr/bin/env bash
# Container entrypoint: model server in the background, UI in the foreground.
set -euo pipefail

# --chat-template chatml: the format the adapter was trained on (Qwen ChatML), the same
# template Ollama uses locally, so local evals match what the Space serves.
/app/llama-server \
  --model models/tripmate.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 8192 --threads "$(nproc)" \
  --chat-template chatml \
  --parallel 1 &

until curl -sf http://127.0.0.1:8080/health >/dev/null; do sleep 1; done
echo "llama-server ready"

exec uv run streamlit run app/streamlit_app.py \
  --server.port 7860 --server.address 0.0.0.0 --server.headless true
