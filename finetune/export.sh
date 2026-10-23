#!/usr/bin/env bash
# Adapter -> fused model -> Ollama model (Q4_K_M) -> standalone GGUF for deployment.
#
# No llama.cpp needed: mlx_lm's own GGUF export does not support Qwen2, but Ollama imports
# safetensors and quantizes them, and the weights it stores are a plain GGUF file that
# llama-server can load on the Space.
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER=${1:-finetune/adapters}
uv run --extra finetune mlx_lm.fuse \
  --model Qwen/Qwen2.5-1.5B-Instruct --adapter-path "$ADAPTER" --save-path finetune/fused

ollama create tripmate-ft -q q4_K_M -f finetune/Modelfile.tuned

blob=$(ollama show tripmate-ft --modelfile | awk '/^FROM \//{print $2; exit}')
cp "$blob" finetune/tripmate-1.5b-q4_k_m.gguf
ls -lh finetune/tripmate-1.5b-q4_k_m.gguf
echo "Serve locally with: TRIPMATE_LLM_MODEL=tripmate-ft uv run streamlit run app/streamlit_app.py"
