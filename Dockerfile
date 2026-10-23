# TripMate on a free Hugging Face Space (Docker, CPU basic: 2 vCPU, 16 GB).
# One container: llama-server serves the fine-tuned GGUF on :8080, Streamlit serves the UI
# on :7860 and talks to it through the same OpenAI-compatible client used locally.
#
# The GGUF is downloaded at build time, so a cold start does not re-download 1 GB.
FROM ghcr.io/ggml-org/llama.cpp:server

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    UV_PYTHON_INSTALL_DIR=/home/user/.python \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    TRIPMATE_LLM_BASE_URL=http://127.0.0.1:8080/v1 \
    TRIPMATE_LLM_MODEL=tripmate-ft \
    TRIPMATE_RERANK=true
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock README.md ./
COPY --chown=user src ./src
RUN uv sync --frozen --no-dev --python 3.12

COPY --chown=user app ./app
COPY --chown=user data/destinations.yaml ./data/destinations.yaml
COPY --chown=user data/kb ./data/kb
COPY --chown=user data/seeds ./data/seeds
COPY --chown=user deploy/start.sh ./deploy/start.sh

# Embed the knowledge base and fetch the retrieval models into the image.
RUN uv run tripmate index && uv run python -c "\
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
from tripmate.config import get_settings; TextCrossEncoder(get_settings().rerank_model)"

ARG GGUF_URL
RUN test -n "$GGUF_URL" || (echo "build with --build-arg GGUF_URL=<Hub file URL>" && exit 1) \
    && mkdir -p models && curl -fL "$GGUF_URL" -o models/tripmate.gguf

EXPOSE 7860
ENTRYPOINT []
CMD ["bash", "deploy/start.sh"]
