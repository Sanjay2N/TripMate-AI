from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Everything is overridable with TRIPMATE_* env vars (or a .env file)."""

    model_config = SettingsConfigDict(env_prefix="TRIPMATE_", env_file=".env", extra="ignore")

    data_dir: Path = ROOT / "data"
    # Kaggle "Indian Tourism Dataset" (destination profiles); optional
    profiles_file: Path = ROOT / "RAG data" / "india_tourism_dataset.json"

    # Retrieval
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank: bool = True
    collection: str = "kb"

    # Serving model: any OpenAI-compatible endpoint (Ollama locally, llama-server when deployed)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "tripmate-base"
    llm_api_key: str = "local"

    # Teacher for synthetic data and judging; same protocol, so it can be swapped for an API model
    teacher_base_url: str = "http://localhost:11434/v1"
    teacher_model: str = "tripmate-teacher"
    teacher_api_key: str = "local"
    # Local "thinking" models spend minutes reasoning before answering unless told not to.
    # Set to empty for endpoints that reject the parameter.
    teacher_reasoning_effort: str = "none"

    @property
    def finetune_dir(self) -> Path:
        return ROOT / "finetune"

    @property
    def kb_dir(self) -> Path:
        return self.data_dir / "kb"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


@lru_cache
def get_settings() -> Settings:
    return Settings()
