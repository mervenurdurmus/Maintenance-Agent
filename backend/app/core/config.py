from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()
BACKEND_DIR = Path(__file__).resolve().parents[2]
RUNTIME_SETTINGS_PATH = BACKEND_DIR / "data" / "runtime_settings.json"
RUNTIME_SETTING_KEYS = {
    "chat_llm_provider",
    "chat_llm_model",
}


class Settings(BaseSettings):
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")
    groq_vision_model: str = Field(default="qwen/qwen3.6-27b", alias="GROQ_VISION_MODEL")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="openai/gpt-4o-mini",
        alias="OPENROUTER_MODEL",
    )
    chat_llm_provider: str = Field(default="groq", alias="CHAT_LLM_PROVIDER")
    chat_llm_model: str = Field(default="", alias="CHAT_LLM_MODEL")
    ragas_llm_provider: str = Field(default="groq", alias="RAGAS_LLM_PROVIDER")
    embedding_model: str = Field(
        default="ggml-org/embeddinggemma-300M-GGUF",
        alias="EMBEDDING_MODEL",
    )
    embedding_provider: str = Field(
        default="llama_server",
        alias="EMBEDDING_PROVIDER",
    )
    embedding_endpoint: str = Field(
        default="http://127.0.0.1:8080/embedding",
        alias="EMBEDDING_ENDPOINT",
    )
    embedding_timeout_seconds: float = Field(
        default=60.0,
        alias="EMBEDDING_TIMEOUT_SECONDS",
    )
    embedding_collection_name: str = Field(
        default="maintenance_documents_embeddinggemma_300m_v1",
        alias="EMBEDDING_COLLECTION_NAME",
    )
    chroma_persist_dir: Path = Field(
        default=BACKEND_DIR / "data" / "chroma",
        alias="CHROMA_PERSIST_DIR",
    )
    upload_dir: Path = Field(
        default=BACKEND_DIR / "data" / "uploads",
        alias="UPLOAD_DIR",
    )
    chat_image_dir: Path = Field(
        default=BACKEND_DIR / "data" / "chat_images",
        alias="CHAT_IMAGE_DIR",
    )
    top_k: int = Field(default=8, alias="TOP_K")
    rerank_top_n: int = Field(default=4, alias="RERANK_TOP_N")
    chunk_size: int = Field(default=350, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    model_config = SettingsConfigDict(populate_by_name=True)


def _load_runtime_settings() -> dict[str, Any]:
    if not RUNTIME_SETTINGS_PATH.exists():
        return {}

    try:
        payload = json.loads(RUNTIME_SETTINGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, dict):
        return {}

    return {
        key: value
        for key, value in payload.items()
        if key in RUNTIME_SETTING_KEYS and isinstance(value, str)
    }


def _apply_runtime_settings(settings: Settings) -> Settings:
    for key, value in _load_runtime_settings().items():
        setattr(settings, key, value)
    return settings


def update_runtime_settings(values: dict[str, str]) -> Settings:
    RUNTIME_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    current = _load_runtime_settings()
    current.update(
        {
            key: value
            for key, value in values.items()
            if key in RUNTIME_SETTING_KEYS
        }
    )
    RUNTIME_SETTINGS_PATH.write_text(
        json.dumps(current, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    get_settings.cache_clear()
    return get_settings()


@lru_cache
def get_settings() -> Settings:
    settings = _apply_runtime_settings(Settings())
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chat_image_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return settings
