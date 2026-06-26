from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    chroma_persist_dir: Path = Field(default=Path("backend/data/chroma"), alias="CHROMA_PERSIST_DIR")
    upload_dir: Path = Field(default=Path("backend/data/uploads"), alias="UPLOAD_DIR")
    top_k: int = Field(default=6, alias="TOP_K")
    rerank_top_n: int = Field(default=3, alias="RERANK_TOP_N")

    model_config = SettingsConfigDict(populate_by_name=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return settings
