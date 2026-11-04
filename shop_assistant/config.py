import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from psycopg.conninfo import make_conninfo

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    database_url: str = ""
    store_backend: str = "postgres"
    ollama_base_url: str = "http://ollama:11434"
    chat_model: str = "qwen3:4b"
    embedding_model: str = "embeddinggemma"
    telegram_bot_token: str = ""
    telegram_proxy_url: str = ""
    model_timeout_seconds: float = 180
    retrieval_top_k: int = 6
    retrieval_min_score: float = 0.30
    max_question_length: int = 1500
    user_cooldown_seconds: float = 3

    def __post_init__(self):
        if self.store_backend != "postgres":
            raise ValueError("This edition uses PostgreSQL/pgvector")
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")
        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must be an HTTP URL")
        if not 1 <= self.retrieval_top_k <= 20:
            raise ValueError("RETRIEVAL_TOP_K must be between 1 and 20")
        if not -1 <= self.retrieval_min_score <= 1:
            raise ValueError("RETRIEVAL_MIN_SCORE must be between -1 and 1")
        if self.model_timeout_seconds <= 0 or self.user_cooldown_seconds < 0:
            raise ValueError("Timeout must be positive and cooldown nonnegative")
        if not 1 <= self.max_question_length <= 3000:
            raise ValueError("MAX_QUESTION_LENGTH must be between 1 and 3000")

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        return cls(
            database_url=os.getenv("DATABASE_URL") or make_conninfo(
                host=os.getenv("PGHOST", "db"), dbname=os.getenv("PGDATABASE", "shop"),
                user=os.getenv("PGUSER", "shop"), password=os.getenv("PGPASSWORD", "shop_local_demo"),
                port=os.getenv("PGPORT", "5432")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/"),
            chat_model=os.getenv("CHAT_MODEL", "qwen3:4b"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "embeddinggemma"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_proxy_url=os.getenv("TELEGRAM_PROXY_URL", ""),
            model_timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "180")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "6")),
            retrieval_min_score=float(os.getenv("RETRIEVAL_MIN_SCORE", "0.30")),
            max_question_length=int(os.getenv("MAX_QUESTION_LENGTH", "1500")),
            user_cooldown_seconds=float(os.getenv("USER_COOLDOWN_SECONDS", "3")),
        )
