"""Configuração centralizada da POC, carregada de variáveis de ambiente / arquivo .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Variável {name} deve ser numérica, recebido: {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Variável {name} deve ser inteira, recebido: {raw!r}") from exc


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.0
    similarity_threshold: float = 0.75
    top_k: int = 3
    knowledge_base_path: Path = PROJECT_ROOT / "base_conhecimento_ifood_genai-exemplo.csv"
    embeddings_cache_dir: Path | None = PROJECT_ROOT / ".cache"
    max_input_chars: int = 1000
    request_timeout_s: float = 30.0
    max_retries: int = 2
    log_level: str = "INFO"
    support_url: str = "https://suporte-interno.exemplo/reembolsos"

    def __post_init__(self) -> None:
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError("SIMILARITY_THRESHOLD deve estar no intervalo (0, 1].")
        if self.top_k < 1:
            raise ValueError("TOP_K deve ser >= 1.")
        if self.max_input_chars < 10:
            raise ValueError("MAX_INPUT_CHARS deve ser >= 10.")
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError("LLM_TEMPERATURE deve estar entre 0 e 2.")

    @classmethod
    def from_env(cls) -> "Settings":
        cache_raw = os.getenv("EMBEDDINGS_CACHE_DIR", ".cache").strip()
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            embedding_model=_env_str("EMBEDDING_MODEL", cls.embedding_model),
            chat_model=_env_str("CHAT_MODEL", cls.chat_model),
            llm_temperature=_env_float("LLM_TEMPERATURE", cls.llm_temperature),
            similarity_threshold=_env_float("SIMILARITY_THRESHOLD", cls.similarity_threshold),
            top_k=_env_int("TOP_K", cls.top_k),
            knowledge_base_path=_resolve_path(
                _env_str("KNOWLEDGE_BASE_PATH", "base_conhecimento_ifood_genai-exemplo.csv")
            ),
            embeddings_cache_dir=_resolve_path(cache_raw) if cache_raw else None,
            max_input_chars=_env_int("MAX_INPUT_CHARS", cls.max_input_chars),
            request_timeout_s=_env_float("REQUEST_TIMEOUT_S", cls.request_timeout_s),
            max_retries=_env_int("MAX_RETRIES", cls.max_retries),
            log_level=_env_str("LOG_LEVEL", cls.log_level).upper(),
            support_url=_env_str("SUPPORT_URL", cls.support_url),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
