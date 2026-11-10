"""Motor de RAG: carga da base CSV, indexação vetorial em memória e busca por cosseno."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("categoria", "pergunta", "resposta", "fonte")


class KnowledgeBaseError(RuntimeError):
    """Base de conhecimento ausente, vazia ou fora do formato esperado."""


@dataclass(frozen=True)
class PolicyMatch:
    """Um registro da base de conhecimento recuperado para uma consulta."""

    categoria: str
    pergunta: str
    resposta: str
    fonte: str
    score: float | None  # similaridade de cosseno; None quando incluído por regra (não por busca)


class Embedder(Protocol):
    """Contrato mínimo de um gerador de embeddings (permite trocar o provedor ou usar fakes em teste)."""

    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Retorna uma matriz (len(texts), dim) de float32."""
        ...


class OpenAIEmbedder:
    """Embeddings via API da OpenAI (padrão: text-embedding-3-small)."""

    def __init__(self, client, model_name: str = "text-embedding-3-small", batch_size: int = 256):
        self._client = client
        self.model_name = model_name
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = self._client.embeddings.create(model=self.model_name, input=batch)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return np.asarray(vectors, dtype=np.float32)


def load_knowledge_base(path: Path | str) -> pd.DataFrame:
    """Carrega e valida o CSV da base de conhecimento."""
    path = Path(path)
    if not path.is_file():
        raise KnowledgeBaseError(f"Base de conhecimento não encontrada: {path}")

    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KnowledgeBaseError(f"Colunas obrigatórias ausentes no CSV: {missing}")

    df = df[list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        df[column] = df[column].astype(str).str.strip()

    invalid = (df["pergunta"] == "") | (df["resposta"] == "") | (df["fonte"] == "")
    if invalid.any():
        logger.warning("Ignorando %d linha(s) sem pergunta/resposta/fonte.", int(invalid.sum()))
    df = df[~invalid].drop_duplicates(subset=["pergunta"]).reset_index(drop=True)

    if df.empty:
        raise KnowledgeBaseError("Base de conhecimento vazia após validação.")
    return df


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class RAGEngine:
    """Índice vetorial em memória sobre a coluna `pergunta` da base.

    Os vetores são normalizados (L2) na indexação, então a similaridade de cosseno
    se reduz a um produto escalar.
    """

    def __init__(
        self,
        embedder: Embedder,
        knowledge_base_path: Path | str,
        cache_dir: Path | str | None = None,
    ):
        self._embedder = embedder
        self._df = load_knowledge_base(knowledge_base_path)
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._matrix = self._build_index()
        logger.info(
            "Índice RAG pronto: %d políticas, dim=%d, modelo=%s",
            len(self._df), self._matrix.shape[1], embedder.model_name,
        )

    @property
    def size(self) -> int:
        return len(self._df)

    @property
    def fontes(self) -> frozenset[str]:
        return frozenset(self._df["fonte"])

    def _cache_file(self, perguntas: list[str]) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha256(
            (self._embedder.model_name + "\x1f" + "\x1e".join(perguntas)).encode("utf-8")
        ).hexdigest()[:20]
        return self._cache_dir / f"embeddings_{digest}.npy"

    def _build_index(self) -> np.ndarray:
        perguntas = self._df["pergunta"].tolist()
        cache_file = self._cache_file(perguntas)

        if cache_file is not None and cache_file.is_file():
            try:
                cached = np.load(cache_file)
                if cached.ndim == 2 and cached.shape[0] == len(perguntas):
                    logger.info("Embeddings carregados do cache: %s", cache_file.name)
                    return cached
            except (OSError, ValueError):
                logger.warning("Cache de embeddings inválido, recalculando: %s", cache_file)

        matrix = _l2_normalize(self._embedder.embed(perguntas))
        if matrix.shape[0] != len(perguntas):
            raise KnowledgeBaseError("Quantidade de embeddings difere da quantidade de perguntas.")

        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, matrix)
        return matrix

    def _row_to_match(self, idx: int, score: float | None) -> PolicyMatch:
        row = self._df.iloc[idx]
        return PolicyMatch(
            categoria=row["categoria"],
            pergunta=row["pergunta"],
            resposta=row["resposta"],
            fonte=row["fonte"],
            score=None if score is None else round(float(score), 4),
        )

    def search(self, query: str, top_k: int = 3) -> list[PolicyMatch]:
        """Retorna os `top_k` registros mais similares à consulta, em ordem decrescente de score."""
        if not query.strip():
            return []
        query_vec = _l2_normalize(self._embedder.embed([query]))[0]
        scores = self._matrix @ query_vec
        top_k = max(1, min(top_k, len(scores)))
        best = np.argsort(-scores, kind="stable")[:top_k]
        return [self._row_to_match(int(i), float(scores[i])) for i in best]

    def get_by_fonte(self, fonte: str) -> list[PolicyMatch]:
        """Todos os registros de uma fonte (usado para garantir que a política da regra esteja no contexto)."""
        indices = self._df.index[self._df["fonte"] == fonte]
        return [self._row_to_match(int(i), None) for i in indices]
