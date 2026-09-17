"""Local embedding model (all-MiniLM-L6-v2), loaded once per process."""

from functools import lru_cache

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, normalize_embeddings=True).tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
