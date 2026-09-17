"""Thin psycopg2 wrapper around the Supabase/pgvector `documents` table."""

import contextlib

import psycopg2

from src.config import DATABASE_URL


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in embedding) + "]"


@contextlib.contextmanager
def connect():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_documents(conn, rows: list[dict]) -> None:
    """Each row: {source, series, period, content, citation, embedding}."""
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO documents (source, series, period, content, citation, embedding)
                VALUES (%(source)s, %(series)s, %(period)s, %(content)s, %(citation)s, %(embedding)s::vector)
                ON CONFLICT (series, period) DO UPDATE SET
                    content = EXCLUDED.content,
                    citation = EXCLUDED.citation,
                    embedding = EXCLUDED.embedding,
                    created_at = now()
                """,
                {**row, "embedding": _vector_literal(row["embedding"])},
            )


def retrieve(conn, query_embedding: list[float], k: int = 5) -> list[dict]:
    vec = _vector_literal(query_embedding)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, citation, 1 - (embedding <=> %s::vector) AS score
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vec, vec, k),
        )
        return [{"content": r[0], "citation": r[1], "score": r[2]} for r in cur.fetchall()]
