"""Indicator Analyst Agent: tags each ingested PNAD commentary chunk with a
theme via zero-shot classification, so future retrieval/analysis can filter
or report by topic instead of treating every chunk as undifferentiated text.

Uses MoritzLaurer/mDeBERTa-v3-base-mnli-xnli, not distilbart-mnli (the model
named in the original plan): distilbart-mnli is English-only and our corpus
is Portuguese. mDeBERTa-v3-base-mnli-xnli is trained on XNLI (multilingual,
includes Portuguese) and confirmed working on real ingested chunks.

Usage:
    python -m src.indicator_analyst
"""

import sys
from functools import lru_cache

from src import db

THEMES = ["emprego", "inflação", "PIB e atividade econômica", "renda", "pobreza", "informalidade"]

_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


@lru_cache(maxsize=1)
def _classifier():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=_MODEL)


def classify_theme(text: str) -> str:
    result = _classifier()(text, THEMES)
    return result["labels"][0]


def run() -> int:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content FROM documents "
                "WHERE source = 'IBGE-PNAD-COMENTARIOS' AND theme IS NULL"
            )
            rows = cur.fetchall()
            for doc_id, content in rows:
                cur.execute(
                    "UPDATE documents SET theme = %s WHERE id = %s",
                    (classify_theme(content), doc_id),
                )
    return len(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    n = run()
    print(f"Classificados {n} documentos por tema.")
