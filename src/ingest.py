"""Week 1 ingestion: pull registered IBGE series, turn each data point into a
short cited sentence, embed it, and upsert into the pgvector `documents` table.

Usage:
    python -m src.ingest [periodos]

`periodos` follows the IBGE API convention, e.g. "-24" for the last 24
periods of each series (default).
"""

import sys

from src import db
from src.embeddings import embed
from src.ibge_client import SERIES, fetch_series, format_period


def build_rows(periodos: str = "-24") -> list[dict]:
    rows = []
    for series in SERIES:
        points = fetch_series(
            series["agregado"], series["variavel"], periodos,
            classificacao=series["classificacao"],
        )
        for period, value in points:
            when = format_period(series["period_kind"], period)
            formatted_value = (
                f"{series['unit']} {value}" if series.get("unit_position") == "prefix"
                else f"{value}{series['unit']}"
            )
            content = (
                f"Segundo o IBGE (SIDRA, tabela {series['sidra_table']}), "
                f"{series['label']} em {when} foi de {formatted_value}."
            )
            citation = f"IBGE/SIDRA tabela {series['sidra_table']} - {series['label']} - {when}"
            rows.append({
                "source": "IBGE-SIDRA",
                "series": series["name"],
                "period": period,
                "content": content,
                "citation": citation,
            })
    return rows


def run(periodos: str = "-24") -> int:
    rows = build_rows(periodos)
    embeddings = embed([r["content"] for r in rows])
    for row, vec in zip(rows, embeddings):
        row["embedding"] = vec

    with db.connect() as conn:
        db.upsert_documents(conn, rows)

    return len(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    periodos = sys.argv[1] if len(sys.argv) > 1 else "-24"
    n = run(periodos)
    print(f"Ingeridos/atualizados {n} documentos.")
