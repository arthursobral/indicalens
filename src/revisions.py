"""Revision Agent: how much has the IBGE revised quarterly GDP since it was
first published?

The API only returns the current (already revised) value of each series, so
first-release values come from the "Contas Nacionais Trimestrais" booklet
the IBGE publishes for each release (ftp.ibge.gov.br, one PDF per quarter,
2020 onward), whose headline states the year-on-year growth as announced
that day. Revision = current API value - first-release value.

Scope is GDP on purpose: it is the series the IBGE actually revises, and the
only one with a per-release archive. IPCA is not revised; PNAD moving
quarters have no per-release archive here.

Usage:
    python -m src.revisions
"""

import io
import re
import sys

import requests
from pypdf import PdfReader

from src import db
from src.embeddings import embed
from src.ibge_client import SERIES, fetch_series, format_period

BASE = "https://ftp.ibge.gov.br/Contas_Nacionais/Contas_Nacionais_Trimestrais/Fasciculo_Indicadores_IBGE"
_PIB = next(s for s in SERIES if s["name"] == "PIB_VARIACAO_TRIMESTRAL")

_NEGATIVE = ("negativ", "queda", "retra", "contra", "recuo", "declínio", "declinio")


def list_booklets() -> list[tuple[str, str]]:
    """[(period 'YYYYQQ', url)], oldest first. 2026 files sit in the folder
    root; earlier years have their own subfolder.
    """
    found = []
    for sub in [""] + [f"{y}/" for y in range(2020, 2026)]:
        html = requests.get(f"{BASE}/{sub}", timeout=30).text
        for name in re.findall(r'href="(pib-vol-val_(\d{6})caderno\.pdf)"', html):
            found.append((name[1], f"{BASE}/{sub}{name[0]}"))
    return sorted(set(found))


def parse_headline(text: str) -> float | None:
    """Year-on-year growth (%) from the booklet headline, e.g. 'Na comparação
    com igual período de 2025, houve crescimento do PIB de 1,8%'. The PDF
    text has stray spaces inside numbers ('201 9'), collapsed first.
    """
    text = re.sub(r"\s+", " ", text)
    while re.search(r"(\d) (\d)", text):
        text = re.sub(r"(\d) (\d)", r"\1\2", text)
    m = re.search(r"igual per[ií]odo de \d{4},? ([^%]{0,90}?)(\d+,\d)%", text)
    if not m:
        return None
    value = float(m.group(2).replace(",", "."))
    return -value if any(w in m.group(1).lower() for w in _NEGATIVE) else value


def first_release_values() -> dict[str, float]:
    values = {}
    for period, url in list_booklets():
        pdf = PdfReader(io.BytesIO(requests.get(url, timeout=60).content))
        for page in pdf.pages[2:6]:
            value = parse_headline(page.extract_text())
            if value is not None:
                values[period] = value
                break
    return values


def build_revisions() -> list[dict]:
    """One dict per quarter with a booklet: period, first, current, revision
    (percentage points, current - first). Machine-readable on purpose: the
    Correlation Agent will regress on `revision`.
    """
    current = dict(fetch_series(_PIB["agregado"], _PIB["variavel"], "all", _PIB["classificacao"]))
    rows = []
    for period, first in first_release_values().items():
        if period in current:
            now = float(current[period])
            rows.append({"period": period, "first": first, "current": now, "revision": round(now - first, 1)})
    return rows


def describe(r: dict) -> str:
    when = format_period("quarterly", r["period"])
    if r["revision"] == 0:
        change = "sem revisao"
    else:
        change = f"revisado {'para cima' if r['revision'] > 0 else 'para baixo'} em {abs(r['revision']):.1f} p.p."
    return (
        f"PIB (variacao contra o mesmo trimestre do ano anterior), {when}: primeira divulgacao "
        f"{r['first']}%, valor atual {r['current']}% - {change}."
    )


def answer(question: str) -> dict | None:
    """Deterministic lookup of a stored revision for a named quarter (the
    documents are written by run()). None if the question isn't about GDP
    revisions or doesn't name one quarter; the RAG path handles the rest.
    """
    from src.table_agent import _extract_period, _mentions_multiple_periods

    q = question.lower()
    if "revis" not in q or "pib" not in q or _mentions_multiple_periods(question):
        return None
    period, _ = _extract_period(question, _PIB)
    if period is None:
        return None
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT content, citation FROM documents WHERE series = 'PIB_REVISAO' AND period = %s", (period,))
        row = cur.fetchone()
    return {"answer": row[0], "citation": row[1]} if row else None


def run() -> int:
    rows = build_revisions()
    docs = [
        {
            "source": "IBGE-PIB-REVISAO",
            "series": "PIB_REVISAO",
            "period": r["period"],
            "content": describe(r),
            "citation": f"IBGE - Contas Nacionais Trimestrais (caderno de {format_period('quarterly', r['period'])}) vs. SIDRA tabela 5932",
        }
        for r in rows
    ]
    for doc, vec in zip(docs, embed([d["content"] for d in docs])):
        doc["embedding"] = vec
    with db.connect() as conn:
        db.upsert_documents(conn, docs)
    return len(docs)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rows = build_revisions()
    revised = [r for r in rows if r["revision"] != 0]
    for r in rows:
        print(describe(r))
    print(f"\n{len(revised)}/{len(rows)} trimestres revisados; maior revisao: "
          f"{max(rows, key=lambda r: abs(r['revision']))['revision']:+.1f} p.p.")
    print(f"Gravados {run()} documentos.")
