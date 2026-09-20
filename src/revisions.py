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
import json
import re
import sys
from pathlib import Path

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


VINTAGES_FILE = Path(__file__).resolve().parent.parent / "data" / "pib_vintages.json"
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


def parse_vintage_table(text: str) -> dict[str, float] | None:
    """Table I.1 of a booklet: year-on-year GDP growth of the LAST FIVE quarters as the IBGE had
    them on that release, e.g. {'202204': 2.7, ..., '202304': 2.1} (oldest first). Only whitespace is
    collapsed here: the digit-gluing used for the headline would merge neighbouring numbers."""
    t = re.sub(r"\s+", " ", text)
    h = re.search(r"((?:\d{4}\.(?:IV|III|II|I) ?){5})", t)
    m = re.search(r"mesmo trimestre do ano anterior < ?Anexo: ?Tabela 2 ?> ?" + " ".join([r"(-?\d+,\d)"] * 5), t)
    if not (h and m):
        return None
    quarters = re.findall(r"(\d{4})\.(IV|III|II|I)", h.group(1))
    return {f"{y}{_ROMAN[q]:02d}": float(v.replace(",", ".")) for (y, q), v in zip(quarters, m.groups())}


def parse_release_date(text: str) -> str | None:
    """ISO date from the cover ('Publicado em 29/05/2020' or 'Atualizado em 01/03/2024')."""
    m = re.search(r"(?:Publicado|Atualizado) em (\d{2})/(\d{2})/(\d{4})", text)
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None


def build_vintages() -> list[dict]:
    """[{period, released, table}] for every booklet. Cross-check: the table's last column (the
    booklet's own quarter) must equal the headline value parsed independently from the text."""
    out = []
    for period, url in list_booklets():
        pdf = PdfReader(io.BytesIO(requests.get(url, timeout=90).content))
        table = next((t for pg in pdf.pages[2:8] if (t := parse_vintage_table(pg.extract_text()))), None)
        headline = next((v for pg in pdf.pages[2:6] if (v := parse_headline(pg.extract_text())) is not None), None)
        released = parse_release_date(pdf.pages[0].extract_text())
        if table is None or released is None or table.get(period) != headline:
            raise ValueError(f"{period}: vintage table/date not parsed or disagrees with headline "
                             f"(table={table and table.get(period)}, headline={headline}, released={released})")
        out.append({"period": period, "released": released, "table": table})
    return out


def load_vintages() -> list[dict]:
    return json.loads(VINTAGES_FILE.read_text(encoding="utf-8"))


def same_age_revisions(vintages: list[dict], k: int) -> list[dict]:
    """Revision of quarter Q at a FIXED age: its value in the booklet k releases later minus its first
    release. Unlike 'current - first' this does not grow with how long ago the quarter was published.
    Event date = release of that later booklet (when the revision became public)."""
    by = {v["period"]: v for v in vintages}
    order = [v["period"] for v in vintages]
    rows = []
    for i, q in enumerate(order):
        if i + k >= len(order):
            break
        later = by[order[i + k]]
        if q in later["table"]:
            rows.append({"period": q, "first": by[q]["table"][q], "later": later["table"][q],
                         "revision": round(later["table"][q] - by[q]["table"][q], 1), "event_date": later["released"]})
    return rows


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
    if "--vintages" in sys.argv:  # refresh data/pib_vintages.json (needed when a new booklet is published)
        v = build_vintages()
        VINTAGES_FILE.parent.mkdir(exist_ok=True)
        VINTAGES_FILE.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(v)} vintages saved to {VINTAGES_FILE}")
        raise SystemExit(0)
    rows = build_revisions()
    revised = [r for r in rows if r["revision"] != 0]
    for r in rows:
        print(describe(r))
    print(f"\n{len(revised)}/{len(rows)} trimestres revisados; maior revisao: "
          f"{max(rows, key=lambda r: abs(r['revision']))['revision']:+.1f} p.p.")
    print(f"Gravados {run()} documentos.")
