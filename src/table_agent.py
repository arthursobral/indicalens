"""Table Agent: answers numeric questions about IBGE series with a direct,
deterministic lookup instead of a table-QA model. See docs/01-decisions.md
for why: google/tapas-base-finetuned-wtq (the model the original plan named)
returned an EMPTY answer on a Portuguese question/table (its BERT backbone
is English-only) and, even in English, got a "most recent value" question
wrong (misfired into an AVERAGE aggregator). Our "tables" are simple
period-to-value series, so a parsed lookup is both more reliable and
simpler than forcing a mis-fit table-QA model into it.
"""

import re

from src.ibge_client import MONTHS_PT, SERIES, fetch_series, format_period

_KEYWORDS = {
    "IPCA_VARIACAO_MENSAL": ["ipca", "inflação", "inflacao"],
    "PIB_VARIACAO_TRIMESTRAL": ["pib"],
    "PNAD_TAXA_DESOCUPACAO": ["desocupação", "desocupacao", "desemprego"],
    "PNAD_RENDIMENTO_MEDIO_REAL": ["rendimento", "renda", "salário", "salario"],
    "PNAD_TAXA_INFORMALIDADE": ["informalidade", "informal"],
    "POBREZA_LINHA_INTERNACIONAL": ["pobreza internacional"],
    "POBREZA_LINHA_NACIONAL": ["pobreza nacional", "pobreza"],
}

# A question naming an indicator isn't necessarily a plain value lookup —
# "como a desocupação varia por nível de instrução?" names a series but
# needs the descriptive RAG path, not a bare number. Any of these markers
# means "defer to RAG", regardless of series match. Deliberately narrow
# (e.g. "varia por", not bare "varia" or "como ") to avoid misfiring on
# ordinary phrasings like "como esta a taxa de desemprego?".
_ANALYTICAL_MARKERS = [
    "varia por", "por que", "porque", "quais fatores", "quais os fatores",
    "distribui", "explica", "compara", "por nivel", "por nível",
    "por regiao", "por região", "por sexo", "por raça", "por raca",
]


def match_series(question: str) -> dict | None:
    """Keyword match only — no embeddings/LLM involved, so a miss here just
    falls back to the vector RAG path rather than guessing. Also backs off
    on questions that look analytical/descriptive rather than a plain
    value lookup, even if they name a known series.
    """
    q = question.lower()
    if any(marker in q for marker in _ANALYTICAL_MARKERS):
        return None
    for series in SERIES:
        if any(kw in q for kw in _KEYWORDS.get(series["name"], [])):
            return series
    return None


def _extract_period(question: str, series: dict) -> str | None:
    """Parses a period the user actually named, in natural phrasing (e.g.
    "maio de 2025", not our own generated "trimestre movel encerrado em
    maio de 2025"). Returns None if the question doesn't name one at all —
    the caller then uses the most recent point. Does NOT check whether that
    period is among the ones fetched; the caller decides what to do if not
    (must not silently substitute a different period).
    """
    q = question.lower()
    year = r"(19\d{2}|20\d{2})"  # IBGE series here only go back to 1979
    if series["period_kind"] == "annual":
        match = re.search(rf"\b{year}\b", q)
        return match.group(1) if match else None
    if series["period_kind"] == "quarterly":
        match = re.search(rf"(\d)\s*[ºo°]?\s*trimestre\s*de\s*{year}", q)
        return f"{match.group(2)}{int(match.group(1)):02d}" if match else None
    # monthly / moving_quarter: both keyed "YYYYMM", named as "<mes> de <ano>"
    for month_num, month_name in MONTHS_PT.items():
        match = re.search(rf"\b{month_name}\b[^0-9]{{0,10}}{year}", q)
        if match:
            return f"{match.group(1)}{month_num}"
    return None


def _format_value(series: dict, value: str) -> str:
    if series.get("unit_position") == "prefix":
        return f"{series['unit']} {value}"
    return f"{value}{series['unit']}"


def answer(question: str) -> dict | None:
    """Returns {"answer": str, "citation": str} or None if the question
    doesn't name a known series, or names a specific period we don't have
    data for (caller should fall back to vector RAG rather than get a
    silently wrong answer for a different period).
    """
    series = match_series(question)
    if series is None:
        return None

    points = fetch_series(series["agregado"], series["variavel"], "all", series["classificacao"])
    if not points:
        return None
    values = dict(points)

    requested = _extract_period(question, series)
    if requested is None:
        period, value = points[-1]  # no period named -> most recent, decided by us, not a model
    elif requested in values:
        period, value = requested, values[requested]
    else:
        return None  # a specific period was named but we don't have it - don't guess

    when = format_period(series["period_kind"], period)
    return {
        "answer": f"{series['label']}, {when}: {_format_value(series, value)}.",
        "citation": f"IBGE/SIDRA tabela {series['sidra_table']} - {series['label']} - {when}",
    }
