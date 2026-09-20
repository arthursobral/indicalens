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
    # Deliberately NOT "desemprego"/"desocupacao" alone - those match the
    # national PNAD Continua series above. PME only covers 6 metro areas
    # and a different age base (10+ vs 14+), so it only triggers when asked
    # for explicitly, never as a silent substitute for the national number.
    "PME_TAXA_DESEMPREGO_METROPOLITANA": ["pme", "regiões metropolitanas", "regioes metropolitanas", "região metropolitana", "regiao metropolitana"],
}

# Series whose distinguishing word can appear anywhere in the question, not
# just glued to "pobreza"/"desemprego" as one exact phrase (e.g. "pobreza
# PELA LINHA internacional" doesn't contain the literal substring "pobreza
# internacional"). Checked before the generic keyword loop below, so an
# explicit qualifier always wins over another series' bare/generic keyword
# in the same question - see docs/01-decisions.md for two real bugs this
# already caused (PME vs. PNAD Continua's "desemprego"; poverty
# internacional vs. nacional's bare "pobreza").
_PRIORITY_MATCHERS = [
    ("PME_TAXA_DESEMPREGO_METROPOLITANA", lambda q: any(kw in q for kw in _KEYWORDS["PME_TAXA_DESEMPREGO_METROPOLITANA"])),
    ("POBREZA_LINHA_INTERNACIONAL", lambda q: "pobreza" in q and "internacional" in q),
]

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
    # asking what the IBGE's commentary/analysis says is descriptive too
    "comentario", "comentário", "o que o ibge", "o que os", "dizem", "diz sobre", "analise", "análise",
    # relational questions ("A inflacao se move junto com a Selic?") are not a single-series lookup
    "se move", "junto com", "relaciona", "relacao entre", "relação entre", "correlac", "afeta", "influenc",
    "impacta", "impacto", "antecede", "depende",
    # series this agent doesn't hold: asking about them must not silently answer the IBGE series named beside them
    "selic", "juros", "cambio", "câmbio", "dolar", "dólar",
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
    priority_names = {name for name, _ in _PRIORITY_MATCHERS}
    for name, matches in _PRIORITY_MATCHERS:
        if matches(q):
            return next(s for s in SERIES if s["name"] == name)
    for series in SERIES:
        if series["name"] in priority_names:
            continue
        if any(kw in q for kw in _KEYWORDS.get(series["name"], [])):
            return series
    return None


def _extract_period(question: str, series: dict) -> tuple[str | None, bool]:
    """Parses a period the user actually named, in natural phrasing (e.g.
    "maio de 2025", not our own generated "trimestre movel encerrado em
    maio de 2025").

    Returns (period, referenced):
    - (code, True): resolved to one exact period.
    - (None, True): the question DID reference a period (e.g. a bare year
      "em 1995" for a monthly series, with no month) but not precisely
      enough to resolve one — caller must not fall back to "most recent"
      here, that would silently answer a different period than asked.
    - (None, False): no period referenced at all — caller should use the
      most recent point.

    Does NOT check whether a resolved period is among the ones fetched;
    the caller decides what to do if not.
    """
    q = question.lower()
    year = r"(19\d{2}|20\d{2})"  # IBGE series here only go back to 1979
    year_match = re.search(rf"\b{year}\b", q)

    if series["period_kind"] == "annual":
        return (year_match.group(1), True) if year_match else (None, False)

    if series["period_kind"] == "quarterly":
        match = re.search(rf"(\d)\s*[ºo°]?\s*trimestre\s*de\s*{year}", q)
        if match:
            return f"{match.group(2)}{int(match.group(1)):02d}", True
        return None, bool(year_match)  # a year alone, with no quarter, is ambiguous

    # monthly / moving_quarter: both keyed "YYYYMM", named as "<mes> de <ano>"
    for month_num, month_name in MONTHS_PT.items():
        match = re.search(rf"\b{month_name}\b[^0-9]{{0,10}}{year}", q)
        if match:
            return f"{match.group(1)}{month_num}", True
    return None, bool(year_match)  # a year alone, with no month, is ambiguous


def _mentions_multiple_periods(question: str) -> bool:
    """"IPCA em janeiro de 1998 e de 1999?" wants two data points; this
    agent only ever resolves and returns one. Silently answering just the
    first-found period would look like a complete answer when it isn't —
    counting distinct years mentioned catches this even when the second
    date elides its month ("e de 1999"), which no single regex would
    otherwise match as its own "<month> de <year>" period.
    """
    years = re.findall(r"\b(?:19\d{2}|20\d{2})\b", question.lower())
    return len(set(years)) > 1


def _format_value(series: dict, value: str) -> str:
    if series.get("unit_position") == "prefix":
        return f"{series['unit']} {value}"
    return f"{value}{series['unit']}"


def answer(question: str) -> dict | None:
    """Returns {"answer": str, "citation": str} or None if the question
    doesn't name a known series, names a specific period we don't have data
    for, or references more than one period (caller should fall back to
    vector RAG, which can pull multiple chunks into one answer, rather than
    get a silently wrong or silently incomplete answer).
    """
    series = match_series(question)
    if series is None:
        return None
    if _mentions_multiple_periods(question):
        return None

    points = fetch_series(
        series["agregado"], series["variavel"], "all", series["classificacao"],
        nivel_territorial=series.get("nivel_territorial", "N1"),
        localidade=series.get("localidade", "1"),
    )
    if not points:
        return None
    values = dict(points)

    requested, referenced = _extract_period(question, series)
    if requested is not None:
        if requested not in values:
            return None  # a specific period was named but we don't have it - don't guess
        period, value = requested, values[requested]
    elif referenced:
        return None  # a period was named (e.g. a bare year) but not precisely - don't guess
    else:
        period, value = points[-1]  # no period named at all -> most recent, decided by us

    when = format_period(series["period_kind"], period)
    return {
        "answer": f"{series['label']}, {when}: {_format_value(series, value)}.",
        "citation": f"IBGE/SIDRA tabela {series['sidra_table']} - {series['label']} - {when}",
    }
